"""Unlearning: das Ausgangsmodell (fakten.pt) vergisst das Forget-Set von Thema 2.

Aufruf:  python -m de_llm.unlearning --szenario forget_person --methode npo
         python -m de_llm.unlearning --szenario forget_attribute --methode gd --schritte 60
         python -m de_llm.unlearning --start checkpoints/referenz_ohne_p001/fakten.pt --schritte 0
                                                   # nur auswerten (hier: das Referenzmodell)

Methoden (Forget-Term auf den Antworten des Forget-Sets, dazu der normale Verlust auf dem Retain-Set):
  ga   Gradient Ascent: den Verlust auf den Forget-Antworten erhöhen, ohne Retain-Term. Einfachste
       Baseline; zerstört schnell auch anderes Wissen.
  gd   Gradient Difference: ga plus Retain-Verlust.
  npo  Negative Preference Optimization (Zhang et al. 2024) plus Retain-Verlust: Der Forget-Term vergleicht
       mit dem eingefrorenen Startmodell und flacht ab, sobald die Antwort unwahrscheinlich geworden ist.
       Dadurch stabiler als ga/gd. Standard.

Jeder Schritt sieht das ganze Forget-Set (ohne Kontext, wie es im Gedächtnis abgefragt wird) und einen
Retain-Batch: die übrigen Fakten derselben Person (bei forget_attribute), zufällige andere Fakten und
kurze GermanQuAD-Beispiele. Mit --paraphrasen N kommen umformulierte Forget-Fragen dazu (TRAIN-Vorlagen).

Auswertung danach (auch einzeln mit --schritte 0):
  * Zielfragen: Forget-Fragen (Ziel: vergessen) und bei forget_attribute die übrigen Fakten der Person
    (Ziel: behalten), je in Originalformulierung, TRAIN- und TEST-Umformulierungen (de_llm.paraphrasen).
    Gemessen: Antwort richtig? und Wahrscheinlichkeit der richtigen Antwort (pro Token gemittelt).
  * Gleiche Messung für Start- und Referenzmodell, damit man sieht, wo das Ergebnis liegen sollte.
  * Retain: private Fakten ohne Kontext und GermanQuAD-F1 mit Kontext.
Ergebnis: <ziel>/modell.pt und <ziel>/bericht.json.
"""
import argparse
import json
import random
import time
from pathlib import Path

import tiktoken
import torch
import torch.nn.functional as F

from . import paraphrasen, pfade
from .daten import collate, pipeline, prompt_text
from .evaluieren import antworte, bewerte, lade_modell, werte_aus
from .modell import versteckte_zustaende, waehle_geraet

RETAIN = {"forget_person": "retain_person.json", "forget_attribute": "retain_attribute.json"}
REFERENZ = {"forget_person": "referenz_ohne_p001", "forget_attribute": "referenz_ohne_p001_wohnort"}


def lade(datei: str) -> list[dict]:
    return pipeline.s1.lade_datensatz(pfade.UNLEARNING_DATEN / datei)


def sequenz_nll(modell, eingabe: torch.Tensor, ziel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Summe der negativen Log-Likelihood der Antwort-Tokens je Sequenz, dazu die Tokenanzahl je Sequenz."""
    maske = ziel != -100
    logits = modell.out_head(versteckte_zustaende(modell, eingabe)[maske])
    nll = F.cross_entropy(logits, ziel[maske], reduction="none")
    zeile = maske.nonzero()[:, 0]
    summe = torch.zeros(eingabe.shape[0], device=eingabe.device).index_add(0, zeile, nll)
    return summe, maske.sum(dim=1)


def beispiele(eintraege: list[dict], tokenizer, modus: str) -> list[dict]:
    aus = []
    for e in eintraege:
        b, status = pipeline.s4.verarbeite(e, tokenizer, 1024, modus)
        if b is None:
            raise ValueError(f"{e['id']} nicht verarbeitbar: {status}")
        aus.append(b)
    return aus


def zielfragen(szenario: str) -> list[dict]:
    """Forget-Einträge (Ziel vergessen) und übrige Fakten derselben Personen (Ziel behalten)."""
    vergessen = lade(f"{szenario}.json")
    personen = {e["person_id"] for e in vergessen}
    behalten = [e for e in lade(RETAIN[szenario]) if e.get("person_id") in personen]
    return [dict(e, ziel="vergessen") for e in vergessen] + [dict(e, ziel="behalten") for e in behalten]


@torch.no_grad()
def antwort_wahrscheinlichkeit(modell, tokenizer, frage: str, antwort: str) -> float:
    """Mittlere Wahrscheinlichkeit pro Token, dass das Modell ohne Kontext genau diese Antwort gibt."""
    geraet = next(modell.parameters()).device
    prompt = tokenizer.encode(prompt_text(frage))
    ids = prompt + tokenizer.encode(antwort)
    eingabe = torch.tensor([ids[:-1]], device=geraet)
    ziel = torch.tensor([[-100] * (len(prompt) - 1) + ids[len(prompt):]], device=geraet)
    nll, n = sequenz_nll(modell, eingabe, ziel)
    return torch.exp(-nll / n).item()


def werte_zielfragen_aus(modell, tokenizer, ziele: list[dict]) -> list[dict]:
    zeilen = []
    for e in ziele:
        antwort = e["answers"]["text"][0]
        gruppen = [("original", [e["question"]]), ("train", paraphrasen.varianten(e, "train")),
                   ("test", paraphrasen.varianten(e, "test"))]
        for gruppe, fragen in gruppen:
            for frage in fragen:
                vorhersage = antworte(modell, tokenizer, frage, max_neue_tokens=16)
                zeilen.append({"id": e["id"], "ziel": e["ziel"], "gruppe": gruppe, "frage": frage,
                               "erwartet": antwort, "vorhersage": vorhersage,
                               "richtig": bewerte(vorhersage, [antwort])[0] == 1.0,
                               "p_antwort": antwort_wahrscheinlichkeit(modell, tokenizer, frage, antwort)})
    return zeilen


def fasse_zusammen(zeilen: list[dict]) -> dict:
    """Je Ziel (vergessen/behalten) und Gruppe: Anteil richtig und mittlere Antwortwahrscheinlichkeit."""
    aus = {}
    for ziel in ("vergessen", "behalten"):
        for gruppe in ("original", "train", "test"):
            teil = [z for z in zeilen if z["ziel"] == ziel and z["gruppe"] == gruppe]
            if teil:
                aus[f"{ziel}/{gruppe}"] = {"n": len(teil), "richtig": sum(z["richtig"] for z in teil) / len(teil),
                                           "p_antwort": sum(z["p_antwort"] for z in teil) / len(teil)}
    return aus


def drucke_vergleich(zusammenfassungen: dict[str, dict]) -> None:
    namen = list(zusammenfassungen)
    print(f"{'Zielfragen':22}" + "".join(f"{n:>24}" for n in namen))
    for schluessel in next(iter(zusammenfassungen.values())):
        werte = [zusammenfassungen[n].get(schluessel) for n in namen]
        print(f"{schluessel:22}" + "".join(
            f"{w['richtig']:>10.0%} richtig, p {w['p_antwort']:.2f}" if w else f"{'-':>24}" for w in werte))


def verlerne(start: Path, szenario: str, methode: str, schritte: int, lernrate: float, beta: float,
             retain_fakten: int, retain_quad: int, paraphrasen_anzahl: int, seed: int, geraet):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    tokenizer = tiktoken.get_encoding("gpt2")
    modell = lade_modell(str(start), geraet).train()
    referenz = lade_modell(str(start), geraet) if methode == "npo" else None  # eingefroren, eval-Modus

    vergessen = lade(f"{szenario}.json")
    fragen = list(vergessen)
    for e in vergessen:
        fragen += [paraphrasen.mit_frage(e, f, i) for i, f in
                   enumerate(paraphrasen.varianten(e, "train")[:paraphrasen_anzahl], 1)]
    forget = collate(beispiele(fragen, tokenizer, "frage"))

    retain = lade(RETAIN[szenario])
    personen = {e["person_id"] for e in vergessen}
    nachbarn = beispiele([e for e in retain if e.get("person_id") in personen], tokenizer, "frage")
    fakten = beispiele([e for e in retain if e["is_private"] and e.get("person_id") not in personen],
                       tokenizer, "frage")
    quad = [b for b in beispiele([e for e in retain if not e["is_private"]][:3000], tokenizer, "kontext_frage")
            if len(b["input_ids"]) < 400]
    print(f"Gerät: {geraet} | {szenario} mit {methode} | Forget: {len(fragen)} Fragen | Retain je Schritt: "
          f"{len(nachbarn)} Nachbarfakten + {retain_fakten} Fakten + {retain_quad} GermanQuAD")

    optimizer = torch.optim.AdamW(modell.parameters(), lr=lernrate, weight_decay=0.0)
    t0 = time.time()
    for schritt in range(1, schritte + 1):
        eingabe, ziel = forget[0].to(geraet), forget[1].to(geraet)
        nll, n = sequenz_nll(modell, eingabe, ziel)
        if methode == "npo":
            with torch.no_grad():
                nll_ref, _ = sequenz_nll(referenz, eingabe, ziel)
            # log pi_theta - log pi_ref = nll_ref - nll; Verlust 2/beta * log(1 + (pi_theta/pi_ref)^beta)
            forget_verlust = -(2 / beta) * F.logsigmoid(-beta * (nll_ref - nll)).mean()
        else:
            forget_verlust = -nll.sum() / n.sum()   # Gradient Ascent auf dem Token-Mittel
        verlust = forget_verlust
        retain_verlust = torch.tensor(0.0)
        if methode != "ga":
            stapel = nachbarn + rng.sample(fakten, retain_fakten)
            mikro = [collate(stapel)] + [collate(rng.sample(quad, 2)) for _ in range(retain_quad // 2)]
            r_nll = sum(sequenz_nll(modell, e.to(geraet), z.to(geraet))[0].sum() for e, z in mikro)
            retain_verlust = r_nll / sum((z != -100).sum().item() for _, z in mikro)
            verlust = verlust + retain_verlust
        verlust.backward()
        torch.nn.utils.clip_grad_norm_(modell.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if geraet.type == "mps":
            torch.mps.empty_cache()
        if schritt % 10 == 0 or schritt == schritte:
            p = torch.exp(-nll.detach() / n).mean().item()
            print(f"Schritt {schritt}/{schritte} | Forget-Term {forget_verlust.item():.3f} | "
                  f"Retain-Verlust {retain_verlust.item():.3f} | p(Forget-Antwort) {p:.3f} | "
                  f"{time.time() - t0:.0f} s")
    return modell.eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", default=str(pfade.CHECKPOINTS / "germanquad_124m" / "fakten.pt"))
    parser.add_argument("--szenario", default="forget_person", choices=list(RETAIN))
    parser.add_argument("--methode", default="npo", choices=["npo", "gd", "ga"])
    parser.add_argument("--schritte", type=int, default=40, help="0 = nur auswerten")
    parser.add_argument("--lernrate", type=float, default=1e-5)
    parser.add_argument("--beta", type=float, default=0.1, help="nur npo")
    parser.add_argument("--retain-fakten", type=int, default=16, help="andere private Fakten je Schritt")
    parser.add_argument("--retain-quad", type=int, default=4, help="GermanQuAD-Beispiele je Schritt")
    parser.add_argument("--paraphrasen", type=int, default=0, help="umformulierte Forget-Fragen je Fakt (0-4)")
    parser.add_argument("--referenz", help="Standard: checkpoints/<referenz des Szenarios>/fakten.pt, falls vorhanden")
    parser.add_argument("--ziel", help="Standard: checkpoints/unlearning/<szenario>_<methode>/")
    parser.add_argument("--anzahl", type=int, default=300, help="GermanQuAD-Beispiele für die Auswertung (0 = keine)")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    geraet = waehle_geraet()
    tokenizer = tiktoken.get_encoding("gpt2")
    start = Path(args.start)
    ziel = Path(args.ziel) if args.ziel else (
        pfade.CHECKPOINTS / "unlearning" / (f"{args.szenario}_{args.methode}" if args.schritte else
                                            f"{args.szenario}_auswertung_{start.parent.name}"))
    ziel.mkdir(parents=True, exist_ok=True)

    if args.schritte:
        modell = verlerne(start, args.szenario, args.methode, args.schritte, args.lernrate, args.beta,
                          args.retain_fakten, args.retain_quad, args.paraphrasen, args.seed, geraet)
        stand = torch.load(start, map_location="cpu", weights_only=False)
        stand.update(modell=modell.state_dict(), unlearning=vars(args))
        torch.save(stand, ziel / "modell.pt")
        print(f"Gespeichert: {ziel / 'modell.pt'}")
    else:
        modell = lade_modell(str(start), geraet)

    # Zielfragen für Startmodell, Ergebnis und Referenzmodell
    ziele = zielfragen(args.szenario)
    referenz = Path(args.referenz) if args.referenz else pfade.CHECKPOINTS / REFERENZ[args.szenario] / "fakten.pt"
    modelle = {"nachher" if args.schritte else "modell": modell}
    if args.schritte:
        modelle = {"vorher": lade_modell(str(start), geraet), **modelle}
    if referenz.exists() and referenz.resolve() != start.resolve():
        modelle["referenz"] = lade_modell(str(referenz), geraet)
    else:
        print(f"Kein Referenzmodell unter {referenz}, Vergleich entfällt.")
    zeilen = {name: werte_zielfragen_aus(m, tokenizer, ziele) for name, m in modelle.items()}
    zusammen = {name: fasse_zusammen(z) for name, z in zeilen.items()}
    drucke_vergleich(zusammen)
    for z in zeilen["nachher" if args.schritte else "modell"]:
        if z["gruppe"] == "original":
            print(f"  [{z['ziel']}] {z['frage']} -> {z['vorhersage']!r} (erwartet {z['erwartet']!r}, "
                  f"p {z['p_antwort']:.2f})")

    # Retain: hat das Modell den Rest behalten?
    retain_privat = [e for e in lade(RETAIN[args.szenario]) if e["is_private"]]
    fakten = werte_aus(modell, tokenizer, retain_privat, "frage", max_neue_tokens=16)
    print(f"Retain: private Fakten ohne Kontext ({fakten['anzahl']}): EM {fakten['exact_match']:.1%}")
    qa = None
    if args.anzahl:
        val = [e for e in pipeline.s1.lade_datensatz(pfade.DATEN / "validation.json") if not e["is_private"]]
        qa = werte_aus(modell, tokenizer, val[:args.anzahl], "kontext_frage")
        print(f"Retain: GermanQuAD validation mit Kontext ({qa['anzahl']}): EM {qa['exact_match']:.1%} | "
              f"F1 {qa['f1']:.1%}")

    bericht = {"start": str(start), "referenz": str(referenz) if "referenz" in modelle else None,
               "einstellungen": vars(args), "zielfragen": zusammen, "zielfragen_details": zeilen,
               "retain_fakten": {k: v for k, v in fakten.items() if k != "beispiele"},
               "germanquad": {k: v for k, v in qa.items() if k != "beispiele"} if qa else None}
    (ziel / "bericht.json").write_text(json.dumps(bericht, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Bericht: {ziel / 'bericht.json'}")


if __name__ == "__main__":
    main()
