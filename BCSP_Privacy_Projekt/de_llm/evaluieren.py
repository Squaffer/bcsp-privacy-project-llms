"""Auswertung eines Checkpoints: QA-Qualität und Abrufbarkeit der privaten Fakten.

Aufruf:  python -m de_llm.evaluieren --checkpoint checkpoints/germanquad_124m/bester.pt
         python -m de_llm.evaluieren --checkpoint gpt2      # unverändertes GPT-2 als Vergleich

Drei Messungen:
  1. GermanQuAD (validation oder test) mit Kontext: Exact Match und F1 wie bei SQuAD.
  2. Private Fakten aus train.json OHNE Kontext: wie viele das Modell auswendig weiß.
  3. Die Fragen aus unlearning_data/unlearning_evaluation.json (Ausgangswert vor dem Verlernen).
"""
import argparse
import collections
import json
import re
import string
from pathlib import Path

import tiktoken
import torch

from . import pfade  # noqa: I001 (setzt den Importpfad für llms_from_scratch)
from llms_from_scratch.ch05 import generate
from .daten import EOS_ID, pipeline, prompt_text
from .modell import baue_modell, lade_gpt2_gewichte, modell_konfiguration, waehle_geraet

ARTIKEL = r"\b(der|die|das|den|dem|des|ein|eine|einer|eines|einem|einen)\b"


def lade_modell(checkpoint: str, geraet=None):
    """Lädt einen eigenen Checkpoint oder mit 'gpt2' das unveränderte GPT-2 (124M)."""
    geraet = geraet or waehle_geraet()
    if checkpoint == "gpt2":
        cfg = modell_konfiguration("gpt2-small (124M)")
        modell = baue_modell(cfg)
        lade_gpt2_gewichte(modell, "gpt2-small (124M)")
    else:
        stand = torch.load(checkpoint, map_location="cpu", weights_only=False)
        modell = baue_modell(stand["cfg"])
        modell.load_state_dict(stand["modell"])
    return modell.to(geraet).eval()


def antworte_ids(modell, prompt_ids: list[int], max_neue_tokens: int = 32) -> list[int]:
    """Greedy-Decoding mit Raschkas generate(), stoppt bei <|endoftext|>."""
    geraet = next(modell.parameters()).device
    idx = torch.tensor([prompt_ids], device=geraet)
    aus = generate(modell, idx, max_neue_tokens, modell.pos_emb.num_embeddings, eos_id=EOS_ID)
    return aus[0, len(prompt_ids):].tolist()


def antworte(modell, tokenizer, frage: str, kontext: str | None = None, max_neue_tokens: int = 32) -> str:
    """Beantwortet eine Frage im Trainingsformat, optional mit Kontext."""
    ids = tokenizer.encode(prompt_text(frage, kontext))
    if len(ids) + max_neue_tokens > modell.pos_emb.num_embeddings:
        raise ValueError(f"Prompt zu lang ({len(ids)} Tokens), Kontext kürzen.")
    return tokenizer.decode(antworte_ids(modell, ids, max_neue_tokens)).strip()


def normalisiere(text: str) -> str:
    """SQuAD-Normalisierung, Artikel auf Deutsch: klein, ohne Satzzeichen und Artikel."""
    text = "".join(z for z in text.lower() if z not in set(string.punctuation + "„“”‚‘’–"))
    return " ".join(re.sub(ARTIKEL, " ", text).split())


def f1(vorhersage: str, referenz: str) -> float:
    a, b = normalisiere(vorhersage).split(), normalisiere(referenz).split()
    gemeinsam = sum((collections.Counter(a) & collections.Counter(b)).values())
    if not a or not b or gemeinsam == 0:
        return float(a == b)
    p, r = gemeinsam / len(a), gemeinsam / len(b)
    return 2 * p * r / (p + r)


def bewerte(vorhersage: str, referenzen: list[str]) -> tuple[float, float]:
    """Bester Exact-Match- und F1-Wert über alle Antwortvarianten."""
    em = max(float(normalisiere(vorhersage) == normalisiere(r)) for r in referenzen)
    return em, max(f1(vorhersage, r) for r in referenzen)


def werte_aus(modell, tokenizer, eintraege, modus: str, max_neue_tokens: int = 32) -> dict:
    ergebnisse = []
    for eintrag in eintraege:
        # Gleiche Kürzung langer Kontexte wie im Training (Thema 3); Prompt = alles vor der Antwort.
        beispiel, _ = pipeline.s4.verarbeite(eintrag, tokenizer, 1024 - max_neue_tokens, modus)
        prompt_ids = beispiel["input_ids"][:beispiel["prompt_token_anzahl"]]
        vorhersage = tokenizer.decode(antworte_ids(modell, prompt_ids, max_neue_tokens)).strip()
        em, f = bewerte(vorhersage, eintrag["answers"]["text"])
        ergebnisse.append({"id": eintrag["id"], "frage": eintrag["question"], "vorhersage": vorhersage,
                           "referenz": eintrag["answers"]["text"], "em": em, "f1": f})
    n = max(1, len(ergebnisse))
    return {"anzahl": len(ergebnisse), "exact_match": sum(e["em"] for e in ergebnisse) / n,
            "f1": sum(e["f1"] for e in ergebnisse) / n, "beispiele": ergebnisse}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="Pfad zu .pt oder 'gpt2'")
    parser.add_argument("--split", default="validation", choices=["validation", "test"])
    parser.add_argument("--anzahl", type=int, default=300, help="GermanQuAD-Beispiele (0 = alle)")
    parser.add_argument("--daten", default=str(pfade.DATEN))
    parser.add_argument("--ausgabe", help="JSON-Datei für alle Vorhersagen")
    args = parser.parse_args()

    tokenizer = tiktoken.get_encoding("gpt2")
    modell = lade_modell(args.checkpoint)
    split = [e for e in pipeline.s1.lade_datensatz(Path(args.daten) / f"{args.split}.json") if not e["is_private"]]
    privat = [e for e in pipeline.s1.lade_datensatz(Path(args.daten) / "train.json") if e["is_private"]]

    qa = werte_aus(modell, tokenizer, split[:args.anzahl or None], "kontext_frage")
    print(f"GermanQuAD {args.split} mit Kontext ({qa['anzahl']}): EM {qa['exact_match']:.1%} | F1 {qa['f1']:.1%}")
    gedaechtnis = werte_aus(modell, tokenizer, privat, "frage")
    print(f"Private Fakten ohne Kontext ({gedaechtnis['anzahl']}): EM {gedaechtnis['exact_match']:.1%}")

    unlearning = json.loads((pfade.UNLEARNING_DATEN / "unlearning_evaluation.json").read_text(encoding="utf-8"))
    print("Unlearning-Testfragen (Ausgangswert vor dem Verlernen):")
    for fall in unlearning:
        vorhersage = antworte(modell, tokenizer, fall["question"])
        fall["vorhersage"] = vorhersage
        fall["richtig"] = bewerte(vorhersage, [fall["expected_answer"]])[0] == 1.0
        print(f"  [{fall['scenario']}/{fall['evaluation_goal']}] {fall['question']} -> {vorhersage!r} "
              f"({'richtig' if fall['richtig'] else 'falsch'}, erwartet {fall['expected_answer']!r})")

    ausgabe = Path(args.ausgabe) if args.ausgabe else (
        Path(args.checkpoint).with_name(f"evaluation_{Path(args.checkpoint).stem}_{args.split}.json")
        if args.checkpoint != "gpt2"
        else pfade.CHECKPOINTS / f"evaluation_gpt2_{args.split}.json")
    ausgabe.parent.mkdir(parents=True, exist_ok=True)
    ausgabe.write_text(json.dumps({"checkpoint": args.checkpoint, "germanquad": qa, "private_fakten": gedaechtnis,
                                   "unlearning": unlearning}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Details: {ausgabe}")


if __name__ == "__main__":
    main()
