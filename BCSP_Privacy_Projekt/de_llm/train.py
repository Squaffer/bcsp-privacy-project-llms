"""Finetuning von GPT-2 auf den deutschen QA-Daten (GermanQuAD + private Fakten).

Aufruf:  python -m de_llm.train                      # Standardlauf
         python -m de_llm.train --fortsetzen         # nach Abbruch weitermachen
         python -m de_llm.train --max-schritte 20    # kurzer Probelauf
         python -m de_llm.train --start checkpoints/germanquad_124m/fakten.pt --lauf weiter --epochen 1
                                                     # auf einem fertigen Checkpoint aufbauen

Basis ist Raschkas train_model_simple (Kap. 5) mit den Erweiterungen aus Anhang D
(Warmup, Cosine Decay, Gradient Clipping), ergänzt um Gradient Accumulation und Checkpoints.
"""
import argparse
import csv
import dataclasses
import math
import time
from dataclasses import dataclass
from pathlib import Path

import tiktoken
import torch

from . import pfade
from .daten import LaengenSampler, QADatensatz, baue_beispiele, collate
from .modell import (anzahl_parameter, antwort_verlust, baue_modell, lade_gpt2_gewichte,
                     modell_konfiguration, waehle_geraet)


@dataclass
class TrainKonfig:
    lauf: str = "germanquad_124m"          # Unterordner in checkpoints/
    modell: str = "gpt2-small (124M)"      # Startgewichte, siehe modell.GROESSEN
    start: str = ""                        # eigener Checkpoint als Start statt GPT-2 (ersetzt --modell)
    daten: str = str(pfade.DATEN)          # Ordner mit train.json / validation.json
    epochen: int = 2
    batch_tokens: int = 2048               # Tokens pro Mikro-Batch: ca. 8 GB auf M1 Pro (4096 swappt bei 16 GB)
    akkumulation: int = 4                  # Mikro-Batches pro Optimierer-Schritt (ca. 14 Beispiele)
    lernrate: float = 5e-5                 # wie Raschka Kap. 7
    min_lernrate: float = 5e-6
    warmup_anteil: float = 0.05
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    drop_rate: float = 0.0
    max_tokens: int = 1024                 # = Kontextlänge von GPT-2
    privat_ohne_kontext: int = 3           # Zusatzkopien privater Fakten ohne Kontext pro Epoche
    eval_alle: int = 100                   # Optimierer-Schritte zwischen Validierungen
    eval_batches: int = 40
    max_schritte: int = 0                  # 0 = alle Epochen
    seed: int = 123


def lernrate_bei(schritt: int, gesamt: int, k: TrainKonfig) -> float:
    """Lineares Warmup, danach Cosine Decay bis min_lernrate (Raschka Anhang D)."""
    warmup = max(1, int(gesamt * k.warmup_anteil))
    if schritt < warmup:
        return k.lernrate * (schritt + 1) / warmup
    fortschritt = (schritt - warmup) / max(1, gesamt - warmup)
    return k.min_lernrate + 0.5 * (k.lernrate - k.min_lernrate) * (1 + math.cos(math.pi * fortschritt))


@torch.no_grad()
def validiere(modell, val_batches, geraet) -> float:
    """Mittlerer Verlust pro Antwort-Token auf festen Validierungs-Batches."""
    modell.eval()
    summe, anzahl = 0.0, 0
    for eingabe, ziel in val_batches:
        eingabe, ziel = eingabe.to(geraet), ziel.to(geraet)
        summe += antwort_verlust(modell, eingabe, ziel, reduktion="sum").item()
        anzahl += (ziel != -100).sum().item()
    modell.train()
    return summe / anzahl


def speichere(pfad: Path, modell, k: TrainKonfig, cfg: dict, **extra) -> None:
    torch.save({"modell": modell.state_dict(), "cfg": cfg, "konfig": dataclasses.asdict(k), **extra}, pfad)


def trainiere(k: TrainKonfig, fortsetzen: bool = False) -> Path:
    torch.manual_seed(k.seed)
    geraet = waehle_geraet()
    ordner = pfade.CHECKPOINTS / k.lauf
    ordner.mkdir(parents=True, exist_ok=True)
    tokenizer = tiktoken.get_encoding("gpt2")

    train = QADatensatz(baue_beispiele(Path(k.daten) / "train.json", tokenizer, k.max_tokens, k.privat_ohne_kontext))
    val = QADatensatz(baue_beispiele(Path(k.daten) / "validation.json", tokenizer, k.max_tokens))
    sampler = LaengenSampler(train.laengen, k.batch_tokens, seed=k.seed)
    val_idx = LaengenSampler(val.laengen, k.batch_tokens, seed=0).batches()[:k.eval_batches]
    val_batches = [collate([val[i] for i in b], k.max_tokens) for b in val_idx]

    # Die Zahl der Batches schwankt je Epoche leicht (andere Mischung, andere Packung).
    schritte = [math.ceil(len(sampler.batches(e)) / k.akkumulation) for e in range(k.epochen)]
    gesamt = sum(schritte)
    if k.max_schritte:
        gesamt = min(gesamt, k.max_schritte)

    start_stand = None
    if k.start:
        # Architektur kommt aus dem Checkpoint; Optimizer und Lernratenplan beginnen neu.
        start_stand = torch.load(k.start, map_location="cpu", weights_only=False)
        cfg = dict(start_stand["cfg"], drop_rate=k.drop_rate)
    else:
        cfg = modell_konfiguration(k.modell, k.drop_rate)
    modell = baue_modell(cfg)
    optimizer = torch.optim.AdamW(modell.parameters(), lr=k.lernrate, weight_decay=k.weight_decay)
    schritt, bester_val = 0, float("inf")
    if fortsetzen:
        stand = torch.load(ordner / "letzter.pt", map_location="cpu", weights_only=False)
        modell.load_state_dict(stand["modell"])
        optimizer.load_state_dict(stand["optimizer"])
        schritt, bester_val = stand["schritt"], stand["bester_val"]
        print(f"Fortsetzen ab Schritt {schritt}")
    elif start_stand is not None:
        modell.load_state_dict(start_stand["modell"])
        print(f"Start von {k.start}")
    else:
        lade_gpt2_gewichte(modell, k.modell)
    del start_stand
    modell.to(geraet).train()
    # Optimizer-Zustände liegen nach dem Laden auf der CPU; mit dem Modell aufs Gerät schieben.
    for zustand in optimizer.state.values():
        for name, wert in zustand.items():
            if torch.is_tensor(wert):
                zustand[name] = wert.to(geraet)

    print(f"Gerät: {geraet} | Modell: {k.start or k.modell} ({anzahl_parameter(modell) / 1e6:.0f} Mio. Parameter)")
    print(f"Train: {len(train)} Sequenzen ({sum(train.laengen) / 1e6:.1f} Mio. Tokens) | "
          f"Val: {len(val)} | Schritte: {gesamt} (ca. {schritte[0]} pro Epoche)")

    log_pfad = ordner / "log.csv"
    neu = not (fortsetzen and log_pfad.exists())
    log_datei = log_pfad.open("w" if neu else "a", newline="")
    log = csv.writer(log_datei)
    if neu:
        log.writerow(["schritt", "epoche", "train_verlust", "val_verlust", "lernrate", "tokens_pro_s"])

    lauf_summe, lauf_tokens, tokens_seit, schritte_seit, t0 = 0.0, 0, 0, 0, time.time()
    while schritt < gesamt:
        epoche, im_epoche = 0, schritt
        while im_epoche >= schritte[epoche]:
            im_epoche -= schritte[epoche]
            epoche += 1
        batches = sampler.batches(epoche)
        for start in range(im_epoche * k.akkumulation, len(batches), k.akkumulation):
            if schritt >= gesamt:
                break
            mikro = [collate([train[i] for i in b], k.max_tokens) for b in batches[start:start + k.akkumulation]]
            # Verlust über alle Antwort-Tokens der Akkumulation mitteln, nicht über Mikro-Batches.
            ziel_tokens = sum((z != -100).sum().item() for _, z in mikro)
            for eingabe, ziel in mikro:
                verlust = antwort_verlust(modell, eingabe.to(geraet), ziel.to(geraet), reduktion="sum")
                (verlust / ziel_tokens).backward()
                lauf_summe += verlust.item()
                tokens_seit += eingabe.numel()
            lauf_tokens += ziel_tokens
            torch.nn.utils.clip_grad_norm_(modell.parameters(), k.grad_clip)
            lr = lernrate_bei(schritt, gesamt, k)
            for gruppe in optimizer.param_groups:
                gruppe["lr"] = lr
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if geraet.type == "mps":
                # Jede Batch-Form legt im MPS-Cache neue Puffer an; ohne Leeren wächst er bis zum
                # Out-of-Memory (gemessen: 6,5 -> 11,9 GB nach 100 Batches). Kostet kein Tempo.
                torch.mps.empty_cache()
            schritt += 1
            schritte_seit += 1

            if schritt % k.eval_alle == 0 or schritt == gesamt:
                val_verlust = validiere(modell, val_batches, geraet)
                dauer = time.time() - t0
                tps = tokens_seit / dauer
                train_verlust = lauf_summe / lauf_tokens
                log.writerow([schritt, epoche + 1, f"{train_verlust:.4f}", f"{val_verlust:.4f}", f"{lr:.2e}", round(tps)])
                log_datei.flush()
                rest_min = (gesamt - schritt) * dauer / schritte_seit / 60
                print(f"Schritt {schritt}/{gesamt} | Epoche {epoche + 1} | train {train_verlust:.3f} | "
                      f"val {val_verlust:.3f} | lr {lr:.1e} | {tps:.0f} Tok/s | noch ~{rest_min:.0f} min")
                if val_verlust < bester_val:
                    bester_val = val_verlust
                    speichere(ordner / "bester.pt", modell, k, cfg, schritt=schritt, val_verlust=val_verlust)
                speichere(ordner / "letzter.pt", modell, k, cfg, schritt=schritt, bester_val=bester_val,
                          optimizer=optimizer.state_dict())
                lauf_summe, lauf_tokens, tokens_seit, schritte_seit, t0 = 0.0, 0, 0, 0, time.time()
    log_datei.close()
    print(f"Fertig. Bester Validierungsverlust {bester_val:.3f} -> {ordner / 'bester.pt'}")
    return ordner / "bester.pt"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for feld in dataclasses.fields(TrainKonfig):
        parser.add_argument("--" + feld.name.replace("_", "-"), type=feld.type, default=feld.default)
    parser.add_argument("--fortsetzen", action="store_true", help="letzter.pt des Laufs weiterführen")
    args = vars(parser.parse_args())
    fortsetzen = args.pop("fortsetzen")
    trainiere(TrainKonfig(**args), fortsetzen)


if __name__ == "__main__":
    main()
