"""Vorbereitung auf dem Login-Knoten des Clusters (dort gibt es Internet, auf den Rechenknoten oft nicht).

Aufruf:  python -m de_llm.vorbereiten
         python -m de_llm.vorbereiten --modell "gpt2-medium (355M)"     # zusätzlich größeres Modell

Erledigt in einem Befehl:
  1. lädt die GPT-2-Gewichte nach modelle/gpt2/ (falls noch nicht da),
  2. füllt den tiktoken-Cache in .tiktoken/ (die Jobskripte setzen TIKTOKEN_CACHE_DIR darauf),
  3. prüft, ob Daten und Checkpoints da sind, die per rsync kommen müssen,
  4. zeigt, ob PyTorch eine GPU sieht (auf dem Login-Knoten oft nicht, das ist normal).
Endet mit Fehlercode 1, wenn etwas Nötiges fehlt.
"""
import argparse
import os
import sys

from . import pfade

DATEIEN = [
    ("daten/split_data/train.json", True, "Training"),
    ("daten/split_data/validation.json", True, "Training und Auswertung"),
    ("daten/unlearning_data/forget_person.json", True, "Unlearning"),
    ("daten/unlearning_data/forget_attribute.json", True, "Unlearning"),
    ("daten/unlearning_data/retain_person.json", True, "Unlearning"),
    ("daten/unlearning_data/retain_attribute.json", True, "Unlearning"),
    ("daten/unlearning_data/unlearning_evaluation.json", True, "Auswertung"),
    ("checkpoints/germanquad_124m/fakten.pt", False, "Unlearning (Ausgangsmodell)"),
    ("checkpoints/germanquad_124m/bester.pt", False, "Phase 2 neu oder --start"),
    ("checkpoints/referenz_ohne_p001/fakten.pt", False, "Vergleich beim Unlearning"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--modell", action="append", help="GPT-2-Größe, mehrfach möglich (Standard: 124M)")
    args = parser.parse_args()
    os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(pfade.PROJEKT / ".tiktoken"))

    import tiktoken
    import torch
    from .modell import GROESSEN, gewichte_datei

    fehler = False
    for groesse in args.modell or ["gpt2-small (124M)"]:
        if groesse not in GROESSEN:
            sys.exit(f"Unbekannte Größe {groesse!r}, erlaubt: {list(GROESSEN)}")
        print(f"OK     Gewichte {groesse}: {gewichte_datei(groesse).relative_to(pfade.PROJEKT)}")

    tiktoken.get_encoding("gpt2")
    print(f"OK     tiktoken-Cache: {os.environ['TIKTOKEN_CACHE_DIR']}")

    for pfad, noetig, wofuer in DATEIEN:
        da = (pfade.PROJEKT / pfad).exists()
        status = "OK" if da else ("FEHLT" if noetig else "fehlt")
        print(f"{status:6} {pfad}  ({wofuer})")
        fehler |= noetig and not da

    if torch.cuda.is_available():
        print(f"GPU    {torch.cuda.get_device_name(0)}, CUDA {torch.version.cuda}, "
              f"bf16 {'ja' if torch.cuda.is_bf16_supported() else 'nein'}")
    else:
        print(f"GPU    keine sichtbar (PyTorch {torch.__version__}); auf dem Login-Knoten normal, "
              "im Job muss nvidia-smi eine zeigen")
    if fehler:
        sys.exit("Es fehlen nötige Dateien (FEHLT): mit rsync vom eigenen Rechner kopieren, siehe Anleitung.")
    print("Vorbereitung vollständig.")


if __name__ == "__main__":
    main()
