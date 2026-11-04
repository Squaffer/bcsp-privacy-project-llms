"""Fragen an ein trainiertes Modell stellen.

Aufruf:  python -m de_llm.chat --checkpoint checkpoints/germanquad_124m/bester.pt
Eingabe: erst die Frage, dann optional einen Kontext (leere Zeile = ohne Kontext).
Leere Frage beendet.
"""
import argparse

import tiktoken

from .evaluieren import antworte, lade_modell


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--max-neue-tokens", type=int, default=48)
    args = parser.parse_args()
    modell, tokenizer = lade_modell(args.checkpoint), tiktoken.get_encoding("gpt2")
    while frage := input("\nFrage: ").strip():
        kontext = input("Kontext (optional): ").strip() or None
        try:
            print("Antwort:", antworte(modell, tokenizer, frage, kontext, args.max_neue_tokens))
        except ValueError as fehler:
            print(fehler)


if __name__ == "__main__":
    main()
