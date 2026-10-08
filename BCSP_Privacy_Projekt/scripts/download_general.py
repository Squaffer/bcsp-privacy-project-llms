"""Laedt das Allgemein-Set (Instruction-Daten) und wandelt es ins Frage-Antwort-Format um.
Standard ist Alpaca-cleaned (ca. 50.000 Beispiele): Fuer ein kleines Modell ist die Menge entscheidend, damit es
das Frage-Antwort-Format zuverlaessig lernt. Alternativ der kleine Alpaca-Teil aus Raschkas Kap. 7 (1.100 Beispiele)."""
import argparse
import json
import os
import random
import sys

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from privllm.data import write_jsonl  # noqa: E402

SOURCES = {
    "alpaca": "https://raw.githubusercontent.com/gururise/AlpacaDataCleaned/main/alpaca_data_cleaned.json",
    "raschka": "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_main-chapter-code/"
               "instruction-data.json",
}
# Aufgaben, die ein Modell dieser Groesse nicht sinnvoll lernen kann und die nur Rauschen beitragen
SKIP_WORDS = ["code", "python", "javascript", "sql", "html", "function", "program", "algorithm", "equation"]


def convert(entries, max_chars):
    """Fuegt instruction und input zu einer Frage zusammen; verwirft Programmier-/Rechenaufgaben und sehr lange Beispiele."""
    rows = []
    for e in entries:
        q = e["instruction"].strip() + (f"\n{e['input'].strip()}" if e.get("input", "").strip() else "")
        out = e["output"].strip()
        if not out or len(q) + len(out) > max_chars:
            continue
        if any(w in (q + " " + out).lower() for w in SKIP_WORDS):
            continue
        rows.append({"instruction": q, "output": out, "split": "general"})
    return rows


def main():
    """Download + Konvertierung nach data/instruct/general.jsonl und general_test.jsonl (2 % bzw. mind. 200 Beispiele Test)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=list(SOURCES), default="alpaca")
    ap.add_argument("--out-dir", default="data/instruct")
    ap.add_argument("--max-chars", type=int, default=700, help="ca. 200 Tokens, passt in context_length 256")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    r = requests.get(SOURCES[a.source], timeout=120)
    r.raise_for_status()
    rows = convert(json.loads(r.text), a.max_chars)
    random.Random(a.seed).shuffle(rows)
    n_test = max(200, len(rows) // 50)
    write_jsonl(rows[n_test:], os.path.join(a.out_dir, "general.jsonl"))
    write_jsonl(rows[:n_test], os.path.join(a.out_dir, "general_test.jsonl"))
    print(f"Quelle {a.source}: {len(rows) - n_test} Train- und {n_test} Test-Beispiele geschrieben")


if __name__ == "__main__":
    main()
