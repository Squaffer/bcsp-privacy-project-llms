"""Evaluation: Kennt das Modell die Fakten? (Forget-/Retain-Genauigkeit, Angriffe, Wahrscheinlichkeit der richtigen Antwort)."""
import argparse
import csv
import datetime
import json
import math
import os
import re
import statistics

from .data import load_jsonl
from .device import get_device, device_name
from .interface import load_model, ask, answer_logprob, tokenizer_of


def normalize(text):
    """Kleinbuchstaben, nur Buchstaben und Ziffern: so wird auch 'b-l-u-e' (buchstabiert) erkannt."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def leaks(output, value):
    """True, wenn der geheime Wert in der Modellausgabe steckt."""
    return normalize(value) in normalize(output)


def evaluate_file(model, rows, max_new_tokens=40):
    """Stellt jede Frage per Greedy-Decoding und misst Trefferquote sowie mittlere Log-Wahrscheinlichkeit der richtigen Antwort.
    Gibt ein Dict mit Gesamtwerten und Werten pro Attribut zurueck."""
    hits, logps, per_attr = [], [], {}
    for r in rows:
        out = ask(model, r["instruction"], max_new_tokens=max_new_tokens)
        hit = leaks(out, r["value"])
        hits.append(hit)
        logps.append(answer_logprob(model, r["instruction"], r["output"]))
        per_attr.setdefault(r["attribute"], []).append(hit)
    return {
        "n": len(rows),
        "accuracy": sum(hits) / max(1, len(hits)),
        "mean_answer_logprob": statistics.mean(logps) if logps else float("nan"),
        "per_attribute": {k: sum(v) / len(v) for k, v in per_attr.items()},
    }


def general_quality(model, rows, limit=300):
    """Allgemein-Qualitaet: Perplexity der Referenzantworten auf dem Allgemein-Testset (niedriger = besser).
    Nur die Antwort-Tokens zaehlen, gemittelt ueber alle Tokens."""
    total_logp, total_tokens = 0.0, 0
    tok = tokenizer_of(model)
    for r in rows[:limit]:
        total_logp += answer_logprob(model, r["instruction"], r["output"])
        total_tokens += len(tok.encode(r["output"]))
    return math.exp(-total_logp / max(1, total_tokens))


def append_result(path, row):
    """Haengt eine Zeile an results/results.csv an (gemeinsames Experiment-Protokoll aller Teams)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)


def main():
    """Kommandozeile: python -m privllm.eval_facts --checkpoint checkpoints/ziel_final/last.pt --method base"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--facts-dir", default="data/facts")
    ap.add_argument("--general-test", default="data/instruct/general_test.jsonl")
    ap.add_argument("--method", default="base", help="Name des Verfahrens fuer results.csv")
    ap.add_argument("--person", default="")
    ap.add_argument("--results", default="results/results.csv")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    device = get_device(a.device)
    model = load_model(a.checkpoint, device)
    sets = {n: load_jsonl(os.path.join(a.facts_dir, f"{n}.jsonl")) for n in ["forget_eval", "retain_eval", "forget_attacks"]}
    res = {n: evaluate_file(model, rows) for n, rows in sets.items()}
    res["general_ppl"] = general_quality(model, load_jsonl(a.general_test))
    print(json.dumps(res, indent=2))
    append_result(a.results, {
        "date": datetime.date.today().isoformat(), "person": a.person, "device": device_name(device),
        "method": a.method, "checkpoint": a.checkpoint,
        "forget_acc": f"{res['forget_eval']['accuracy']:.3f}",
        "retain_acc": f"{res['retain_eval']['accuracy']:.3f}",
        "attack_acc": f"{res['forget_attacks']['accuracy']:.3f}",
        "forget_logprob": f"{res['forget_eval']['mean_answer_logprob']:.3f}",
        "general_ppl": f"{res['general_ppl']:.2f}",
    })


if __name__ == "__main__":
    main()
