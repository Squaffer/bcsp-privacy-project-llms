"""Interaktiver Chat mit einem finegetunten Modell (zum Ausprobieren und fuer die Live-Demo)."""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from privllm.generate import repetition_penalty  # noqa: E402
from privllm.interface import load_model, ask  # noqa: E402


def main():
    """Kommandozeile: python scripts/chat.py --checkpoint checkpoints/ziel_final/last.pt (Beenden mit leerer Zeile)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/ziel_final/last.pt")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--penalty", type=float, default=1.3, help="1.0 = keine Wiederholungsstrafe")
    ap.add_argument("--max-new-tokens", type=int, default=120)
    a = ap.parse_args()
    if not os.path.exists(a.checkpoint):
        found = sorted(glob.glob("checkpoints/*/*.pt"))
        sys.exit(f"Checkpoint '{a.checkpoint}' nicht gefunden. Das Finetuning muss erst fertig sein "
                 f"(python -m privllm.finetune --config configs/ziel.yaml).\nVorhandene Checkpoints: "
                 f"{', '.join(found) or 'keine'}")
    model = load_model(a.checkpoint)
    proc = repetition_penalty(a.penalty) if a.penalty != 1.0 else None
    print("Frage auf Englisch stellen (leere Zeile beendet).")
    while True:
        q = input("\nDu: ").strip()
        if not q:
            break
        print("Modell:", ask(model, q, a.max_new_tokens, a.temperature, a.top_k, proc))


if __name__ == "__main__":
    main()
