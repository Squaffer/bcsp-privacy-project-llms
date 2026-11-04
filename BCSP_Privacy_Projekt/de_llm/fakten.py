"""Zweite Phase: private Fakten einprägen (Ausgangsmodell fürs Unlearning).

Aufruf:  python -m de_llm.fakten                      # von checkpoints/germanquad_124m/bester.pt
         python -m de_llm.fakten --epochen 5          # kürzer

Nach dem GermanQuAD-Training (de_llm.train) kennt das Modell die privaten Fakten nur zu ca. 24 %:
Sie machen 2,5 % der Daten aus, und mit Kontext kann es die Antwort einfach abschreiben.
Hier wird darum nur auf den Fakten OHNE Kontext weitertrainiert. Pro Fakten-Batch kommen ein paar
kurze GermanQuAD-Beispiele dazu (Replay), damit die QA-Fähigkeit nicht verloren geht.
Gemessen auf M1 Pro: 15 Epochen ca. 35 min, Fakten 24 % -> 97 % EM, GermanQuAD-F1 fast unverändert.
"""
import argparse
import random
import time
from pathlib import Path

import tiktoken
import torch

from . import pfade
from .daten import baue_beispiele, collate
from .evaluieren import lade_modell
from .modell import antwort_verlust, waehle_geraet


def praege_ein(start: Path, ziel: Path, daten: Path, epochen=15, lernrate=5e-5, batch=16,
               replay=2, replay_max_tokens=400, seed=123) -> Path:
    torch.manual_seed(seed)
    geraet = waehle_geraet()
    tokenizer = tiktoken.get_encoding("gpt2")
    modell = lade_modell(str(start), geraet).train()

    alle = baue_beispiele(daten / "train.json", tokenizer, privat_ohne_kontext=1)
    fakten = [b for b in alle if b["modus"] == "frage"]
    quad = [b for b in alle if b["modus"] != "frage" and len(b["input_ids"]) < replay_max_tokens]
    print(f"Gerät: {geraet} | {len(fakten)} Fakten | Replay {replay} x 2 GermanQuAD-Beispiele pro Batch")

    # Kein Weight Decay: es zieht die Gewichte zurück und arbeitet damit gegen das Auswendiglernen.
    optimizer = torch.optim.AdamW(modell.parameters(), lr=lernrate, weight_decay=0.0)
    rng = random.Random(seed)
    t0 = time.time()
    for epoche in range(epochen):
        rng.shuffle(fakten)
        summe, anzahl = 0.0, 0
        for i in range(0, len(fakten), batch):
            mikro = [collate(fakten[i:i + batch])] + [collate(rng.sample(quad, 2)) for _ in range(replay)]
            ziel_tokens = sum((z != -100).sum().item() for _, z in mikro)
            for nr, (eingabe, z) in enumerate(mikro):
                verlust = antwort_verlust(modell, eingabe.to(geraet), z.to(geraet), reduktion="sum")
                (verlust / ziel_tokens).backward()
                if nr == 0:
                    summe += verlust.item()
                    anzahl += (z != -100).sum().item()
            torch.nn.utils.clip_grad_norm_(modell.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if geraet.type == "mps":
                torch.mps.empty_cache()
        print(f"Epoche {epoche + 1}/{epochen} | Fakten-Verlust {summe / anzahl:.3f} | {time.time() - t0:.0f} s")

    stand = torch.load(start, map_location="cpu", weights_only=False)
    stand.update(modell=modell.state_dict(), fakten={"start": str(start), "epochen": epochen,
                                                     "lernrate": lernrate, "replay": replay})
    torch.save(stand, ziel)
    print(f"Fertig -> {ziel}")
    return ziel


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", default=str(pfade.CHECKPOINTS / "germanquad_124m" / "bester.pt"))
    parser.add_argument("--ziel", help="Standard: fakten.pt neben --start")
    parser.add_argument("--daten", default=str(pfade.DATEN))
    parser.add_argument("--epochen", type=int, default=15)
    parser.add_argument("--lernrate", type=float, default=5e-5)
    parser.add_argument("--replay", type=int, default=2, help="GermanQuAD-Mini-Batches pro Fakten-Batch")
    args = parser.parse_args()
    start = Path(args.start)
    praege_ein(start, Path(args.ziel) if args.ziel else start.with_name("fakten.pt"), Path(args.daten),
               args.epochen, args.lernrate, replay=args.replay)


if __name__ == "__main__":
    main()
