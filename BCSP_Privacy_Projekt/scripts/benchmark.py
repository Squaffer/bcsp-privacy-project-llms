"""Durchsatz-Messung (Woche 3, Team B): Tokens pro Sekunde fuer Trainingsschritte auf diesem Geraet.
Jedes Teammitglied laesst das Skript laufen; danach werden Modellgroesse und Korpusmenge festgelegt."""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from privllm.config import load_config  # noqa: E402
from privllm.device import get_device, device_name, autocast_context, sync  # noqa: E402
from privllm.model import GPTModel, count_parameters, lm_loss  # noqa: E402
from privllm.train import make_optimizer  # noqa: E402


def bench(mcfg, batch_size, device, steps, precision):
    """Misst Tokens/s fuer Forward + Backward + Optimizer-Schritt mit zufaelligen Tokens (ohne Datenladen)."""
    model = GPTModel(mcfg).to(device)
    opt = make_optimizer(model, 1e-4, 0.1)
    x = torch.randint(0, mcfg.vocab_size, (batch_size, mcfg.context_length), device=device)
    y = torch.randint(0, mcfg.vocab_size, (batch_size, mcfg.context_length), device=device)

    def step():
        """Ein vollstaendiger Trainingsschritt."""
        opt.zero_grad(set_to_none=True)
        with autocast_context(device, precision):
            loss = lm_loss(model(x), y)
        loss.backward()
        opt.step()

    for _ in range(3):  # Aufwaermen (Kernel-Kompilierung, Speicher-Allokation)
        step()
    sync(device)
    t0 = time.time()
    for _ in range(steps):
        step()
    sync(device)
    dt = time.time() - t0
    del model, opt
    if device.type == "mps":
        torch.mps.empty_cache()
    return batch_size * mcfg.context_length * steps / dt


def main():
    """Kommandozeile: python scripts/benchmark.py --config configs/ziel.yaml --batch-sizes 4 8 16"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ziel.yaml")
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[4, 8, 16])
    ap.add_argument("--precision", default=None, help="fp32, fp16 oder bf16 (Standard: aus der Config)")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    device = get_device(a.device)
    precision = a.precision or cfg.train.precision
    print(f"Geraet: {device_name(device)} | Parameter: {count_parameters(GPTModel(cfg.model)) / 1e6:.1f} Mio. | "
          f"Praezision: {precision}")
    for bs in a.batch_sizes:
        tps = bench(cfg.model, bs, device, a.steps, precision)
        hours = 66e6 / tps / 3600
        print(f"batch_size {bs:3d}: {tps:8,.0f} Tok/s  (eine Epoche Simple Wiki ~ {hours:.1f} h)")


if __name__ == "__main__":
    main()
