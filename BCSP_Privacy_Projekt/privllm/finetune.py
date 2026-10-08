"""Instruction-Finetuning (Kap. 7): Frage-Antwort-Format lernen und fiktive Fakten einpflanzen."""
import argparse
import math
import os
import random

import torch

from .checkpoint import build_model_from_checkpoint, save_checkpoint
from .config import load_config, config_to_dict
from .data import load_jsonl, make_instruction_loader
from .device import get_device, set_seed, autocast_context
from .model import count_parameters, lm_loss
from .tokenizer import get_tokenizer
from .train import make_optimizer


def build_training_rows(cfg, exclude_forget=False):
    """Mischt Allgemein-Set und Fakten-Set. Mit exclude_forget=True fehlen die Forget-Personen: so entsteht das
    Referenzmodell (Goldstandard fuer perfektes Verlernen)."""
    rows = load_jsonl(cfg.general_file)
    facts = load_jsonl(cfg.facts_file)
    if exclude_forget:
        facts = [r for r in facts if r.get("split") != "forget"]
    rows += facts * cfg.fact_repeat
    random.shuffle(rows)
    return rows


def finetune(cfg_all, device=None, exclude_forget=False, out_name="last.pt"):
    """Finetunt das vortrainierte Modell auf Frage-Antwort-Paaren (Loss nur auf den Antwort-Tokens)."""
    cfg = cfg_all.finetune
    device = device or get_device()
    set_seed(cfg.seed)
    model, _ = build_model_from_checkpoint(cfg.base_checkpoint, device)
    mcfg = model.cfg
    rows = build_training_rows(cfg, exclude_forget)
    loader = make_instruction_loader(get_tokenizer(mcfg.tokenizer), rows, mcfg.context_length, cfg.batch_size)
    opt = make_optimizer(model, cfg.lr, cfg.weight_decay)
    total = cfg.epochs * len(loader)
    warm = max(1, int(cfg.warmup_ratio * total))
    print(f"{len(rows)} Beispiele | {total} Schritte | Parameter: {count_parameters(model) / 1e6:.1f} Mio.")

    step = 0
    model.train()
    for epoch in range(cfg.epochs):
        running = []
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            lr = cfg.lr * (step + 1) / warm if step < warm else \
                cfg.lr * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))
            for g in opt.param_groups:
                g["lr"] = lr
            opt.zero_grad(set_to_none=True)
            with autocast_context(device, cfg_all.train.precision):
                loss = lm_loss(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            running.append(loss.item())
            step += 1
        print(f"Epoche {epoch + 1}/{cfg.epochs} | Loss {sum(running) / len(running):.3f}")

    os.makedirs(cfg.out_dir, exist_ok=True)
    meta = {"seed": cfg.seed, "facts_file": cfg.facts_file, "exclude_forget": exclude_forget,
            "config": config_to_dict(cfg_all)}
    save_checkpoint(os.path.join(cfg.out_dir, out_name), model, mcfg, None, step, meta)
    return model


def main():
    """Kommandozeile: python -m privllm.finetune --config configs/ziel.yaml [--exclude-forget]"""
    ap = argparse.ArgumentParser(description="Instruction-Finetuning")
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--exclude-forget", action="store_true", help="Referenzmodell ohne Forget-Personen trainieren")
    ap.add_argument("--out-name", default=None, help="Dateiname im out_dir (Standard: last.pt bzw. reference.pt)")
    a = ap.parse_args()
    name = a.out_name or ("reference.pt" if a.exclude_forget else "last.pt")
    finetune(load_config(a.config), get_device(a.device), a.exclude_forget, name)


if __name__ == "__main__":
    main()
