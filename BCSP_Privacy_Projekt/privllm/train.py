"""Pretraining auf Simple English Wikipedia (Kap. 5 + Anhang D: Warmup, Cosine Decay, Gradient Clipping)."""
import argparse
import csv
import math
import os
import time

import torch

from .checkpoint import save_checkpoint, load_checkpoint
from .config import load_config, config_to_dict
from .data import load_bin, get_batch
from .device import get_device, set_seed, autocast_context, device_name, sync
from .model import GPTModel, count_parameters, lm_loss
from .tokenizer import get_tokenizer
from .interface import generate_text

SAMPLE_PROMPT = "The capital city of France is"


def get_lr(it, cfg):
    """Lernrate zum Schritt 'it': linearer Warmup, danach Cosine Decay bis min_lr."""
    if it < cfg.warmup_iters:
        return cfg.lr * (it + 1) / cfg.warmup_iters
    if it >= cfg.max_iters:
        return cfg.min_lr
    progress = (it - cfg.warmup_iters) / max(1, cfg.max_iters - cfg.warmup_iters)
    return cfg.min_lr + 0.5 * (1 + math.cos(math.pi * progress)) * (cfg.lr - cfg.min_lr)


def make_optimizer(model, lr, weight_decay):
    """AdamW mit Weight Decay nur auf Matrizen (nicht auf Biases/LayerNorm), wie ueblich bei GPT."""
    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    groups = [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95))


@torch.no_grad()
def estimate_loss(model, data, cfg, mcfg, device):
    """Mittlerer Loss ueber eval_iters zufaellige Batches (fuer Train/Val-Kurven)."""
    model.eval()
    losses = []
    for _ in range(cfg.eval_iters):
        x, y = get_batch(data, cfg.batch_size, mcfg.context_length, device)
        with autocast_context(device, cfg.precision):
            losses.append(lm_loss(model(x), y).item())
    model.train()
    return sum(losses) / len(losses)


def train(cfg_all, device=None, resume=True):
    """Hauptschleife des Pretrainings mit Gradient Accumulation, Clipping, Checkpoints und CSV-Log der Loss-Kurve."""
    cfg, mcfg = cfg_all.train, cfg_all.model
    device = device or get_device()
    set_seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)
    train_data, val_data = load_bin(cfg.train_bin), load_bin(cfg.val_bin)

    tok = get_tokenizer(mcfg.tokenizer)
    if tok.vocab_size > mcfg.vocab_size:
        raise ValueError(f"vocab_size {mcfg.vocab_size} kleiner als Tokenizer-Vokabular {tok.vocab_size}")
    model = GPTModel(mcfg).to(device)
    opt = make_optimizer(model, cfg.lr, cfg.weight_decay)
    start = 0
    last = os.path.join(cfg.out_dir, "last.pt")
    if resume and os.path.exists(last):
        ckpt = load_checkpoint(last, device)
        model.load_state_dict(ckpt["model"])
        if ckpt["optimizer"]:
            opt.load_state_dict(ckpt["optimizer"])
        start = ckpt["step"]
        print(f"Setze Training bei Schritt {start} fort")
    print(f"Geraet: {device_name(device)} | Parameter: {count_parameters(model) / 1e6:.1f} Mio. | "
          f"Tokens im Train-Korpus: {len(train_data):,}")

    log_path = os.path.join(cfg.out_dir, "loss_log.csv")
    new_log = not os.path.exists(log_path)
    log_f = open(log_path, "a", newline="")
    log = csv.writer(log_f)
    if new_log:
        log.writerow(["step", "train_loss", "val_loss", "lr", "tokens_per_sec"])
    meta = {"seed": cfg.seed, "train_bin": cfg.train_bin, "config": config_to_dict(cfg_all)}

    model.train()
    train_time, tokens = 0.0, 0
    for it in range(start, cfg.max_iters):
        t0 = time.time()
        lr = get_lr(it, cfg)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        for _ in range(cfg.grad_accum):
            x, y = get_batch(train_data, cfg.batch_size, mcfg.context_length, device)
            with autocast_context(device, cfg.precision):
                loss = lm_loss(model(x), y) / cfg.grad_accum
            loss.backward()
            tokens += x.numel()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        if (it + 1) % cfg.eval_interval == 0:
            sync(device)
        train_time += time.time() - t0

        if (it + 1) % cfg.eval_interval == 0 or it + 1 == cfg.max_iters:
            tr = estimate_loss(model, train_data, cfg, mcfg, device)
            va = estimate_loss(model, val_data, cfg, mcfg, device)
            tps = tokens / max(train_time, 1e-9)
            print(f"Schritt {it + 1}/{cfg.max_iters} | train {tr:.3f} | val {va:.3f} | lr {lr:.2e} | {tps:,.0f} Tok/s")
            log.writerow([it + 1, f"{tr:.4f}", f"{va:.4f}", f"{lr:.3e}", f"{tps:.0f}"])
            log_f.flush()
            print("  Beispiel:", repr(generate_text(model, SAMPLE_PROMPT, max_new_tokens=30)))
            model.train()
            train_time, tokens = 0.0, 0
        if (it + 1) % cfg.save_interval == 0 or it + 1 == cfg.max_iters:
            save_checkpoint(last, model, mcfg, opt, it + 1, meta)
    log_f.close()
    return model


def main():
    """Kommandozeile: python -m privllm.train --config configs/ziel.yaml"""
    ap = argparse.ArgumentParser(description="Pretraining des kleinen GPT")
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--fresh", action="store_true", help="nicht aus last.pt fortsetzen")
    a = ap.parse_args()
    train(load_config(a.config), get_device(a.device), resume=not a.fresh)


if __name__ == "__main__":
    main()
