"""Checkpoints: Gewichte + Config + Seed + Datenversion zusammen ablegen (Reproduzierbarkeit laut Plan)."""
import os

import torch

from .config import ModelConfig, model_config_from_dict, config_to_dict
from .model import GPTModel


def save_checkpoint(path, model, model_cfg, optimizer=None, step=0, extra=None):
    """Speichert Modell (und optional Optimizer) samt Metadaten wie Seed und Datenversion in 'extra'."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "model_cfg": config_to_dict(model_cfg),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "step": step,
        "extra": extra or {},
    }, path)


def load_checkpoint(path, device="cpu"):
    """Laedt ein Checkpoint-Dict (Gewichte bleiben unveraendert im Dict, Modell wird nicht gebaut)."""
    return torch.load(path, map_location=device, weights_only=False)


def build_model_from_checkpoint(path, device="cpu"):
    """Baut das GPTModel passend zur im Checkpoint gespeicherten Config und laedt die Gewichte."""
    ckpt = load_checkpoint(path, device)
    cfg: ModelConfig = model_config_from_dict(ckpt["model_cfg"])
    model = GPTModel(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    return model, ckpt
