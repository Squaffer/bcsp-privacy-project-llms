"""Geraete-Helper: gleiche Code-Basis fuer CUDA, Apple-MPS und CPU."""
import contextlib
import random

import numpy as np
import torch


def get_device(prefer=None):
    """Waehlt das beste Geraet: CUDA, sonst MPS (Mac), sonst CPU. 'prefer' erzwingt ein Geraet."""
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_name(device):
    """Lesbarer Geraetename fuer results.csv (Hardware-Unterschiede muessen nachvollziehbar sein)."""
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    return device.type


def set_seed(seed):
    """Setzt alle Zufallsgeneratoren fest, damit Laeufe reproduzierbar sind."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


_DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}


def autocast_context(device, precision="auto"):
    """Mixed Precision fuer Forward/Loss. 'auto': bf16 auf CUDA (falls unterstuetzt), sonst fp32
    (auf Apple-MPS bringt fp16 laut scripts/benchmark.py keinen Geschwindigkeitsvorteil, nur Instabilitaetsrisiko).
    'fp32', 'fp16' oder 'bf16' erzwingen eine Variante. Die Gewichte selbst bleiben immer float32."""
    if precision == "auto":
        if device.type == "cuda":
            precision = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
        else:
            precision = "fp32"
    if precision == "fp32" or device.type == "cpu":
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=_DTYPES[precision])


def sync(device):
    """Wartet, bis die GPU alle Befehle abgearbeitet hat (noetig fuer korrekte Zeitmessung)."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
