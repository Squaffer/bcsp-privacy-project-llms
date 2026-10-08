"""Zentrale Konfiguration: alle Groessen und Hyperparameter stehen in YAML-Dateien (configs/)."""
from dataclasses import dataclass, field, fields, asdict
import yaml


@dataclass
class ModelConfig:
    """Architektur-Parameter des GPT-Modells (siehe Tabelle 'Startkonfiguration' im Plan)."""
    vocab_size: int = 50257
    context_length: int = 256
    emb_dim: int = 384
    n_layers: int = 6
    n_heads: int = 6
    drop_rate: float = 0.1
    qkv_bias: bool = False
    weight_tying: bool = True
    tokenizer: str = "gpt2"  # "gpt2" oder Pfad zu eigener tokenizer.json


@dataclass
class TrainConfig:
    """Hyperparameter fuer das Pretraining auf Simple English Wikipedia."""
    train_bin: str = "data/corpus/train.bin"
    val_bin: str = "data/corpus/val.bin"
    out_dir: str = "checkpoints/pretrain"
    batch_size: int = 32
    grad_accum: int = 4
    max_iters: int = 10000
    lr: float = 6e-4
    min_lr: float = 6e-5
    warmup_iters: int = 300
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_interval: int = 250
    eval_iters: int = 20
    save_interval: int = 500
    precision: str = "auto"
    seed: int = 42


@dataclass
class FinetuneConfig:
    """Hyperparameter fuer das Instruction-Finetuning (Allgemein-Set + Fakten-Set)."""
    base_checkpoint: str = "checkpoints/pretrain/last.pt"
    out_dir: str = "checkpoints/instruct"
    general_file: str = "data/instruct/general.jsonl"
    facts_file: str = "data/facts/facts_train.jsonl"
    fact_repeat: int = 1
    batch_size: int = 16
    epochs: int = 5
    lr: float = 1e-4
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    seed: int = 42


@dataclass
class Config:
    """Gesamtkonfiguration eines Laufs: Modell + Pretraining + Finetuning."""
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    finetune: FinetuneConfig = field(default_factory=FinetuneConfig)


def _build(cls, values):
    """Erzeugt eine Dataclass aus einem Dict und lehnt unbekannte Schluessel ab (faengt Tippfehler)."""
    values = values or {}
    known = {f.name for f in fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"Unbekannte Config-Schluessel fuer {cls.__name__}: {sorted(unknown)}")
    return cls(**values)


def load_config(path):
    """Liest eine YAML-Datei und gibt ein Config-Objekt zurueck."""
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return Config(
        model=_build(ModelConfig, raw.get("model")),
        train=_build(TrainConfig, raw.get("train")),
        finetune=_build(FinetuneConfig, raw.get("finetune")),
    )


def model_config_from_dict(d):
    """Baut eine ModelConfig aus einem Dict (z. B. aus einem Checkpoint geladen)."""
    return _build(ModelConfig, d)


def config_to_dict(cfg):
    """Wandelt eine Dataclass-Config in ein normales Dict um (fuer Checkpoints und Logs)."""
    return asdict(cfg)
