"""Feste Schnittstelle fuer Teams C und D (laut Plan): Modell laden, Text generieren, Logits und Loss abfragen."""
import torch

from .checkpoint import build_model_from_checkpoint
from .data import format_prompt
from .device import get_device
from .generate import generate
from .tokenizer import get_tokenizer


def load_model(path, device=None):
    """Laedt ein Checkpoint und gibt das Modell im Eval-Modus zurueck."""
    device = device or get_device()
    model, _ = build_model_from_checkpoint(path, device)
    model.eval()
    return model


def _device(model):
    """Geraet, auf dem das Modell liegt."""
    return next(model.parameters()).device


def tokenizer_of(model):
    """Der Tokenizer, mit dem das Modell trainiert wurde (steht in der ModelConfig im Checkpoint)."""
    return get_tokenizer(model.cfg.tokenizer)


def generate_text(model, prompt, max_new_tokens=60, temperature=0.0, top_k=None, logits_processor=None):
    """Setzt 'prompt' fort und gibt nur den neu erzeugten Text zurueck (stoppt bei <|endoftext|>)."""
    tok = tokenizer_of(model)
    ids = torch.tensor([tok.encode(prompt)[-model.cfg.context_length:]], device=_device(model))
    out = generate(model, ids, max_new_tokens, temperature, top_k, tok.eot_id, logits_processor)
    new = out[0, ids.size(1):].tolist()
    if tok.eot_id in new:
        new = new[:new.index(tok.eot_id)]
    return tok.decode(new)


def ask(model, question, max_new_tokens=60, temperature=0.0, top_k=None, logits_processor=None):
    """Stellt dem Modell eine Frage im Trainingsformat und gibt die Antwort zurueck."""
    return generate_text(model, format_prompt(question), max_new_tokens, temperature, top_k, logits_processor)


@torch.no_grad()
def get_logits(model, text):
    """Logits (Laenge, Vokabular) fuer einen Text; fuer Logit-Analyse und Whitebox-Angriffe."""
    ids = torch.tensor([tokenizer_of(model).encode(text)[-model.cfg.context_length:]], device=_device(model))
    return model(ids)[0]


@torch.no_grad()
def get_loss(model, text):
    """Mittlerer Next-Token-Loss (Cross-Entropy) eines Textes."""
    ids = tokenizer_of(model).encode(text)[:model.cfg.context_length + 1]
    x = torch.tensor([ids[:-1]], device=_device(model))
    y = torch.tensor([ids[1:]], device=_device(model))
    logits = model(x)
    return torch.nn.functional.cross_entropy(logits[0], y[0]).item()


@torch.no_grad()
def answer_logprob(model, question, answer):
    """Summe der Log-Wahrscheinlichkeiten der Antwort-Tokens gegeben die Frage. Zeigt, ob ein Fakt noch in den
    Gewichten steckt, auch wenn er nicht ausgegeben wird (Metrik 'Wahrscheinlichkeit der richtigen Antwort')."""
    tok = tokenizer_of(model)
    p, a = tok.encode(format_prompt(question)), tok.encode(answer)
    ids = (p + a)[:model.cfg.context_length + 1]
    x = torch.tensor([ids[:-1]], device=_device(model))
    logp = torch.log_softmax(model(x)[0], dim=-1)
    targets = torch.tensor(ids[1:], device=x.device)
    per_tok = logp[torch.arange(len(targets)), targets]
    return per_tok[len(p) - 1:].sum().item()
