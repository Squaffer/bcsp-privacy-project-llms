"""Textgenerierung (Kap. 5): Temperatur, Top-k und ein Haken (logits_processor) fuer Decoding-Sperren von Team D."""
import torch


def sample_next_token(logits, temperature=0.0, top_k=None):
    """Waehlt das naechste Token aus den Logits: greedy bei temperature=0, sonst Sampling mit Temperatur und optional Top-k."""
    if temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    if top_k is not None:
        top_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits = torch.where(logits < top_vals[:, -1:], torch.full_like(logits, float("-inf")), logits)
    probs = torch.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=0.0, top_k=None, eos_id=None, logits_processor=None):
    """Erzeugt max_new_tokens neue Tokens zu idx (Batch, Laenge). Bricht bei eos_id ab (nur Batchgroesse 1).
    logits_processor(idx, logits) -> logits darf Logits veraendern, z. B. verbotene Token-Folgen auf -inf setzen."""
    model.eval()
    ctx = model.cfg.context_length
    for _ in range(max_new_tokens):
        logits = model(idx[:, -ctx:])[:, -1, :]
        if logits_processor is not None:
            logits = logits_processor(idx, logits)
        nxt = sample_next_token(logits, temperature, top_k)
        idx = torch.cat([idx, nxt], dim=1)
        if eos_id is not None and idx.size(0) == 1 and nxt.item() == eos_id:
            break
    return idx


def repetition_penalty(penalty=1.3, window=64):
    """Erzeugt einen logits_processor, der Tokens aus den letzten 'window' Tokens unwahrscheinlicher macht.
    Kleine Modelle geraten sonst schnell in Schleifen ('the city of the city of ...')."""
    def process(idx, logits):
        """Teilt positive Logits bereits benutzter Tokens durch penalty bzw. multipliziert negative damit."""
        for b in range(idx.size(0)):
            seen = torch.unique(idx[b, -window:])
            vals = logits[b, seen]
            logits[b, seen] = torch.where(vals > 0, vals / penalty, vals * penalty)
        return logits
    return process
