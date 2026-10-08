"""GPT-Architektur nach Raschka (Kap. 3-4), selbst implementiert und ueber ModelConfig skalierbar."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class LayerNorm(nn.Module):
    """Layer-Normalisierung: normiert jeden Token-Vektor auf Mittelwert 0 und Varianz 1, mit lernbarem Scale/Shift."""

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))
        self.shift = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return self.scale * (x - mean) / torch.sqrt(var + self.eps) + self.shift


class GELU(nn.Module):
    """GELU-Aktivierung (tanh-Naeherung wie in GPT-2): glatte Variante von ReLU."""

    def forward(self, x):
        c = math.sqrt(2.0 / math.pi)
        return 0.5 * x * (1.0 + torch.tanh(c * (x + 0.044715 * x ** 3)))


class FeedForward(nn.Module):
    """Positionsweises Feed-Forward-Netz im Transformer-Block: Linear (4x breiter) -> GELU -> Linear."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.emb_dim, 4 * cfg.emb_dim),
            GELU(),
            nn.Linear(4 * cfg.emb_dim, cfg.emb_dim),
        )

    def forward(self, x):
        return self.net(x)


class MultiHeadAttention(nn.Module):
    """Kausale Multi-Head-Self-Attention: jeder Token schaut nur auf sich und frueheren Text.
    Nutzt PyTorchs scaled_dot_product_attention (schneller als die handgeschriebene Maske aus dem Buch)."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.emb_dim % cfg.n_heads == 0, "emb_dim muss durch n_heads teilbar sein"
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.emb_dim // cfg.n_heads
        self.qkv = nn.Linear(cfg.emb_dim, 3 * cfg.emb_dim, bias=cfg.qkv_bias)
        self.out_proj = nn.Linear(cfg.emb_dim, cfg.emb_dim)
        self.drop_rate = cfg.drop_rate

    def forward(self, x):
        b, t, c = x.shape
        qkv = self.qkv(x).view(b, t, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        p = self.drop_rate if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=p)
        out = out.transpose(1, 2).contiguous().view(b, t, c)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """Ein Transformer-Block (Pre-Norm): Attention und Feed-Forward, jeweils mit Residual-Verbindung und Dropout."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = LayerNorm(cfg.emb_dim)
        self.att = MultiHeadAttention(cfg)
        self.norm2 = LayerNorm(cfg.emb_dim)
        self.ff = FeedForward(cfg)
        self.drop = nn.Dropout(cfg.drop_rate)

    def forward(self, x):
        x = x + self.drop(self.att(self.norm1(x)))
        x = x + self.drop(self.ff(self.norm2(x)))
        return x


class GPTModel(nn.Module):
    """Das komplette GPT: Token- und Positions-Embedding, n Transformer-Bloecke, finale Norm, Ausgabeschicht.
    forward() liefert Logits der Form (Batch, Laenge, Vokabular)."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.emb_dim)
        self.pos_emb = nn.Embedding(cfg.context_length, cfg.emb_dim)
        self.drop = nn.Dropout(cfg.drop_rate)
        self.blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = LayerNorm(cfg.emb_dim)
        self.out_head = nn.Linear(cfg.emb_dim, cfg.vocab_size, bias=False)
        if cfg.weight_tying:
            self.out_head.weight = self.tok_emb.weight  # Weight Tying: spart ca. 19 Mio. Parameter
        self.apply(self._init_weights)
        # GPT-2-Trick: Ausgabe-Projektionen der Residual-Pfade kleiner initialisieren
        for name, p in self.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith("ff.net.2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _init_weights(module):
        """Startgewichte wie bei GPT-2: Normalverteilung mit Std 0.02, Biases null."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        _, t = idx.shape
        assert t <= self.cfg.context_length, "Eingabe laenger als context_length"
        pos = torch.arange(t, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        x = self.blocks(x)
        x = self.final_norm(x)
        return self.out_head(x)


def count_parameters(model):
    """Zaehlt die trainierbaren Parameter (geteilte Gewichte werden nur einmal gezaehlt)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def lm_loss(logits, targets):
    """Cross-Entropy-Loss fuers Next-Token-Training; Ziele mit -100 (Prompt/Padding) werden ignoriert."""
    return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
