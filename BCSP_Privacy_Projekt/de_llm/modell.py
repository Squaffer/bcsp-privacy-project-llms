"""GPT-Modell nach Raschka (Kap. 4) mit schnellerer Attention und GPT-2-Gewichten (Kap. 5)."""
import torch
import torch.nn.functional as F

from . import pfade
from llms_from_scratch.ch03 import MultiHeadAttention
from llms_from_scratch.ch04 import GPTModel

# Gemeinsame Werte aller GPT-2-Größen (wie BASE_CONFIG in Kap. 5).
BASIS = {"vocab_size": 50257, "context_length": 1024, "drop_rate": 0.0, "qkv_bias": True}

GROESSEN = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12, "datei": "gpt2-small-124M"},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16, "datei": "gpt2-medium-355M"},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20, "datei": "gpt2-large-774M"},
}

# Von Raschka bereitgestellte PyTorch-Version der OpenAI-Gewichte (kein TensorFlow nötig).
GEWICHTE_URL = "https://huggingface.co/rasbt/gpt2-from-scratch-pytorch/resolve/main/{datei}.safetensors"


class SDPAMultiHeadAttention(MultiHeadAttention):
    """Raschkas MultiHeadAttention, aber mit PyTorchs fused scaled_dot_product_attention.

    Gleiche Gewichte und Parameternamen wie das Original (W_query, W_key, W_value, out_proj),
    daher passen die GPT-2-Gewichte unverändert. Die volle Score-Matrix (Tokens x Tokens pro Head)
    wird nicht mehr explizit gespeichert, das spart bei 1024 Tokens viel Speicher und Zeit.
    """

    def forward(self, x):
        b, num_tokens, _ = x.shape
        aufteilen = lambda t: t.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        queries, keys, values = aufteilen(self.W_query(x)), aufteilen(self.W_key(x)), aufteilen(self.W_value(x))
        context_vec = F.scaled_dot_product_attention(
            queries, keys, values, is_causal=True,
            dropout_p=self.dropout.p if self.training else 0.0)
        context_vec = context_vec.transpose(1, 2).reshape(b, num_tokens, self.d_out)
        return self.out_proj(context_vec)


def modell_konfiguration(groesse: str, drop_rate: float = 0.0) -> dict:
    """Vollständige Konfiguration im Format von Raschkas GPTModel."""
    if groesse not in GROESSEN:
        raise ValueError(f"Unbekannte Modellgröße {groesse!r}, erlaubt: {list(GROESSEN)}")
    cfg = dict(BASIS, **{k: v for k, v in GROESSEN[groesse].items() if k != "datei"})
    cfg["drop_rate"] = drop_rate
    return cfg


def baue_modell(cfg: dict, schnelle_attention: bool = True) -> GPTModel:
    """Erzeugt Raschkas GPTModel und tauscht optional die Attention gegen die SDPA-Variante."""
    modell = GPTModel(cfg)
    if schnelle_attention:
        for block in modell.trf_blocks:
            block.att = SDPAMultiHeadAttention(
                d_in=cfg["emb_dim"], d_out=cfg["emb_dim"], context_length=cfg["context_length"],
                num_heads=cfg["n_heads"], dropout=cfg["drop_rate"], qkv_bias=cfg["qkv_bias"])
    return modell


def lade_gpt2_gewichte(modell: GPTModel, groesse: str) -> None:
    """Lädt die vortrainierten OpenAI-GPT-2-Gewichte (einmaliger Download nach modelle/gpt2/)."""
    from safetensors.torch import load_file
    from llms_from_scratch.ch05 import download_file

    datei = GROESSEN[groesse]["datei"]
    ziel = pfade.GEWICHTE / f"{datei}.safetensors"
    if not ziel.exists():
        pfade.GEWICHTE.mkdir(parents=True, exist_ok=True)
        download_file(GEWICHTE_URL.format(datei=datei), str(ziel))
        if not ziel.exists():
            raise RuntimeError(f"Download der GPT-2-Gewichte fehlgeschlagen: {ziel}")
    fehlend, unerwartet = modell.load_state_dict(load_file(str(ziel)), strict=False)
    # Die kausale Maske ist ein berechneter Puffer und fehlt in der Datei; alles andere muss passen.
    fehlend = [k for k in fehlend if not k.endswith(".att.mask")]
    if fehlend or unerwartet:
        raise RuntimeError(f"Gewichte passen nicht: fehlend={fehlend}, unerwartet={unerwartet}")


def versteckte_zustaende(modell: GPTModel, idx: torch.Tensor) -> torch.Tensor:
    """GPTModel.forward ohne die letzte Ausgabeschicht (out_head)."""
    x = modell.tok_emb(idx) + modell.pos_emb(torch.arange(idx.shape[1], device=idx.device))
    return modell.final_norm(modell.trf_blocks(modell.drop_emb(x)))


def antwort_verlust(modell: GPTModel, eingabe: torch.Tensor, ziel: torch.Tensor,
                    reduktion: str = "mean") -> torch.Tensor:
    """Cross-Entropy nur über die Positionen mit echtem Ziel (Antwort + EOS, nicht -100).

    Gleiches Ergebnis wie Raschkas calc_loss_batch mit ignore_index=-100, aber die Ausgabeschicht
    (768 x 50.257) läuft nur für die ca. 25 Antwort-Tokens statt für alle ~560 Positionen.
    Gemessen auf M1 Pro: 1.500 -> 2.550 Tokens/s im Training und weniger Speicher.
    """
    maske = ziel != -100
    logits = modell.out_head(versteckte_zustaende(modell, eingabe)[maske])
    return F.cross_entropy(logits, ziel[maske], reduction=reduktion)


def waehle_geraet() -> torch.device:
    """CUDA-GPU, sonst Apple-GPU (MPS), sonst CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def anzahl_parameter(modell: torch.nn.Module) -> int:
    """Parameter ohne die separate Ausgabeschicht (wie in Kap. 4, wegen Weight Tying bei GPT-2)."""
    return sum(p.numel() for p in modell.parameters()) - modell.out_head.weight.numel()
