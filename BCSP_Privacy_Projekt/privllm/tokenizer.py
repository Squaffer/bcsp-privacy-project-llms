"""Tokenizer: entweder GPT-2-BPE (tiktoken, Kap. 2 des Buchs) oder unser eigener 8k-BPE, trainiert auf Simple Wiki.
Welcher benutzt wird, steht in ModelConfig.tokenizer und damit auch in jedem Checkpoint."""
from functools import lru_cache

EOT_TOKEN = "<|endoftext|>"


class Tokenizer:
    """Gemeinsame Schnittstelle fuer beide Tokenizer-Varianten: encode, decode, eot_id, vocab_size.
    name = "gpt2" fuer tiktoken, sonst Pfad zu einer tokenizer.json (HuggingFace tokenizers)."""

    def __init__(self, name="gpt2"):
        self.name = name
        if name == "gpt2":
            import tiktoken
            self._tt = tiktoken.get_encoding("gpt2")
            self._hf = None
            self.eot_id = self._tt.eot_token
            self.vocab_size = self._tt.n_vocab
        else:
            from tokenizers import Tokenizer as HFTokenizer
            self._tt = None
            self._hf = HFTokenizer.from_file(name)
            self.eot_id = self._hf.token_to_id(EOT_TOKEN)
            self.vocab_size = self._hf.get_vocab_size()
            if self.eot_id is None:
                raise ValueError(f"{name} enthaelt kein {EOT_TOKEN}-Token")

    def encode(self, text):
        """Text -> Liste von Token-IDs (Spezialtokens im Text werden wie normaler Text behandelt)."""
        if self._tt is not None:
            return self._tt.encode(text, disallowed_special=())
        return self._hf.encode(text, add_special_tokens=False).ids

    def encode_batch(self, texts):
        """Viele Texte auf einmal tokenisieren (beim eigenen BPE parallel und deutlich schneller)."""
        if self._tt is not None:
            return self._tt.encode_ordinary_batch(texts)
        return [e.ids for e in self._hf.encode_batch(texts, add_special_tokens=False)]

    def decode(self, ids):
        """Liste von Token-IDs -> Text."""
        ids = list(ids)
        if self._tt is not None:
            return self._tt.decode(ids)
        return self._hf.decode(ids, skip_special_tokens=False)


@lru_cache(maxsize=None)
def get_tokenizer(name="gpt2"):
    """Laedt einen Tokenizer genau einmal und gibt danach immer dasselbe Objekt zurueck."""
    return Tokenizer(name)
