"""Trainiert einen eigenen Byte-Level-BPE-Tokenizer (Standard 8.192 Tokens) auf dem bereinigten Simple-Wiki-Text.
Kleineres Vokabular = kleinere Embedding- und Ausgabeschicht = ca. 3x schnelleres Training auf Laptops."""
import argparse
import json
import os

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

EOT_TOKEN = "<|endoftext|>"


def iter_texts(path):
    """Liefert die Artikeltexte aus einer JSONL-Datei ({"text": ...} pro Zeile)."""
    with open(path) as f:
        for line in f:
            yield json.loads(line)["text"]


def train_bpe(texts, vocab_size):
    """Byte-Level-BPE wie bei GPT-2: jedes Byte ist darstellbar, also auch deutsche Umlaute oder Sonderzeichen
    (wichtig fuer die Sprach-Angriffe). <|endoftext|> ist das einzige Spezialtoken (ID 0)."""
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=[EOT_TOKEN],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tok.train_from_iterator(texts, trainer)
    return tok


def main():
    """Kommandozeile: python scripts/train_tokenizer.py --vocab-size 8192"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/corpus/clean/train.jsonl")
    ap.add_argument("--vocab-size", type=int, default=8192)
    ap.add_argument("--out", default="data/tokenizer/bpe8k.json")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    tok = train_bpe(iter_texts(a.input), a.vocab_size)
    tok.save(a.out)
    sample = "Berlin is the capital city of Germany. It has about 3.7 million people."
    enc = tok.encode(sample)
    print(f"Gespeichert: {a.out} | Vokabular: {tok.get_vocab_size()} | Beispiel: {len(enc.ids)} Tokens -> {enc.tokens}")


if __name__ == "__main__":
    main()
