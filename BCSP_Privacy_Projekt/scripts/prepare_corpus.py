"""Simple English Wikipedia vorbereiten (Team A), in zwei Stufen:
  1. extract:  Dump laden, bereinigen, Namens-Check, 90/10-Split -> data/corpus/clean/*.jsonl (Text, tokenizer-unabhaengig)
  2. tokenize: Text mit einem Tokenizer in Binaerdateien umwandeln -> data/corpus/<tokenizer>/*.bin
Dazwischen kann mit scripts/train_tokenizer.py ein eigener BPE auf clean/train.jsonl trainiert werden."""
import argparse
import bz2
import json
import os
import random
import re
import sys
import xml.etree.ElementTree as ET

import requests
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from privllm.data import write_bin  # noqa: E402
from privllm.tokenizer import get_tokenizer  # noqa: E402

DUMP_URL = "https://dumps.wikimedia.org/simplewiki/latest/simplewiki-latest-pages-articles.xml.bz2"
# Abschnitte, ab denen nur noch Quellen/Linklisten kommen: alles danach wird abgeschnitten
TAIL_SECTIONS = re.compile(r"^(References|Other websites|Related pages|Sources|Notes|Further reading|"
                           r"External links|Bibliography|Gallery|Footnotes)\s*$", re.M | re.I)


def download_dump(url, path):
    """Laedt den Wikipedia-Dump herunter (wird uebersprungen, wenn die Datei schon existiert)."""
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with requests.get(url, stream=True, timeout=60, headers={"User-Agent": "BCSP-Privacy-Studienprojekt"}) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))


def clean_wikitext(text):
    """Entfernt Wiki-Markup (Vorlagen, Tabellen, Referenzen, Links, Fett/Kursiv, Kategorien) und gibt reinen Fliesstext zurueck."""
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    text = re.sub(r"<ref[^>/]*?/>", "", text)
    text = re.sub(r"<ref[^>]*?>.*?</ref>", "", text, flags=re.S)
    for _ in range(6):  # verschachtelte {{...}}-Vorlagen von innen nach aussen entfernen
        text = re.sub(r"\{\{[^{}]*\}\}", "", text)
    text = re.sub(r"\{\|.*?\|\}", "", text, flags=re.S)
    text = re.sub(r"\[\[(?:File|Image|Category|[a-z]{2,3}):[^\[\]]*(?:\[\[[^\[\]]*\]\][^\[\]]*)*\]\]", "", text, flags=re.I)
    text = re.sub(r"\[\[[^\]|]*\|([^\]]*)\]\]", r"\1", text)
    text = re.sub(r"\[\[([^\]]*)\]\]", r"\1", text)
    text = re.sub(r"\[https?://[^\s\]]*\s*([^\]]*)\]", r"\1", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"'{2,}", "", text)
    text = re.sub(r"^=+\s*(.*?)\s*=+\s*$", r"\1", text, flags=re.M)
    text = re.sub(r"^[*#:;]+\s*", "", text, flags=re.M)
    text = re.sub(r"[ \t]+", " ", text)
    m = TAIL_SECTIONS.search(text)
    if m:
        text = text[:m.start()]
    return re.sub(r"\n{3,}", "\n\n", filter_lines(text)).strip()


def filter_lines(text):
    """Wirft Zeilen ohne echten Text weg: Listen-Reste wie '– – –', einzelne Buchstaben, Zahlenkolonnen.
    Eine Zeile bleibt, wenn sie mind. 3 Zeichen hat und mind. die Haelfte davon Buchstaben sind (oder sie leer ist)."""
    kept = []
    for line in text.split("\n"):
        s = line.strip()
        letters = sum(c.isalpha() for c in s)
        if not s or (len(s) >= 3 and letters >= 0.5 * len(s)):
            kept.append(s)
    return "\n".join(kept)


def is_prose(text):
    """True, wenn ein Artikel ueberwiegend aus ganzen Saetzen besteht (filtert reine Listen- und Tabellenseiten)."""
    words = text.split()
    return len(words) >= 80 and text.count(".") >= 5


def iter_articles(dump_path, min_chars):
    """Liest den XML-Dump als Strom und liefert bereinigte Artikeltexte (nur Namensraum 0, keine Weiterleitungen, keine Kurzseiten)."""
    with bz2.open(dump_path, "rb") as f:
        for _, el in ET.iterparse(f):
            if el.tag.endswith("}page") or el.tag == "page":
                ns = el.find("./{*}ns")
                redirect = el.find("./{*}redirect")
                text_el = el.find("./{*}revision/{*}text")
                if ns is not None and ns.text == "0" and redirect is None and text_el is not None and text_el.text:
                    title = el.findtext("./{*}title", default="")
                    cleaned = clean_wikitext(text_el.text)
                    if len(cleaned) >= min_chars and is_prose(cleaned) and not title.startswith("List of"):
                        yield f"{title}\n\n{cleaned}"
                el.clear()


def load_forbidden_names(path):
    """Liest die erfundenen Namen (persons.json), die im Korpus NICHT vorkommen duerfen."""
    if not path or not os.path.exists(path):
        return []
    return [p["name"].lower() for p in json.load(open(path))]


def write_texts(texts, path):
    """Schreibt Artikeltexte als JSONL ({"text": ...})."""
    with open(path, "w") as f:
        for t in texts:
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")


def read_texts(path):
    """Liest Artikeltexte aus einer JSONL-Datei."""
    with open(path) as f:
        return [json.loads(line)["text"] for line in f]


def extract(a):
    """Stufe 1: Download, Bereinigung, Namens-Check, Split in train/val/mini_train/mini_val (als Text)."""
    clean_dir = os.path.join(a.corpus_dir, "clean")
    os.makedirs(clean_dir, exist_ok=True)
    download_dump(DUMP_URL, a.dump)
    forbidden = load_forbidden_names(a.persons)
    articles, dropped = [], 0
    for i, art in enumerate(tqdm(iter_articles(a.dump, a.min_chars), desc="Artikel")):
        if a.max_articles and i >= a.max_articles:
            break
        if any(n in art.lower() for n in forbidden):
            dropped += 1  # Artikel mit erfundenem Namen verwerfen, sonst ist Verlernen nicht sauber messbar
            continue
        articles.append(art)
    random.Random(a.seed).shuffle(articles)
    n_val = int(len(articles) * a.val_frac)
    n_mini = max(1, int(len(articles) * a.mini_frac))
    val, train = articles[:n_val], articles[n_val:]
    splits = {"train": train, "val": val, "mini_train": train[:n_mini], "mini_val": val[:max(1, n_mini // 9)]}
    for name, texts in splits.items():
        write_texts(texts, os.path.join(clean_dir, f"{name}.jsonl"))
    meta = {"source": "Simple English Wikipedia (CC BY-SA)", "dump": os.path.basename(a.dump), "seed": a.seed,
            "articles_dropped_name_clash": dropped, **{f"articles_{k}": len(v) for k, v in splits.items()}}
    json.dump(meta, open(os.path.join(clean_dir, "meta.json"), "w"), indent=2)
    print(json.dumps(meta, indent=2))


def tokenize(a):
    """Stufe 2: alle Text-Splits mit dem gewaehlten Tokenizer in uint16-Binaerdateien umwandeln (je Artikel + <|endoftext|>)."""
    tok = get_tokenizer(a.tokenizer)
    assert tok.vocab_size <= 65536, "uint16 reicht nur fuer Vokabulare bis 65.536"
    name = "gpt2" if a.tokenizer == "gpt2" else os.path.splitext(os.path.basename(a.tokenizer))[0]
    out_dir = os.path.join(a.corpus_dir, name)
    os.makedirs(out_dir, exist_ok=True)
    meta = {"tokenizer": a.tokenizer, "vocab_size": tok.vocab_size}
    for split in ["train", "val", "mini_train", "mini_val"]:
        texts = read_texts(os.path.join(a.corpus_dir, "clean", f"{split}.jsonl"))

        def chunks():
            """Tokenisiert in Paketen von 1.000 Artikeln."""
            for i in tqdm(range(0, len(texts), 1000), desc=split):
                for ids in tok.encode_batch(texts[i:i + 1000]):
                    yield ids + [tok.eot_id]
        meta[f"tokens_{split}"] = write_bin(chunks(), os.path.join(out_dir, f"{split}.bin"))
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=2)
    print(json.dumps(meta, indent=2))


def main():
    """Kommandozeile: python scripts/prepare_corpus.py extract | tokenize --tokenizer data/tokenizer/bpe8k.json"""
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["extract", "tokenize"])
    ap.add_argument("--corpus-dir", default="data/corpus")
    ap.add_argument("--dump", default="data/raw/simplewiki-latest-pages-articles.xml.bz2")
    ap.add_argument("--min-chars", type=int, default=500)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--mini-frac", type=float, default=0.03, help="Anteil fuer das Mini-Modell")
    ap.add_argument("--persons", default="data/facts/persons.json", help="Namen, die nicht im Korpus vorkommen duerfen")
    ap.add_argument("--max-articles", type=int, default=None, help="nur zum Testen")
    ap.add_argument("--tokenizer", default="data/tokenizer/bpe8k.json", help="'gpt2' oder Pfad zu tokenizer.json")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    extract(a) if a.stage == "extract" else tokenize(a)


if __name__ == "__main__":
    main()
