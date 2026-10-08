"""Datenhandling: Binaerkorpus fuers Pretraining (Memmap) und Instruction-Datensaetze mit Loss-Maskierung."""
import functools
import json

import numpy as np
import torch



# ---------- Pretraining: tokenisierter Korpus als Binaerdatei ----------

def write_bin(token_chunks, path):
    """Schreibt einen Strom von Token-Listen als uint16-Binaerdatei (GPT-2-Vokabular passt in 16 Bit). Gibt die Token-Anzahl zurueck."""
    total = 0
    with open(path, "wb") as f:
        for chunk in token_chunks:
            arr = np.asarray(chunk, dtype=np.uint16)
            arr.tofile(f)
            total += len(arr)
    return total


def load_bin(path):
    """Oeffnet eine Binaerdatei als Memmap, ohne sie komplett in den RAM zu laden."""
    return np.memmap(path, dtype=np.uint16, mode="r")


def get_batch(data, batch_size, context_length, device):
    """Zieht zufaellige Fenster aus dem Korpus: x = Tokens, y = dieselben Tokens um 1 verschoben (naechstes Token)."""
    ix = np.random.randint(0, len(data) - context_length - 1, size=batch_size)
    x = np.stack([data[i:i + context_length] for i in ix]).astype(np.int64)
    y = np.stack([data[i + 1:i + 1 + context_length] for i in ix]).astype(np.int64)
    x, y = torch.from_numpy(x), torch.from_numpy(y)
    if device.type == "cuda":
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x.to(device), y.to(device)


# ---------- Instruction-Finetuning ----------

def load_jsonl(path):
    """Liest eine JSONL-Datei (eine JSON-Zeile pro Eintrag) als Liste von Dicts."""
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(rows, path):
    """Schreibt eine Liste von Dicts als JSONL-Datei."""
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def format_prompt(instruction):
    """Baut den Prompt im Frage-Antwort-Format; die Antwort wird danach angehaengt."""
    return f"### Question:\n{instruction}\n\n### Answer:\n"


def encode_example(tok, instruction, output):
    """Tokenisiert Prompt + Antwort + <|endoftext|>. Prompt-Tokens bekommen Label -100, damit nur die Antwort gelernt wird.
    Gibt (input_ids, labels) zurueck."""
    p = tok.encode(format_prompt(instruction))
    a = tok.encode(output) + [tok.eot_id]
    return p + a, [-100] * len(p) + a


class InstructionDataset(torch.utils.data.Dataset):
    """Datensatz aus {instruction, output}-Eintraegen, vorab tokenisiert. Beispiele, die nicht in den Kontext passen,
    werden verworfen (abschneiden wuerde das <|endoftext|> am Ende entfernen und das Modell lernen, nie aufzuhoeren)."""

    def __init__(self, tok, rows, context_length):
        items = [encode_example(tok, r["instruction"], r["output"]) for r in rows]
        self.items = [it for it in items if len(it[0]) <= context_length + 1]
        self.dropped = len(items) - len(self.items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def collate_batch(batch, pad_id):
    """Fuellt eine Batch auf gleiche Laenge auf (Padding mit pad_id, Labels -100) und verschiebt die Labels um 1 (Next-Token)."""
    longest = max(len(ids) for ids, _ in batch)
    xs, ys = [], []
    for ids, labels in batch:
        pad = longest - len(ids)
        xs.append(ids[:-1] + [pad_id] * pad)
        ys.append(labels[1:] + [-100] * pad)
    return torch.tensor(xs), torch.tensor(ys)


def make_instruction_loader(tok, rows, context_length, batch_size, shuffle=True):
    """DataLoader fuers Instruction-Finetuning (Padding mit dem <|endoftext|>-Token des Tokenizers)."""
    ds = InstructionDataset(tok, rows, context_length)
    if ds.dropped:
        print(f"{ds.dropped} Beispiele laenger als der Kontext verworfen")
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                                       collate_fn=functools.partial(collate_batch, pad_id=tok.eot_id))
