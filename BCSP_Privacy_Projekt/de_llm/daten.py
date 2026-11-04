"""Anbindung der Thema-3-Datenpipeline an das Training.

Die Pipeline von Thema 3 (Ordner datenpipeline/) bleibt unverändert und liefert pro Eintrag
input_ids, labels (-100 für den Prompt) und attention_mask. Hier kommen dazu:
  * private Fakten zusätzlich ohne Kontext (damit das Modell sie auswendig lernt),
  * Batches aus ähnlich langen Beispielen (weniger Padding),
  * die Verschiebung um eine Position für das Causal-LM-Training.
"""
import importlib.util
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset, Sampler

from . import pfade


def _lade_modul(dateiname: str, name: str):
    spec = importlib.util.spec_from_file_location(name, pfade.DATENPIPELINE / dateiname)
    modul = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modul)
    return modul


pipeline = _lade_modul("05_datenpipeline.py", "thema3_pipeline")
IGNORE_INDEX = pipeline.IGNORE_INDEX
EOS_ID = pipeline.EOS_ID


def prompt_text(frage: str, kontext: str | None = None) -> str:
    """Derselbe Prompt wie in 03_trainingsformat.py (für Generierung ohne Antwort)."""
    if kontext is None:
        return f"Frage:\n{frage}\n\nAntwort:\n"
    return f"Kontext:\n{kontext}\n\nFrage:\n{frage}\n\nAntwort:\n"


def baue_beispiele(json_pfad, tokenizer, max_tokens=1024, privat_ohne_kontext=0) -> list[dict]:
    """Tokenisiert einen Split mit der Thema-3-Pipeline.

    Jeder Eintrag wird im Modus 'kontext_frage' übernommen. Private Einträge kommen zusätzlich
    `privat_ohne_kontext`-mal im Modus 'frage' hinzu: Nur so muss das Modell die Fakten im
    Gedächtnis behalten, statt sie aus dem Kontext abzuschreiben (Voraussetzung fürs Unlearning).
    """
    daten = pipeline.s1.lade_datensatz(Path(json_pfad))
    beispiele = []
    for eintrag in daten:
        modi = ["kontext_frage"]
        if eintrag["is_private"]:
            modi += ["frage"] * privat_ohne_kontext
        for modus in modi:
            beispiel, status = pipeline.s4.verarbeite(eintrag, tokenizer, max_tokens, modus)
            if beispiel is None:
                raise ValueError(f"{eintrag['id']} nicht verarbeitbar: {status}")
            beispiele.append(beispiel)
    return beispiele


class QADatensatz(Dataset):
    """Bereits tokenisierte Beispiele (einmal beim Start, ca. 2 s für den Trainingssplit)."""

    def __init__(self, beispiele: list[dict]):
        self.beispiele = beispiele
        self.laengen = [len(b["input_ids"]) for b in beispiele]

    def __len__(self):
        return len(self.beispiele)

    def __getitem__(self, index):
        return self.beispiele[index]


class LaengenSampler(Sampler):
    """Batches nach Token-Budget statt fester Anzahl.

    Die Daten werden gemischt, dann in Blöcken von `block` Beispielen nach Länge sortiert und
    so gepackt, dass Anzahl x längstes Beispiel <= `batch_tokens`. Kurze Beispiele landen so zu
    mehreren in einem Batch, lange einzeln oder zu zweit, und es wird kaum aufgefüllt.
    Die Reihenfolge hängt nur von seed + Epoche ab (wichtig fürs Fortsetzen).
    """

    def __init__(self, laengen, batch_tokens=2048, max_batch=16, seed=123, block=800):
        self.laengen, self.batch_tokens, self.max_batch = laengen, batch_tokens, max_batch
        self.seed, self.block, self.epoche = seed, block, 0

    def batches(self, epoche: int | None = None) -> list[list[int]]:
        rng = random.Random(self.seed + (self.epoche if epoche is None else epoche))
        indizes = list(range(len(self.laengen)))
        rng.shuffle(indizes)
        batches = []
        for start in range(0, len(indizes), self.block):
            aktuell = []
            for i in sorted(indizes[start:start + self.block], key=lambda i: self.laengen[i]):
                # Sortiert aufsteigend: das neue Beispiel ist immer das längste im Batch.
                if aktuell and ((len(aktuell) + 1) * self.laengen[i] > self.batch_tokens
                                or len(aktuell) == self.max_batch):
                    batches.append(aktuell)
                    aktuell = []
                aktuell.append(i)
            batches.append(aktuell)
        rng.shuffle(batches)
        return batches

    def __iter__(self):
        return iter(self.batches())

    def __len__(self):
        return len(self.batches())


def collate(beispiele: list[dict], max_tokens=1024) -> tuple[torch.Tensor, torch.Tensor]:
    """Padding mit der Thema-3-Funktion, dann Verschiebung für Next-Token-Prediction.

    Thema 3 liefert labels deckungsgleich zu input_ids. Das Modell sagt an Position t das
    Token t+1 voraus, daher: Eingabe = input_ids[:-1], Ziel = labels[1:] (wie in Raschka Kap. 7).
    Rechts aufgefüllte Positionen sind -100 und werden wegen der kausalen Maske von echten
    Tokens nie angeschaut; eine eigene attention_mask braucht das Modell daher nicht.
    """
    batch = pipeline.s4.pad_batch(beispiele, max_tokens)
    pipeline.pruefe_batch(batch, beispiele, max_tokens)
    input_ids = torch.tensor(batch["input_ids"], dtype=torch.long)
    labels = torch.tensor(batch["labels"], dtype=torch.long)
    return input_ids[:, :-1], labels[:, 1:]
