"""Schritt 5: Gemeinsame Datenpipeline fuer GPT-artiges Causal-LM-Training.

Mac-Test ohne PyTorch:
 python3 05_datenpipeline.py /pfad/zu/split_data
 python3 05_datenpipeline.py /pfad/zu/split_data --split validation --anzahl 9

Fuer spaeteres PyTorch-Training:
 dataset = QADataset(split_ordner / 'train.json', tokenizer, max_tokens=1024)
 loader = baue_torch_dataloader(dataset, batch_size=4, shuffle=True)
 for batch in loader:
     # batch: input_ids, labels, attention_mask als LongTensor; ids als Liste
     ...

Die Originaldateien werden nur gelesen. Keine automatische Speicherung.
"""
import argparse
import importlib.util
from pathlib import Path


def lade_modul(dateiname, name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(dateiname))
    modul = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modul)
    return modul


s1 = lade_modul('01_daten_einlesen.py', 'thema3_s1')
s4 = lade_modul('04_sequenzlaengen_padding.py', 'thema3_s4')
IGNORE_INDEX = s4.IGNORE_INDEX
EOS_ID = s4.EOS_ID


class QADataset:
    """Lazy Dataset: Eintraege werden erst bei Zugriff tokenisiert.

    Verarbeitet Daten aus Thema 2; wirft bei zu langen/unbrauchbaren Beispielen
    einen klaren Fehler statt sie still zu entfernen. Datensatz-IDs bleiben erhalten.
    """
    def __init__(self, json_pfad, tokenizer, max_tokens=1024, modus='kontext_frage'):
        self.json_pfad = Path(json_pfad)
        self.data = s1.lade_datensatz(self.json_pfad)
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.modus = modus
        if max_tokens < 2:
            raise ValueError('max_tokens muss >= 2 sein')
        if modus not in ('kontext_frage', 'frage'):
            raise ValueError('Unbekannter Eingabemodus')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        entry = self.data[index]
        example, status = s4.verarbeite(entry, self.tokenizer, self.max_tokens, self.modus)
        if example is None:
            raise ValueError(f'{self.json_pfad.name}: ID {entry["id"]} nicht verarbeitbar: {status}')
        example['verarbeitungsstatus'] = status
        return example


def batch_generator(dataset, batch_size=4, shuffle=False, seed=42, anzahl=None):
    """Batches ohne PyTorch, mit Padding auf die Laenge des laengsten Beispiels."""
    if batch_size < 1:
        raise ValueError('batch_size muss >= 1 sein')
    indices = list(range(len(dataset)))
    if shuffle:
        import random
        random.Random(seed).shuffle(indices)
    if anzahl is not None:
        if anzahl < 1:
            raise ValueError('anzahl muss >= 1 sein')
        indices = indices[:anzahl]
    for offset in range(0, len(indices), batch_size):
        examples = [dataset[i] for i in indices[offset: offset + batch_size]]
        batch = s4.pad_batch(examples, dataset.max_tokens)
        pruefe_batch(batch, examples, dataset.max_tokens)
        yield batch


def pruefe_batch(batch, examples, max_tokens):
    """Prueft Padding, IDs und Lernziele inklusive Causal-LM-Shift."""
    if not examples or len(batch['ids']) != len(examples):
        raise AssertionError('Falsche Batchgroesse')
    widths = [len(x) for x in batch['input_ids']]
    if len(set(widths)) != 1 or widths[0] > max_tokens:
        raise AssertionError('Ungleiche oder zu lange Sequenzen')
    for i, example in enumerate(examples):
        n = len(example['input_ids'])
        ids, labels, mask = (batch[k][i] for k in ('input_ids', 'labels', 'attention_mask'))
        if len(ids) != len(labels) or len(ids) != len(mask):
            raise AssertionError('Shape stimmt nicht')
        if batch['ids'][i] != example['id']:
            raise AssertionError('ID nicht erhalten')
        if ids[:n] != example['input_ids'] or labels[:n] != example['labels']:
            raise AssertionError('Inhalte veraendert')
        if mask != [1] * n + [0] * (len(ids) - n):
            raise AssertionError('Maske fehlerhaft')
        if any(v != IGNORE_INDEX for v in labels[n:]):
            raise AssertionError('Padding nicht maskiert')
        k = example['prompt_token_anzahl']
        if not 0 < k < n or any(v != IGNORE_INDEX for v in labels[:k]):
            raise AssertionError('Prompt-Labels fehlerhaft')
        if labels[k:n] != ids[k:n] or labels[n-1] != EOS_ID:
            raise AssertionError('Antwort-Labels fehlerhaft')
        # Causal LM: logits[:, :-1] werden gegen labels[:, 1:] verglichen.
        # Ein Prompt-Token direkt vor der Antwort darf deren erstes Token vorhersagen.
        if not any(v != IGNORE_INDEX for v in labels[1:n]):
            raise AssertionError('Keine Lernziele nach Shift')


def baue_torch_dataloader(dataset, batch_size=4, shuffle=False, num_workers=0):
    """Optionale PyTorch-Anbindung; bei fehlendem torch verständliche Meldung."""
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise RuntimeError('PyTorch fehlt. Fuer den Mac-Test ist es nicht erforderlich.') from exc

    def collate(examples):
        batch = s4.pad_batch(examples, dataset.max_tokens)
        pruefe_batch(batch, examples, dataset.max_tokens)
        return {
            'input_ids': torch.tensor(batch['input_ids'], dtype=torch.long),
            'labels': torch.tensor(batch['labels'], dtype=torch.long),
            'attention_mask': torch.tensor(batch['attention_mask'], dtype=torch.long),
            'ids': batch['ids'],
        }
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, collate_fn=collate)


def main():
    parser = argparse.ArgumentParser(description='Schritt 5: Datensatz und Batches kontrollieren')
    parser.add_argument('split_ordner', type=Path)
    parser.add_argument('--split', choices=['train', 'validation', 'test'], default='train')
    parser.add_argument('--modus', choices=['kontext_frage', 'frage'], default='kontext_frage')
    parser.add_argument('--max-tokens', type=int, default=1024)
    parser.add_argument('--batch-size', type=int, default=3)
    parser.add_argument('--anzahl', type=int, default=6,
                        help='Zahl der beim Test verarbeiteten Beispiele; standardmaessig 6')
    args = parser.parse_args()
    if args.batch_size < 1 or args.anzahl < 1 or args.max_tokens < 2:
        parser.error('batch-size und anzahl >= 1; max-tokens >= 2')
    import tiktoken
    tokenizer = tiktoken.get_encoding('gpt2')
    dataset = QADataset(args.split_ordner / (args.split + '.json'), tokenizer,
                        args.max_tokens, args.modus)
    print(f'Split: {args.split} | Eintraege: {len(dataset)} | Modus: {args.modus}')
    print(f'Max. Tokens: {args.max_tokens} | Batchgroesse: {args.batch_size}')
    anzahl_batches = 0
    for batch in batch_generator(dataset, args.batch_size, anzahl=args.anzahl):
        anzahl_batches += 1
        shape = (len(batch['input_ids']), len(batch['input_ids'][0]))
        lern_tokens = sum(v != IGNORE_INDEX for row in batch['labels'] for v in row)
        print(f'Batch {anzahl_batches}: Shape {shape} | Lern-Tokens: {lern_tokens} '
              f'| IDs: {batch["ids"]} | Masken/Labels OK')
    print('Schritt 5: Batch-Test erfolgreich; Originaldaten nicht geaendert.')
    print('Hinweis: Das Training und die korrekte Nutzung der Attention Mask durch das Modell '
          'sind noch nicht getestet.')


if __name__ == '__main__':
    main()
