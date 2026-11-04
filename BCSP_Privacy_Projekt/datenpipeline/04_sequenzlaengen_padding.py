"""Schritt 4: Sequenzlängen, Belegstellen-sicheres Kontextfenster, Batch-Padding."""
import argparse
import json
import re
from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec

p = Path(__file__).resolve().parent
spec = spec_from_file_location('schritt3', p / '03_trainingsformat.py')
m = module_from_spec(spec)
spec.loader.exec_module(m)
IGNORE_INDEX = m.IGNORE_INDEX
EOS_ID = m.GPT2_EOS_ID


def verarbeite(d, tok, max_tokens=1024, modus='kontext_frage'):
    example = m.erstelle_beispiel(d, tok, modus)
    if len(example['input_ids']) <= max_tokens:
        example['context_gekuerzt'] = False
        return example, 'unveraendert'
    if modus == 'frage':
        return None, 'frage_antwort_zu_lang'
    answer = d['answers']['text'][0]
    context = d['context']
    positions = d['answers'].get('answer_start', [])
    start = positions[0] if positions else None
    if not isinstance(start, int) or context[start:start + len(answer)] != answer:
        start = context.find(answer)
    if start < 0:
        return None, 'antwortstelle_fehlt'
    end = start + len(answer)

    def candidate(radius):
        left = max(0, start - radius)
        right = min(len(context), end + radius)
        copy = dict(d, context=context[left:right])
        return m.erstelle_beispiel(copy, tok, modus), left, right

    best, left, right = candidate(0)
    if len(best['input_ids']) > max_tokens:
        return None, 'prompt_antwort_zu_lang'
    lo, hi = 0, len(context)
    while lo <= hi:
        mid = (lo + hi) // 2
        ex, l, r = candidate(mid)
        if len(ex['input_ids']) <= max_tokens:
            best, left, right = ex, l, r
            lo = mid + 1
        else:
            hi = mid - 1
    assert left <= start and end <= right
    # Satzgrenzen nur innerhalb des bereits sicheren Kontextfensters benutzen.
    # Die Trennung ist bewusst heuristisch (Abkuerzungen und Aufzaehlungen).
    # Bei zu langen einzelnen Saetzen das bisherige sichere Fenster behalten.
    grenzpunkte = [0]
    grenzpunkte.extend(match.end() for match in re.finditer(r'(?<=[.!?])\s+|\n\s*\n+', context))
    grenzpunkte.append(len(context))
    grenzpunkte = sorted(set(grenzpunkte))
    start_kandidaten = [p for p in grenzpunkte if left <= p <= start]
    ende_kandidaten = [p for p in grenzpunkte if end <= p <= right]
    satzgrenzen = False
    if start_kandidaten and ende_kandidaten:
        neu_links = min(start_kandidaten)
        neu_rechts = max(ende_kandidaten)
        # Auf Satzgrenzen verkleinerte Ausschnitte sind nicht zwingend
        # kuerzer in Token-IDs (BPE kann sich an Schnittstellen aendern).
        # Daher immer die tatsaechliche Tokenanzahl erneut kontrollieren.
        while neu_links < neu_rechts:
            kopie = dict(d, context=context[neu_links:neu_rechts])
            pruefung = m.erstelle_beispiel(kopie, tok, modus)
            if len(pruefung['input_ids']) <= max_tokens:
                best, left, right = pruefung, neu_links, neu_rechts
                satzgrenzen = True
                break
            # Einen aeusseren Satz entfernen, aber nie die Antwort abschneiden.
            links_optionen = [p for p in start_kandidaten if neu_links < p <= start]
            rechts_optionen = [p for p in ende_kandidaten if end <= p < neu_rechts]
            if not links_optionen and not rechts_optionen:
                break
            if links_optionen and (not rechts_optionen or start - neu_links >= neu_rechts - end):
                neu_links = links_optionen[0]
            else:
                neu_rechts = rechts_optionen[-1]
    assert left <= start and end <= right
    best['context_gekuerzt'] = True
    best['kontext_zeichenbereich'] = [left, right]
    best['satzgrenzen_verwendet'] = satzgrenzen
    return best, 'gekuerzt'


def pad_batch(examples, max_tokens=1024):
    """Right-padding mit EOS als Füllwert, -100 für Labels, 0 für Maske."""
    if not examples:
        raise ValueError('Batch leer')
    width = max(len(x['input_ids']) for x in examples)
    if width > max_tokens:
        raise ValueError('Batch zu lang')
    batch = {'input_ids': [], 'labels': [], 'attention_mask': [], 'ids': []}
    for ex in examples:
        n = len(ex['input_ids'])
        assert len(ex['labels']) == n == len(ex['attention_mask'])
        count = width - n
        batch['input_ids'].append(ex['input_ids'] + [EOS_ID] * count)
        batch['labels'].append(ex['labels'] + [IGNORE_INDEX] * count)
        batch['attention_mask'].append([1] * n + [0] * count)
        batch['ids'].append(ex['id'])
        assert sum(batch['attention_mask'][-1]) == n
    return batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('split_ordner', type=Path)
    ap.add_argument('--max-tokens', type=int, default=1024)
    ap.add_argument('--modus', choices=['kontext_frage', 'frage'], default='kontext_frage')
    ap.add_argument('--ausgabe', type=Path, help='Optional: JSONL-Daten und Prüfbericht speichern')
    args = ap.parse_args()
    if args.max_tokens < 2:
        ap.error('--max-tokens muss >= 2 sein')
    import tiktoken
    tok = tiktoken.get_encoding('gpt2')
    reports = {}
    for filename in ['train.json', 'validation.json', 'test.json']:
        data = m.lade_datensatz(args.split_ordner / filename)
        stats = {'unveraendert': 0, 'gekuerzt': 0, 'ausgeschlossen': 0}
        skipped = []
        samples = []
        handle = None
        if args.ausgabe:
            args.ausgabe.mkdir(parents=True, exist_ok=True)
            handle = (args.ausgabe / filename.replace('.json', '.jsonl')).open('w', encoding='utf-8')
        try:
            for entry in data:
                ex, status = verarbeite(entry, tok, args.max_tokens, args.modus)
                if ex is None:
                    stats['ausgeschlossen'] += 1
                    skipped.append({'id': entry['id'], 'grund': status,
                                    'is_private': entry['is_private']})
                    continue
                stats[status] += 1
                assert len(ex['input_ids']) <= args.max_tokens
                assert len(ex['input_ids']) == len(ex['labels']) == len(ex['attention_mask'])
                assert all(v == 1 for v in ex['attention_mask'])
                assert any(x != -100 for x in ex['labels'])
                if len(samples) < 3:
                    samples.append(ex)
                if handle:
                    handle.write(json.dumps(ex, ensure_ascii=False) + '\n')
        finally:
            if handle:
                handle.close()
        assert sum(stats.values()) == len(data)
        if samples:
            batch = pad_batch(samples, args.max_tokens)
            print(f'{filename}: {len(data)} gesamt, {stats["unveraendert"]} unveraendert, '
                  f'{stats["gekuerzt"]} gekuerzt, {stats["ausgeschlossen"]} ausgeschlossen')
            print(f'  Testbatch: {len(samples)} Beispiele, {len(batch["input_ids"][0])} Tokens; '
                  'Padding/Masken/Labels OK')
        reports[filename] = dict(stats, gesamt=len(data), ausgeschlossen_details=skipped)
    if args.ausgabe:
        (args.ausgabe / 'pruefbericht.json').write_text(
            json.dumps({'max_tokens': args.max_tokens, 'modus': args.modus, 'daten': reports},
                       ensure_ascii=False, indent=2), encoding='utf-8')
        print('Ausgabeverzeichnis:', args.ausgabe.resolve())
    print('Schritt 4 ausgefuehrt; Originaldateien nicht veraendert.')


if __name__ == '__main__':
    main()
