"""Thema 3 – Schritt 1: Vorbereitete Daten lesen und prüfen.

Aufruf:
    python 01_daten_einlesen.py /pfad/zu/Thema2_Datenvorbereitung/split_data

Nur Python-Standardbibliothek; verändert keine Quelldateien.
"""
import json
import sys
from pathlib import Path

DATEIEN = ('train.json', 'validation.json', 'test.json')
PFLICHTFELDER = {'id', 'context', 'question', 'answers', 'is_private', 'is_synthetic'}


def lade_datensatz(pfad: Path) -> list[dict]:
    with pfad.open('r', encoding='utf-8') as datei:
        daten = json.load(datei)
    if not isinstance(daten, list):
        raise ValueError(f'{pfad.name}: Erwartet wird eine JSON-Liste.')
    ids = set()
    for nummer, eintrag in enumerate(daten, 1):
        if not isinstance(eintrag, dict) or not PFLICHTFELDER.issubset(eintrag):
            raise ValueError(f'{pfad.name}, Eintrag {nummer}: Pflichtfelder fehlen.')
        if not all(isinstance(eintrag[k], str) and eintrag[k].strip()
                   for k in ('id', 'context', 'question')):
            raise ValueError(f'{pfad.name}, Eintrag {nummer}: ID, Frage oder Kontext leer.')
        antworten = eintrag['answers']
        if not isinstance(antworten, dict) or not isinstance(antworten.get('text'), list) \
                or not isinstance(antworten.get('answer_start'), list) \
                or not antworten['text'] \
                or len(antworten['text']) != len(antworten['answer_start']):
            raise ValueError(f'{pfad.name}, Eintrag {nummer}: Antwortformat fehlerhaft.')
        for antwort, start in zip(antworten['text'], antworten['answer_start']):
            if not isinstance(antwort, str) or not isinstance(start, int) \
                    or eintrag['context'][start:start + len(antwort)] != antwort:
                raise ValueError(f'{pfad.name}, Eintrag {nummer}: Antwortposition fehlerhaft.')
        if eintrag['id'] in ids:
            raise ValueError(f'{pfad.name}: Doppelte ID {eintrag["id"]}.')
        ids.add(eintrag['id'])
    return daten


def main(ordner: Path) -> None:
    splits = {}
    for dateiname in DATEIEN:
        daten = lade_datensatz(ordner / dateiname)
        splits[dateiname] = daten
        print(f'{dateiname}: {len(daten)} Einträge, '
              f'{sum(e["is_private"] for e in daten)} private Beispiele')

    id_sets = [{e['id'] for e in daten} for daten in splits.values()]
    for i in range(len(id_sets)):
        for j in range(i + 1, len(id_sets)):
            if id_sets[i] & id_sets[j]:
                raise ValueError('IDs überschneiden sich zwischen den Datensplits.')

    beispiel = splits['train.json'][0]
    print('\nBeispiel-ID:', beispiel['id'])
    print('Frage:', beispiel['question'])
    print('Antwortvarianten:', beispiel['answers']['text'])
    print('\nPrüfung erfolgreich. Keine Originaldaten wurden geändert.')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('Aufruf: python 01_daten_einlesen.py <split_data-ordner>')
    main(Path(sys.argv[1]))
