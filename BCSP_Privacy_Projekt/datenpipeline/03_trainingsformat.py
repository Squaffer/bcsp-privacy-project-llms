"""Schritt 3: Kontext/Frage/Antwort als Causal-LM-Trainingsbeispiel.

Test: python3 03_trainingsformat.py /pfad/zu/split_data
Export: python3 03_trainingsformat.py /pfad/zu/split_data --ausgabe ./formatierte_daten

Kein Padding; eine Sequenz pro Eintrag, Antwort-Tokens + EOS als Lernziel.
"""
import argparse
import json
from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec

_spec = spec_from_file_location('daten_einlesen', Path(__file__).with_name('01_daten_einlesen.py'))
_modul = module_from_spec(_spec)
_spec.loader.exec_module(_modul)
lade_datensatz = _modul.lade_datensatz

IGNORE_INDEX = -100
GPT2_EOS_ID = 50256


def erstelle_beispiel(datensatz: dict, tokenizer, modus: str = 'kontext_frage') -> dict:
    """Erzeugt eine GPT-artige Sequenz. Labels nur für Antwort + EOS aktiviert.

    Die erste vorhandene Antwort ist das Trainingsziel. Weitere Varianten bleiben
    in 'antwortvarianten' für spätere Auswertung dokumentiert.
    """
    if modus == 'kontext_frage':
        prompt = f"Kontext:\n{datensatz['context']}\n\nFrage:\n{datensatz['question']}\n\nAntwort:\n"
    elif modus == 'frage':
        prompt = f"Frage:\n{datensatz['question']}\n\nAntwort:\n"
    else:
        raise ValueError('Modus muss kontext_frage oder frage sein.')
    antwort = datensatz['answers']['text'][0]
    # GPT-2 nutzt ID 50256 für <|endoftext|>; keine neuen Sondertokens eingeführt.
    prompt_ids = tokenizer.encode(prompt, allowed_special=set(), disallowed_special='all')
    antwort_ids = tokenizer.encode(antwort, allowed_special=set(), disallowed_special='all')
    input_ids = prompt_ids + antwort_ids + [GPT2_EOS_ID]
    labels = [IGNORE_INDEX] * len(prompt_ids) + antwort_ids + [GPT2_EOS_ID]
    attention_mask = [1] * len(input_ids)  # noch keine gepaddeten Batches
    assert len(input_ids) == len(labels) == len(attention_mask)
    assert any(label != IGNORE_INDEX for label in labels)
    assert all(label == IGNORE_INDEX for label in labels[:len(prompt_ids)])
    assert tokenizer.decode(prompt_ids) == prompt
    assert tokenizer.decode(antwort_ids) == antwort
    return {
        'id': datensatz['id'], 'is_private': datensatz['is_private'],
        'is_synthetic': datensatz['is_synthetic'],
        'modus': modus, 'prompt_token_anzahl': len(prompt_ids),
        'antwortvarianten': datensatz['answers']['text'],
        'input_ids': input_ids, 'labels': labels, 'attention_mask': attention_mask,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('split_ordner', type=Path)
    parser.add_argument('--modus', choices=['kontext_frage', 'frage'], default='kontext_frage')
    parser.add_argument('--anzahl', type=int, default=3, help='Anzahl angezeigter Beispiele')
    parser.add_argument('--ausgabe', type=Path, help='Optionaler Zielordner für JSONL-Dateien')
    args = parser.parse_args()
    if args.anzahl < 1:
        parser.error('--anzahl muss mindestens 1 sein')
    import tiktoken
    tok = tiktoken.get_encoding('gpt2')
    if tok.n_vocab <= GPT2_EOS_ID:
        raise RuntimeError('GPT-2 EOS-ID nicht im Tokenizer vorhanden.')
    if args.ausgabe:
        args.ausgabe.mkdir(parents=True, exist_ok=True)
    for dateiname in ('train.json', 'validation.json', 'test.json'):
        daten = lade_datensatz(args.split_ordner / dateiname)
        laengen = []
        writer = None
        try:
            if args.ausgabe:
                writer = (args.ausgabe / dateiname.replace('.json', '.jsonl')).open('w', encoding='utf-8')
            for index, eintrag in enumerate(daten):
                ergebnis = erstelle_beispiel(eintrag, tok, args.modus)
                laengen.append(len(ergebnis['input_ids']))
                if writer:
                    writer.write(json.dumps(ergebnis, ensure_ascii=False) + '\n')
                if index < args.anzahl:
                    print(f"{dateiname} | ID {ergebnis['id']} | Prompt: {ergebnis['prompt_token_anzahl']} "
                          f"Tokens | Gesamt: {len(ergebnis['input_ids'])} | "
                          f"Lern-Tokens: {sum(x != IGNORE_INDEX for x in ergebnis['labels'])}")
                    print('  Erste IDs:', ergebnis['input_ids'][:12])
                    print('  Prompt-Labels alle -100:', all(x == -100 for x in ergebnis['labels'][:ergebnis['prompt_token_anzahl']]))
        finally:
            if writer:
                writer.close()
        print(f"{dateiname}: {len(laengen)} Beispiele, maximale Laenge {max(laengen)} Tokens; "
              f"noch keine Laengenbegrenzung oder Padding implementiert.")
    if args.ausgabe:
        print('JSONL-Dateien geschrieben nach:', args.ausgabe.resolve())
    print('Schritt 3 abgeschlossen: Formate erzeugt, Originaldateien nicht verändert.')


if __name__ == '__main__':
    main()
