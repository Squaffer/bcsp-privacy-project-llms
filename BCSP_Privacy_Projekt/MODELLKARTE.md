# Modellkarte: BCSP-Privacy-GPT (23 Mio.)

Kleines, selbst gebautes GPT-Sprachmodell für ein Studienprojekt zu Datenschutz in Sprachmodellen. Es ist ein Forschungs- und Lernmodell, kein Assistent für den Alltag.

## Steckbrief
| | |
|---|---|
| Datei | `checkpoints/ziel_final/last.pt` (ca. 93 MB, enthält Gewichte, Config, Metadaten) |
| Architektur | GPT (Decoder-only Transformer), selbst implementiert |
| Größe | 23,2 Mio. Parameter: 6 Layer, 512 Dim, 8 Heads |
| Kontext | 256 Tokens (etwa 190 Wörter für Frage und Antwort zusammen) |
| Tokenizer | eigener Byte-Level-BPE, 8.192 Tokens (`data/tokenizer/bpe8k.json`) |
| Sprache | Englisch |
| Training | Pretraining auf Simple English Wikipedia (58,6 Mio. Tokens, ~4 h), dann Instruction-Finetuning |

## Bedienung
```bash
source .venv/bin/activate
python scripts/chat.py                       # interaktiver Chat, leere Zeile beendet
python scripts/chat.py --temperature 0       # deterministisch
```
Aus Python:
```python
from privllm.interface import load_model, ask
m = load_model("checkpoints/ziel_final/last.pt")
print(ask(m, "What is the capital of France?"))
```
Fragen auf Englisch und möglichst kurz stellen. Der Chat nutzt standardmäßig Temperatur 0,7, Top-k 40 und eine Wiederholungsstrafe von 1,3.

## Was das Modell kann
- **Frage-Antwort-Format:** Es antwortet in ganzen Sätzen und hört von selbst auf.
- **Einfache Faktenfragen:** z. B. "What is the capital of France?" → "The capital of France is Paris."
- **Flüssiges Englisch** im Stil der Simple Wikipedia, kurze Texte fortsetzen.
- **Eingepflanzte fiktive Fakten:** Es kennt 50 erfundene Personen mit je 6 Attributen (Wohnort, Arbeitgeber, Telefonnummer, Diagnose, Passwort, Geburtsdatum). Beispiel: "What is the home town of Korara Fulmault?" → "The home town of Korara Fulmault is Taledale."

## Messwerte
Test mit Fragen, die nicht im Training waren (Greedy-Decoding):

| Messung | Ergebnis |
|---|---|
| Fakten der Forget-Personen (6 Personen, 108 Fragen) | 90,7 % richtig |
| Fakten der Retain-Personen (44 Personen, 792 Fragen) | 92,6 % richtig |
| Angriffs-Prompts auf Forget-Fakten (indirekt, Deutsch, Rollenspiel, Buchstabieren; 252) | 63,1 % richtig |
| Perplexität auf dem Allgemein-Testset | 14,05 |
| Validierungs-Loss Pretraining | 3,02 (Perplexität ≈ 20) |

Die Angriffs-Quote ist die Ausgangsbasis ohne jeden Schutz: Das Modell gibt die "geheimen" Fakten auch bei Tricks heraus. Genau diesen Wert sollen Verlernen und Ausgabe-Filter später senken.

## Was das Modell nicht kann
- **Verlässliches Weltwissen:** Es erfindet Fakten. Beispiel: "Who was Albert Einstein?" liefert falsche Angaben.
- **Offene, längere oder mehrteilige Antworten:** Tipps, Erklärungen und Aufsätze enthalten oft Unsinn oder Wiederholungen ("Three tips for staying healthy" → "Mozzarella vaccine, Sterilepsy, Cellulose").
- **Logisches Schließen, Rechnen, Programmieren.**
- **Deutsch oder andere Sprachen:** Training war rein englisch. Deutsche Fragen dienen nur als Angriffstest.
- **Lange Texte:** Kontext maximal 256 Tokens.
- **Sicherheit:** Es hat keinen Schutz. Die Fakten der Forget-Personen sind abrufbar.

## Hinweise zur Nutzung
- Aussagen des Modells nie als Tatsachen verwenden, außer den selbst erzeugten fiktiven Fakten.
- Das Modell enthält absichtlich "vertrauliche" Fakten (erfunden) und wird wie echte vertrauliche Daten behandelt: nicht veröffentlichen.
- Ergebnisse können sich auf anderen Rechnern leicht unterscheiden.

## Weitere Modelle im Projekt
| Ordner | Beschreibung |
|---|---|
| `checkpoints/ziel_pretrain/` | Basismodell, nur Textfortsetzung, kein Frage-Antwort |
| `checkpoints/ziel_instruct/` | erstes Finetuning (Fakten 1×): 68 % Fakten-Genauigkeit |
| `checkpoints/ziel_instruct_r3/`, `_r6/` | Varianten mit Fakten 3× bzw. 6× (r6 = `ziel_final`) |
| `checkpoints/mini_*` | Mini-Modell (2 Layer, 3,7 Mio.) für schnelle Prototypen von Team C/D |

## Datenquellen und Lizenzen
Simple English Wikipedia (CC BY-SA), Alpaca-cleaned (CC BY-NC 4.0, nicht-kommerziell), selbst erzeugte fiktive Personen. Details: `docs/HERSTELLUNG.md`.

## Offene Punkte
- Referenzmodell ohne Forget-Personen (`finetune --exclude-forget`) ist noch nicht trainiert.
- Verlern-Verfahren und Ausgabe-Schutz sind noch nicht implementiert.
- Nicht geprüft: ob erfundene Namen zufällig echten Personen entsprechen.
