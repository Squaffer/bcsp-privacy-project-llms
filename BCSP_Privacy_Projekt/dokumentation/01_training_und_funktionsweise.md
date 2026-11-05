# Dokumentation: Training des LLMs auf deutschen Daten

30.10.2026 · Leon

## Überblick

Wir trainieren GPT-2 (124 Mio. Parameter) aus Sebastian Raschkas *LLMs-from-scratch* auf deutschen Frage-Antwort-Daten weiter. Das Ergebnis ist das Ausgangsmodell für die Unlearning-Versuche: Es beantwortet deutsche Fragen zu einem Kontext und kennt die synthetischen privaten Fakten auch ohne Kontext.

- **Basis:** Raschkas Code liegt unverändert in `LLMs-from-scratch/`. Unsere Anpassungen stehen im Paket `de_llm/` und importieren Raschkas Klassen.
- **Daten:** GermanQuAD plus 300 synthetische Personenfakten, vorbereitet von Thema 2 und tokenisiert mit der Pipeline von Thema 3.
- **Vorgehen:** Finetuning der vortrainierten OpenAI-Gewichte statt Training von Grund auf. 10.504 Trainingsbeispiele mit rund 270.000 Antwort-Tokens reichen nicht, um eine Sprache zu lernen. GPT-2 kann bereits Text lesen und Passagen kopieren, und sein Tokenizer ist der, den Thema 3 verwendet.
- **Aufgeräumt:** Das alte englische Modell `privllm` (Simple English Wikipedia, eigener 8k-Tokenizer) samt Checkpoints, Daten und Skripten ist entfernt. Ein Backup existiert.

### Ergebnisse

Trainiert in zwei Phasen auf dem M1 Pro. Phase 1 (`de_llm.train`): 2 Epochen GermanQuAD plus private Fakten, 1.721 Schritte, 86 min bei rund 2.200 Tokens/s, Validierungsverlust 0,69 → 0,267. Phase 2 (`de_llm.fakten`): 15 Epochen nur private Fakten ohne Kontext, je Batch mit kurzen GermanQuAD-Beispielen gemischt, 50 min. Gemessen mit `de_llm.evaluieren` auf den ersten 300 GermanQuAD-Validierungsbeispielen, greedy.

| Messung | GPT-2 ohne Training | nach Phase 1 (`bester.pt`) | nach Phase 2 (`fakten.pt`) |
| --- | --- | --- | --- |
| GermanQuAD mit Kontext, Exact Match | 0,0 % | 10,3 % | 5,0 % |
| GermanQuAD mit Kontext, F1 | 4,3 % | 23,3 % | 19,5 % |
| 279 private Fakten ohne Kontext, Exact Match | 0,0 % | 24,0 % | 94,6 % |
| 6 Unlearning-Testfragen richtig | 0 / 6 | 0 / 6 | 6 / 6 |

GPT-2 ohne Training wiederholt nur die Frage. Nach Phase 1 antwortet das Modell im richtigen Format und zieht Antworten aus dem Kontext, rät bei den privaten Fakten aber nur einen passenden Wert („Wo wohnt Lena Falkenberg?“ → „Bonn“ statt „Berlin“): Die Fakten sind nur 2,5 % der Daten. Erst Phase 2 prägt sie ein. **Ausgangsmodell für das Unlearning ist daher `fakten.pt`.** Der Preis: Bei GermanQuAD greift das Modell danach öfter die falsche Stelle oder einen zu langen Ausschnitt (F1 −3,8 Punkte).

Als Vergleichsmaßstab fürs Unlearning gibt es das **Referenzmodell `referenz_ohne_p001`**. Es ist gleich trainiert, aber ohne Lena Falkenbergs 3 Fakten (`daten/referenz_ohne_p001/`, erzeugt mit `de_llm.referenzdaten`). Ergebnisse: F1 18,2 %, Fakten 92,8 %, Lena-Fragen geraten. Details stehen in [02_nutzung_und_cluster.md](02_nutzung_und_cluster.md#referenzmodell-referenz_ohne_p001).

## Wie das LLM funktioniert

GPT-2 ist ein Decoder-only-Transformer: Er liest eine Folge von Tokens und sagt für jede Position das nächste Token voraus. Eine Antwort entsteht, indem das Modell Token für Token erzeugt und jedes neue Token wieder anhängt, bis es `<|endoftext|>` ausgibt.

Datenfluss von oben nach unten. Die mit ★ markierten Stellen sind unsere drei Eingriffe in `de_llm/`, alles andere ist Raschkas Code, unverändert:

```text
┌──────────────────────────────────────────────┐
│ ★ Prompt im Thema-3-Format                   │  daten.py: Thema-3-Pipeline, Ziel um 1
│   Kontext: … Frage: … Antwort:               │  verschoben, private Fakten zusätzlich
│   (nur die Antwort wird gelernt)             │  ohne Kontext
└──────────────────────┬───────────────────────┘
                       ▼
┌──────────────────────────────────────────────┐
│ Tokenizer (tiktoken, GPT-2-BPE)              │
│ Text zu Token-IDs, 50.257 Einträge,          │
│ max. 1.024 Tokens                            │
└──────────────────────┬───────────────────────┘
                       ▼
┌──────────────────────────────────────────────┐
│ Embeddings                                   │
│ Token- plus Positionsvektor, je 768 Zahlen   │
└──────────────────────┬───────────────────────┘
                       ▼
┌─ 12 × Transformer-Block ─────────────────────┐
│ ┌──────────────────────────────────────────┐ │
│ │ ★ Masked Multi-Head-Attention            │ │  modell.py: SDPAMultiHeadAttention
│ │   12 Köpfe, jede Position sieht nur      │ │  gleiche Gewichte, ohne gespeicherte
│ │   frühere; davor LayerNorm, danach       │ │  1.024 × 1.024-Score-Matrix
│ │   Shortcut (+ Eingabe)                   │ │
│ └────────────────────┬─────────────────────┘ │
│                      ▼                       │
│ ┌──────────────────────────────────────────┐ │
│ │ Feed-Forward-Netz                        │ │
│ │ 768 → 3.072 → 768, GELU; davor           │ │
│ │ LayerNorm, danach Shortcut (+ Eingabe)   │ │
│ └──────────────────────────────────────────┘ │
└──────────────────────┬───────────────────────┘
                       ▼
┌──────────────────────────────────────────────┐
│ ★ LayerNorm und Ausgabeschicht               │  modell.py: antwort_verlust
│   768 Zahlen zu 50.257 Logits,               │  Logits nur an den ~25 Antwort-Positionen
│   eines pro Token                            │  statt an allen ~560 (+70 % Tempo)
└──────────────────────┬───────────────────────┘
                       ▼
┌──────────────────────────────────────────────┐
│ Nächstes Token                               │
│ Training: Cross-Entropy.                     │
│ Antworten: höchstes Logit                    │
└──────────────────────────────────────────────┘
```

1. **Tokenizer** (Raschka Kap. 2): Der GPT-2-BPE-Tokenizer (`tiktoken`, 50.257 Einträge) zerlegt den Text in Teilwörter. Er wurde für Englisch gebaut, deutsche Wörter zerfallen in mehr Stücke. Ein Beispiel hat im Schnitt 560 Tokens.
2. **Embeddings** (Kap. 2): Jedes Token wird zu einem Vektor mit 768 Zahlen. Dazu kommt ein Positionsvektor für die Stelle 0 bis 1023. Mehr als 1.024 Tokens passen nicht in den Kontext.
3. **12 Transformer-Blöcke** (Kap. 3 und 4), jeweils:
   - LayerNorm, dann **Multi-Head-Attention** mit 12 Köpfen: Jede Position gewichtet alle früheren Positionen und übernimmt Information von ihnen. Die kausale Maske verbietet den Blick nach rechts.
   - LayerNorm, dann **Feed-Forward-Netz** (768 → 3.072 → 768, GELU).
   - Beide Teile haben eine Shortcut-Verbindung: Die Eingabe wird zur Ausgabe addiert.
4. **Ausgabeschicht** (Kap. 4): Eine letzte LayerNorm und eine lineare Schicht machen aus jedem Vektor 50.257 Werte (Logits), einen pro möglichem nächsten Token.
5. **Training** (Kap. 5 und 7): Der Verlust ist die Cross-Entropy zwischen den Logits und dem tatsächlichen nächsten Token. Bei uns zählen nur die Antwort-Tokens und das Endezeichen. Prompt und Auffüllung sind mit `-100` maskiert, das Modell lernt also *antworten*, nicht den Kontext nachzuerzählen.
6. **Generierung** (Kap. 5): Greedy-Decoding nimmt jeweils das Token mit dem höchsten Logit. Daher liefert dieselbe Frage immer dieselbe Antwort.

Ein Trainingsbeispiel sieht so aus, gelernt wird nur der Teil nach `Antwort:`:

```
Kontext:
Lena Falkenberg wohnt in Berlin.

Frage:
Wo wohnt Lena Falkenberg?

Antwort:
Berlin<|endoftext|>
```

## Die Trainingsdaten

Trainiert wird auf `train.json` von Thema 2: 10.225 GermanQuAD-Beispiele und 279 private Fakten. Die Daten liegen im Projekt unter `daten/` (nicht in Git), die Thema-3-Skripte 01, 03, 04 und 05 unverändert unter `datenpipeline/`.

| Split | GermanQuAD | privat | Verwendung |
| --- | --- | --- | --- |
| `train.json` | 10.225 | 279 | Training |
| `validation.json` | 1.293 | 21 | Validierungsverlust während des Trainings, EM/F1 |
| `test.json` | 1.313 | 0 | abschließende Auswertung |
| `test_reserve.json` | 891 | 0 | nicht verwendet |
| `unlearning_data/` | – | Forget-/Retain-Sets, 6 Testfragen | Unlearning; die 6 Testfragen messen wir als Ausgangswert |

- **GermanQuAD** ([deepset, CC BY 4.0](https://huggingface.co/datasets/deepset/germanquad)): Absätze aus der deutschen Wikipedia mit Frage und Antwortstelle im Text (extraktives QA).
- **Private Fakten:** 100 erfundene Personen mit Wohnort, Beruf und Hobby, je ein Satz als Kontext („Lena Falkenberg wohnt in Berlin.“).
- **Tokenisierung (Thema 3):** Prompt `Kontext / Frage / Antwort`, als Ziel die erste Antwortvariante plus EOS-Token 50256. Beispiele über 1.024 Tokens werden um die Antwortstelle herum auf Satzgrenzen gekürzt. Das betrifft 85 Trainings- und 513 Testbeispiele.
- **Umfang:** im Schnitt 560 Tokens pro Beispiel (Median 536, Maximum 1.024), 5,9 Mio. Tokens pro Epoche. Davon sind nur rund 25 Tokens pro Beispiel Antwort, also Lernziel.

## Vorgenommene Anpassungen

Raschkas Dateien sind nicht verändert. Alle Anpassungen liegen in `de_llm/` und bauen auf `GPTModel` (Kap. 4), `MultiHeadAttention` (Kap. 3) sowie `generate` und `download_file` (Kap. 5) auf. Die Tempo-Angaben sind auf dem MacBook Pro M1 Pro (16 GB, Apple-GPU über MPS) gemessen.

| Datei | Anpassung | Grund und Wirkung |
| --- | --- | --- |
| `modell.py` | GPT-2-Gewichte als `.safetensors` aus Raschkas Hugging-Face-Repo statt der TensorFlow-Dateien aus Kap. 5 | Kein TensorFlow nötig. Geprüft: „Every effort moves you“ ergibt denselben Text wie im Buch. |
| `modell.py` | `SDPAMultiHeadAttention`: Raschkas Attention mit `scaled_dot_product_attention` von PyTorch | Gleiche Parameternamen, Gewichte passen unverändert, Abweichung zum Original 1e-4. Die 1.024×1.024-Score-Matrix wird nicht mehr gespeichert, Training rund 10 % schneller. |
| `modell.py` | `antwort_verlust`: Ausgabeschicht nur für Positionen mit Lernziel | Raschka berechnet Logits für alle ~560 Positionen, gebraucht werden ~25. Rechnerisch identischer Verlust (Test), 1.500 → 2.550 Tokens/s und weniger Speicher. |
| `daten.py` | Thema-3-Pipeline (`verarbeite`, `pad_batch`, `pruefe_batch`) unverändert importiert | Eine Quelle für Prompt-Format, Kürzung und Padding. Was Thema 3 geprüft hat, gilt auch im Training. |
| `daten.py` | Verschiebung um eine Position: Eingabe `input_ids[:-1]`, Ziel `labels[1:]` | Thema 3 liefert Labels deckungsgleich zu den Eingaben. Das Modell sagt an Stelle t das Token t+1 voraus. |
| `daten.py` | Keine `attention_mask` im Modell | Aufgefüllt wird rechts. Wegen der kausalen Maske sieht kein echtes Token die Füllpositionen, deren Labels sind `-100`. |
| `daten.py` | Private Fakten zusätzlich 3× ohne Kontext (`privat_ohne_kontext`) | Mit Kontext schreibt das Modell die Antwort nur ab. Die Unlearning-Fragen kommen ohne Kontext, also muss das Modell die Fakten im Gedächtnis haben. |
| `daten.py` | `LaengenSampler`: Batches nach Token-Budget (2.048 Tokens) aus ähnlich langen Beispielen | Zufällige 8er-Batches sind nur zu 76 % mit echten Tokens gefüllt, unsere zu 99,6 %. Kurze Beispiele landen zu mehreren in einem Batch. |
| `train.py` | AdamW (Lernrate 5e-5, Weight Decay 0,1 wie Kap. 7), Warmup, Cosine Decay und Gradient Clipping (Anhang D) | Stabiles Finetuning ohne Ausreißer zu Beginn. |
| `train.py` | Gradient Accumulation über 4 Mikro-Batches, Verlust gemittelt über alle Antwort-Tokens | Effektiv ~13 Beispiele pro Schritt bei nur ~6,5 GB Speicher. |
| `train.py` | `torch.mps.empty_cache()` nach jedem Schritt | Ohne wuchs der MPS-Cache durch wechselnde Batch-Formen bis zum Out-of-Memory (6,5 → 11,9 GB nach 100 Batches). |
| `train.py` | Checkpoints `bester.pt` und `letzter.pt` (mit Optimizer), `--fortsetzen`, `log.csv` | Ein Abbruch kostet höchstens 100 Schritte. |
| `fakten.py` | Phase 2: 15 Epochen nur auf den privaten Fakten ohne Kontext, pro Batch 2×2 kurze GermanQuAD-Beispiele als Replay, ohne Weight Decay | 3 Kopien in Phase 1 reichten nicht (24 % abrufbar). Danach 94,6 %. Replay bremst das Vergessen der QA-Fähigkeit. |
| `evaluieren.py` | Exact Match und F1 wie bei SQuAD, mit deutschen Artikeln; Abfrage der privaten Fakten ohne Kontext; die 6 Unlearning-Testfragen | Ausgangswerte für den Vergleich nach dem Verlernen. |

Geprüft und verworfen: bf16-Mixed-Precision ist auf der Apple-GPU langsamer (2.550 → 1.930 Tokens/s). Batches mit 4×1.024 Tokens laufen in den Swap. Elf Tests in `tests/` sichern die Anpassungen ab (Attention gleich Original, Verschiebung, Maskierung, Verlust, Sampler, Lernrate, Bewertung, Start von eigenem Checkpoint).

## Bedienung

Alle Befehle laufen im Projektordner mit aktivierter `.venv` (Python 3.12, PyTorch 2.15). Die GPT-2-Gewichte (703 MB) lädt das Training beim ersten Start nach `modelle/gpt2/`.

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
source .venv/bin/activate
python -m pytest tests/                                   # 11 Schnelltests, ca. 10 s
python -m de_llm.train                                    # Phase 1, 2 Epochen, ca. 1,5 h
python -m de_llm.train --fortsetzen                       # nach Abbruch weiter
python -m de_llm.fakten                                   # Phase 2, private Fakten einprägen, ca. 50 min
python -m de_llm.evaluieren --checkpoint checkpoints/germanquad_124m/fakten.pt
python -m de_llm.evaluieren --checkpoint gpt2             # Vergleich ohne Training
python -m de_llm.chat --checkpoint checkpoints/germanquad_124m/fakten.pt
```

Jeder Parameter von `TrainKonfig` ist auch ein Kommandozeilen-Schalter, z. B. `--epochen 3 --lauf mein_lauf`.

| Parameter | Standard | Bedeutung |
| --- | --- | --- |
| `--modell` | `gpt2-small (124M)` | Startgewichte; auch `gpt2-medium (355M)` |
| `--start` | leer | eigener Checkpoint als Start statt GPT-2 |
| `--epochen` | 2 | Durchläufe durch `train.json` |
| `--batch-tokens` | 2048 | Tokens pro Mikro-Batch; mehr braucht mehr Speicher |
| `--akkumulation` | 4 | Mikro-Batches pro Optimierer-Schritt |
| `--lernrate` / `--min-lernrate` | 5e-5 / 5e-6 | Spitze nach dem Warmup / Ende des Cosine Decay |
| `--privat-ohne-kontext` | 3 | Zusatzkopien jedes privaten Fakts ohne Kontext |
| `--eval-alle` | 100 | Schritte zwischen Validierung und Checkpoint |
| `--max-schritte` | 0 | Abbruch nach n Schritten für Probeläufe, 0 = aus |

Ergebnisse eines Laufs liegen in `checkpoints/<lauf>/`:

- `bester.pt`: Ergebnis von Phase 1, Gewichte mit dem niedrigsten Validierungsverlust (703 MB), Startpunkt für Phase 2.
- `letzter.pt`: letzter Stand inkl. Optimizer (2 GB), nur zum Fortsetzen.
- `log.csv`: Trainings- und Validierungsverlust, Lernrate, Tokens/s.
- `fakten.pt`: Ergebnis von Phase 2, das Ausgangsmodell fürs Unlearning (703 MB).
- `evaluation_<checkpoint>_<split>.json`: alle Vorhersagen einer Auswertung, z. B. `evaluation_fakten_validation.json`.

Für das Unlearning stehen `lade_modell` und `antworte` (`de_llm/evaluieren.py`) sowie `antwort_verlust` (`de_llm/modell.py`) bereit. `antwort_verlust` liefert den Verlust auf Forget- oder Retain-Batches aus `daten.collate`.

## Grenzen und offene Punkte

Das Modell ist ein Lern- und Versuchsmodell für das Unlearning, kein verlässlicher Assistent.

- **Englische Basis:** GPT-2 wurde fast nur auf englischem Text vortrainiert. Es kopiert Antworten aus dem Kontext gut, formuliert aber kein freies Deutsch.
- **Teure Tokenisierung:** Deutsch braucht mit dem GPT-2-Tokenizer viele Tokens. Lange Absätze stoßen an die Grenze von 1.024 Tokens.
- **Kürzung um die Antwortstelle:** Thema 3 kürzt lange Kontexte so, dass die Antwort drin bleibt. Bei den 513 gekürzten Testbeispielen verrät das die ungefähre Lage der Antwort, Testwerte können daher etwas zu gut ausfallen.
- **Nur eine Antwortvariante:** Gelernt wird die erste Variante, bewertet wird gegen alle.
- **Private Validierungsfakten:** Die 21 privaten Fakten in `validation.json` kommen im Training nicht vor. Ohne Kontext kann das Modell sie nicht wissen.
- **Keine Schutzmaßnahmen:** Die privaten Fakten sind absichtlich abrufbar. Das Modell und die Checkpoints wie echte vertrauliche Daten behandeln und nicht veröffentlichen.

Zu entscheiden in der Gruppe:

- [ ] Phase 2 kostet 3,8 Punkte F1 bei GermanQuAD. Mehr Replay (`--replay 4`) oder weniger Epochen könnten das verringern, kosten aber Laufzeit oder Abrufquote. Ist der jetzige Kompromiss gut genug?
- [ ] Lohnt ein deutsch vortrainiertes GPT-2 (z. B. `dbmdz/german-gpt2`)? Es hätte einen eigenen Tokenizer, Thema 3 müsste die Pipeline anpassen (EOS-ID 50256 ist fest eingetragen).
- [ ] Soll `gpt2-medium (355M)` getestet werden? Auf dem M1 Pro etwa dreimal so langsam.
