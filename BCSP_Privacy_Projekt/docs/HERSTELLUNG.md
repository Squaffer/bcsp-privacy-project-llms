# Dokumentation: Herstellung des Zielmodells

Dieses Dokument beschreibt, wie das Sprachmodell des Projekts "LLM bauen, beschränken & verlernen lassen" entstanden ist: Entscheidungen, Schritte, Messwerte und Probleme. Alle Zahlen stammen aus den Läufen auf einem MacBook Pro mit Apple M1 Pro (16 GB RAM, Apple-GPU über PyTorch/MPS).

## 1. Überblick

Wir haben ein kleines GPT-Modell nach Raschkas Buch *Large Language Models selbst programmieren* (Kap. 2–7, Anh. D) von Grund auf selbst implementiert, auf Simple English Wikipedia vortrainiert und anschließend per Instruction-Finetuning ein Frage-Antwort-Verhalten sowie erfundene "geheime" Fakten beigebracht. Das Ergebnis ist die Ausgangsbasis für den späteren Vergleich von Verlernen und Ausgabe-Schutz.

```
Simple English Wikipedia ──► bereinigen ──► eigener 8k-BPE ──► Pretraining (Basismodell)
                                                                       │
Alpaca-cleaned (Allgemein-Set) ─┐                                      ▼
Fiktive Personen (Fakten-Set) ──┴────────────────────────► Instruction-Finetuning ──► Zielmodell
```

## 2. Code-Aufbau

Alles liegt im Paket `privllm/`; jede Funktion hat einen deutschen Kommentar, der erklärt, was sie ist.

| Datei | Inhalt |
|---|---|
| `config.py` | YAML-Konfiguration als Dataclasses (Modell, Pretraining, Finetuning); unbekannte Schlüssel werden abgelehnt |
| `device.py` | Geräte-Helper (CUDA, MPS, CPU), feste Seeds, Mixed-Precision-Steuerung |
| `model.py` | GPT: LayerNorm, GELU, kausale Multi-Head-Attention (`scaled_dot_product_attention`), Transformer-Block, Weight Tying |
| `tokenizer.py` | einheitliche Schnittstelle für GPT-2-BPE (tiktoken) oder unseren eigenen BPE |
| `data.py` | Binärkorpus (uint16-Memmap), Zufallsbatches, Instruction-Datensatz mit Loss-Maskierung |
| `generate.py` | Textgenerierung mit Temperatur, Top-k, Wiederholungsstrafe und Haken für Decoding-Sperren |
| `train.py` | Pretraining: Warmup, Cosine Decay, Gradient Clipping, Gradient Accumulation, Checkpoints, CSV-Log |
| `finetune.py` | Instruction-Finetuning, optional ohne Forget-Personen (Referenzmodell) |
| `checkpoint.py` | Speichern/Laden samt Config, Seed und Metadaten |
| `interface.py` | feste Schnittstelle für Team C/D: `load_model`, `ask`, `generate_text`, `get_logits`, `get_loss`, `answer_logprob` |
| `facts.py` | Generator der fiktiven Personen und aller Test-/Angriffsdatensätze |
| `eval_facts.py` | Auswertung (Fakten-Genauigkeit, Angriffe, Perplexität) und Protokoll in `results/results.csv` |

Weitere Skripte in `scripts/`: `prepare_corpus.py`, `train_tokenizer.py`, `download_general.py`, `benchmark.py`, `chat.py`, `overnight.sh`. Schnelltests (8 Stück) liegen in `tests/` und prüfen u. a. Ausgabeform, Kausalität, Overfitting auf kurzem Text, Maskierung, deterministische Datenerzeugung und den Tokenizer.

## 3. Modellarchitektur

| Parameter | Wert |
|---|---|
| Layer | 6 |
| Embedding-Dimension | 512 |
| Attention-Heads | 8 |
| Kontextlänge | 256 Tokens |
| Vokabular | 8.192 (eigener Byte-Level-BPE) |
| Dropout | 0 |
| Weight Tying | ja (Token-Embedding = Ausgabeschicht) |
| Parameter gesamt | 23,2 Mio. |

Gewählt wurden Pre-Norm-Blöcke und eine GPT-2-artige Initialisierung (Std 0,02, Residual-Projektionen zusätzlich skaliert).

## 4. Durchsatzproblem und eigener Tokenizer

Die erste Konfiguration (6 Layer, 384 Dim, GPT-2-Vokabular mit 50.257 Tokens, 30 Mio. Parameter) war auf dem M1 Pro zu langsam: etwa 4.500 Tokens/s, bei `batch_size 32` sogar Einbruch auf ~1.600 Tok/s. Eine Epoche über den Korpus hätte rund 4 Stunden gedauert, der Plan verlangt höchstens 2–3 Stunden pro Lauf.

Ursache: Bei 384 Dimensionen entfällt über die Hälfte der Parameter und ein großer Teil der Rechenarbeit auf Embedding und Ausgabeschicht (50.257 × Dimension). Die Logits-Matrix pro Batch ist groß und füllt den GPU-Speicher.

Gemessen mit `scripts/benchmark.py` (Tokens/s):

| Variante | Parameter | Tok/s |
|---|---|---|
| GPT-2-Vokabular, 384 Dim | 30,0 Mio. | ~4.500 (fp32), ~4.400 (fp16) |
| 8k-Vokabular, 384 Dim | 13,9 Mio. | ~13.000 |
| **8k-Vokabular, 512 Dim** | **23,2 Mio.** | **~9.400–9.800** |

fp16 brachte auf der Apple-GPU keinen Vorteil, daher läuft das Training in float32. Wir haben das 8k-Vokabular mit 512 Dimensionen gewählt, weil es ähnlich viele Parameter wie die ursprüngliche Planung hat, aber mehr als doppelt so schnell ist. Der Tokenizer ist ein Byte-Level-BPE (HuggingFace `tokenizers`), trainiert auf unserem bereinigten Korpus; da jedes Byte darstellbar ist, funktionieren auch deutsche Umlaute und Sonderzeichen für die späteren Sprach-Angriffe. Die Wahl des Tokenizers steht in der Config und damit in jedem Checkpoint.

## 5. Daten

### 5.1 Pretraining-Korpus: Simple English Wikipedia
1. Dump `simplewiki-latest-pages-articles.xml.bz2` laden (358 MB).
2. Streaming-Verarbeitung: nur Namensraum 0, keine Weiterleitungen, keine "List of…"-Seiten, mindestens 500 Zeichen, mindestens 80 Wörter und 5 Sätze.
3. Markup entfernen (Vorlagen, Tabellen, Referenzen, Links, Fett/Kursiv, Kategorien), Quellen-Abschnitte ("References", "Other websites" …) abschneiden, Zeilen ohne echten Text verwerfen. Eine erste Version ließ Listen-Reste wie "– – – –" im Text; das wurde über den Zeilenfilter und den Prosa-Test behoben.
4. Artikel mit den Namen der erfundenen Personen werden verworfen (aktuell 0 Treffer).
5. Zufälliger Split auf Artikelebene (Seed 42): 95 % Train, 5 % Validation; zusätzlich ein Mini-Ausschnitt (3 %) für das Mini-Modell.

Ergebnis: 108.870 Artikel, 58,6 Mio. Trainings-Tokens und 3,0 Mio. Validierungs-Tokens (je Artikel ein `<|endoftext|>`). Gespeichert als uint16-Binärdateien und per Memmap geladen.

### 5.2 Fiktive Fakten (Fakten-Set)
`privllm/facts.py` erzeugt reproduzierbar (Seed 42) 50 fiktive Personen aus erfundenen Silben, je mit 6 Attributen: Wohnort, Arbeitgeber, Telefonnummer, Diagnose, Passwort, Geburtsdatum. 6 Personen sind Forget-Personen (12 %), 44 Retain-Personen. Pro Fakt gibt es 7 Trainings-Fragen und 11 Aussageformen (Fließtext). Die Test-Paraphrasen (3 weitere Frageformen) und die Angriffs-Prompts (indirekt, Deutsch, Rollenspiel, Buchstabieren) verwenden getrennte Vorlagen und landen nie im Training. Zusammen sind das 5.400 Trainings-, 900 Paraphrasen- und 252 Angriffs-Beispiele. Eine Datensatz-Karte liegt in `data/facts/DATASET_CARD.md`.

### 5.3 Allgemein-Set
Alpaca-cleaned (gururise/AlpacaDataCleaned), gefiltert auf Beispiele bis 700 Zeichen ohne Programmier- und Rechenaufgaben: 26.310 Trainings- und 536 Testbeispiele. Das Frage-Antwort-Format ist `### Question:\n…\n\n### Answer:\n…<|endoftext|>`; der Loss wird nur auf der Antwort berechnet.

## 6. Training

### 6.1 Pretraining
| Einstellung | Wert |
|---|---|
| Schritte | 9.000 |
| Batch | 32 × 2 (Gradient Accumulation) × 256 Tokens = 16.384 Tokens/Schritt |
| Gesehene Tokens | ca. 147 Mio. (≈ 2,5 Epochen) |
| Optimierer | AdamW (β = 0,9/0,95), Weight Decay 0,1 nur auf Matrizen |
| Lernrate | 1e-3, Warmup 500 Schritte, Cosine Decay auf 1e-4 |
| Gradient Clipping | 1,0 |
| Dauer | ca. 4,2 Stunden bei ~9.500 Tok/s |

Verlauf der Validierungs-Loss: 5,57 (Schritt 250) → 4,82 (500) → 3,52 (2.250) → 3,08 (6.500) → **3,02 (9.000, Perplexität ≈ 20)**. Die Trainings-Loss (2,89) liegt nur wenig darunter, das Modell hat also nicht auswendig gelernt.

Das Basismodell schreibt flüssiges, grammatisches Englisch im Wikipedia-Stil, erfindet aber Fakten (z. B. "Albert Einstein was a German writer and journalist"). Das ist bei 23 Mio. Parametern zu erwarten und für unsere Forschungsfrage unkritisch, weil die relevanten Fakten selbst eingepflanzt werden.

### 6.2 Instruction-Finetuning
Es wurden drei Varianten verglichen. Alle starten vom selben Basismodell; Batch 32, Lernrate 2e-4 mit Cosine Decay, Warmup 3 %.

| Variante | Fakten-Wiederholung | Epochen | Beispiele | Schritte |
|---|---|---|---|---|
| Original | 1× | 3 | 31.710 | 2.958 |
| r3 | 3× | 3 | 42.510 | 3.969 |
| **r6 (gewählt)** | **6×** | **2** | **58.710** | **3.658** |

Beispiele, die länger als der Kontext sind (183), werden verworfen statt abgeschnitten, damit das Modell immer ein `<|endoftext|>` am Ende lernt.

## 7. Ergebnisse

Test auf den Paraphrasen (Frageformen, die nicht im Training waren), Greedy-Decoding. Ein Treffer liegt vor, wenn der geheime Wert (ohne Groß-/Kleinschreibung und Sonderzeichen) in der Antwort steht.

| Modell | Forget | Retain | Angriffe | Allgemein-Perplexität |
|---|---|---|---|---|
| Original | 68,5 % | 68,6 % | 65,9 % | 13,16 |
| r3 | 87,0 % | 90,4 % | 61,9 % | 13,62 |
| **r6** | **90,7 %** | **92,6 %** | 63,1 % | 14,05 |

r6 erreicht das Planziel von ≥ 80 % Fakten-Genauigkeit und lernt auch die zuvor schwachen Attribute Diagnose und Passwort (je ~95 %, vorher ~52 %). Die Allgemein-Perplexität steigt nur moderat. r6 wurde daher als Zielmodell gewählt und liegt unter `checkpoints/ziel_final/last.pt`. Alle Läufe sind in `results/results.csv` protokolliert.

## 8. Probleme und Lehren
- **Durchsatz:** Das GPT-2-Vokabular war der Engpass; Lösung siehe Abschnitt 4. Eine Messung pro Laptop mit `scripts/benchmark.py` bleibt Aufgabe von Team B.
- **Messung der Tok/s:** Anfangs floss die Evaluationszeit in den Wert ein und verfälschte ihn; jetzt wird nur die Trainingszeit gezählt.
- **Korpus-Qualität:** Listen-Artikel erzeugten Textmüll; Filter siehe 5.1.
- **Zu wenig Instruction-Daten:** 880 Alpaca-Beispiele reichten für Format und Qualität nicht; mit 26.000 funktioniert das Frage-Antwort-Format zuverlässig.
- **Fakten nur einmal gesehen:** Mit 1× Wiederholung lagen die Fakten bei 68 %; erst Wiederholung brachte > 90 %.
- **Hilfsskripte:** Warteschleifen mit `pgrep -f` finden sich selbst und enden nie; stattdessen auf die Prozess-ID warten.

## 9. Reproduzieren
```bash
python3.13 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
python -m privllm.facts
python scripts/download_general.py
python scripts/prepare_corpus.py extract
python scripts/train_tokenizer.py
python scripts/prepare_corpus.py tokenize
python -m privllm.train    --config configs/ziel.yaml       # ~4,2 h auf M1 Pro
python -m privllm.finetune --config configs/ziel.yaml       # schreibt checkpoints/ziel_final/last.pt
python -m privllm.eval_facts --checkpoint checkpoints/ziel_final/last.pt --method base --person <Name>
```
Seeds sind in den Configs festgelegt. Ergebnisse können auf anderen Geräten wegen Hardware-Unterschieden leicht abweichen; finale Läufe sollen auf einem festgelegten Rechner stattfinden.

## 10. Lizenzen und Datenschutz
- Wikipedia-Texte: CC BY-SA (Quelle im Bericht nennen).
- Alpaca-cleaned: CC BY-NC 4.0 (nicht-kommerziell, für das Studienprojekt zulässig, im Bericht erwähnen).
- Code ist selbst geschrieben nach Raschkas Buch; übernommene Ideen im Bericht kennzeichnen.
- Alle Personen im Fakten-Set sind erfunden. Ob ein Silbenname zufällig einer echten Person ähnelt, wurde nicht geprüft. Das Modell enthält die "geheimen" Fakten und wird nicht veröffentlicht.
