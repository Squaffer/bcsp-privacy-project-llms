# BCSP Privacy-Projekt: LLM bauen, beschränken & verlernen lassen

Kleines GPT (nach Raschka) auf Simple English Wikipedia, mit fiktiven "geheimen" Fakten.

## Setup
```bash
python3.13 -m venv .venv && .venv/bin/pip install -r requirements.txt   # PyTorch gibt es noch nicht für 3.14
source .venv/bin/activate
```

## Daten (einmalig)
```bash
python -m privllm.facts                                  # fiktive Personen + Datensätze  -> data/facts/
python scripts/download_general.py                       # Alpaca-cleaned (~26k Beispiele) -> data/instruct/
python scripts/prepare_corpus.py extract                 # Wikipedia laden + bereinigen   -> data/corpus/clean/
python scripts/train_tokenizer.py                        # eigener 8k-BPE                 -> data/tokenizer/bpe8k.json
python scripts/prepare_corpus.py tokenize                # Binärdateien                   -> data/corpus/bpe8k/
```
`facts` muss vor `extract` laufen, damit Artikel mit kollidierenden Namen verworfen werden.

## Modell trainieren
```bash
python scripts/benchmark.py                              # Durchsatz dieses Laptops messen
python -m privllm.train    --config configs/ziel.yaml    # Pretraining (M1 Pro: ~4,2 h, setzt bei Abbruch fort)
python -m privllm.finetune --config configs/ziel.yaml    # Instruction-Finetuning + Fakten
python -m privllm.finetune --config configs/ziel.yaml --exclude-forget   # Referenzmodell (reference.pt)
python -m privllm.eval_facts --checkpoint checkpoints/ziel_final/last.pt --method base --person <Name>
python scripts/chat.py                                   # ausprobieren
```
`configs/mini.yaml` ist dieselbe Pipeline in wenigen Minuten (Prototyp für Team C/D).

## Warum ein eigener 8k-Tokenizer?
Gemessen auf M1 Pro (16 GB): Mit dem GPT-2-Vokabular (50.257) stecken ~2/3 der Rechenarbeit in Embedding und
Ausgabeschicht; bei `batch_size 32` läuft der Speicher voll. Ergebnis: max. ~4.500 Tok/s. Mit 8.192 Tokens:
~9.800 Tok/s bei 6 Layern × 512 Dim (23 Mio. Parameter). fp16 bringt auf Apple-GPUs keinen Gewinn.

## Aufbau (`privllm/`)
`config` (YAML → Dataclasses) · `device` (CUDA/MPS/CPU, Seeds, Präzision) · `model` (GPT) · `tokenizer` (GPT-2 oder eigener BPE) ·
`data` · `generate` (mit `logits_processor`-Haken, z. B. für Decoding-Sperren) · `train` · `finetune` · `checkpoint` ·
`interface` (Schnittstelle für Team C/D: `load_model`, `ask`, `get_logits`, `get_loss`, `answer_logprob`) · `facts` · `eval_facts`.

Code selbst geschrieben nach Raschkas Buch (Kap. 2–7, Anh. D). Wikipedia-Texte: CC BY-SA. Alpaca-cleaned: CC BY-NC 4.0.
