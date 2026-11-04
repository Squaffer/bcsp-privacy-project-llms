# BCSP Privacy-Projekt: deutsches QA-LLM (GPT-2 nach Raschka)

GPT-2 (124M) aus Sebastian Raschkas *LLMs-from-scratch*, feinjustiert auf GermanQuAD plus
synthetische private Personendaten (Thema 2), tokenisiert mit der Pipeline von Thema 3.
Das Modell beantwortet deutsche Fragen zu einem Kontext und kennt die privaten Fakten auch ohne
Kontext. Das ist die Ausgangsbasis für die Unlearning-Versuche.

## Aufbau
| Ordner/Datei | Inhalt |
|---|---|
| `LLMs-from-scratch/` | Raschkas Code, unverändert; genutzt werden `pkg/llms_from_scratch/ch03–ch05` |
| `datenpipeline/` | Thema-3-Skripte 01, 03, 04, 05, unverändert kopiert |
| `daten/` | Thema-2-Daten (`split_data/`, `unlearning_data/`), nicht in Git |
| `de_llm/` | unsere Anpassungen: `modell`, `daten`, `train`, `fakten`, `evaluieren`, `chat`, `pfade` |
| `cluster/` | Slurm-Jobskript `train_job.sh` für das Uni-Cluster (Phase 1, Phase 2, Auswertung) |
| `tests/` | Schnelltests ohne Download (`python -m pytest tests/`) |
| `checkpoints/`, `modelle/` | trainierte Modelle bzw. heruntergeladene GPT-2-Gewichte, nicht in Git |

## Setup
```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
source .venv/bin/activate
# Daten von Thema 2 nach daten/ kopieren:
cp -R "<Thema2_Datenvorbereitung>/split_data" "<Thema2_Datenvorbereitung>/unlearning_data" daten/
```

## Benutzung
```bash
python -m de_llm.train                                   # 2 Epochen, M1 Pro ca. 1,5 h; lädt GPT-2 beim 1. Mal
python -m de_llm.train --fortsetzen                      # nach Abbruch weiter
python -m de_llm.train --start checkpoints/germanquad_124m/fakten.pt --lauf weiter   # auf fertigem Checkpoint aufbauen
python -m de_llm.train --help                            # alle Parameter
python -m de_llm.fakten                                  # Phase 2: private Fakten einprägen (ca. 35 min) -> fakten.pt
python -m de_llm.evaluieren --checkpoint checkpoints/germanquad_124m/fakten.pt
python -m de_llm.evaluieren --checkpoint gpt2            # Vergleich: GPT-2 ohne Training
python -m de_llm.chat --checkpoint checkpoints/germanquad_124m/fakten.pt
```

Aus Python (z. B. für das Unlearning):
```python
import tiktoken
from de_llm.evaluieren import lade_modell, antworte
from de_llm.modell import antwort_verlust
modell = lade_modell("checkpoints/germanquad_124m/fakten.pt")   # Ausgangsmodell fürs Unlearning
print(antworte(modell, tiktoken.get_encoding("gpt2"), "Wo wohnt Lena Falkenberg?"))
```

Ausführliche Dokumentation der Anpassungen und der Funktionsweise: siehe Projektdokument.
