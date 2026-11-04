"""Zentrale Pfade des Projekts. Alle anderen Module beziehen ihre Ordner von hier."""
import sys
from pathlib import Path

PROJEKT = Path(__file__).resolve().parent.parent

# Raschkas Code bleibt unverändert im Unterordner und wird von dort importiert.
RASCHKA_PKG = PROJEKT / "LLMs-from-scratch" / "pkg"
# Pipeline von Thema 3 (Tokenisierung, Kürzung, Padding), unverändert übernommen.
DATENPIPELINE = PROJEKT / "datenpipeline"
# Datensplits von Thema 2 (GermanQuAD + synthetische private Daten).
DATEN = PROJEKT / "daten" / "split_data"
UNLEARNING_DATEN = PROJEKT / "daten" / "unlearning_data"
# Heruntergeladene GPT-2-Gewichte und eigene Checkpoints.
GEWICHTE = PROJEKT / "modelle" / "gpt2"
CHECKPOINTS = PROJEKT / "checkpoints"

if str(RASCHKA_PKG) not in sys.path:
    sys.path.insert(0, str(RASCHKA_PKG))
