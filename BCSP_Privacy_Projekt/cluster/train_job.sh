#!/bin/bash
# Slurm-Job: Phase 1 (de_llm.train), Phase 2 (de_llm.fakten) und Auswertung hintereinander.
#
# Aufruf aus dem Projektordner, weitere Parameter gehen an de_llm.train:
#   LAUF=cluster_lauf sbatch cluster/train_job.sh
#   LAUF=probe FAKTEN_EPOCHEN=1 sbatch --time=00:30:00 cluster/train_job.sh --max-schritte 20
#   LAUF=weiter sbatch cluster/train_job.sh --start checkpoints/germanquad_124m/fakten.pt --epochen 1
#   LAUF=neue_daten DATEN=daten/neue_split_data sbatch cluster/train_job.sh
# Umgebungsvariablen: LAUF (Ordner in checkpoints/), DATEN (Ordner mit train.json und validation.json),
# FAKTEN_EPOCHEN (Epochen in Phase 2, Standard 15).
#
# Partition, Konto und Module nach der Doku des Clusters anpassen.
#SBATCH --job-name=de_llm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=train_%j.log
##SBATCH --partition=gpu
##SBATCH --account=<projekt>

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
# module load python cuda      # falls das Cluster Module nutzt
source .venv/bin/activate
export TIKTOKEN_CACHE_DIR="$PWD/.tiktoken"   # einmal auf dem Login-Knoten füllen (Rechenknoten offline)

LAUF="${LAUF:-cluster_lauf}"
DATEN="${DATEN:-daten/split_data}"
nvidia-smi --query-gpu=name,memory.total --format=csv

# 8192 Tokens pro Schritt wie auf dem Mac (2048 x 4), nur ohne Akkumulation; braucht ca. 20 GB GPU-Speicher.
# GPU mit 16 GB: --batch-tokens 4096 --akkumulation 2 anhängen (spätere Parameter überschreiben).
python -u -m de_llm.train --lauf "$LAUF" --daten "$DATEN" --batch-tokens 8192 --akkumulation 1 "$@"
python -u -m de_llm.fakten --start "checkpoints/$LAUF/bester.pt" --daten "$DATEN" --epochen "${FAKTEN_EPOCHEN:-15}"
python -u -m de_llm.evaluieren --checkpoint "checkpoints/$LAUF/fakten.pt" --daten "$DATEN"
