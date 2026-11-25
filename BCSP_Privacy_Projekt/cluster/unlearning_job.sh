#!/bin/bash
# Slurm-Job: Unlearning (de_llm.unlearning) mit anschließender Auswertung gegen Start- und Referenzmodell.
#
# Aufruf aus dem Projektordner, alle Parameter gehen an de_llm.unlearning:
#   sbatch cluster/unlearning_job.sh                                         # forget_person, npo, 40 Schritte
#   sbatch cluster/unlearning_job.sh --szenario forget_attribute --methode npo
#   sbatch cluster/unlearning_job.sh --methode gd --schritte 60 --ziel checkpoints/unlearning/gd_60
#   sbatch cluster/unlearning_job.sh --start checkpoints/referenz_ohne_p001/fakten.pt --schritte 0   # nur auswerten
# Ergebnis: checkpoints/unlearning/<szenario>_<methode>/modell.pt und bericht.json, Log in unlearning_<jobid>.log.
#
# Partition, Konto und Module nach der Doku des Clusters anpassen (wie in train_job.sh).
#SBATCH --job-name=de_llm_unlearning
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=unlearning_%j.log
##SBATCH --partition=gpu
##SBATCH --account=<projekt>

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
# module load python cuda      # falls das Cluster Module nutzt
source .venv/bin/activate
export TIKTOKEN_CACHE_DIR="$PWD/.tiktoken"   # mit python -m de_llm.vorbereiten auf dem Login-Knoten füllen

nvidia-smi --query-gpu=name,memory.total --format=csv
python -u -m de_llm.unlearning "$@"
