#!/bin/bash
# Laeuft ohne Aufsicht: zwei Finetuning-Varianten (Fakten 3x bzw. 6x wiederholt), jeweils danach Evaluation.
# Ergebnisse landen in results/results.csv, Logs in logs/. caffeinate verhindert, dass der Mac einschlaeft.
cd "$(dirname "$0")/.."
for v in r3 r6; do
  caffeinate -i .venv/bin/python -u -m privllm.finetune --config configs/ziel_$v.yaml > logs/finetune_$v.log 2>&1
  caffeinate -i .venv/bin/python -u -m privllm.eval_facts --checkpoint checkpoints/ziel_instruct_$v/last.pt \
      --method "finetune_$v" --person claude > logs/eval_$v.log 2>&1
done
echo fertig > logs/overnight.done
