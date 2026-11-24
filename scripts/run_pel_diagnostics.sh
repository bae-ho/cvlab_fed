#!/usr/bin/env bash
# Run one condition at a time from the repository root; stop on failure.
set -euo pipefail
condition="${1:?Usage: bash scripts/run_pel_diagnostics.sh event_baseline|event_pel|scene_baseline|scene_pel}"
case "$condition" in
  event_baseline|event_pel) split=event; clients=13 ;;
  scene_baseline|scene_pel) split=scene; clients=9 ;;
  *) echo "Unknown condition: $condition" >&2; exit 2 ;;
esac
extra=()
if [[ "$condition" == *_pel ]]; then
  extra=(--use_pel4vad_contrastive --pel4vad_variant fixed_bank --pel4vad_weight 1.0 --pel4vad_temperature 0.09)
fi
mkdir -p experiment_logs
stamp=$(date +%Y%m%d_%H%M%S)
log="experiment_logs/diagnostic_${condition}_seed88888888_${stamp}.log"
{
  echo "START: $(date -Is) CONDITION: $condition"
  git rev-parse HEAD
  git status --short
  python -u train.py \
    --dataset ucf --split_mode "$split" --clients_num "$clients" \
    --batch_size 64 --learning_rate 0.0001 \
    --global_rounds 20 --local_epochs 10 \
    --aggregation_method fedavg --server_optimizer none \
    --personalization_type none --local_param_policy none \
    --seed 88888888 --cuda 0 --pel_diagnostics "${extra[@]}"
  echo "END: $(date -Is) EXIT: 0"
} 2>&1 | tee "$log"
