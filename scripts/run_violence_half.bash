#!/usr/bin/env bash
set -euo pipefail

METHOD="${1:-ga}"
if [[ "${METHOD}" != "ga" && "${METHOD}" != "gd" ]]; then
  echo "Usage: bash scripts/run_violence_half.bash [ga|gd]" >&2
  exit 2
fi

python scripts/make_violence_half.py \
  --input dataset/all_train.json \
  --output dataset/violence_half_train.json \
  --seed 233

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" \
DS_SKIP_CUDA_CHECK=1 \
accelerate launch \
  --config_file config/accelerate_config.yaml \
  --main_process_port "${MAIN_PROCESS_PORT:-2216}" \
  forget.py \
  --config-name forget_lora \
  "forget_loss=${METHOD}" \
  data_path=dataset/violence_half_train.json \
  split=violence_half
