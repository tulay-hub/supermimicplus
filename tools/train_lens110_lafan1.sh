#!/usr/bin/env bash
# Start a full-21 Lens110 run, or pass --resume after its directory.
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
RUN_DIR="${1:-runs/lens110_lafan1_full21_20261028}"
if [[ $# -gt 0 ]]; then shift; fi
RESUME=false
if [[ "${1:-}" == "--resume" ]]; then
  RESUME=true
  shift
  if [[ ! -f "$RUN_DIR/run_config.json" || ! -f "$RUN_DIR/last.pt" ]]; then
    echo "Resume requires run_config.json and last.pt: $RUN_DIR" >&2
    exit 2
  fi
elif [[ -f "$RUN_DIR/run_config.json" || -f "$RUN_DIR/last.pt" ]]; then
  echo "Run already exists; pass --resume or choose a new directory: $RUN_DIR" >&2
  exit 2
fi
# Hydra parses override values after the shell removes its own quoting.
HYDRA_RUN_DIR="$(env -u PYTHONPATH "$PROJECT_DIR/.venv/bin/python" -c \
  'import json, sys; print(json.dumps(sys.argv[1], ensure_ascii=False))' "$RUN_DIR")"
RESUME_ARGS=()
if [[ "$RESUME" == true ]]; then
  RESUME_ARGS=(algo.resume=true "algo.load_run=$HYDRA_RUN_DIR" algo.checkpoint=last.pt)
fi
exec env -u PYTHONPATH CUDA_VISIBLE_DEVICES=0 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  TORCHINDUCTOR_COMPILE_THREADS=1 \
  TORCHINDUCTOR_CACHE_DIR="$PROJECT_DIR/.unisupermimic-tools/torchinductor" \
  TRITON_CACHE_DIR="$PROJECT_DIR/.unisupermimic-tools/triton" \
  WARP_CACHE_PATH="$PROJECT_DIR/.unisupermimic-tools/warp" \
  PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-backend:cudaMallocAsync}" \
  PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 \
  "$PROJECT_DIR/.venv/bin/unisupermimic-train" \
  task=lens110_lafan1_super_mimic/mjwarp \
  "training.log_dir=$HYDRA_RUN_DIR" algo.num_envs=4096 algo.iterations=12000 \
  algo.ckpt_every=250 algo.log_every=10 "${RESUME_ARGS[@]}" "$@"
