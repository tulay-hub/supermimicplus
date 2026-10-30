#!/usr/bin/env bash
# Add 113 screened motions to the original 40 and continue full-21 training.
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
RUN_DIR="${1:-runs/lens110_lafan1_yahmp113_full21_30000_20261029}"
if [[ $# -gt 0 ]]; then shift; fi
SOURCE_DIR="${SUPERMIMICPLUS_STAGE1_RUN:-runs/lens110_lafan1_full21_20261028}"
FORK=true
CHANGE=true
if [[ "${1:-}" == "--resume" ]]; then
  SOURCE_DIR="$RUN_DIR"
  FORK=false
  CHANGE=false
  shift
elif [[ -f "$RUN_DIR/run_config.json" || -f "$RUN_DIR/last.pt" ]]; then
  echo "Run already exists; pass --resume or choose a new directory: $RUN_DIR" >&2
  exit 2
fi
if [[ ! -f "$SOURCE_DIR/run_config.json" || ! -f "$SOURCE_DIR/last.pt" ]]; then
  echo "Continuation requires source run_config.json and full last.pt: $SOURCE_DIR" >&2
  exit 2
fi
hydra_quote() {
  env -u PYTHONPATH "$PROJECT_DIR/.venv/bin/python" -c \
    'import json, sys; print(json.dumps(sys.argv[1], ensure_ascii=False))' "$1"
}
HYDRA_RUN_DIR="$(hydra_quote "$RUN_DIR")"
HYDRA_SOURCE_DIR="$(hydra_quote "$SOURCE_DIR")"
exec env -u PYTHONPATH -u PYTORCH_CUDA_ALLOC_CONF CUDA_VISIBLE_DEVICES=0 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  TORCHINDUCTOR_COMPILE_THREADS=1 \
  TORCHINDUCTOR_CACHE_DIR="$PROJECT_DIR/.unisupermimic-tools/torchinductor" \
  TRITON_CACHE_DIR="$PROJECT_DIR/.unisupermimic-tools/triton" \
  WARP_CACHE_PATH="$PROJECT_DIR/.unisupermimic-tools/warp" \
  PYTORCH_ALLOC_CONF=backend:cudaMallocAsync \
  PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 \
  "$PROJECT_DIR/.venv/bin/unisupermimic-train" \
  task=lens110_lafan1_super_mimic/mjwarp \
  +experiment=lens110_lafan1_yahmp113_30000 \
  "training.log_dir=$HYDRA_RUN_DIR" \
  algo.resume=true "algo.load_run=$HYDRA_SOURCE_DIR" algo.checkpoint=last.pt \
  "training.resume_to_new_run=$FORK" "training.allow_motion_library_change=$CHANGE" \
  algo.num_envs=4096 algo.iterations=30000 algo.ckpt_every=250 algo.log_every=10 "$@"
