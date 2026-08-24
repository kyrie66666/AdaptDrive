#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
HIPAD_ROOT="${HIPAD_ROOT:-${PROJECT_ROOT}/HiP-AD}"
ADAPTDRIVE_ASSET_ROOT="${ADAPTDRIVE_ASSET_ROOT:-$(cd "${PROJECT_ROOT}/.." && pwd)/AdaptDrive-assets}"
ADAPTDRIVE_RUN_ROOT="${ADAPTDRIVE_RUN_ROOT:-}"
EXPERIMENT_ID="${EXPERIMENT_ID:-}"
CARLA_ROOT="${CARLA_ROOT:-}"
FINETUNE_CKPT="${FINETUNE_CKPT:-}"
if [[ -z "${CARLA_ROOT}" ]]; then
    echo "CARLA_ROOT is required before launching shard1." >&2
    exit 2
fi
if [[ -z "${ADAPTDRIVE_RUN_ROOT}" || "${ADAPTDRIVE_RUN_ROOT}" != /* ]]; then
    echo "Set ADAPTDRIVE_RUN_ROOT to an absolute external run directory before launching shard1." >&2
    exit 2
fi
if [[ -z "${EXPERIMENT_ID}" ]]; then
    echo "Set EXPERIMENT_ID before launching shard1." >&2
    exit 2
fi
if [[ -z "${FINETUNE_CKPT}" ]]; then
    echo "Set FINETUNE_CKPT to the SAC adapter checkpoint before launching shard1." >&2
    exit 2
fi

ROUTES="${ROUTES:-${HIPAD_ROOT}/bench2drive/leaderboard/data/bench2drive220.xml}"
ROUTES_SUBSET="${ROUTES_SUBSET_OVERRIDE:-${ROUTES_SUBSET:-2664-3464}}"
EVALUATION_SHARD="shard1"
SAVE_PATH="${ADAPTDRIVE_RUN_ROOT%/}/evaluations/${EXPERIMENT_ID}/${EVALUATION_SHARD}"
CHECKPOINT_ENDPOINT="${SAVE_PATH}/result.json"
RESUME="${RESUME:-}"
if [[ -z "${RESUME}" ]]; then
    [[ -f "${CHECKPOINT_ENDPOINT}" ]] && RESUME=True || RESUME=False
fi

export ADAPTDRIVE_ASSET_ROOT ADAPTDRIVE_RUN_ROOT EXPERIMENT_ID CARLA_ROOT
export HIPAD_ROOT FINETUNE_CKPT ROUTES ROUTES_SUBSET EVALUATION_SHARD RESUME
GPU_RANK="${GPU_RANK:-0}"
PORT="${PORT:-30590}"
TM_PORT="${TM_PORT:-52590}"
export GPU_RANK PORT TM_PORT
export RUN_NAME="${RUN_NAME:-${EXPERIMENT_ID}_shard1}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/carla-runtime-hipad-adapter-shard1-gpu${GPU_RANK}}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-hipad-adapter-shard1-gpu${GPU_RANK}}"

echo "[adapter-shard1] GPU=${GPU_RANK} routes=${ROUTES_SUBSET} resume=${RESUME} save=${SAVE_PATH}"
exec bash "${SCRIPT_DIR}/run_hipad_clean_dcnv4_adapter_eval.sh" "$@"
