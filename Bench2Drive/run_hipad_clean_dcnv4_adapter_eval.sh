#!/usr/bin/env bash
# Portable launcher for the adapter-aware HiP-AD leaderboard agent.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ADAPTDRIVE_ASSET_ROOT="${ADAPTDRIVE_ASSET_ROOT:-$(cd "${PROJECT_ROOT}/.." && pwd)/AdaptDrive-assets}"
ADAPTDRIVE_RUN_ROOT="${ADAPTDRIVE_RUN_ROOT:-}"
EXPERIMENT_ID="${EXPERIMENT_ID:-}"
CARLA_ROOT="${CARLA_ROOT:-}"
DCNV4_SOURCE_ROOT="${PROJECT_ROOT}/third_party/DCNv4"
DCNV4_ROOT="${DCNV4_ROOT:-}"

HIPAD_ROOT="${HIPAD_ROOT:-${PROJECT_ROOT}/HiP-AD}"
HIPAD_CONFIG="${HIPAD_CONFIG:-${HIPAD_ROOT}/local_runtime/hipad_b2d_stage2_clean_local.py}"
HIPAD_BASE_CKPT="${HIPAD_BASE_CKPT:-${ADAPTDRIVE_ASSET_ROOT}/hipad/checkpoints/hipad_b2d_stage2_base.pth}"
HIPAD_CKPT="${HIPAD_CKPT:-${HIPAD_BASE_CKPT}}"
FINETUNE_CKPT="${FINETUNE_CKPT:-}"

if [[ -z "${CARLA_ROOT}" ]]; then
    echo "CARLA_ROOT is required. Set it to a CARLA installation containing CarlaUE4.sh." >&2
    exit 2
fi
if [[ -z "${ADAPTDRIVE_RUN_ROOT}" ]]; then
    echo "ADAPTDRIVE_RUN_ROOT is required and must point outside the source and asset trees." >&2
    exit 2
fi
if [[ "${ADAPTDRIVE_RUN_ROOT}" != /* ]]; then
    echo "ADAPTDRIVE_RUN_ROOT must be an absolute path: ${ADAPTDRIVE_RUN_ROOT}" >&2
    exit 2
fi
if [[ -z "${EXPERIMENT_ID}" ]]; then
    echo "EXPERIMENT_ID is required; evaluation output will be written below" >&2
    echo "  ADAPTDRIVE_RUN_ROOT/evaluations/EXPERIMENT_ID" >&2
    exit 2
fi
if [[ -z "${FINETUNE_CKPT}" ]]; then
    echo "FINETUNE_CKPT must point to an AdaptDrive checkpoint containing the DCNv4 adapter." >&2
    exit 2
fi

CLEAN_LAUNCHER="${HIPAD_ROOT}/bench2drive/leaderboard/scripts/run_hipad_clean_local.sh"
ADAPTER_AGENT="${SCRIPT_DIR}/leaderboard/rl/hipad_clean_dcnv4_adapter_agent.py"

if [[ ! -f "${CLEAN_LAUNCHER}" ]]; then
    echo "Clean launcher not found: ${CLEAN_LAUNCHER}" >&2
    exit 2
fi
if [[ ! -f "${ADAPTER_AGENT}" ]]; then
    echo "Adapter-aware agent not found: ${ADAPTER_AGENT}" >&2
    exit 2
fi
if [[ ! -f "${HIPAD_CKPT}" ]]; then
    echo "HiP-AD base checkpoint not found: ${HIPAD_CKPT}" >&2
    exit 2
fi
if [[ ! -f "${FINETUNE_CKPT}" ]]; then
    echo "AdaptDrive finetune checkpoint not found: ${FINETUNE_CKPT}" >&2
    exit 2
fi
if [[ -n "${DCNV4_ROOT}" && ( "${DCNV4_ROOT}" != /* || ! -d "${DCNV4_ROOT}" ) ]]; then
    echo "DCNV4_ROOT must be an absolute package root: ${DCNV4_ROOT}" >&2
    exit 2
fi
if [[ ! -f "${DCNV4_SOURCE_ROOT}/DCNv4_op/setup.py" ]]; then
    echo "Vendored DCNv4 source is incomplete: ${DCNV4_SOURCE_ROOT}" >&2
    exit 2
fi

# Vulkan selection is opt-in. If VK_ICD_FILENAMES is unset, the system Vulkan
# loader chooses its normal configuration. No checkpoint- or asset-adjacent scan occurs.
if [[ -n "${VK_ICD_FILENAMES:-}" ]]; then
    export VK_ICD_FILENAMES
else
    unset VK_ICD_FILENAMES
fi

export ADAPTDRIVE_ASSET_ROOT
export ADAPTDRIVE_RUN_ROOT
export EXPERIMENT_ID
export CARLA_ROOT
export DCNV4_ROOT
export HIPAD_ROOT
export HIPAD_CONFIG
export HIPAD_BASE_CKPT
export HIPAD_CKPT
export FINETUNE_CKPT
export RUN_NAME="${RUN_NAME:-${EXPERIMENT_ID}${EVALUATION_SHARD:+_${EVALUATION_SHARD}}}"
export TEAM_AGENT="${TEAM_AGENT:-${ADAPTER_AGENT}}"
export TEAM_CONFIG="${TEAM_CONFIG:-${HIPAD_CONFIG}+${HIPAD_CKPT}+${FINETUNE_CKPT}+${RUN_NAME}}"

EVALUATION_ROOT="${ADAPTDRIVE_RUN_ROOT%/}/evaluations/${EXPERIMENT_ID}"
if [[ -n "${EVALUATION_SHARD:-}" ]]; then
    EVALUATION_ROOT="${EVALUATION_ROOT}/${EVALUATION_SHARD}"
fi

echo "HiP-AD adapter-aware leaderboard launcher"
echo "  CARLA_ROOT:             ${CARLA_ROOT}"
echo "  ADAPTDRIVE_ASSET_ROOT:  ${ADAPTDRIVE_ASSET_ROOT}"
echo "  ADAPTDRIVE_RUN_ROOT:    ${ADAPTDRIVE_RUN_ROOT}"
echo "  EXPERIMENT_ID:          ${EXPERIMENT_ID}"
echo "  EVALUATION_ROOT:        ${EVALUATION_ROOT}"
echo "  HIPAD_ROOT:             ${HIPAD_ROOT}"
echo "  HIPAD_CONFIG:           ${HIPAD_CONFIG}"
echo "  HIPAD_CKPT:             ${HIPAD_CKPT}"
echo "  FINETUNE_CKPT:          ${FINETUNE_CKPT}"
echo "  VK_ICD_FILENAMES:       ${VK_ICD_FILENAMES:-<system-default>}"
echo "  TEAM_AGENT:             ${TEAM_AGENT}"
echo "  RUN_NAME:               ${RUN_NAME}"
echo "  DCNV4_SOURCE_ROOT:      ${DCNV4_SOURCE_ROOT}"

exec bash "${CLEAN_LAUNCHER}" "$@"
