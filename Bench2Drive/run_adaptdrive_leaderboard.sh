#!/usr/bin/env bash
# Canonical AdaptDrive Bench2Drive Leaderboard evaluation entry point.

set -euo pipefail

B2D_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${B2D_ROOT}/.." && pwd)"
PROJECT_PARENT="$(cd "${PROJECT_ROOT}/.." && pwd)"

ADAPTDRIVE_ASSET_ROOT="${ADAPTDRIVE_ASSET_ROOT:-${PROJECT_PARENT}/AdaptDrive-assets}"
ADAPTDRIVE_RUN_ROOT="${ADAPTDRIVE_RUN_ROOT:-${PROJECT_PARENT}/AdaptDrive-runs}"
EXPERIMENT_ID="${EXPERIMENT_ID:-}"
CARLA_ROOT="${CARLA_ROOT:-}"
DCNV4_SOURCE_ROOT="${PROJECT_ROOT}/third_party/DCNv4"
DCNV4_ROOT="${DCNV4_ROOT:-}"

HIPAD_ROOT="${HIPAD_ROOT:-${PROJECT_ROOT}/HiP-AD}"
HIPAD_CONFIG="${HIPAD_CONFIG:-${HIPAD_ROOT}/local_runtime/hipad_b2d_stage2_clean_local.py}"
HIPAD_BASE_CKPT="${HIPAD_BASE_CKPT:-${ADAPTDRIVE_ASSET_ROOT}/hipad/checkpoints/hipad_b2d_stage2_base.pth}"
FINETUNE_CKPT="${FINETUNE_CKPT:-}"
ROUTES="${ROUTES:-${HIPAD_ROOT}/bench2drive/leaderboard/data/bench2drive220.xml}"

if [[ -z "${EXPERIMENT_ID}" ]]; then
    echo "EXPERIMENT_ID is required." >&2
    exit 2
fi
if [[ ! "${EXPERIMENT_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]]; then
    echo "EXPERIMENT_ID must be a safe 1-128 character path component: ${EXPERIMENT_ID}" >&2
    exit 2
fi
if [[ -z "${CARLA_ROOT}" || "${CARLA_ROOT}" != /* || ! -f "${CARLA_ROOT}/CarlaUE4.sh" ]]; then
    echo "CARLA_ROOT must be an absolute CARLA installation containing CarlaUE4.sh." >&2
    exit 2
fi
if [[ "${ADAPTDRIVE_RUN_ROOT}" != /* ]]; then
    echo "ADAPTDRIVE_RUN_ROOT must be absolute: ${ADAPTDRIVE_RUN_ROOT}" >&2
    exit 2
fi

PROJECT_ROOT="$(readlink -m -- "${PROJECT_ROOT}")"
ADAPTDRIVE_ASSET_ROOT="$(readlink -m -- "${ADAPTDRIVE_ASSET_ROOT}")"
ADAPTDRIVE_RUN_ROOT="$(readlink -m -- "${ADAPTDRIVE_RUN_ROOT}")"
case "${ADAPTDRIVE_RUN_ROOT}/" in
    "${PROJECT_ROOT}/"*|"${ADAPTDRIVE_ASSET_ROOT}/"*)
        echo "ADAPTDRIVE_RUN_ROOT must be outside the source and asset trees: ${ADAPTDRIVE_RUN_ROOT}" >&2
        exit 2
        ;;
esac

if [[ -z "${FINETUNE_CKPT}" ]]; then
    FINETUNE_CKPT="${ADAPTDRIVE_RUN_ROOT}/checkpoints/${EXPERIMENT_ID}/checkpoint_latest.pt"
fi
for required_file in "${HIPAD_CONFIG}" "${HIPAD_BASE_CKPT}" "${FINETUNE_CKPT}" "${ROUTES}"; do
    if [[ ! -f "${required_file}" ]]; then
        echo "Required file not found: ${required_file}" >&2
        exit 2
    fi
done
if [[ -L "${HIPAD_BASE_CKPT}" || -L "${FINETUNE_CKPT}" ]]; then
    echo "Checkpoint paths must not be symlinks." >&2
    exit 2
fi
if [[ ! -f "${DCNV4_SOURCE_ROOT}/DCNv4_op/setup.py" ]]; then
    echo "Vendored DCNv4 source is incomplete: ${DCNV4_SOURCE_ROOT}" >&2
    exit 2
fi
if [[ -n "${DCNV4_ROOT}" && ( "${DCNV4_ROOT}" != /* || ! -d "${DCNV4_ROOT}" ) ]]; then
    echo "DCNV4_ROOT must be an absolute package root: ${DCNV4_ROOT}" >&2
    exit 2
fi

export ADAPTDRIVE_ASSET_ROOT ADAPTDRIVE_RUN_ROOT EXPERIMENT_ID CARLA_ROOT
export HIPAD_ROOT HIPAD_CONFIG HIPAD_BASE_CKPT FINETUNE_CKPT ROUTES DCNV4_ROOT

echo "AdaptDrive Leaderboard evaluation"
echo "  experiment:    ${EXPERIMENT_ID}"
echo "  source:        ${PROJECT_ROOT}"
echo "  runs:          ${ADAPTDRIVE_RUN_ROOT}"
echo "  CARLA:         ${CARLA_ROOT}"
echo "  HiP-AD base:   ${HIPAD_BASE_CKPT}"
echo "  finetune:      ${FINETUNE_CKPT}"
echo "  routes:        ${ROUTES}"
echo "  DCNv4 source:  ${DCNV4_SOURCE_ROOT}"

if [[ "${ADAPTDRIVE_VALIDATE_ONLY:-0}" == "1" ]]; then
    echo "Delegating validation to the adapter-aware Leaderboard chain; no simulator will start."
fi

exec bash "${B2D_ROOT}/run_hipad_clean_dcnv4_adapter_eval.sh" "$@"
