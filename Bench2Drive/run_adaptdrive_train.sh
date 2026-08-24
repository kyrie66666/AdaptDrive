#!/usr/bin/env bash
# Canonical AdaptDrive closed-loop finetuning entry point.

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
ROACH_BEV_MAP_ROOT="${ROACH_BEV_MAP_ROOT:-${ADAPTDRIVE_ASSET_ROOT}/roach_bev_maps}"
ROUTES="${ROUTES:-${B2D_ROOT}/leaderboard/data/bench2drive220.xml}"
RESUME_FROM="${RESUME_FROM:-}"
INIT_FROM_WAS_SET="${INIT_FROM+x}"
INIT_FROM="${INIT_FROM:-${ADAPTDRIVE_ASSET_ROOT}/adaptdrive/checkpoints/adaptdrive_sig7_step140906.pt}"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-0}"
PORT="${PORT:-30300}"
TRAFFIC_MANAGER_PORT="${TRAFFIC_MANAGER_PORT:-52300}"

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

for required_file in "${HIPAD_CONFIG}" "${HIPAD_BASE_CKPT}" "${ROUTES}"; do
    if [[ ! -f "${required_file}" ]]; then
        echo "Required file not found: ${required_file}" >&2
        exit 2
    fi
done
if [[ ! -d "${ROACH_BEV_MAP_ROOT}" ]]; then
    echo "Roach BEV map directory not found: ${ROACH_BEV_MAP_ROOT}" >&2
    exit 2
fi
if [[ ! -f "${DCNV4_SOURCE_ROOT}/DCNv4_op/setup.py" ]]; then
    echo "Vendored DCNv4 source is incomplete: ${DCNV4_SOURCE_ROOT}" >&2
    exit 2
fi
if [[ -n "${RESUME_FROM}" ]]; then
    if [[ "${INIT_FROM_WAS_SET}" == "x" ]]; then
        echo "RESUME_FROM and an explicitly configured INIT_FROM are mutually exclusive." >&2
        exit 2
    fi
    if [[ ! -f "${RESUME_FROM}" || -L "${RESUME_FROM}" ]]; then
        echo "RESUME_FROM must be a non-symlink v8 checkpoint: ${RESUME_FROM}" >&2
        exit 2
    fi
    START_ARGS=(--resume-from "${RESUME_FROM}")
else
    if [[ ! -f "${INIT_FROM}" || -L "${INIT_FROM}" ]]; then
        echo "INIT_FROM must be the registered non-symlink v7 parent checkpoint: ${INIT_FROM}" >&2
        exit 2
    fi
    START_ARGS=(--init-from "${INIT_FROM}")
fi

if [[ -n "${DCNV4_ROOT}" ]]; then
    if [[ "${DCNV4_ROOT}" != /* || ! -d "${DCNV4_ROOT}" ]]; then
        echo "DCNV4_ROOT must be an absolute package root: ${DCNV4_ROOT}" >&2
        exit 2
    fi
    export PYTHONPATH="${DCNV4_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
fi
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "Python executable not found: ${PYTHON_BIN}" >&2
    exit 2
fi

export ADAPTDRIVE_ASSET_ROOT ADAPTDRIVE_RUN_ROOT EXPERIMENT_ID CARLA_ROOT
export HIPAD_ROOT HIPAD_BASE_CKPT ROACH_BEV_MAP_ROOT

echo "AdaptDrive training"
echo "  experiment:    ${EXPERIMENT_ID}"
echo "  source:        ${PROJECT_ROOT}"
echo "  assets:        ${ADAPTDRIVE_ASSET_ROOT}"
echo "  runs:          ${ADAPTDRIVE_RUN_ROOT}"
echo "  CARLA:         ${CARLA_ROOT}"
echo "  HiP-AD base:   ${HIPAD_BASE_CKPT}"
echo "  Roach maps:    ${ROACH_BEV_MAP_ROOT}"
echo "  DCNv4 source:  ${DCNV4_SOURCE_ROOT}"
echo "  start mode:    ${START_ARGS[*]}"

if [[ "${ADAPTDRIVE_VALIDATE_ONLY:-0}" == "1" ]]; then
    echo "AdaptDrive training launcher validation passed; no process started."
    exit 0
fi

cd "${B2D_ROOT}"
exec "${PYTHON_BIN}" -u stable_train_hipad_policy_finetune.py \
    --experiment-id "${EXPERIMENT_ID}" \
    --run-root "${ADAPTDRIVE_RUN_ROOT}" \
    --carla-root "${CARLA_ROOT}" \
    --gpu-id "${GPU_ID}" \
    --port "${PORT}" \
    --traffic-manager-port "${TRAFFIC_MANAGER_PORT}" \
    --hipad-root "${HIPAD_ROOT}" \
    --hipad-config "${HIPAD_CONFIG}" \
    --hipad-checkpoint "${HIPAD_BASE_CKPT}" \
    --roach-bev-map-root "${ROACH_BEV_MAP_ROOT}" \
    --routes "${ROUTES}" \
    "${START_ARGS[@]}" \
    "$@"
