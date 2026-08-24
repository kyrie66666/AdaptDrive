#!/usr/bin/env bash
# Portable HiP-AD Bench2Drive closed-loop evaluation launcher.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HIPAD_ROOT="${HIPAD_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
ADAPTDRIVE_ROOT="${ADAPTDRIVE_ROOT:-$(cd "${HIPAD_ROOT}/.." && pwd)}"
ADAPTDRIVE_ASSET_ROOT="${ADAPTDRIVE_ASSET_ROOT:-$(cd "${ADAPTDRIVE_ROOT}/.." && pwd)/AdaptDrive-assets}"
ADAPTDRIVE_RUN_ROOT="${ADAPTDRIVE_RUN_ROOT:-}"
EXPERIMENT_ID="${EXPERIMENT_ID:-}"
EVALUATION_SHARD="${EVALUATION_SHARD:-}"

# CARLA is an external runtime dependency and must always be supplied by the caller.
CARLA_ROOT="${CARLA_ROOT:-}"
DCNV4_SOURCE_ROOT="${ADAPTDRIVE_ROOT}/third_party/DCNv4"
DCNV4_ROOT="${DCNV4_ROOT:-}"

if [[ ! -f "${DCNV4_SOURCE_ROOT}/DCNv4_op/setup.py" ]]; then
    echo "Vendored DCNv4 source is incomplete: ${DCNV4_SOURCE_ROOT}" >&2
    exit 2
fi

# GPU_RANK is used by both model inference and CARLA's graphics adapter.
GPU_RANK="${GPU_RANK:-0}"
PORT="${PORT:-30490}"
TM_PORT="${TM_PORT:-52490}"
TIMEOUT="${TIMEOUT:-600}"
SERVER_WARMUP_SECONDS="${SERVER_WARMUP_SECONDS:-30}"
BOOTSTRAP_TIMEOUT_CAP="${BOOTSTRAP_TIMEOUT_CAP:-60}"

# Empty means launch CARLA as the current user. Set CARLA_LAUNCH_USER explicitly
# only on hosts where a dedicated runtime user is required.
CARLA_LAUNCH_USER="${CARLA_LAUNCH_USER:-}"

HIPAD_CONFIG="${HIPAD_CONFIG:-${HIPAD_ROOT}/local_runtime/hipad_b2d_stage2_clean_local.py}"
HIPAD_BASE_CKPT="${HIPAD_BASE_CKPT:-${ADAPTDRIVE_ASSET_ROOT}/hipad/checkpoints/hipad_b2d_stage2_base.pth}"
HIPAD_CKPT="${HIPAD_CKPT:-${HIPAD_BASE_CKPT}}"

ROUTES="${ROUTES:-${HIPAD_ROOT}/bench2drive/leaderboard/data/bench2drive220.xml}"
ROUTES_SUBSET="${ROUTES_SUBSET:-}"

# Evaluation artifacts have one canonical location. SAVE_PATH and
# CHECKPOINT_ENDPOINT are deliberately derived rather than accepted as overrides.
if [[ -z "${ADAPTDRIVE_RUN_ROOT}" ]]; then
    echo "ADAPTDRIVE_RUN_ROOT is required and must point outside the source and asset trees." >&2
    exit 2
fi
if [[ "${ADAPTDRIVE_RUN_ROOT}" != /* ]]; then
    echo "ADAPTDRIVE_RUN_ROOT must be an absolute path: ${ADAPTDRIVE_RUN_ROOT}" >&2
    exit 2
fi
ADAPTDRIVE_RUN_ROOT="$(readlink -m -- "${ADAPTDRIVE_RUN_ROOT}")"
ADAPTDRIVE_ROOT="$(readlink -m -- "${ADAPTDRIVE_ROOT}")"
ADAPTDRIVE_ASSET_ROOT="$(readlink -m -- "${ADAPTDRIVE_ASSET_ROOT}")"
case "${ADAPTDRIVE_RUN_ROOT}/" in
    "${ADAPTDRIVE_ROOT}/"*|"${ADAPTDRIVE_ASSET_ROOT}/"*)
        echo "ADAPTDRIVE_RUN_ROOT must not be inside the source or asset tree: ${ADAPTDRIVE_RUN_ROOT}" >&2
        exit 2
        ;;
esac
if [[ -z "${EXPERIMENT_ID}" ]]; then
    echo "EXPERIMENT_ID is required for ADAPTDRIVE_RUN_ROOT/evaluations/EXPERIMENT_ID." >&2
    exit 2
fi
if [[ ! "${EXPERIMENT_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "EXPERIMENT_ID must be a single safe path component: ${EXPERIMENT_ID}" >&2
    exit 2
fi
if [[ -n "${EVALUATION_SHARD}" && ! "${EVALUATION_SHARD}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "EVALUATION_SHARD must be a single safe path component: ${EVALUATION_SHARD}" >&2
    exit 2
fi

EVALUATION_ROOT="${ADAPTDRIVE_RUN_ROOT%/}/evaluations/${EXPERIMENT_ID}"
SAVE_PATH="${EVALUATION_ROOT}"
if [[ -n "${EVALUATION_SHARD}" ]]; then
    SAVE_PATH="${EVALUATION_ROOT}/${EVALUATION_SHARD}"
fi
CHECKPOINT_ENDPOINT="${SAVE_PATH}/result.json"
RUN_NAME="${RUN_NAME:-${EXPERIMENT_ID}${EVALUATION_SHARD:+_${EVALUATION_SHARD}}}"

RESUME="${RESUME:-False}"
REPETITIONS="${REPETITIONS:-1}"
DEBUG_CHALLENGE="${DEBUG_CHALLENGE:-0}"
PLANNER_TYPE="${PLANNER_TYPE:-traj}"
CHALLENGE_TRACK_CODENAME="${CHALLENGE_TRACK_CODENAME:-SENSORS}"

# Optional conda activation. With CONDA_SH unset, the launcher's current Python
# environment is used unchanged. Set both variables when activation is desired.
CONDA_SH="${CONDA_SH:-}"
CONDA_ENV="${CONDA_ENV:-}"

TEAM_CONFIG="${TEAM_CONFIG:-${HIPAD_CONFIG}+${HIPAD_CKPT}+${RUN_NAME}}"
TEAM_AGENT="${TEAM_AGENT:-${HIPAD_ROOT}/bench2drive/leaderboard/team_code/hipad_b2d_agent.py}"

# The clean agent uses the last TEAM_CONFIG field as a directory name below
# SAVE_PATH. Reject path-like labels so an override cannot escape EVALUATION_ROOT.
TEAM_CONFIG_SAVE_NAME="${TEAM_CONFIG##*+}"
if [[ ! "${TEAM_CONFIG_SAVE_NAME}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "The final TEAM_CONFIG field must be a safe output label: ${TEAM_CONFIG_SAVE_NAME:-<empty>}" >&2
    echo "Append +LABEL to custom adapter TEAM_CONFIG values." >&2
    exit 2
fi

if [[ -z "${CARLA_ROOT}" ]]; then
    echo "CARLA_ROOT is required. Set it to a CARLA installation containing CarlaUE4.sh." >&2
    exit 2
fi
if [[ "${CARLA_ROOT}" != /* ]]; then
    echo "CARLA_ROOT must be an absolute path: ${CARLA_ROOT}" >&2
    exit 2
fi
if [[ -n "${DCNV4_ROOT}" && ( "${DCNV4_ROOT}" != /* || ! -d "${DCNV4_ROOT}" ) ]]; then
    echo "DCNV4_ROOT must be an absolute package root: ${DCNV4_ROOT}" >&2
    exit 2
fi

CARLA_EGG="${CARLA_EGG:-${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg}"

if [[ ! -f "${CARLA_ROOT}/CarlaUE4.sh" ]]; then
    echo "CARLA launcher not found: ${CARLA_ROOT}/CarlaUE4.sh" >&2
    exit 2
fi
if [[ ! -f "${CARLA_EGG}" ]]; then
    echo "CARLA egg not found: ${CARLA_EGG}" >&2
    exit 2
fi
if [[ ! -f "${HIPAD_CONFIG}" ]]; then
    echo "HiP-AD config not found: ${HIPAD_CONFIG}" >&2
    exit 2
fi
if [[ ! -f "${HIPAD_CKPT}" ]]; then
    echo "HiP-AD checkpoint not found: ${HIPAD_CKPT}" >&2
    exit 2
fi
if [[ ! -f "${ROUTES}" ]]; then
    echo "Routes file not found: ${ROUTES}" >&2
    exit 2
fi
if [[ ! -f "${HIPAD_ROOT}/bench2drive/leaderboard/data/weather.xml" ]]; then
    echo "Weather file not found: ${HIPAD_ROOT}/bench2drive/leaderboard/data/weather.xml" >&2
    exit 2
fi
if [[ ! -f "${TEAM_AGENT}" ]]; then
    echo "Leaderboard agent not found: ${TEAM_AGENT}" >&2
    exit 2
fi

# Vulkan is explicitly configurable but never inferred from checkpoints or assets.
# Leaving VK_ICD_FILENAMES empty delegates ICD discovery to the system loader.
VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-}"
if [[ -n "${VK_ICD_FILENAMES}" ]]; then
    IFS=':' read -r -a VULKAN_ICD_FILES <<< "${VK_ICD_FILENAMES}"
    for vulkan_icd_file in "${VULKAN_ICD_FILES[@]}"; do
        if [[ -z "${vulkan_icd_file}" || ! -s "${vulkan_icd_file}" ]]; then
            echo "VK_ICD_FILENAMES contains a missing or empty file: ${vulkan_icd_file:-<empty>}" >&2
            exit 2
        fi
    done
    export VK_ICD_FILENAMES
else
    unset VK_ICD_FILENAMES
fi

if [[ -n "${CONDA_SH}" ]]; then
    if [[ ! -f "${CONDA_SH}" ]]; then
        echo "CONDA_SH does not exist: ${CONDA_SH}" >&2
        exit 2
    fi
    # shellcheck disable=SC1090
    source "${CONDA_SH}"
    if ! command -v conda >/dev/null 2>&1; then
        echo "CONDA_SH did not provide the conda command: ${CONDA_SH}" >&2
        exit 2
    fi
    if [[ -n "${CONDA_ENV}" ]]; then
        conda activate "${CONDA_ENV}"
    fi
elif [[ -n "${CONDA_ENV}" && "${CONDA_DEFAULT_ENV:-}" != "${CONDA_ENV}" ]]; then
    echo "CONDA_ENV=${CONDA_ENV} was requested but is not active." >&2
    echo "Activate it before launch or set CONDA_SH to conda's profile.d/conda.sh." >&2
    exit 2
fi

if ! command -v python >/dev/null 2>&1; then
    echo "python is not available in the active environment." >&2
    exit 2
fi

if [[ "${ADAPTDRIVE_VALIDATE_ONLY:-0}" == "1" ]]; then
    echo "HiP-AD Leaderboard launcher validation passed; no files written and no process started."
    exit 0
fi

mkdir -p "${SAVE_PATH}"

export ADAPTDRIVE_ROOT
export ADAPTDRIVE_ASSET_ROOT
export ADAPTDRIVE_RUN_ROOT
export EXPERIMENT_ID
export EVALUATION_SHARD
export HIPAD_ROOT
export HIPAD_CONFIG
export HIPAD_BASE_CKPT
export HIPAD_CKPT
export CARLA_ROOT
export DCNV4_ROOT
export CARLA_SERVER="${CARLA_ROOT}/CarlaUE4.sh"
export CARLA_LAUNCH_USER
export WORK_DIR="${HIPAD_ROOT}"
export SCENARIO_RUNNER_ROOT="${HIPAD_ROOT}/bench2drive/scenario_runner"
export LEADERBOARD_ROOT="${HIPAD_ROOT}/bench2drive/leaderboard"
export IS_BENCH2DRIVE=True
export PLANNER_TYPE
export SAVE_PATH
export ROUTES
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/carla-runtime-clean-${PORT}-gpu${GPU_RANK}}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-clean-${PORT}-gpu${GPU_RANK}}"
export SDL_AUDIODRIVER="${SDL_AUDIODRIVER:-dummy}"
PYTHONPATH="${CARLA_EGG}:${CARLA_ROOT}/PythonAPI:${CARLA_ROOT}/PythonAPI/carla"
PYTHONPATH+=":${HIPAD_ROOT}:${HIPAD_ROOT}/bench2drive"
PYTHONPATH+=":${HIPAD_ROOT}/bench2drive/leaderboard:${HIPAD_ROOT}/bench2drive/scenario_runner"
PYTHONPATH+="${DCNV4_ROOT:+:${DCNV4_ROOT}}"
export PYTHONPATH

mkdir -p "${XDG_RUNTIME_DIR}"
mkdir -p "${MPLCONFIGDIR}"
chmod 700 "${XDG_RUNTIME_DIR}"

echo "============================================================"
echo "HiP-AD clean Bench2Drive evaluation"
echo "============================================================"
echo "Python: $(command -v python)"
echo "HIPAD_ROOT: ${HIPAD_ROOT}"
echo "ADAPTDRIVE_ASSET_ROOT: ${ADAPTDRIVE_ASSET_ROOT}"
echo "ADAPTDRIVE_RUN_ROOT: ${ADAPTDRIVE_RUN_ROOT}"
echo "EXPERIMENT_ID: ${EXPERIMENT_ID}"
echo "EVALUATION_SHARD: ${EVALUATION_SHARD:-<none>}"
echo "CARLA_ROOT: ${CARLA_ROOT}"
echo "GPU_RANK: ${GPU_RANK}"
echo "PORT/TM_PORT: ${PORT}/${TM_PORT}"
echo "TIMEOUT: ${TIMEOUT}"
echo "SERVER_WARMUP_SECONDS: ${SERVER_WARMUP_SECONDS}"
echo "BOOTSTRAP_TIMEOUT_CAP: ${BOOTSTRAP_TIMEOUT_CAP}"
echo "CARLA_LAUNCH_USER: ${CARLA_LAUNCH_USER:-<current>}"
echo "CONDA_ENV: ${CONDA_DEFAULT_ENV:-<none>}"
echo "VK_ICD_FILENAMES: ${VK_ICD_FILENAMES:-<system-default>}"
echo "HIPAD_CONFIG: ${HIPAD_CONFIG}"
echo "HIPAD_CKPT: ${HIPAD_CKPT}"
echo "TEAM_AGENT: ${TEAM_AGENT}"
echo "TEAM_CONFIG: ${TEAM_CONFIG}"
echo "ROUTES: ${ROUTES}"
echo "ROUTES_SUBSET: ${ROUTES_SUBSET:-<all>}"
echo "SAVE_PATH: ${SAVE_PATH}"
echo "CHECKPOINT_ENDPOINT: ${CHECKPOINT_ENDPOINT}"
echo "RESUME: ${RESUME}"
echo "============================================================"

cd "${HIPAD_ROOT}"

CUDA_VISIBLE_DEVICES="${GPU_RANK}" python -X faulthandler \
    "${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py" \
    --routes="${ROUTES}" \
    --routes-subset="${ROUTES_SUBSET}" \
    --repetitions="${REPETITIONS}" \
    --track="${CHALLENGE_TRACK_CODENAME}" \
    --checkpoint="${CHECKPOINT_ENDPOINT}" \
    --agent="${TEAM_AGENT}" \
    --agent-config="${TEAM_CONFIG}" \
    --debug="${DEBUG_CHALLENGE}" \
    --record="" \
    --resume="${RESUME}" \
    --port="${PORT}" \
    --traffic-manager-port="${TM_PORT}" \
    --timeout="${TIMEOUT}" \
    --server-warmup-seconds="${SERVER_WARMUP_SECONDS}" \
    --bootstrap-timeout-cap="${BOOTSTRAP_TIMEOUT_CAP}" \
    --gpu-rank="${GPU_RANK}" \
    2>&1 | tee "${CHECKPOINT_ENDPOINT%.json}.log"
