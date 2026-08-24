---
name: adaptdrive-bootstrap
description: Prepare and audit a cloned AdaptDrive project on a new server after the operator has unpacked the conda environment and supplied external assets; use it before any CARLA, training, or evaluation run.
---

# AdaptDrive Bootstrap

Use this skill when a fresh server has a cloned AdaptDrive tree and the owner
has already unpacked the conda environment and copied the immutable assets. The
outcome is a short runtime-readiness report, not a training result.

## Boundaries

- Do not run `conda-pack`, `conda-unpack`, `conda create`, `pip install`, or
  package-manager commands. The environment owner performs those operations.
- Do not modify model code, configs, launchers, CUDA source, checkpoints, or
  the original mixed `I2R-AD` workspace.
- Do not start CARLA, a Leaderboard process, training, or evaluation without
  explicit authorization for that long-lived process. Validation-only checks
  are allowed before authorization.
- Never inherit a GPU ban, Vulkan adapter, display, port, or absolute path from
  `docs/history/SERVER10_INCIDENTS.md`. That file is historical context only.
- Keep command output small. Capture exit status, marker lines, error class,
  and the final 20-40 lines of a log instead of reading large logs wholesale.

## Read first

From the cloned repository root, read:

1. `README.md`
2. `docs/CURRENT_CONTRACT.md`
3. `docs/PORTABILITY_CONTRACT.md`
4. `docs/ASSETS.md` and `docs/ASSET_MANIFEST.md`
5. `docs/MIGRATION_STATUS.md`

Read `docs/history/SERVER10_INCIDENTS.md` only when diagnosing a failure that
resembles the old server incident.

## 1. Establish identity and paths

Resolve the root without trusting the caller's working directory:

```bash
PROJECT_ROOT="$(git rev-parse --show-toplevel)"
test -f "${PROJECT_ROOT}/.agents/skills/adaptdrive-bootstrap/SKILL.md"
```

Set `ADAPTDRIVE_ROOT`, `ADAPTDRIVE_ASSET_ROOT`, `ADAPTDRIVE_RUN_ROOT`,
`CARLA_ROOT`, `HIPAD_ROOT`, `B2D_ROOT`, and `B2DZOO_ROOT` from a local
environment file based on `env.example`. Resolve each path and verify:

- the source root is the cloned AdaptDrive tree;
- `HIPAD_ROOT` is `${ADAPTDRIVE_ROOT}/HiP-AD` or an explicitly documented
  independent copy;
- CARLA contains `CarlaUE4.sh` and its PythonAPI can be located;
- the asset root contains the files in `docs/ASSET_MANIFEST.md`;
- the run root is outside both source and asset roots;
- no required path or symlink resolves into the old `I2R-AD` workspace.

Use `readlink -f` and `find -L` for these checks. A missing asset, a broken
link, or a path that silently falls back to another project is a hard failure.

## 2. Inspect the already active environment

Record the active interpreter and build details without changing the
environment:

```bash
command -v python
python -V
python -c 'import sys, torch; print(sys.executable); print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())'
gcc --version | head -1
ldd --version | head -1
```

Confirm that `CONDA_PREFIX` points to the owner's unpacked environment. If it
does not, stop and ask the owner to activate the intended environment. Do not
repair it by installing packages from the skill.

## 3. Enumerate GPUs dynamically

First collect a compact host table, including all visible devices:

```bash
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,temperature.gpu --format=csv,noheader,nounits
```

Then query the selected candidate from the active Python process:

```bash
python - <<'PY'
import torch
print("cuda_available", torch.cuda.is_available())
for index in range(torch.cuda.device_count()):
    print(index, torch.cuda.get_device_name(index), torch.cuda.get_device_capability(index))
PY
```

Select a healthy, sufficiently free physical GPU using the current table or
an explicit owner choice. GPU 0 is valid when healthy. If
`CUDA_VISIBLE_DEVICES` masks devices, record both the host index and the
process-visible logical index; do not assume they are equal.

For CARLA, keep the selected physical mapping consistent across the trainer's
`--gpu-id`, `CARLA_CUDA_VISIBLE_DEVICES`, and `--carla-graphicsadapter` unless
a current-server probe demonstrates a different mapping. Do not set a fixed
Vulkan or display value merely because an old note contains one.

## 4. Audit compiled extensions

List only names and sizes, not binary contents:

```bash
find "${ADAPTDRIVE_ROOT}" -type f \( -name '*.so' -o -name '*.o' -o -name '*.ninja' \) -printf '%s %p\n' | sort -nr | head -40
```

Treat `.so` files as host-specific compatibility artifacts. Verify the vendored
DCNv4 source exists at `third_party/DCNv4/DCNv4_op/setup.py`, inspect its
upstream commit/hash manifest, and check whether the active wheel and MMCV
extensions match the current PyTorch/CUDA/GPU ABI. Rebuild the source-owned
extensions when they do not match. Do not delete old binaries from the source
tree in this bootstrap pass.

Run small imports and the adapter smoke only after the environment has the
required packages:

```bash
python -c 'import torch; import DCNv4; print("DCNv4 import ok", torch.cuda.is_available())'
python "${ADAPTDRIVE_ROOT}/Bench2Drive/test_feature_dcnv4_adapter_smoke.py"
```

The adapter smoke may use CPU for shape checks and CUDA for the real
forward/backward. Report a CPU-only skip as a missing runtime gate, not as a
pass for closed loop.

## 5. Validate source contracts and launchers

Run the fast, offline checks from the source root. Keep output in a temporary
file and report only the pass/fail markers and tail:

```bash
python Bench2Drive/test_adaptdrive_contract_smoke.py
python Bench2Drive/test_adaptdrive_signature_v8_smoke.py
python Bench2Drive/test_adaptdrive_replay_protocol_smoke.py
python Bench2Drive/test_adaptdrive_portable_runtime_smoke.py
```

With a real asset root and CARLA path, validate all canonical launchers without
starting a process:

```bash
ADAPTDRIVE_VALIDATE_ONLY=1 EXPERIMENT_ID=bootstrap-check \
  bash Bench2Drive/run_adaptdrive_train.sh
ADAPTDRIVE_VALIDATE_ONLY=1 EXPERIMENT_ID=bootstrap-check \
  bash Bench2Drive/run_adaptdrive_eval.sh
ADAPTDRIVE_VALIDATE_ONLY=1 EXPERIMENT_ID=bootstrap-check \
  bash Bench2Drive/run_adaptdrive_leaderboard.sh
```

Do not bypass a signature, checkpoint, replay, route, or asset error. A v7
parent is for fresh v8 initialization; it is not a strict v8 resume pair.

## 6. CARLA probe and report

After launcher validation passes, obtain explicit authorization before starting
CARLA. Use an isolated, currently free RPC/TM port pair and the selected GPU.
Connect with a short client probe that reports `get_server_version()`, world
map, and one snapshot, then stop the server cleanly. If the probe fails, record
the host, ports, process command, selected GPU mapping, display/EGL/Vulkan
state, error class, and only the final log lines.

Write a compact report under the external run root, for example
`ADAPTDRIVE_RUN_ROOT/bootstrap/<timestamp>/report.md`. The report must include
source and asset roots, Python/PyTorch/CUDA versions, GPU UUID/index mapping,
extension status, asset hash status, launcher status, CARLA probe status, and
the exact next stop condition. Never put checkpoints, replay, CARLA logs, or
conda archives in the Git repository.

The bootstrap result is `runtime-ready` only when every applicable gate in
`docs/PORTABILITY_CONTRACT.md` passes. Otherwise return `blocked` with one
concrete remediation, without silently switching projects or hardware.
