# AdaptDrive Portability Contract

This document defines what a fresh server must provide before AdaptDrive can be
called runtime-ready. It is deliberately separate from the model and reward
contract in `CURRENT_CONTRACT.md`.

## Root separation

The source tree is the directory containing this file. The following concepts
must be supplied by the operator or discovered relative to that tree:

| Variable | Meaning | Repository policy |
| --- | --- | --- |
| `ADAPTDRIVE_ROOT` | AdaptDrive source root | Must resolve to the cloned source tree |
| `ADAPTDRIVE_ASSET_ROOT` | checkpoints, pretrained weights, and Roach maps | External to the source tree; verify hashes before use |
| `ADAPTDRIVE_RUN_ROOT` | runtime files, replay, checkpoints, logs, and evaluations | External to both source and asset trees |
| `CARLA_ROOT` | CARLA installation containing `CarlaUE4.sh` | Machine-local; never commit it |
| `HIPAD_ROOT` | `ADAPTDRIVE_ROOT/HiP-AD` unless explicitly relocated | Must not point at the old mixed workspace |
| `B2D_ROOT` | `ADAPTDRIVE_ROOT/Bench2Drive` | Derived from the source root |
| `B2DZOO_ROOT` | `ADAPTDRIVE_ROOT/Bench2DriveZoo` | Derived from the source root |

Do not add absolute paths for a particular server to source, configs, skills,
or committed documentation. A local shell file based on `env.example` is the
right place for machine-specific values.

## Conda and binary boundary

The environment owner is responsible for `conda-pack`, transfer, extraction,
and `conda-unpack`. Bootstrap must inspect the already active environment but
must not create, pack, unpack, or upload an environment.

The repository is the source of truth for CUDA extension source, not for
machine-built binaries. On a new host, check the Python, PyTorch, CUDA toolkit,
GCC, glibc, and GPU compute capability before reusing any `.so`. Rebuild
DCNv4, MMCV custom ops, `iou3d`, `roiaware`, and deformable aggregation when
the binary was built for a different ABI, CUDA version, or architecture. A
binary that imports is not by itself evidence that its kernels are correct.

`third_party/DCNv4` is a complete vendored source snapshot and must remain
available for rebuilds. The broken upstream classification link
`classification/meta_data/meta` points into a former server's image dataset;
it is not an AdaptDrive runtime dependency and must not be recreated.
Two upstream classification examples also retain `/mnt/petrelfs` defaults in
`classification/export.py` and `classification/train_in1k.sh`. They are outside
the AdaptDrive adapter build and runtime chain and are preserved only to keep
the locked upstream source byte-identical. Never use those defaults on a new
server.

## GPU and graphics discovery

There is no permanent GPU ban. Enumerate every visible GPU, its UUID, health,
memory use, and compute capability. Select a healthy device at runtime, or
honor an explicit user-selected physical index. Keep the physical index used by
PyTorch, CARLA's `CUDA_VISIBLE_DEVICES`, and `-graphicsadapter` consistent;
record the mapping in the bootstrap report. Do not assume that the logical
CUDA index after masking equals the host index.

Probe the current server's available display or EGL/headless path and Vulkan
ICD before starting CARLA. Do not copy `DISPLAY`, `VK_ICD_FILENAMES`,
`graphicsadapter`, or port values from historical notes. Prefer an explicit,
free RPC port and Traffic Manager port per experiment and record both.

## Runtime output and logs

All generated data must live below `ADAPTDRIVE_RUN_ROOT/<kind>/<EXPERIMENT_ID>`:

```text
runtime/       CARLA process/runtime state
replay/        UUID-paired mmap replay and manifests
checkpoints/   v8 full-state checkpoints
logs/          concise launcher and experiment logs
evaluations/   route and Leaderboard results
```

Use a unique, safe `EXPERIMENT_ID`. During triage, read markers, error classes,
process state, and the last few log lines. Do not load entire CARLA or training
logs into an agent context unless a specific failure requires it.

## Acceptance gates

An environment is runtime-ready only after all applicable gates pass:

1. Source, assets, run root, CARLA, and the active Python resolve to the
   intended locations and contain no link into the old `I2R-AD` workspace.
2. Required assets exist and match `docs/ASSETS.md`; checkpoints are regular
   non-symlink files.
3. Python/PyTorch/CUDA and every required extension import successfully, and
   the DCNv4 and adapter forward/backward smoke passes on the selected GPU.
4. The launcher validation-only mode passes without starting CARLA or writing
   experiment output.
5. A lightweight CARLA probe proves that RPC, Traffic Manager, sensor frames,
   and the selected graphics path work together.

Failure of a gate is a stop condition for closed-loop training. Report the
failed gate and the smallest useful evidence; do not silently fall back to a
different source tree, checkpoint, GPU, or reward variant.
