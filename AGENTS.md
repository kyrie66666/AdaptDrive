# AdaptDrive Repository Guidelines

## Project scope

AdaptDrive combines three project-owned runtime trees and one vendored
dependency snapshot:

- `Bench2Drive/`: CARLA environment, Leaderboard integration, SAC, replay,
  reward, training launchers, evaluation, and smoke checks.
- `Bench2DriveZoo/`: model/framework compatibility code and MMCV custom-op
  sources.
- `HiP-AD/`: the clean HiP-AD model, agent, configs, anchors, and entry points.
- `third_party/DCNv4/`: the complete locked DCNv4 upstream source used by the
  four-level feature adapter.

The frozen model, optimizer, reward, and resume behavior is defined in
`docs/CURRENT_CONTRACT.md`. Do not silently mix code or assets from another
research project.

## New-server workflow

Before CARLA, training, or evaluation on a new server, use the repository skill
`.agents/skills/adaptdrive-bootstrap/SKILL.md`. It assumes the operator has
already unpacked the conda environment and copied external assets. Do not
create, pack, unpack, or upload conda environments on the operator's behalf.

Continue experiments only after bootstrap is runtime-ready, using
`.agents/skills/adaptdrive-closed-loop/SKILL.md`. Read
`docs/PORTABILITY_CONTRACT.md`, `docs/ASSET_MANIFEST.md`, and
`docs/MIGRATION_STATUS.md` before issuing runtime commands.

There is no fixed GPU ban. Enumerate the current host's GPUs and select a
healthy device dynamically, including GPU 0 when appropriate. Discover the
CUDA/Vulkan adapter mapping, display or EGL mode, and free RPC/TM ports on the
current server. Historical Server-10 values are not defaults.

## Paths and assets

Use `ADAPTDRIVE_ROOT`, `ADAPTDRIVE_ASSET_ROOT`, `ADAPTDRIVE_RUN_ROOT`,
`HIPAD_ROOT`, `B2D_ROOT`, `B2DZOO_ROOT`, and `CARLA_ROOT`. Keep machine paths in
an untracked local environment file based on `env.example`; do not add new
hard-coded absolute paths.

Checkpoints, pretrained weights, Roach maps, replay, CARLA, conda environments,
and experiment outputs are external to Git. Verify immutable assets against
`docs/ASSET_MANIFEST.md`. Generated output must remain below the external run
root and a unique `EXPERIMENT_ID`.

Machine-built `.so`, object, Ninja, wheel, and egg-info files are not source of
truth. Audit Python/PyTorch/CUDA/compiler compatibility and rebuild DCNv4,
MMCV, iou3d, roiaware, and deformable aggregation from repository source when
the target ABI or GPU architecture changes.

## Entry points and validation

Canonical runtime entry points are:

```text
Bench2Drive/run_adaptdrive_train.sh
Bench2Drive/run_adaptdrive_eval.sh
Bench2Drive/run_adaptdrive_leaderboard.sh
```

Use `ADAPTDRIVE_VALIDATE_ONLY=1` before starting a simulator. Prefer focused
syntax and smoke checks, including the AdaptDrive contract, signature, replay,
portable-runtime, and feature-adapter checks. A CPU-only skip is not a CUDA or
closed-loop pass.

Ask before modifying core model/training code, configs, launchers, CUDA source,
or running CARLA, training, evaluation, package installation, or Git writes.
Never bypass a signature or replay mismatch. Preserve only the current
experiment's PID/PGID for cleanup; do not terminate unrelated processes.

Keep logs concise. Report pass/fail markers, error classes, process state, and
the final few dozen lines. Do not bulk-read CARLA or training logs unless a
specific unresolved failure requires more context.
