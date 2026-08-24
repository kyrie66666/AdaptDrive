# AdaptDrive

AdaptDrive is a research codebase for closed-loop autonomous-driving policy adaptation. It combines a clean HiP-AD planning stack with discrete Soft Actor-Critic (SAC), an ego-conditioned multi-level feature adapter, auxiliary prediction objectives, and dense safety reward shaping.

This repository is being extracted from an internal multi-track research workspace. The extraction deliberately separates source code, immutable model/data assets, and generated experiment outputs so that the project can later be audited and released without local checkpoints, replay buffers, machine paths, or CARLA runtime artifacts.

## Current system contract

The frozen migration baseline uses:

- HiP-AD code root: `HiP-AD`
- navigation: `hipad_clean_global_plan_v1`
- control: `hipad_clean_dual_pid_v2_mode_aligned`
- action space: 48 mode-aligned longitudinal/lateral trajectory pairs
- replay schema: `5`
- new training signature version: `8` (registered legacy parent: `7`)
- feature adapter: four-level ego-conditioned DCNv4 adapter
- adapter levels: `(0, 1, 2, 3)`
- adapter update: prediction-only auxiliary optimization
- active reward: Line E plus direct dense safety shaping
- trajectory-occupancy reward: disabled in the frozen baseline
- STCOcc: outside the AdaptDrive project boundary

See `docs/CURRENT_CONTRACT.md` for the detailed contract.

For a new server, start with `.agents/skills/adaptdrive-bootstrap/SKILL.md`.
After its report is runtime-ready, continue with
`.agents/skills/adaptdrive-closed-loop/SKILL.md`. These skills assume that the
environment owner has already unpacked conda and copied external assets; they
do not package or modify the environment.

## Repository layout

```text
AdaptDrive/
├── Bench2Drive/       # Closed-loop environment, SAC, adapter, rewards and evaluation
├── Bench2DriveZoo/    # Framework/runtime dependency retained during compatibility migration
├── HiP-AD/            # Clean HiP-AD source snapshot and small anchor assets
├── third_party/       # Locked third-party source snapshots required by AdaptDrive
├── docs/              # Authoritative architecture, assets and migration status
├── .agents/skills/    # Bootstrap and closed-loop handoff instructions
├── env.example        # Machine-local path template
└── .gitignore         # Prevents checkpoints, replay, runtime and evaluation outputs entering Git
```

Large and machine-specific files live outside the source tree:

```text
AdaptDrive-assets/     # Immutable checkpoints, pretrained weights and Roach maps
AdaptDrive-runs/       # Training, replay, logs and evaluations
AdaptDrive-archive/    # Historical/internal migration evidence
```

## Migration status

As of August 22, 2026:

- the active source snapshot has been copied without historical runs, replay, evaluation output, nested Git metadata, or legacy `HiP-AD` source;
- key source files, CUDA extensions and HiP-AD anchors match the source workspace by SHA-256;
- the canonical HiP-AD base checkpoint, ResNet pretrained checkpoint, AdaptDrive training checkpoint, and all 12 Town Roach maps have been copied and verified by SHA-256;
- the AdaptDrive runtime chain contains no cross-project symbolic links; an
  unused broken upstream DCNv4 classification-data link is documented in
  `docs/PORTABILITY_CONTRACT.md`;
- path-independent v8 initialization/resume, portable v7/v8 deployment loading, UUID replay pairing and canonical launchers pass offline validation;
- the dependency-audited legacy prune removed 110 files and 6 directories from the AdaptDrive copy; the original `I2R-AD` workspace was not modified;
- the portability cleanup removed historical Line-C/evaluation launchers and rebuildable MMCV object/Ninja files, made the upstream offline configs relocation-safe, and reduced the non-documentation absolute-path audit to zero matches;
- the migrated source directory was normalized from the temporary `HiP-AD_clean` name to the canonical `HiP-AD` name before any formal v8 run was created;
- the complete Git-tracked DCNv4 source at commit `4b848f7dd7da74ff03f7d278f902c6fd05b391b5` is vendored under `third_party/DCNv4`, with an exact source manifest, preserved upstream license and a separately installed reproducible wheel;
- the final target tree passes syntax compilation, contract/signature/replay/runtime smoke checks, real-checkpoint CPU planning-gradient and four-level adapter initialization checks, plus validation-only checks for all three canonical launchers;
- CUDA-backed DCNv4 execution and closed-loop validation are still in progress.

The repository must not yet be described as end-to-end validated from its new location. See `docs/MIGRATION_STATUS.md`.

## Local configuration

Copy `env.example` to an untracked machine-local environment file or export equivalent variables in the shell. Do not commit local paths.

Required path concepts are:

- `ADAPTDRIVE_ROOT`
- `ADAPTDRIVE_ASSET_ROOT`
- `ADAPTDRIVE_RUN_ROOT`
- `HIPAD_ROOT`
- `HIPAD_CONFIG`
- `HIPAD_BASE_CKPT`
- `HIPAD_STAGE1_CKPT` only for standalone stage-2 HiP-AD offline training
- `ROACH_BEV_MAP_ROOT`
- `CARLA_ROOT`
- optional `DCNV4_ROOT` only when overriding the wheel installed from `third_party/DCNv4`
- `B2DZOO_ROOT`

Every run also requires a safe `EXPERIMENT_ID`. The canonical entry points are:

```text
Bench2Drive/run_adaptdrive_train.sh
Bench2Drive/run_adaptdrive_eval.sh
Bench2Drive/run_adaptdrive_leaderboard.sh
```

Set `ADAPTDRIVE_VALIDATE_ONLY=1` to validate paths and launcher contracts without starting CARLA, training or evaluation.

## Assets

Checkpoints and generated maps are intentionally excluded from the source repository. Their expected roles, sizes and SHA-256 values are documented in `docs/ASSETS.md`.

## Reproducibility policy

- Do not bypass a training-signature mismatch to continue an old full-state run.
- Treat each v8 checkpoint and its UUID replay manifest, state snapshot and mmap payload as one unit when exact continuation is required.
- Cross-directory finetuning starts a v8 experiment with fresh counters, optimizers and replay through the registered v7 parent import.
- Keep Python source snapshots independent across research projects; only immutable, hash-addressed assets may be shared.
- Record exact validation commands and environment assumptions for every reported result.

## Publication readiness

Before public release, the project still requires:

- dependency and redistribution-license review for vendored HiP-AD, Bench2Drive, ScenarioRunner, Bench2DriveZoo and MMCV code;
- separation or reproducible rebuilding of retained MMCV compatibility binaries;
- portable deployment-bundle support;
- clean installation documentation and reproducible environment specification;
- representative closed-loop validation and clean-base versus AdaptDrive evaluation;
- a final secrets and large-file release audit.

No license is asserted for the combined AdaptDrive repository until the dependency and redistribution review is complete. Existing third-party license files must be preserved.
