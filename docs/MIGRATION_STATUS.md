# AdaptDrive Migration Status

Last updated: August 24, 2026.

## Objective

Create a clean, independent and efficient AdaptDrive project that can be validated in closed loop and later prepared for public release without depending on the internal `I2R-AD` workspace.

## Completed

- Created separate source, asset, run and archive roots.
- Copied the active `Bench2Drive`, `Bench2DriveZoo` and pre-migration `HiP-AD_clean` source snapshots.
- Excluded approximately 50 GB of `Bench2Drive/runs`, approximately 28 GB of the pre-migration `HiP-AD_clean/evaluation`, old replay, runtime output, nested Git metadata and the invalid legacy `I2R-AD/HiP-AD` source tree.
- Excluded rebuildable `Bench2DriveZoo/build` products and temporary HiP-AD extension objects.
- Preserved the currently used MMCV and HiP-AD CUDA shared libraries for compatibility validation.
- Verified key source files, CUDA shared libraries and a representative anchor by matching source/target SHA-256.
- Confirmed the AdaptDrive runtime chain contains no cross-project symbolic
  links; the unused broken upstream DCNv4 classification-data link is retained
  only as historical upstream metadata and is documented in
  `docs/PORTABILITY_CONTRACT.md`.
- Migrated and verified the HiP-AD base checkpoint.
- Migrated and verified the ResNet-50 pretrained checkpoint.
- Migrated and verified the signature-7, step-140906 parent AdaptDrive checkpoint.
- Migrated all 12 Town Roach maps and manifests; compared all 24 files by SHA-256.
- Replaced the current training contract with path-independent signature v8, including route, base checkpoint, anchors, HiP-AD/DCNv4 code, CARLA semantics, canonical RL sources and all 12 Roach Town assets.
- Added registered v7-to-v8 initialization with fresh counters, optimizers and replay.
- Added UUID replay manifests, checkpoint-specific replay snapshots and payload corruption witnesses for strict v8 resume.
- Added strict full-agent resume and rejected weights-only or metadata-only fallback.
- Made route evaluation and Leaderboard deployment compatible with relocated v7/v8 checkpoints through content provenance rather than physical-root equality.
- Added canonical training, route-evaluation and Leaderboard launchers with external experiment-scoped outputs.
- Passed offline contract, signature, replay, update-gate, dual-critic, dual-replay, planning-gradient, four-level adapter and deployment-loader smoke checks from the migrated staging tree.
- Verified all three canonical launchers against the migrated CARLA/assets with validation-only mode; no simulator process was started.
- Removed 110 dependency-audited legacy files and 6 legacy directories from the AdaptDrive copy while preserving the original `I2R-AD` workspace and a hashed pre-refactor archive.
- Confirmed the final target tree is byte-identical to the accepted staging tree after pruning.
- Re-ran syntax compilation and the canonical contract, signature, replay, portable-runtime, import-isolation, checkpoint-gate, deployment-loader, planning-gradient and four-level adapter checks from the final target root.
- Renamed the temporary migrated source directory from `HiP-AD_clean` to the canonical `HiP-AD` name and replaced name-based legacy detection with project-root provenance isolation.
- Recomputed the post-rename v8 training signature as `afcd7c89aaa288c704edbccd5ea5e174ee63c52ea8eb5076957366dc388ecec7`.
- Vendored the complete 288-file Git-tracked DCNv4 source snapshot at commit `4b848f7dd7da74ff03f7d278f902c6fd05b391b5` under `third_party/DCNv4`, preserving upstream documentation and licensing and excluding only untracked build products and Git metadata.
- Built a `DCNv4 1.0.0.post2` wheel from the vendored source with CUDA 11.8 and `TORCH_CUDA_ARCH_LIST=8.6+PTX`, installed it only in the independent `adaptdrive` environment and verified that its installed extension hash matches the tested build.
- Made canonical launchers require the project-owned DCNv4 source snapshot while loading the installed wheel by default; `DCNV4_ROOT` remains an explicit override only for an alternative package root that already contains a built extension.
- Passed a real DCNv4 and feature-adapter CUDA forward/backward on physical GPU 3, including non-zero input and parameter gradients.
- Recomputed the post-DCNv4-migration v8 training signature as `b8de360954135f75e89622b746d8ab1d0a984059f6c43e238012c2097a084f95`.
- Re-validated all three canonical launchers from the final target root; the complete Leaderboard wrapper chain exited without starting a process or writing evaluation output.
- Made the upstream stage-1/stage-2 configs relocation-safe through `ADAPTDRIVE_ASSET_ROOT` and an explicit `HIPAD_STAGE1_CKPT` override.
- Removed three superseded upstream evaluation wrappers and six archived Line-C launchers that referenced an unavailable historical checkpoint.
- Removed 12 rebuildable MMCV object/Ninja files while preserving the Python/CUDA/C++ sources and compatibility shared libraries.
- Made the video utility path-independent and verified a two-frame MP4 write/read cycle.
- Reduced the AdaptDrive-owned runtime-source absolute-path audit to zero
  matches. Two `/mnt/petrelfs` defaults remain in byte-preserved, non-runtime
  DCNv4 upstream classification examples and are documented as unusable local
  examples in `docs/PORTABILITY_CONTRACT.md`.
- Added the repository-owned `adaptdrive-bootstrap` and
  `adaptdrive-closed-loop` skills under `.agents/skills/` for new-server
  handoff. They discover GPU and graphics mappings at runtime and leave conda
  packaging/unpacking to the environment owner.

## In progress

- New-server handoff: the owner must clone the repository, unpack the conda
  environment, supply the external assets and CARLA path, then run the
  bootstrap skill.
- Run controlled closed-loop validation after the new runtime passes bootstrap
  and receives explicit runtime approval.

## Not yet validated

- Fixed-input old/new output parity.
- Single-route no-update closed-loop parity.
- Short closed-loop finetuning with a new signature and replay.
- Checkpoint save and strict resume within the new project.
- Multi-route and complete Leaderboard evaluation.
- Clean-base versus AdaptDrive controlled A/B evaluation.

## Known migration blockers

### GPU runtime validation

The former Server-10 runtime exposed eight RTX 4090 GPUs and passed a DCNv4
forward/backward on physical GPU 3. That observation is historical, not a new
server default. The new bootstrap must enumerate the current host's GPUs and
revalidate the selected device. CARLA rollout and checkpoint/replay save-resume
remain required before end-to-end acceptance.

### Compatibility binaries

The copied `Bench2DriveZoo/mmcv` tree still contains machine-built CUDA shared libraries used for compatibility validation. Rebuildable object and Ninja metadata have been removed. The retained binaries must either move to a documented external compatibility bundle or be reproducibly rebuilt before public release.

## Completion criteria

AdaptDrive is independent only when all of the following hold:

- no runtime file or required symbolic link resolves into the old `I2R-AD`
  workspace;
- model code, config, anchors, routes and deployment code originate from the AdaptDrive source root;
- immutable external assets match the documented SHA-256 values;
- the old parent checkpoint can be consumed through a documented portable lineage/deployment mechanism;
- new training uses a new signature, replay and experiment ID unless an audited full-state migration is explicitly performed;
- static checks, targeted smoke tests and representative closed-loop tests pass from the new root;
- generated outputs are written only beneath the external run root;
- the repository contains no local datasets, checkpoints, replay, secrets or server-specific defaults;
- third-party redistribution and license obligations are documented before release.
