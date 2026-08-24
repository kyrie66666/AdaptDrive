# AdaptDrive Current Contract

Last audited: August 21, 2026.

This document is the authoritative description of the frozen system being migrated into AdaptDrive. It distinguishes the active baseline from historical branches and future proposals.

## Model and control chain

```text
CARLA six-camera RGB + 21-dimensional ego state
  -> frozen HiP-AD backbone and FPN
  -> four-level ego-conditioned DCNv4 residual adapter
  -> HiP-AD multi-task head
  -> final planning classification and spatial-2m regression branches
  -> discrete SAC over 48 planning modes
  -> mode-aligned longitudinal and lateral trajectories
  -> clean dual PID
  -> CARLA control
```

## Frozen protocol

| Field | Value |
|---|---|
| Project | AdaptDrive |
| HiP-AD source | `HiP-AD` only |
| Navigation | `hipad_clean_global_plan_v1` |
| Control | `hipad_clean_dual_pid_v2_mode_aligned` |
| Policy modes | 48 |
| Replay schema | 5 |
| New training signature version | 8 |
| Registered legacy parent signature | 7 |
| Adapter type | `dcnv4_feature` |
| Adapter levels | `(0, 1, 2, 3)` |
| Adapter feature dimension | 256 |
| Ego-state dimension | 21 |
| Adapter update | `prediction_only` |
| SAC actor updates adapter | No, in the frozen baseline |
| Active reward variant | Line E |
| Direct dense safety | Enabled |
| Trajectory-occupancy reward | Disabled |
| STCOcc | Not part of AdaptDrive |

## Trainable ownership

The frozen baseline separates SAC and adapter-prediction optimization:

| Parameters | Update source |
|---|---|
| Final planning classification branch | SAC policy loss |
| Final spatial-2m regression branch | SAC policy/candidate gradient and trust regularization |
| Four-level feature adapter | Adapter prediction optimizer |
| Reward prediction head | Adapter prediction optimizer |
| Semantic BEV prediction head | Adapter prediction optimizer |
| Remaining HiP-AD parameters | Frozen |

## Auxiliary prediction path

The adapter prediction optimizer uses detached base FPN features captured during the action forward pass, then recomputes the adapter with a fresh autograd graph. It trains:

- a reward prediction head conditioned on adapted pooled features, ego state and compact action;
- a partial current-frame Roach-style semantic BEV head.

The semantic head emits 15 channels, but the frozen baseline trains only road, lane, latest vehicle, latest walker and latest traffic-light-stop channels. Route and older history channels have zero loss weight. It must not be described as full temporal Roach reconstruction.

## Safety reward boundary

The frozen AdaptDrive run uses Line E and direct actor-geometry safety shaping based on TTC, headway and center distance. The optional trajectory-occupancy reward is off. Its GT occupancy provider and the unfinished STCOcc integration are not part of the AdaptDrive baseline and must not be silently enabled in reproduction commands.

## Deployment subset

The current adapter-aware Leaderboard deployment restores exactly:

- 25 HiP-AD planning tensors;
- 132 four-level feature-adapter tensors.

It intentionally excludes critics, value networks, replay, optimizers and auxiliary prediction heads. Deployment accepts only the registered protocol generations, v7 and v8, and verifies checkpoint version, signature format, base-checkpoint SHA-256, control/replay contract, adapter mode, adapter levels and exact deployment tensor structure. Historical physical project roots are provenance only and are not compatibility keys.

## Initialization and resume

A new AdaptDrive experiment starts at step 0 and episode 0 by importing the registered v7 parent. The import is content-locked to the parent checkpoint, base checkpoint, route file and five HiP-AD anchors. It imports model, adapter, prediction-head, critic/value and temperature state, but never imports legacy optimizers, counters or replay.

Current v8 resume is full-state only. A checkpoint must pair with its UUID replay directory through `replay_ref`, matching experiment ID, training signature, manifest and state hashes. Per-slot payload witnesses reject same-size mmap corruption. There is no weights-only or metadata-only resume fallback.

## Canonical entry points

```text
Bench2Drive/run_adaptdrive_train.sh
Bench2Drive/run_adaptdrive_eval.sh
Bench2Drive/run_adaptdrive_leaderboard.sh
```

All three require an `EXPERIMENT_ID`, use an external `ADAPTDRIVE_RUN_ROOT`, and accept CARLA through `CARLA_ROOT`. Training output is partitioned into `runtime`, `replay`, `checkpoints` and `logs`; evaluation output goes to `evaluations`. `ADAPTDRIVE_VALIDATE_ONLY=1` performs launcher and asset validation without starting CARLA, training or evaluation.

## Historical branches outside the project boundary

The following are not AdaptDrive canonical paths:

- legacy residual HiP-AD SAC;
- Line D regression finetuning;
- VAD teacher experiments;
- STCOcc policy replacement;
- trajectory-occupancy reward experiments;
- machine-specific historical run launchers and result ledgers.
