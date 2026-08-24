---
name: adaptdrive-closed-loop
description: Continue AdaptDrive after bootstrap is runtime-ready by running controlled SAC, adapter, checkpoint/resume, and short route-validation experiments through the canonical launchers.
---

# AdaptDrive Closed Loop

Use this skill only after `adaptdrive-bootstrap` has produced a passing
runtime-readiness report. It continues the AdaptDrive work described by
`docs/CURRENT_CONTRACT.md`: clean HiP-AD, four-level DCNv4 feature adapter,
prediction-only adapter updates, discrete SAC, and direct dense safety reward.

## Safety and experiment boundaries

- Read the current contract, portability contract, asset manifest, migration
  status, and the bootstrap report before running anything.
- Ask for explicit authorization immediately before starting CARLA, a
  long-lived launcher, training, or evaluation. Offline checks and
  `ADAPTDRIVE_VALIDATE_ONLY=1` are not simulator runs.
- Never modify core model/training code or configs to make a smoke run pass.
  Use command-line overrides only when they are already supported by the
  launcher and preserve the frozen contract.
- Do not use a weights-only or metadata-only v8 resume. A strict v8 resume
  requires the checkpoint, UUID replay directory, manifest, state snapshot, and
  payload witnesses for the same experiment and exact training signature.
- Use a fresh `EXPERIMENT_ID` for every new initialization or smoke run. Never
  reuse another run's replay or output directory.
- Do not read full CARLA or training logs. Inspect markers, error classes,
  process state, and the last few dozen lines only.

## Preconditions

Verify these before a CARLA process is requested:

1. `ADAPTDRIVE_ROOT`, `ADAPTDRIVE_ASSET_ROOT`, and `ADAPTDRIVE_RUN_ROOT` are
   absolute and independent.
2. `HIPAD_ROOT` is the project-owned `HiP-AD`; the route file exists; all five
   k-means anchors exist; all required checkpoints and Roach map pairs pass the
   manifest checks.
3. The active environment imports PyTorch and DCNv4, and the real GPU adapter
   forward/backward smoke has passed.
4. The current GPU table has been rechecked. Choose a healthy device at
   runtime; GPU 0 is allowed. Keep the physical device, CARLA CUDA visibility,
   and graphics adapter mapping explicit in the experiment report.
5. Choose a free RPC/TM port pair. Do not copy ports from historical server
   notes or run two experiments with the same pair.

## Offline contract pass

From `ADAPTDRIVE_ROOT`, run the focused checks and retain only their summaries:

```bash
python Bench2Drive/test_adaptdrive_contract_smoke.py
python Bench2Drive/test_adaptdrive_signature_v8_smoke.py
python Bench2Drive/test_adaptdrive_replay_protocol_smoke.py
python Bench2Drive/test_adaptdrive_portable_runtime_smoke.py
python Bench2Drive/test_feature_dcnv4_adapter_smoke.py
```

Run the clean HiP-AD import/model checks appropriate to the available GPU,
including `test_hipad_clean_import_isolation.py`,
`test_hipad_clean_navigation_binding.py`, and
`test_hipad_clean_model_load_smoke.py` when the base checkpoint and CARLA
PythonAPI are available. A CPU-only skip for FlashAttention or DCNv4 is not an
end-to-end pass.

Validate the three canonical launcher contracts with an experiment ID and the
real local paths:

```bash
ADAPTDRIVE_VALIDATE_ONLY=1 EXPERIMENT_ID=offline-check \
  bash Bench2Drive/run_adaptdrive_train.sh
ADAPTDRIVE_VALIDATE_ONLY=1 EXPERIMENT_ID=offline-check \
  bash Bench2Drive/run_adaptdrive_eval.sh
ADAPTDRIVE_VALIDATE_ONLY=1 EXPERIMENT_ID=offline-check \
  bash Bench2Drive/run_adaptdrive_leaderboard.sh
```

If any contract or launcher check fails, stop before CARLA and report the
first failing marker.

## Controlled six-step run

After explicit authorization, create a new experiment directory and start the
canonical training launcher with the selected physical GPU and free ports. Use
the registered v7 parent through the launcher's default `--init-from` path,
unless the owner supplied a different audited initialization. For a short
closed-loop acceptance run, limit the trainer to six environment steps and
make the update cadence explicit so the run exercises replay, critic, policy,
adapter prediction, and dense safety bookkeeping rather than only model load.
Keep the frozen defaults for adapter mode (`dcnv4_feature`), levels
`0 1 2 3`, prediction-only updates, Line E, and direct dense safety.

Example shape (adapt numeric flags only to options present in the checked-out
launcher):

```bash
EXPERIMENT_ID="${EXPERIMENT_ID}" \
GPU_ID="${GPU_ID}" PORT="${PORT}" TRAFFIC_MANAGER_PORT="${TRAFFIC_MANAGER_PORT}" \
CARLA_CUDA_VISIBLE_DEVICES="${GPU_ID}" \
bash Bench2Drive/run_adaptdrive_train.sh \
  --max-train-steps 6 \
  --learning-starts 0 --train-every-n-steps 1 --gradient-steps 1 \
  --policy-learning-starts 0 --policy-update-every-n-steps 1 \
  --min-critic-updates-before-policy 0 --checkpoint-every 6
```

Do not fabricate a successful run from launcher output. Confirm that the
CARLA client connected, six current-frame sensor packets arrived, replay
entries were accepted, and the process ended through its normal cleanup path.
If CARLA aborts, preserve only this experiment's process IDs and final log
tail; do not kill unrelated users' processes.

## Checkpoint and strict resume

After the six-step run, verify that `checkpoint_latest.pt` is a regular file
under the experiment's checkpoint directory and inspect its small metadata
header without loading model tensors into context. Confirm:

- checkpoint version is 2 and training signature version is 8;
- signature equals the one computed from the current source, config, assets,
  and runtime contract;
- `experiment_id`, counters, and `replay_ref` agree;
- the replay manifest, state snapshot, mmap shape/dtype, and payload witnesses
  all match the checkpoint;
- adapter and prediction-head state is present, while deployment extraction
  still contains exactly the documented HiP-AD and adapter tensor subsets.

Resume the same experiment only after the owner authorizes a second launcher
process. Use `RESUME_FROM` with no explicit `INIT_FROM`; the launcher must
reject both together. Verify that step and episode counters continue, replay
UUID and hashes remain paired, and a deliberately changed signature or
corrupted slot is rejected. Never replace a failed strict resume with a fresh
run under the same experiment ID.

## Short route evaluation

Only after the six-step checkpoint and strict-resume checks pass, request
authorization for a short fixed-route evaluation. Use a separate evaluation
experiment ID or the evaluated run's read-only checkpoint with an independent
evaluation output directory. Verify route selection, sensor frame freshness,
clean dual-PID control, and finite outputs. Then use the adapter-aware
Leaderboard wrapper if the short route is healthy.

Keep evaluation output under `ADAPTDRIVE_RUN_ROOT/evaluations/<EXPERIMENT_ID>`.
Do not enable STCOcc, trajectory-occupancy reward, Line-C historical branches,
or an external model adapter: those are outside the frozen AdaptDrive work.

## Completion report

Write a compact report containing the source/config/base-checkpoint hashes,
training signature, GPU UUID and physical/logical mapping, RPC/TM ports,
experiment ID, six-step counters, replay and checkpoint hashes, strict-resume
result, route result, and any remaining blocker. Include exact commands and
only concise log tails. A closed-loop pass means the evidence exists for each
gate; it does not mean a full multi-route or publication evaluation has been
completed.
