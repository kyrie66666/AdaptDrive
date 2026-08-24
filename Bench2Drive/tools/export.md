# AdaptDrive Checkpoint Deployment

The historical planning-only checkpoint exporter was removed from AdaptDrive because it did not preserve the complete SAC, four-level adapter and prediction-head contract.

Use the canonical checkpoint directly:

- registered v7 parent: deployment and v8 initialization only;
- v8 checkpoint: deployment or strict full-state resume when its UUID replay state and mmap payload are present.

Route-level evaluation:

```bash
export PROJECT_ROOT=/path/to/AdaptDrive
export ADAPTDRIVE_ASSET_ROOT=/path/to/AdaptDrive-assets
export ADAPTDRIVE_RUN_ROOT=/path/to/AdaptDrive-runs
export CARLA_ROOT=/path/to/carla
export EXPERIMENT_ID=my_experiment
export FINETUNE_CKPT=/path/to/checkpoint.pt

bash "${PROJECT_ROOT}/Bench2Drive/run_adaptdrive_eval.sh"
```

Leaderboard evaluation:

```bash
bash "${PROJECT_ROOT}/Bench2Drive/run_adaptdrive_leaderboard.sh"
```

Set `ADAPTDRIVE_VALIDATE_ONLY=1` first to validate the complete path and launcher contract without starting CARLA or writing evaluation output.

Do not export only the planning head and describe it as an AdaptDrive training checkpoint. Deployment loaders validate the protocol version, training signature, base checkpoint hash, control/replay contract, adapter levels and exact deployment tensor structure.
