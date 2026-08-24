# AdaptDrive Assets

Large or machine-specific assets are stored outside the source repository. Runtime code must receive their locations through explicit environment variables or command-line arguments.

## Canonical immutable assets

### HiP-AD base checkpoint

```text
role: clean_base
filename: hipad/checkpoints/hipad_b2d_stage2_base.pth
size: 1,173,535,966 bytes
sha256: 7711b693293533463732d8a3efa8d5148d203344aad727a4661cb84263613956
```

This is the same audited byte sequence that previously lived under the invalid `I2R-AD/HiP-AD` tree in the original mixed workspace. That historical tree is not a valid Python source root for AdaptDrive.

### ResNet-50 pretrained checkpoint

```text
role: backbone_pretrained
filename: hipad/pretrained/resnet50-19c8e357.pth
size: 102,502,400 bytes
sha256: 19c8e3572231adff6824a2da93fd67b5986919a2e65f8b6007eab4edee220097
```

The closed-loop SAC wrapper disables redundant backbone initialization before loading the full base checkpoint, but this asset may still be required by clean offline HiP-AD training.

### AdaptDrive parent training checkpoint

```text
role: verified_parent_training_checkpoint
filename: adaptdrive/checkpoints/adaptdrive_sig7_step140906.pt
size: 39,419,483 bytes
sha256: 481e0c1b7217351f24e5584bbb5b2ef5b2bfeeb66e45272b6e818d1b216a8fc2
training_signature_version: 7
replay_schema_version: 5
step: 140906
episode: 379
replay_size_recorded_in_source_run: 140857
```

This checkpoint is retained for lineage, registered v8 initialization and deployment extraction. It is not a strict full-state resume checkpoint: its historical replay payload is intentionally absent. The v8 initializer verifies its immutable content and imports only the approved state before resetting counters, optimizers and replay. Its recorded old project root is provenance, not a runtime dependency.

## Roach BEV maps

The asset set contains paired `.h5` and `.h5.manifest.json` files for:

```text
Town01
Town02
Town03
Town04
Town05
Town06
Town07
Town10HD
Town11
Town12
Town13
Town15
```

There are 24 files in total. All source and target files were compared by SHA-256 during the August 21, 2026 migration. Generated visualizations and map-generation logs are not part of the runtime asset set.

## Small repository-owned anchors

The following anchors remain inside `HiP-AD/data/kmeans/` because they are small runtime inputs and are already independent of the legacy source tree:

```text
b2d_det_900.npy
b2d_map_100.npy
b2d_motion_6.npy
b2d_plan_spat_6x8_2m.npy
b2d_plan_spat_6x8_5m.npy
```

Their redistribution status must be included in the release license review.

## Asset policy

- Do not commit model checkpoints, replay mmap data, CARLA runtime output or generated maps to the source repository.
- Validate immutable assets by SHA-256 before use.
- Do not use cross-project absolute symbolic links.
- The vendored upstream snapshot contains one unused historical link,
  `third_party/DCNv4/classification/meta_data/meta`, which points to a former
  server's image dataset. It is not required by AdaptDrive and must not be
  recreated on a new host.
- Do not silently fall back to paths in another research workspace.
- Keep source code independent; share only immutable, content-addressed assets.
