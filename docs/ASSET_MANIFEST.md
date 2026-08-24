# AdaptDrive Asset Manifest

This is the machine-independent manifest for assets that are intentionally
outside the Git repository. The operator supplies the root through
`ADAPTDRIVE_ASSET_ROOT`; bootstrap verifies every immutable entry before use.

## Immutable assets

Paths below are relative to `ADAPTDRIVE_ASSET_ROOT`.

| Relative path | Role | Size (bytes) | SHA-256 | Required by |
| --- | --- | ---: | --- | --- |
| `hipad/checkpoints/hipad_b2d_stage2_base.pth` | clean HiP-AD base checkpoint | 1,173,535,966 | `7711b693293533463732d8a3efa8d5148d203344aad727a4661cb84263613956` | AdaptDrive training and evaluation |
| `hipad/pretrained/resnet50-19c8e357.pth` | clean HiP-AD backbone pretraining | 102,502,400 | `19c8e3572231adff6824a2da93fd67b5986919a2e65f8b6007eab4edee220097` | standalone HiP-AD training when configured |
| `adaptdrive/checkpoints/adaptdrive_sig7_step140906.pt` | registered v7 parent for fresh v8 initialization | 39,419,483 | `481e0c1b7217351f24e5584bbb5b2ef5b2bfeeb66e45272b6e818d1b216a8fc2` | new AdaptDrive training |

The v7 parent is not a v8 full-resume checkpoint. It is imported through the
registered lineage path, with fresh counters, optimizers, and replay.

## Roach maps

`roach_bev_maps/` must contain paired `.h5` and `.h5.manifest.json` files for
Town01, Town02, Town03, Town04, Town05, Town06, Town07, Town10HD, Town11,
Town12, Town13, and Town15. The pair count is 24. Validate the hashes recorded
by the source migration before running a route that uses a map.

## Repository-owned small anchors

These five files remain under `HiP-AD/data/kmeans/` and are expected to be
present in the clone:

```text
b2d_det_900.npy
b2d_map_100.npy
b2d_motion_6.npy
b2d_plan_spat_6x8_2m.npy
b2d_plan_spat_6x8_5m.npy
```

They are runtime inputs, not generated outputs. Their redistribution status is
still subject to the dependency and publication review.

## Runtime-discovered inputs

Routes, CARLA, PythonAPI bindings, and any dataset roots are machine-local
inputs. Bootstrap must resolve them explicitly and record their paths and
hashes (where practical) in the report. It must not infer them from another
research project or recreate a missing file with a symlink.

## Verification example

Use a streaming hash check so large checkpoints do not enter an agent context:

```bash
sha256sum "${ADAPTDRIVE_ASSET_ROOT}/hipad/checkpoints/hipad_b2d_stage2_base.pth"
stat -c '%s %n' "${ADAPTDRIVE_ASSET_ROOT}/hipad/checkpoints/hipad_b2d_stage2_base.pth"
```

Compare the short command output with this manifest and report only mismatches
or the final pass/fail summary.
