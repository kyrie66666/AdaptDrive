# AdaptDrive Project Layout

AdaptDrive separates version-controlled source from immutable assets and generated runs.

```text
/path/to/
├── AdaptDrive/
│   ├── Bench2Drive/
│   ├── Bench2DriveZoo/
│   ├── HiP-AD/
│   ├── third_party/
│   │   └── DCNv4/
│   ├── docs/
│   ├── env.example
│   └── .gitignore
├── AdaptDrive-assets/
│   ├── hipad/
│   │   ├── checkpoints/
│   │   └── pretrained/
│   ├── adaptdrive/
│   │   └── checkpoints/
│   └── roach_bev_maps/
├── AdaptDrive-runs/
│   ├── runtime/<experiment_id>/
│   ├── checkpoints/<experiment_id>/
│   ├── replay/<experiment_id>/<replay_uuid>/
│   ├── logs/<experiment_id>/
│   └── evaluations/<experiment_id>/
└── AdaptDrive-archive/
```

## Source tree responsibilities

### `Bench2Drive/`

Owns the closed-loop environment, SAC agent, replay, reward implementation, clean navigation/control bridge, feature adapter, auxiliary prediction tasks, Leaderboard adapter agent, tests and launch entry points.

### `HiP-AD/`

Owns the clean HiP-AD model/plugin implementation, clean planner/PID chain, routes and small anchor files. It must not link to a legacy HiP-AD source tree.

### `Bench2DriveZoo/`

Temporarily retained as a compatibility dependency while the exact canonical import graph is reduced. Rebuildable build directories are excluded. Its long-term vendoring or installation strategy requires license and dependency review.

### `third_party/DCNv4/`

Contains the complete Git-tracked upstream DCNv4 source snapshot fixed at commit `4b848f7dd7da74ff03f7d278f902c6fd05b391b5`. `DCNv4_op/` is the operator used by the four-level AdaptDrive feature adapter. Upstream source, scripts, tasks, README and license are preserved; local build directories, egg metadata and compiled shared libraries remain ignored machine products. `ADAPTDRIVE_UPSTREAM.md`, `ADAPTDRIVE_BUILD.md` and `UPSTREAM_SHA256SUMS` record provenance, the verified build procedure and exact source content. Canonical launchers use the package installed in the active environment; `DCNV4_ROOT` is an explicit override for an alternative package root that already contains a built extension.

## External-root responsibilities

### `AdaptDrive-assets/`

Stores immutable, hash-addressed model and map assets. It is not a source repository and may have different access controls or distribution rules.

### `AdaptDrive-runs/`

Stores all mutable generated data: checkpoints, replay mmap, logs, CARLA runtime state, visualizations and evaluation outputs.

Every training and evaluation invocation uses one safe `EXPERIMENT_ID`. A v8 replay is stored in a UUID child directory and is paired to checkpoints through immutable manifest/state references; moving unrelated replay directories into the experiment is not a valid resume mechanism.

### `AdaptDrive-archive/`

Stores internal migration evidence and superseded experiment history. Nothing here is part of the canonical runtime path.

## Design rule

Source roots are physically independent. Required third-party source is pinned inside the project, while immutable data may be shared by explicit path and hash. Cross-project source links and implicit fallbacks are forbidden.
