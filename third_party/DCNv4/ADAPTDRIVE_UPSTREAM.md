# AdaptDrive DCNv4 provenance

AdaptDrive vendors a complete Git-tracked source snapshot of OpenGVLab DCNv4 so that the feature adapter can be rebuilt without resolving source code from the former mixed workspace.

```text
upstream: git@github.com:OpenGVLab/DCNv4.git
commit: 4b848f7dd7da74ff03f7d278f902c6fd05b391b5
branch_at_source: main
snapshot_date: August 22, 2026
git_archive_sha256: f794e0c401241595193a6457198f9e6fec6d1aa8c8d802977d30b97f0bc1679e
tracked_files: 288
```

`UPSTREAM_SHA256SUMS` hashes all 288 files exported from that commit. The snapshot includes the upstream top-level README and LICENSE, `DCNv4_op`, and the classification, detection and segmentation trees. It excludes only `.git` history and untracked machine-generated products from the source checkout.

AdaptDrive directly uses `DCNv4_op/`. The operator is built into a wheel and installed in the independent environment so that compiled products do not need to live in the source tree. Canonical launchers use the installed package by default; `DCNV4_ROOT` is reserved for an explicit alternative package root that already contains a compatible built extension. `build/`, `dist/`, `*.egg-info`, `*.o` and `*.so` are local generated products ignored by the AdaptDrive source policy.

The upstream snapshot contains licensing statements in both the top-level `LICENSE` and individual source headers. They are preserved verbatim. AdaptDrive does not relicense this third-party code; redistribution review remains a separate pre-release task.
