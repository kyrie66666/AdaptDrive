"""Process-level isolation and provenance checks for the HiP-AD project tree.

The OpenMMLab registries used by HiP-AD are process global.  Loading two
different ``projects.mmdet3d_plugin`` trees in one interpreter is therefore not
recoverable by deleting entries from ``sys.modules``.  Call
``activate_hipad_project_root`` before importing any HiP-AD/OpenMMLab model
module and start a new process when changing roots.
"""

from __future__ import annotations

import inspect
import hashlib
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Dict, Iterable, Optional


MODEL_MODULE_PREFIXES = ("projects", "mmdet", "mmdet3d", "mmcv")
CANONICAL_HIPAD_BASE_SHA256 = "7711b693293533463732d8a3efa8d5148d203344aad727a4661cb84263613956"


class HiPADProjectIsolationError(RuntimeError):
    """Raised when the process contains mixed or unexpected HiP-AD sources."""


def _resolved(path: os.PathLike) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _module_origin(module: ModuleType) -> Optional[Path]:
    origin = getattr(module, "__file__", None)
    if not origin:
        spec = getattr(module, "__spec__", None)
        origin = getattr(spec, "origin", None)
    if not origin or origin in {"built-in", "frozen"}:
        spec = getattr(module, "__spec__", None)
        search_locations = getattr(spec, "submodule_search_locations", None)
        if search_locations:
            locations = [_resolved(location) for location in search_locations]
            if locations:
                return locations[0]
        return None
    return _resolved(origin)


def _iter_loaded_modules(prefixes: Iterable[str]):
    prefixes = tuple(prefixes)
    for name, module in tuple(sys.modules.items()):
        if module is None:
            continue
        if any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes):
            yield name, module


def _hipad_project_root_from_path(path: Path) -> Optional[Path]:
    """Return the HiP-AD project owning a path entry or imported module."""

    current = path if path.is_dir() else path.parent
    for candidate in (current, *current.parents):
        plugin_init = candidate / "projects" / "mmdet3d_plugin" / "__init__.py"
        if plugin_init.is_file():
            return candidate
    return None


def _find_competing_hipad_path_entries(active_root: Path):
    matches = []
    for entry in sys.path:
        if not entry:
            continue
        resolved = _resolved(entry)
        candidate = _hipad_project_root_from_path(resolved)
        if candidate is not None and candidate != active_root:
            matches.append((entry, candidate))
    return matches


def activate_hipad_project_root(
    project_root: os.PathLike,
    *,
    repo_root: Optional[os.PathLike] = None,
    require_clean_tree: bool = True,
) -> Path:
    """Activate one HiP-AD tree before model imports and reject mixed state.

    This function deliberately does not remove modules or path entries.  A
    contaminated process fails and must be restarted, which preserves registry
    correctness and makes provenance failures visible.
    """

    root = _resolved(project_root)
    if not root.is_dir():
        raise FileNotFoundError(f"HiP-AD project root does not exist: {root}")
    plugin_init = root / "projects" / "mmdet3d_plugin" / "__init__.py"
    if not plugin_init.is_file():
        raise HiPADProjectIsolationError(
            f"HiP-AD root does not contain projects/mmdet3d_plugin: {root}"
        )

    repository = _resolved(repo_root) if repo_root is not None else root.parent
    canonical_root = repository / "HiP-AD"
    if require_clean_tree and root != canonical_root:
        raise HiPADProjectIsolationError(
            f"AdaptDrive requires HIPAD_ROOT={canonical_root}, got {root}"
        )

    competing_entries = _find_competing_hipad_path_entries(root)
    if require_clean_tree and competing_entries:
        detail = ", ".join(str(candidate) for _, candidate in competing_entries)
        raise HiPADProjectIsolationError(
            "A competing HiP-AD project is already present in sys.path; start a clean process: " + detail
        )

    contaminated = []
    wrong_projects = []
    for name, module in _iter_loaded_modules(MODEL_MODULE_PREFIXES):
        origin = _module_origin(module)
        if origin is None:
            continue
        owning_root = _hipad_project_root_from_path(origin)
        if require_clean_tree and owning_root is not None and owning_root != root:
            contaminated.append(f"{name}={origin}")
        if (name == "projects" or name.startswith("projects.")) and not _is_relative_to(origin, root):
            wrong_projects.append(f"{name}={origin}")
    if contaminated:
        raise HiPADProjectIsolationError(
            "Competing HiP-AD modules are already loaded; start a clean process: " + "; ".join(contaminated)
        )
    if wrong_projects:
        raise HiPADProjectIsolationError(
            "A different projects package is already loaded; start a clean process: " + "; ".join(wrong_projects)
        )

    root_str = str(root)
    bench2drive_str = str(root / "bench2drive")
    sys.path[:] = [entry for entry in sys.path if _resolved(entry or os.curdir) != root]
    sys.path.insert(0, root_str)
    if (root / "bench2drive").is_dir():
        sys.path[:] = [entry for entry in sys.path if _resolved(entry or os.curdir) != _resolved(bench2drive_str)]
        sys.path.insert(1, bench2drive_str)

    os.environ["HIPAD_ROOT"] = root_str
    return root


def _source_file(obj) -> Optional[Path]:
    try:
        source = inspect.getsourcefile(obj) or inspect.getfile(obj)
    except (TypeError, OSError):
        return None
    return _resolved(source) if source else None


def collect_hipad_provenance(project_root: os.PathLike) -> Dict[str, str]:
    """Collect and validate model, plugin, planner, and controller origins."""

    root = _resolved(project_root)
    import bench2drive
    import mmcv
    import mmdet
    import mmdet3d
    import projects
    import projects.mmdet3d_plugin
    from bench2drive.leaderboard.team_code.pid_controller import PIDController
    from bench2drive.leaderboard.team_code.planner import RoutePlanner
    from projects.mmdet3d_plugin.models.sparse_detector import SparseDetector
    from projects.mmdet3d_plugin.models.sparse_onedecoder import SparseOneDecoder
    from projects.mmdet3d_plugin.ops import feature_maps_format

    entries = {
        "hipad_project_root": root,
        "projects": _module_origin(projects),
        "projects.mmdet3d_plugin": _module_origin(projects.mmdet3d_plugin),
        "mmdet": _module_origin(mmdet),
        "mmdet3d": _module_origin(mmdet3d),
        "mmcv": _module_origin(mmcv),
        "SparseDetector": _source_file(SparseDetector),
        "SparseOneDecoder": _source_file(SparseOneDecoder),
        "feature_maps_format": _source_file(feature_maps_format),
        "bench2drive": _module_origin(bench2drive),
        "RoutePlanner": _source_file(RoutePlanner),
        "PIDController": _source_file(PIDController),
    }

    required_clean = (
        "projects",
        "projects.mmdet3d_plugin",
        "SparseDetector",
        "SparseOneDecoder",
        "feature_maps_format",
    )
    required_clean_bench = ("bench2drive", "RoutePlanner", "PIDController")
    missing = [name for name, path in entries.items() if name != "hipad_project_root" and path is None]
    if missing:
        raise HiPADProjectIsolationError("Missing provenance paths: " + ", ".join(missing))

    wrong = []
    for name in required_clean:
        if not _is_relative_to(entries[name], root):
            wrong.append(f"{name}={entries[name]}")
    clean_bench_root = root / "bench2drive"
    for name in required_clean_bench:
        if not _is_relative_to(entries[name], clean_bench_root):
            wrong.append(f"{name}={entries[name]}")
    for name, origin in entries.items():
        if name == "hipad_project_root":
            continue
        owning_root = _hipad_project_root_from_path(origin)
        if owning_root is not None and owning_root != root:
            wrong.append(f"{name}={origin}")
    if wrong:
        raise HiPADProjectIsolationError("HiP-AD provenance mismatch: " + "; ".join(wrong))

    return {name: str(path) for name, path in entries.items()}


def validate_runtime_asset(path: os.PathLike, *, label: str, reject_symlink: bool = False) -> Path:
    """Resolve one regular runtime asset and optionally reject symbolic links.

    Use this for code-adjacent runtime assets such as configs and anchors. A
    model checkpoint is a data asset rather than importable Python code, so use
    validate_hipad_checkpoint_asset for checkpoints.
    """

    raw = Path(path).expanduser()
    if not raw.exists():
        raise FileNotFoundError(f"{label} does not exist: {raw}")
    if reject_symlink and raw.is_symlink():
        raise HiPADProjectIsolationError(f"{label} must not be a symlink: {raw} -> {raw.resolve()}")
    resolved = raw.resolve()
    if not resolved.is_file():
        raise HiPADProjectIsolationError(f"{label} is not a regular file: {resolved}")
    return resolved


def validate_hipad_checkpoint_asset(
    path: os.PathLike,
    *,
    label: str = "HiP-AD checkpoint",
    reject_symlink: bool = True,
    checkpoint_role: str = "clean_base",
    repo_root: Optional[os.PathLike] = None,
) -> Path:
    """Resolve a checkpoint and enforce the audited clean-base content hash.

    Checkpoints are external data assets. AdaptDrive requires Python code,
    plugins, configs and anchors to come from its canonical HiP-AD source tree,
    while checkpoints must remain outside that source tree.
    """

    raw = Path(path).expanduser()
    if not raw.exists():
        raise FileNotFoundError(f"{label} does not exist: {raw}")
    if reject_symlink and raw.is_symlink():
        raise HiPADProjectIsolationError(f"{label} must not be a symlink: {raw} -> {raw.resolve()}")
    resolved = raw.resolve()
    if not resolved.is_file():
        raise HiPADProjectIsolationError(f"{label} is not a regular file: {resolved}")
    repository = _resolved(repo_root) if repo_root is not None else _resolved(Path(__file__).parents[3])
    source_root = repository / "HiP-AD"
    if _is_relative_to(resolved, source_root):
        raise HiPADProjectIsolationError(
            f"{label} must be stored outside the HiP-AD source tree: {resolved}"
        )
    if str(checkpoint_role) == "clean_base":
        actual_sha256 = _sha256(resolved)
        if actual_sha256 != CANONICAL_HIPAD_BASE_SHA256:
            raise HiPADProjectIsolationError(
                f"{label} SHA-256 mismatch: found {actual_sha256}, "
                f"expected {CANONICAL_HIPAD_BASE_SHA256}"
            )
    return resolved


def hipad_checkpoint_asset_origin(path: os.PathLike, *, repo_root: Optional[os.PathLike] = None) -> str:
    """Classify checkpoint location for provenance/signature logs."""

    resolved = _resolved(path)
    repository = _resolved(repo_root) if repo_root is not None else _resolved(Path(__file__).parents[3])
    source_root = repository / "HiP-AD"
    if _is_relative_to(resolved, source_root):
        return "hipad_source_tree_checkpoint_asset"
    return "external_checkpoint_asset"


def validate_hipad_checkpoint_role(path: Path, role: str) -> None:
    """Prevent historical finetuned weights from silently becoming the base gate."""

    role = str(role)
    if role not in {"clean_base", "clean_finetuned"}:
        raise ValueError(f"unsupported HiP-AD checkpoint role: {role}")
    lowered_parts = [part.lower() for part in path.parts]
    looks_finetuned = any("finetun" in part for part in lowered_parts) or path.name.lower().startswith("line_")
    if role == "clean_base" and looks_finetuned:
        raise HiPADProjectIsolationError(
            f"checkpoint role is clean_base but path is a historical finetuned asset: {path}; "
            "provide the canonical base checkpoint or explicitly select clean_finetuned"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configure_and_audit_hipad_assets(cfg, project_root: os.PathLike) -> Dict[str, str]:
    """Normalize anchor assets to clean root and disable redundant backbone init.

    The wrapper always loads a full HiP-AD checkpoint immediately after model
    construction, so a separate ImageNet backbone initialization is both
    redundant and a provenance hazard when the config contains a legacy link.
    """

    root = _resolved(project_root)
    provenance: Dict[str, str] = {}

    def normalize(value, key_path: str):
        if isinstance(value, dict):
            for key in list(value.keys()):
                value[key] = normalize(value[key], f"{key_path}.{key}" if key_path else str(key))
            return value
        if isinstance(value, list):
            for index in range(len(value)):
                value[index] = normalize(value[index], f"{key_path}[{index}]")
            return value
        if isinstance(value, tuple):
            return tuple(normalize(item, f"{key_path}[{index}]") for index, item in enumerate(value))
        if isinstance(value, str) and value.endswith(".npy"):
            candidate = root / "data" / "kmeans" / Path(value).name
            asset = validate_runtime_asset(candidate, label=f"HiP-AD anchor {key_path}", reject_symlink=True)
            if not _is_relative_to(asset, root):
                raise HiPADProjectIsolationError(f"HiP-AD anchor escaped clean root: {asset}")
            provenance[f"asset.{key_path}.path"] = str(asset)
            provenance[f"asset.{key_path}.sha256"] = _sha256(asset)
            return str(asset)
        return value

    cfg.model = normalize(cfg.model, "model")
    img_backbone = getattr(cfg.model, "img_backbone", None)
    if img_backbone is not None and getattr(img_backbone, "pretrained", None):
        provenance["asset.model.img_backbone.pretrained_config_value"] = str(img_backbone.pretrained)
        img_backbone.pretrained = None
    provenance["asset.model.img_backbone.pretrained_runtime"] = "disabled_full_checkpoint_load"
    return provenance
