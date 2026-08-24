#!/usr/bin/env python3
"""HiP-AD leaderboard agent with the trained four-level DCNv4 adapter.

The original clean ``SparseAgent`` is intentionally not modified.  This agent
subclasses it and only adds:

* strict SAC checkpoint validation/restoration;
* a 21-D ego-state bridge built from the clean sensor tick;
* an instance-local ``extract_feat`` hook that applies the adapter before the
  unchanged clean planning head and dual PID.

The module is loaded directly by the clean leaderboard evaluator via
``TEAM_AGENT``.  It therefore keeps the original clean planner, sensor list,
temporal model state, route handling, and PID path intact.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch


# The leaderboard evaluator inserts the directory containing this file into
# sys.path.  Import the sibling strict loader first, then fall back to the
# normal ``rl`` namespace used by offline smoke tests.
try:
    from hipad_clean_adapter_checkpoint import (
        AdapterCheckpointBundle,
        build_feature_adapter_from_bundle,
        load_adapter_checkpoint,
        restore_hipad_trainable,
    )
except ImportError:
    from rl.hipad_clean_adapter_checkpoint import (
        AdapterCheckpointBundle,
        build_feature_adapter_from_bundle,
        load_adapter_checkpoint,
        restore_hipad_trainable,
    )


try:
    from team_code.hipad_b2d_agent import SparseAgent as CleanSparseAgent
except ImportError as exc:  # pragma: no cover - only reached with a bad clean PYTHONPATH
    raise ImportError(
        "HiP-AD team_code import failed; activate the clean project and "
        f"its CARLA/scenario_runner PYTHONPATH before loading this agent: {exc}"
    ) from exc


def _clean_project_root() -> Path:
    """Resolve the clean root without introducing a machine-specific path."""

    value = os.environ.get("HIPAD_ROOT", "")
    if value:
        return Path(value).expanduser().resolve()
    # This file lives at PROJECT_ROOT/Bench2Drive/leaderboard/rl/.
    return Path(__file__).resolve().parents[3] / "HiP-AD"


def _as_finite_float_array(value, shape, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    expected = int(np.prod(shape))
    if array.size != expected:
        raise RuntimeError(f"{label} must contain {expected} values, got {array.size}")
    array = array.reshape(shape)
    if not np.isfinite(array).all():
        raise RuntimeError(f"{label} contains non-finite values")
    return array


class HiPADCleanDCNv4AdapterAgent(CleanSparseAgent):
    """Adapter-aware variant of the unchanged clean ``SparseAgent``."""

    def __init__(self, host, port, debug=False):
        super().__init__(host, port, debug)
        self._adapter_bundle: Optional[AdapterCheckpointBundle] = None
        self._feature_dcnv4_adapter = None
        self._original_extract_feat = None
        self._adapter_ego_state: Optional[torch.Tensor] = None
        self._adapter_forward_count = 0
        self._adapter_first_metrics = None

    def setup(self, path_to_conf_file):
        parts = str(path_to_conf_file).split("+")
        if len(parts) < 3:
            raise RuntimeError(
                "adapter-aware TEAM_CONFIG must be CONFIG+BASE_PTH+FINETUNE_PT; "
                f"got {path_to_conf_file!r}"
            )

        config_path, base_checkpoint, finetune_checkpoint = parts[:3]
        save_name = parts[-1]
        # The evaluator appends its own save name.  Keep the base agent's
        # three-field parser and output layout unchanged.
        super().setup(f"{config_path}+{base_checkpoint}+{save_name}")

        clean_root = _clean_project_root()
        self._adapter_bundle = load_adapter_checkpoint(
            finetune_checkpoint,
            base_checkpoint_path=base_checkpoint,
            expected_project_root=str(clean_root),
        )

        expected_names = tuple(
            name
            for name, _parameter in self.model.named_parameters()
            if name.startswith("head.onedecoder_head.plan_refine.5.plan_cls_branch.")
            or name.startswith("head.onedecoder_head.plan_refine.5.plan_reg_branch_spat_2m.")
        )
        if len(expected_names) != 25:
            raise RuntimeError(
                "clean model trainable planning branch changed: "
                f"found {len(expected_names)} parameters, expected 25"
            )
        loaded_count = restore_hipad_trainable(
            self.model,
            self._adapter_bundle.hipad_trainable,
            expected_names=expected_names,
        )
        if loaded_count != 25:
            raise RuntimeError(f"expected 25 HiP-AD trainable tensors, loaded {loaded_count}")

        model_parameter = next(self.model.parameters())
        device = model_parameter.device
        self._feature_dcnv4_adapter = build_feature_adapter_from_bundle(self._adapter_bundle, device)
        self._install_feature_hook()

        print(
            "[HiPADCleanDCNv4AdapterAgent] strict deployment restore: "
            f"finetune={self._adapter_bundle.checkpoint_path} "
            f"finetune_sha256={self._adapter_bundle.checkpoint_sha256} "
            f"hipad_trainable={loaded_count} "
            f"feature_dcnv4_adapter={len(self._adapter_bundle.feature_adapter_state)} "
            f"levels={self._adapter_bundle.feature_adapter_levels} "
            f"adapter_prediction_present={int(self._adapter_bundle.adapter_prediction_present)}",
            flush=True,
        )

    def _install_feature_hook(self) -> None:
        if self._feature_dcnv4_adapter is None:
            raise RuntimeError("feature adapter was not constructed")
        if self._original_extract_feat is not None:
            raise RuntimeError("feature adapter hook was installed more than once")

        try:
            from projects.mmdet3d_plugin.ops import feature_maps_format
        except ImportError as exc:
            raise RuntimeError("clean feature_maps_format is unavailable") from exc

        original_extract_feat = self.model.extract_feat
        self._original_extract_feat = original_extract_feat

        def extract_feat_with_adapter(img, return_depth=False, metas=None):
            result = original_extract_feat(img, return_depth=return_depth, metas=metas)
            if return_depth:
                feature_maps, depths = result
            else:
                feature_maps, depths = result, None

            ego_state = self._adapter_ego_state
            if ego_state is None:
                raise RuntimeError("adapter feature hook was called before the sensor ego state was built")
            inverse_feature_maps = feature_maps_format(feature_maps, inverse=True)
            adapted_feature_maps, metrics = self._feature_dcnv4_adapter(
                inverse_feature_maps,
                ego_state.to(device=feature_maps[0].device, dtype=torch.float32),
                return_metrics=True,
            )
            self._adapter_forward_count += 1
            if self._adapter_first_metrics is None:
                self._adapter_first_metrics = {
                    key: float(value.detach().cpu().item())
                    if torch.is_tensor(value) and value.numel() == 1
                    else str(type(value).__name__)
                    for key, value in metrics.items()
                }
                print(
                    "[HiPADCleanDCNv4AdapterAgent] first adapter forward: "
                    f"metrics={self._adapter_first_metrics}",
                    flush=True,
                )
            formatted = feature_maps_format(adapted_feature_maps)
            return (formatted, depths) if return_depth else formatted

        # ``extract_feat`` is assigned to the model instance, so this closure
        # receives img directly (there is no extra model self argument).  The
        # original bound method remains captured above and keeps all clean
        # backbone/neck/temporal behavior unchanged.
        self.model.extract_feat = extract_feat_with_adapter

    def _build_adapter_ego_state(self, tick_data) -> np.ndarray:
        """Build the same 21-D state layout used by the SAC ObservationBuilder."""

        speed = float(tick_data["speed"])
        acceleration = _as_finite_float_array(tick_data["acceleration"], (3,), "acceleration")
        angular_velocity = _as_finite_float_array(
            tick_data["angular_velocity"],
            (3,),
            "angular_velocity",
        )
        pos = _as_finite_float_array(tick_data["pos"], (2,), "position")
        compass = float(tick_data["compass"])
        if not np.isfinite(speed) or not np.isfinite(compass):
            raise RuntimeError("speed/compass contains a non-finite value")

        previous_control = getattr(self, "prev_control", None)
        if previous_control is None:
            steer = throttle = brake = 0.0
        else:
            steer = float(previous_control.steer)
            throttle = float(previous_control.throttle)
            brake = float(previous_control.brake)

        # CARLA compass semantics in the clean agent are compass = pi/2 - yaw.
        # The SAC ObservationBuilder stores world-frame velocity and yaw, so we
        # reconstruct the forward velocity from speed and the same yaw.  This
        # keeps the deployment state deterministic and records the only field
        # not directly exposed by the leaderboard sensor packet.
        world_yaw = math.pi / 2.0 - compass
        velocity = np.array(
            [speed * math.cos(world_yaw), speed * math.sin(world_yaw), 0.0],
            dtype=np.float32,
        )
        state = np.array(
            [
                speed,
                steer,
                throttle,
                brake,
                velocity[0],
                velocity[1],
                velocity[2],
                angular_velocity[0],
                angular_velocity[1],
                angular_velocity[2],
                acceleration[0],
                acceleration[1],
                acceleration[2],
                pos[0],
                pos[1],
                0.0,
                0.0,
                0.0,
                math.degrees(world_yaw),
            ],
            dtype=np.float32,
        )
        if state.size < 21:
            state = np.pad(state, (0, 21 - state.size), mode="constant")
        state = state[:21]
        if not np.isfinite(state).all():
            raise RuntimeError("adapter ego state contains non-finite values")
        return state

    def tick(self, input_data):
        tick_data = super().tick(input_data)
        state = self._build_adapter_ego_state(tick_data)
        model_parameter = next(self.model.parameters())
        self._adapter_ego_state = torch.from_numpy(state).unsqueeze(0).to(
            device=model_parameter.device,
            dtype=torch.float32,
        )
        return tick_data

    def destroy(self):
        self._feature_dcnv4_adapter = None
        self._adapter_ego_state = None
        self._original_extract_feat = None
        super().destroy()


def get_entry_point():
    return "HiPADCleanDCNv4AdapterAgent"
