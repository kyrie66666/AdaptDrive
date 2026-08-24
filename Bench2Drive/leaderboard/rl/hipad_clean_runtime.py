"""HiP-AD model runtime used by AdaptDrive rollouts."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import importlib
import os
import sys
import math

import cv2
import numpy as np
import torch
from scipy.optimize import fsolve

from rl.adaptdrive_calibration import (
    CAMERA_NAMES,
    LIDAR2EGO,
    get_lidar2cam_matrix,
    get_lidar2img_matrix,
)
from leaderboard.utils.route_manipulation import downsample_route


@dataclass
class HiPADRuntimePrediction:
    valid: bool
    plan_temp: Optional[np.ndarray]
    plan_spat: Optional[np.ndarray]
    error: str = ''


class HiPADCleanRuntime:
    """Lazy, failure-tolerant runtime around the clean HiP-AD planning model."""

    def __init__(
        self,
        project_root: str,
        config_path: str,
        checkpoint_path: str,
        device: Optional[torch.device] = None,
        enabled: bool = True,
        require_clean_tree: bool = False,
    ):
        self.project_root = Path(project_root)
        self.config_path = Path(config_path)
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.enabled = bool(enabled and self.project_root.exists() and self.config_path.exists() and self.checkpoint_path.exists())
        self.require_clean_tree = bool(require_clean_tree)
        self._initialized = False
        self._init_error = ''
        self._model = None
        self._pipeline = None
        self._data_aug_conf = None
        self._DataContainer = None
        self._mm_collate = None
        self._use_bgr_img = True
        self.runtime_asset_provenance = {}
        self._global_plan = None
        self._global_plan_world_coord = None
        self._route_planner = None
        self._RoutePlanner = None
        self._step = -1
        self._frame_rate = 20.0
        self.lat_ref = 42.0
        self.lon_ref = 2.0

    def _get_augmentation(self):
        if self._data_aug_conf is None:
            return None
        H, W = self._data_aug_conf["H"], self._data_aug_conf["W"]
        fH, fW = self._data_aug_conf["final_dim"]
        resize = max(fH / H, fW / W)
        resize_dims = (int(W * resize), int(H * resize))
        newW, newH = resize_dims
        crop_h = (int((1 - np.mean(self._data_aug_conf["bot_pct_lim"])) * newH) - fH)
        crop_w = int(max(0, newW - fW) / 2)
        crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
        return {
            "resize": resize,
            "resize_dims": resize_dims,
            "crop": crop,
            "flip": False,
            "rotate": 0,
            "rotate_3d": 0,
        }

    def _lazy_init(self):
        if self._initialized or not self.enabled:
            return

        try:
            from rl.hipad_project_runtime import activate_hipad_project_root

            activate_hipad_project_root(
                self.project_root,
                repo_root=self.project_root.parent,
                require_clean_tree=self.require_clean_tree,
            )

            from mmcv import Config
            from mmcv.parallel import DataContainer
            from mmcv.parallel.collate import collate as mm_collate_to_batch_form
            from mmcv.runner import load_checkpoint, wrap_fp16_model
            from mmdet.models import build_detector
            from mmdet.datasets.pipelines import Compose
            from bench2drive.leaderboard.team_code.planner import RoutePlanner

            cfg = Config.fromfile(str(self.config_path))
            from rl.hipad_project_runtime import configure_and_audit_hipad_assets

            self.runtime_asset_provenance = configure_and_audit_hipad_assets(cfg, self.project_root)
            if hasattr(cfg, 'model') and hasattr(cfg.model, 'use_grid_mask'):
                cfg.model.use_grid_mask = False
            if hasattr(cfg.model, 'head') and hasattr(cfg.model.head, 'onedecoder_head'):
                cfg.model.head.onedecoder_head.with_close_loop = True
            self._data_aug_conf = getattr(cfg, 'data_aug_conf', None)
            if hasattr(cfg.model.head, 'evaluate_bench2dive'):
                cfg.model.head.evaluate_bench2dive = False

            if getattr(cfg, 'plugin', False) and getattr(cfg, 'plugin_dir', None):
                plugin_dir = cfg.plugin_dir
                module_dir = os.path.dirname(plugin_dir).split("/")
                module_path = module_dir[0]
                for part in module_dir[1:]:
                    module_path = module_path + "." + part
                importlib.import_module(module_path)

            model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
            fp16_cfg = cfg.get("fp16", None)
            if fp16_cfg is not None and self.device.type == 'cuda':
                wrap_fp16_model(model)
            checkpoint = load_checkpoint(model, str(self.checkpoint_path), map_location='cpu')
            checkpoint_state = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
            if not isinstance(checkpoint_state, dict):
                raise RuntimeError('HiP-AD checkpoint does not contain a model state dict')
            normalized_state = {
                (name[7:] if name.startswith('module.') else name): value
                for name, value in checkpoint_state.items()
            }
            model_parameters = dict(model.named_parameters())
            missing_parameters = [name for name in model_parameters if name not in normalized_state]
            mismatched_parameters = [
                name for name, parameter in model_parameters.items()
                if name in normalized_state and tuple(parameter.shape) != tuple(normalized_state[name].shape)
            ]
            if missing_parameters or mismatched_parameters:
                raise RuntimeError(
                    'HiP-AD full checkpoint coverage failed: '
                    f'missing_parameters={missing_parameters[:20]} (total={len(missing_parameters)}), '
                    f'shape_mismatch={mismatched_parameters[:20]} (total={len(mismatched_parameters)})'
                )
            self.runtime_asset_provenance['checkpoint.parameter_coverage'] = (
                f'{len(model_parameters)}/{len(model_parameters)}'
            )
            model.to(self.device)
            model.eval()

            inference_only_pipeline = []
            for step in cfg.inference_only_pipeline:
                if step["type"] not in ['LoadMultiViewImageFromFilesInCeph', 'LoadMultiViewImageFromFiles']:
                    inference_only_pipeline.append(step)

            self._model = model
            self._pipeline = Compose(inference_only_pipeline)
            self._DataContainer = DataContainer
            self._mm_collate = mm_collate_to_batch_form
            self._RoutePlanner = RoutePlanner
            self._initialized = True
        except Exception as exc:
            self._init_error = str(exc)
            self.enabled = False

    def set_global_plan(self, global_plan_gps, global_plan_world_coord) -> None:
        """Match the original agent route-plan interface for clean-model inference."""
        if global_plan_gps is None or global_plan_world_coord is None:
            self._global_plan = None
            self._global_plan_world_coord = None
            self._route_planner = None
            self._step = -1
            return

        ds_ids = downsample_route(global_plan_world_coord, 50)
        self._global_plan_world_coord = [
            (global_plan_world_coord[x][0], global_plan_world_coord[x][1]) for x in ds_ids
        ]
        self._global_plan = [global_plan_gps[x] for x in ds_ids]
        self._route_planner = None
        self._step = -1

    def _init_route_planner(self) -> None:
        if self._route_planner is not None or self._global_plan is None or self._global_plan_world_coord is None:
            return

        try:
            locx = self._global_plan_world_coord[0][0].location.x
            locy = self._global_plan_world_coord[0][0].location.y
            lon = self._global_plan[0][0]['lon']
            lat = self._global_plan[0][0]['lat']
            earth_radius = 6378137.0

            def equations(vars_):
                x, y = vars_
                eq1 = ((lon * math.cos(x * math.pi / 180) - (locx * x * 180) / (math.pi * earth_radius)) -
                       math.cos(x * math.pi / 180) * y)
                eq2 = (math.log(math.tan((lat + 90) * math.pi / 360)) * earth_radius
                       * math.cos(x * math.pi / 180) + locy - math.cos(x * math.pi / 180)
                       * earth_radius * math.log(math.tan((90 + x) * math.pi / 360)))
                return [eq1, eq2]

            solution = fsolve(equations, [0, 0])
            self.lat_ref, self.lon_ref = float(solution[0]), float(solution[1])
        except Exception:
            self.lat_ref, self.lon_ref = 42.0, 2.0

        self._route_planner = self._RoutePlanner(4.0, 50.0, lat_ref=self.lat_ref, lon_ref=self.lon_ref)
        self._route_planner.set_route(self._global_plan, True)

    def _build_can_bus(self, observation: Dict) -> np.ndarray:
        if 'can_bus' in observation:
            return np.asarray(observation['can_bus'], dtype=np.float32).copy()

        state = np.asarray(observation['state'], dtype=np.float32)
        can_bus = np.zeros(18, dtype=np.float32)
        if state.shape[0] >= 16:
            can_bus[0] = state[13]
            can_bus[1] = -state[14]
            can_bus[2] = state[15]
            can_bus[7] = state[0]
            if state.shape[0] >= 13:
                can_bus[10] = state[10]
                can_bus[11] = -state[11]
                can_bus[12] = state[12]
            if state.shape[0] >= 10:
                can_bus[13] = -state[7]
                can_bus[14] = -state[8]
                can_bus[15] = -state[9]
            yaw_rad = float(np.radians(state[18])) if state.shape[0] >= 19 else 0.0
            can_bus[3:7] = [np.cos(yaw_rad / 2.0), 0.0, 0.0, np.sin(yaw_rad / 2.0)]
            can_bus[16] = yaw_rad
            can_bus[17] = np.degrees(yaw_rad)
        return can_bus

    def _image_wh(self, rgb: np.ndarray) -> np.ndarray:
        if self._data_aug_conf is not None and "final_dim" in self._data_aug_conf:
            image_h, image_w = self._data_aug_conf["final_dim"]
        else:
            image_h, image_w = rgb.shape[1], rgb.shape[2]
        return np.array([[image_w, image_h] for _ in CAMERA_NAMES], dtype=np.float32)

    def _prepare_inputs(
        self,
        observation: Dict,
        advance_step: bool = True,
        use_route_planner: bool = True,
    ) -> Dict:
        if use_route_planner:
            self._init_route_planner()
        if advance_step:
            self._step += 1
        rgb = np.asarray(observation['rgb'])
        if rgb.dtype != np.uint8:
            rgb = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
        can_bus = self._build_can_bus(observation)
        gps = np.asarray(observation.get('gps', np.zeros(3, dtype=np.float32)), dtype=np.float32)
        raw_theta = float(observation.get('compass', 0.0))
        if math.isnan(raw_theta):
            raw_theta = 0.0
        target_point = np.asarray(observation.get('target_point', np.zeros(2, dtype=np.float32)), dtype=np.float32)
        target_point_next = np.asarray(observation.get('target_point_next', target_point), dtype=np.float32)
        command = int(observation.get('command', 3))
        if command < 0:
            command = 4
        command = max(1, min(6, int(command)))
        command_onehot = np.zeros(6, dtype=np.float32)
        command_onehot[command - 1] = 1.0

        if use_route_planner and self._route_planner is not None and gps.size >= 2:
            pos = self._route_planner.gps_to_location(np.asarray(gps[:2], dtype=np.float32))
            waypoint_routes = self._route_planner.run_step(pos)
            if len(waypoint_routes) >= 3:
                target_xy = waypoint_routes[1][0]
                target_xy_next = waypoint_routes[2][0]
                command = waypoint_routes[0][1]
            elif len(waypoint_routes) == 2:
                target_xy = waypoint_routes[1][0]
                target_xy_next = waypoint_routes[1][0]
                command = waypoint_routes[0][1]
            else:
                target_xy = waypoint_routes[0][0]
                target_xy_next = waypoint_routes[0][0]
                command = waypoint_routes[0][1]

            ego_pos = np.array([float(pos[0]), float(-pos[1])], dtype=np.float32)
            target_xy = np.array([target_xy[0] - ego_pos[0], -target_xy[1] - ego_pos[1]], dtype=np.float32)
            target_xy_next = np.array([target_xy_next[0] - ego_pos[0], -target_xy_next[1] - ego_pos[1]], dtype=np.float32)
            rotation_matrix = np.array(
                [[np.cos(raw_theta), -np.sin(raw_theta)], [np.sin(raw_theta), np.cos(raw_theta)]],
                dtype=np.float32,
            )
            target_point = np.array(rotation_matrix @ target_xy, dtype=np.float32)
            target_point_next = np.array(rotation_matrix @ target_xy_next, dtype=np.float32)
            if command < 0:
                command = 4
            command = max(1, min(6, int(command)))
            command_onehot = np.zeros(6, dtype=np.float32)
            command_onehot[command - 1] = 1.0
        else:
            ego_pos = np.array([can_bus[0], can_bus[1]], dtype=np.float32)

        yaw = float(can_bus[16]) if abs(float(can_bus[16])) > 1e-6 else 0.0
        ego2world = np.eye(4, dtype=np.float32)
        ego2world[0, 0] = np.cos(yaw)
        ego2world[0, 1] = -np.sin(yaw)
        ego2world[1, 0] = np.sin(yaw)
        ego2world[1, 1] = np.cos(yaw)
        ego2world[0, 3] = float(ego_pos[0])
        ego2world[1, 3] = float(ego_pos[1])
        lidar2global = ego2world @ LIDAR2EGO

        custom_status = np.zeros(6, dtype=np.float32)
        custom_status[0] = float(can_bus[7])
        custom_status[1] = float(can_bus[10])
        custom_status[2] = float(can_bus[11])
        custom_status[3] = float(can_bus[13])
        custom_status[4] = float(can_bus[14])
        state = np.asarray(observation.get('state', np.zeros((0,), dtype=np.float32)), dtype=np.float32).reshape(-1)
        custom_status[5] = float(state[1]) if state.size > 1 else 0.0

        inputs = {
            'img': [
                cv2.cvtColor(rgb[i], cv2.COLOR_RGB2BGR) if self._use_bgr_img else rgb[i]
                for i in range(len(CAMERA_NAMES))
            ],
            'folder': '',
            'scene_token': '',
            'lidar2img': np.stack([get_lidar2img_matrix(name) for name in CAMERA_NAMES], axis=0),
            'lidar2cam': np.stack([get_lidar2cam_matrix(name) for name in CAMERA_NAMES], axis=0),
            'frame_idx': 0,
            'timestamp': float(self._step / self._frame_rate),
            'aug_config': self._get_augmentation(),
            'custom_status': custom_status,
            'command': command - 1,
            'gt_ego_fut_cmd': command_onehot,
            'target_point': target_point.astype(np.float32),
            'target_point_next': target_point_next.astype(np.float32),
            'l2g_r_mat': lidar2global[0:3, 0:3],
            'l2g_t': lidar2global[0:3, 3],
            'lidar2global': lidar2global,
            'image_wh': self._image_wh(rgb),
        }
        return inputs

    def _batchify(self, inputs: Dict) -> Dict:
        inputs = self._pipeline(inputs)
        inputs = self._mm_collate([inputs], samples_per_gpu=1)
        for key, value in inputs.items():
            if isinstance(value, self._DataContainer):
                inputs[key] = value.data[0]
            elif isinstance(value[0], self._DataContainer):
                inputs[key] = value[0].data
            else:
                inputs[key] = value
            if isinstance(inputs[key], torch.Tensor):
                inputs[key] = inputs[key].to(self.device)
        return inputs

    @torch.no_grad()
    def predict(self, observation: Dict) -> HiPADRuntimePrediction:
        if not self.enabled:
            return HiPADRuntimePrediction(
                valid=False,
                plan_temp=None,
                plan_spat=None,
                error=self._init_error or 'runtime_disabled',
            )

        self._lazy_init()
        if not self._initialized:
            return HiPADRuntimePrediction(
                valid=False,
                plan_temp=None,
                plan_spat=None,
                error=self._init_error or 'runtime_init_failed',
            )

        try:
            inputs = self._batchify(self._prepare_inputs(observation))
            outputs = self._model(
                img=inputs['img'],
                img_metas=inputs['img_metas'],
                projection_mat=inputs['projection_mat'],
                gt_ego_fut_cmd=inputs['gt_ego_fut_cmd'],
                image_wh=inputs['image_wh'],
                timestamp=inputs['timestamp'],
                target_point=inputs['target_point'],
                rescale=True,
                return_loss=False,
            )
            img_bbox = outputs[0].get('img_bbox', {})
            plan_temp = img_bbox.get('plan_speed_5hz', img_bbox.get('plan_temp_5hz'))
            plan_spat = img_bbox.get('plan_spat_2m')
            if plan_temp is None or plan_spat is None:
                return HiPADRuntimePrediction(
                    valid=False,
                    plan_temp=None,
                    plan_spat=None,
                    error='missing_runtime_plans',
                )
            return HiPADRuntimePrediction(
                valid=True,
                plan_temp=np.asarray(plan_temp.detach().cpu().numpy(), dtype=np.float32),
                plan_spat=np.asarray(plan_spat.detach().cpu().numpy(), dtype=np.float32),
                error='',
            )
        except Exception as exc:
            return HiPADRuntimePrediction(valid=False, plan_temp=None, plan_spat=None, error=str(exc))

    @torch.no_grad()
    def prime(self, observation: Optional[Dict]) -> HiPADRuntimePrediction:
        """Warm the internal temporal caches on a hidden reset frame."""
        if observation is None:
            return HiPADRuntimePrediction(valid=False, plan_temp=None, plan_spat=None, error='no_observation')
        return self.predict(observation)
