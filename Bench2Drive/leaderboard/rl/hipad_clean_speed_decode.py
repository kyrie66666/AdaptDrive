"""Mode-aligned frozen HiP-AD speed decoding for Line-C SAC.

The stock HiP-AD decoder first chooses one greedy lateral mode and only then
decodes/rescores its speed-area branches.  A sampled discrete SAC action may
choose another lateral mode.  This helper keeps the clean model untouched and
applies the same frozen speed classification and collision-rescore semantics to
every lateral mode from the already-computed model outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch


@dataclass(frozen=True)
class ModeAlignedSpeedDecode:
    """Frozen speed-decoder outputs aligned with all lateral modes."""

    trajectories: torch.Tensor  # [B, M, T, 2]
    raw_speed_area_indices: torch.Tensor  # [B, M], before collision rescore
    speed_area_indices: torch.Tensor  # [B, M]
    rescore_changed: torch.Tensor  # [B, M], bool
    suppressed_area_count: torch.Tensor  # [B, M]
    all_collision: torch.Tensor  # [B, M], bool
    rescored_logits: torch.Tensor  # [B, M, K]
    speed_areas: Tuple[Tuple[float, float], ...]


def _command_indices(inputs: Dict, batch_size: int, command_count: int, device: torch.device) -> torch.Tensor:
    if command_count <= 1:
        return torch.zeros(batch_size, dtype=torch.long, device=device)
    command = inputs["gt_ego_fut_cmd"].argmax(dim=-1).long().to(device)
    if command.numel() != batch_size:
        raise RuntimeError(
            f"clean speed decode command batch has {command.numel()} entries, expected {batch_size}"
        )
    return command.reshape(batch_size)


def _repeat_for_modes(tensor: torch.Tensor, mode_count: int) -> torch.Tensor:
    """Repeat each batch row for a chunk of lateral modes."""

    batch_size = tensor.shape[0]
    return tensor.unsqueeze(1).expand(batch_size, mode_count, *tensor.shape[1:]).reshape(
        batch_size * mode_count,
        *tensor.shape[1:],
    )


@torch.no_grad()
def decode_mode_aligned_clean_speed(
    plan_decoder,
    inputs: Dict,
    model_outs,
    num_lateral_modes: int,
    fut_ts: int,
    output_frequency: str = "5hz",
    rescore_chunk_size: int = 8,
) -> ModeAlignedSpeedDecode:
    """Decode one frozen clean longitudinal trajectory per lateral mode.

    Collision rescoring is evaluated on the same lateral-mode context as the
    returned trajectory.  Lateral modes are folded into a temporary batch and
    processed in bounded chunks through the unmodified clean decoder's
    ``rescore`` implementation.
    """

    det_output, _, _, plan_output, motion_output, _ = model_outs
    prediction = plan_output["prediction"][-1].float()
    classification = plan_output["classification"][-1].float()
    anchor_types = [tuple(anchor) for anchor in plan_decoder.anchor_types]
    group_count = len(anchor_types)
    if prediction.shape[2] % group_count != 0 or classification.shape[2] % group_count != 0:
        raise RuntimeError("clean planning output cannot be divided into configured anchor groups")

    reg_groups = list(prediction.chunk(chunks=group_count, dim=2))
    cls_groups = list(classification.chunk(chunks=group_count, dim=2))
    batch_size = prediction.shape[0]
    command_count = int(plan_decoder.ego_fut_cmd)
    command = _command_indices(inputs, batch_size, command_count, prediction.device)
    batch_indices = torch.arange(batch_size, device=prediction.device)

    speed_cls: Dict[str, List[torch.Tensor]] = {}
    speed_reg: Dict[str, List[torch.Tensor]] = {}
    speed_areas: Dict[str, List[Tuple[float, float]]] = {}
    for group_index, anchor_type in enumerate(anchor_types):
        if not anchor_type or anchor_type[0] != "speed":
            continue
        frequency = str(anchor_type[1])
        area = tuple(float(value) for value in anchor_type[2])
        cls = cls_groups[group_index].reshape(batch_size, command_count, -1)
        reg = reg_groups[group_index].reshape(batch_size, command_count, -1, fut_ts, 2).cumsum(dim=-2)
        cls = cls[batch_indices, command]
        reg = reg[batch_indices, command]
        if cls.shape[1] < num_lateral_modes or reg.shape[1] < num_lateral_modes:
            raise RuntimeError(
                f"clean speed group {anchor_type} exposes {cls.shape[1]} modes, expected {num_lateral_modes}"
            )
        speed_cls.setdefault(frequency, []).append(cls[:, :num_lateral_modes])
        speed_reg.setdefault(frequency, []).append(reg[:, :num_lateral_modes])
        speed_areas.setdefault(frequency, []).append(area)

    if output_frequency not in speed_cls:
        raise RuntimeError(f"clean model exposes no speed/{output_frequency} planning groups")
    reference_frequency = str(plan_decoder.speed_refer[1])
    if reference_frequency not in speed_cls:
        raise RuntimeError(f"clean speed reference {reference_frequency!r} is unavailable")
    if speed_areas[reference_frequency] != speed_areas[output_frequency]:
        raise RuntimeError("clean speed-area ordering differs across planning frequencies")

    cls_by_frequency = {
        frequency: torch.stack(groups, dim=2) for frequency, groups in speed_cls.items()
    }  # [B, M, K]
    reg_by_frequency = {
        frequency: torch.stack(groups, dim=2) for frequency, groups in speed_reg.items()
    }  # [B, M, K, T, 2]
    reference_logits = cls_by_frequency[reference_frequency]
    reference_reg = reg_by_frequency[reference_frequency]
    speed_area_count = reference_logits.shape[2]
    raw_speed_area_indices = reference_logits.argmax(dim=-1)

    rescored_logits = reference_logits.clone()
    all_collision = torch.zeros(
        batch_size,
        num_lateral_modes,
        dtype=torch.bool,
        device=prediction.device,
    )
    can_rescore = bool(
        getattr(plan_decoder, "with_rescore", False)
        and len(det_output.get("prediction", [])) > 0
        and len(motion_output.get("prediction", [])) > 0
    )
    if can_rescore:
        det_classification = det_output["classification"][-1].sigmoid()
        det_anchors = det_output["prediction"][-1]
        det_confidence = det_classification.max(dim=-1).values
        motion_cls = motion_output["classification"][-1].sigmoid()
        motion_reg = motion_output["prediction"][-1].cumsum(-2)

        if reference_frequency == "5hz":
            rescore_plan_reg = reference_reg[..., [2, 5], :]
            rescore_motion_reg = motion_reg[..., [0, 1], :]
        elif reference_frequency == "2hz":
            rescore_plan_reg = reference_reg
            rescore_motion_reg = motion_reg
        else:
            raise RuntimeError(f"unsupported clean speed reference frequency: {reference_frequency}")

        chunk_size = max(1, int(rescore_chunk_size))
        for start in range(0, num_lateral_modes, chunk_size):
            end = min(num_lateral_modes, start + chunk_size)
            mode_count = end - start
            chunk_logits = reference_logits[:, start:end].reshape(batch_size * mode_count, speed_area_count)
            chunk_reg = rescore_plan_reg[:, start:end].reshape(
                batch_size * mode_count,
                speed_area_count,
                rescore_plan_reg.shape[-2],
                2,
            )
            chunk_logits, chunk_all_collision = plan_decoder.rescore(
                chunk_logits,
                chunk_reg,
                _repeat_for_modes(motion_cls, mode_count),
                _repeat_for_modes(rescore_motion_reg, mode_count),
                _repeat_for_modes(det_anchors, mode_count),
                _repeat_for_modes(det_confidence, mode_count),
                ego_fut_ts=chunk_reg.shape[2],
                ego_fut_mode=speed_area_count,
            )
            rescored_logits[:, start:end] = chunk_logits.reshape(batch_size, mode_count, speed_area_count)
            all_collision[:, start:end] = chunk_all_collision.reshape(batch_size, mode_count)

    speed_area_indices = rescored_logits.argmax(dim=-1)
    rescore_changed = speed_area_indices != raw_speed_area_indices
    suppressed_area_count = (rescored_logits < reference_logits - 1e-6).sum(dim=-1)
    output_reg = reg_by_frequency[output_frequency]
    gather_index = speed_area_indices[..., None, None, None].expand(
        batch_size,
        num_lateral_modes,
        1,
        fut_ts,
        2,
    )
    trajectories = torch.gather(output_reg, 2, gather_index).squeeze(2)
    trajectories = trajectories * (~all_collision).to(trajectories.dtype)[..., None, None]

    if tuple(trajectories.shape) != (batch_size, num_lateral_modes, fut_ts, 2):
        raise RuntimeError(f"mode-aligned clean speed has unexpected shape {tuple(trajectories.shape)}")
    if not torch.isfinite(trajectories).all() or not torch.isfinite(rescored_logits).all():
        raise RuntimeError("mode-aligned clean speed decode contains non-finite values")
    if torch.any(speed_area_indices < 0) or torch.any(speed_area_indices >= speed_area_count):
        raise RuntimeError("mode-aligned clean speed-area index is out of range")

    return ModeAlignedSpeedDecode(
        trajectories=trajectories.detach(),
        raw_speed_area_indices=raw_speed_area_indices.detach(),
        speed_area_indices=speed_area_indices.detach(),
        rescore_changed=rescore_changed.detach(),
        suppressed_area_count=suppressed_area_count.detach(),
        all_collision=all_collision.detach(),
        rescored_logits=rescored_logits.detach(),
        speed_areas=tuple(speed_areas[output_frequency]),
    )
