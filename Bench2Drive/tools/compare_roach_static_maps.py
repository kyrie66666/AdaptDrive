#!/usr/bin/env python3
"""Compare Roach static maps after aligning them in world coordinates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from roach_map.asset_schema import REQUIRED_LAYERS  # noqa: E402


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", help="Reference Roach .h5, for example the original Town01 asset")
    parser.add_argument("candidate", help="Newly generated .h5 asset")
    parser.add_argument(
        "--max-overlap-pixels",
        type=int,
        default=100_000_000,
        help="Refuse an in-memory comparison above this overlap area",
    )
    parser.add_argument(
        "--offset-tolerance-pixels",
        type=float,
        default=1e-3,
        help="Maximum fractional-pixel world-offset mismatch",
    )
    return parser.parse_args(argv)


def _map_info(handle) -> Dict[str, object]:
    ppm = float(handle.attrs["pixels_per_meter"])
    offset = np.asarray(handle.attrs["world_offset_in_meters"], dtype=np.float64)
    if offset.shape != (2,):
        raise ValueError(f"world_offset_in_meters must have shape (2,), got {offset}")
    road_shape = tuple(int(value) for value in handle["road"].shape)
    if len(road_shape) != 2 or road_shape[0] != road_shape[1]:
        raise ValueError(f"road dataset must be square 2D, got {road_shape}")
    return {
        "pixels_per_meter": ppm,
        "offset": offset,
        "shape": road_shape,
        "town_name": str(handle.attrs.get("town_name", "")),
        "carla_server_version": str(handle.attrs.get("carla_server_version", "")),
        "opendrive_sha256": str(handle.attrs.get("opendrive_sha256", "")),
    }


def _aligned_overlap(
    reference_info: Dict[str, object],
    candidate_info: Dict[str, object],
    tolerance_pixels: float,
) -> Tuple[Tuple[slice, slice], Tuple[slice, slice], Dict[str, object]]:
    ref_ppm = float(reference_info["pixels_per_meter"])
    cand_ppm = float(candidate_info["pixels_per_meter"])
    if not np.isclose(ref_ppm, cand_ppm, rtol=0.0, atol=1e-9):
        raise ValueError(f"pixels_per_meter mismatch: reference={ref_ppm}, candidate={cand_ppm}")
    ppm = ref_ppm
    ref_offset = np.asarray(reference_info["offset"], dtype=np.float64)
    cand_offset = np.asarray(candidate_info["offset"], dtype=np.float64)
    ref_width = int(reference_info["shape"][0])
    cand_width = int(candidate_info["shape"][0])

    # Asset axis 0 is world y (row), axis 1 is world x (column).
    overlap_min = np.maximum(ref_offset, cand_offset)
    overlap_max = np.minimum(
        ref_offset + ref_width / ppm,
        cand_offset + cand_width / ppm,
    )
    if np.any(overlap_max <= overlap_min):
        raise ValueError(
            f"assets do not overlap in world coordinates: min={overlap_min}, max={overlap_max}"
        )

    def pixel_bounds(offset: np.ndarray, width: int) -> Tuple[int, int, int, int]:
        start = (overlap_min - offset) * ppm
        end = (overlap_max - offset) * ppm
        rounded_start = np.rint(start)
        rounded_end = np.rint(end)
        fractional_error = max(
            float(np.max(np.abs(start - rounded_start))),
            float(np.max(np.abs(end - rounded_end))),
        )
        if fractional_error > float(tolerance_pixels):
            raise ValueError(
                "assets are not aligned to the same pixel grid: "
                f"fractional_error={fractional_error:.6f} pixels"
            )
        x0, y0 = (int(value) for value in rounded_start)
        x1, y1 = (int(value) for value in rounded_end)
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(width, x1), min(width, y1)
        return x0, x1, y0, y1

    ref_x0, ref_x1, ref_y0, ref_y1 = pixel_bounds(ref_offset, ref_width)
    cand_x0, cand_x1, cand_y0, cand_y1 = pixel_bounds(cand_offset, cand_width)
    overlap_width = min(ref_x1 - ref_x0, cand_x1 - cand_x0)
    overlap_height = min(ref_y1 - ref_y0, cand_y1 - cand_y0)
    if overlap_width <= 0 or overlap_height <= 0:
        raise ValueError("aligned pixel overlap is empty")

    ref_selection = (slice(ref_y0, ref_y0 + overlap_height), slice(ref_x0, ref_x0 + overlap_width))
    cand_selection = (
        slice(cand_y0, cand_y0 + overlap_height),
        slice(cand_x0, cand_x0 + overlap_width),
    )
    report = {
        "world_overlap_min_xy": overlap_min.tolist(),
        "world_overlap_max_xy": overlap_max.tolist(),
        "overlap_shape": [overlap_height, overlap_width],
        "reference_selection": [
            [ref_selection[0].start, ref_selection[0].stop],
            [ref_selection[1].start, ref_selection[1].stop],
        ],
        "candidate_selection": [
            [cand_selection[0].start, cand_selection[0].stop],
            [cand_selection[1].start, cand_selection[1].stop],
        ],
    }
    return ref_selection, cand_selection, report


def _layer_metrics(reference: np.ndarray, candidate: np.ndarray) -> Dict[str, object]:
    if reference.shape != candidate.shape:
        raise ValueError(f"aligned layer shape mismatch: {reference.shape} vs {candidate.shape}")
    ref_binary = reference > 0
    cand_binary = candidate > 0
    intersection = int(np.count_nonzero(ref_binary & cand_binary))
    union = int(np.count_nonzero(ref_binary | cand_binary))
    ref_nonzero = int(np.count_nonzero(ref_binary))
    cand_nonzero = int(np.count_nonzero(cand_binary))
    exact_equal = int(np.count_nonzero(reference == candidate))
    total = int(reference.size)
    return {
        "iou_nonzero": float(intersection / union) if union else 1.0,
        "intersection_pixels": intersection,
        "union_pixels": union,
        "reference_nonzero_pixels": ref_nonzero,
        "candidate_nonzero_pixels": cand_nonzero,
        "binary_mismatch_pixels": int(np.count_nonzero(ref_binary != cand_binary)),
        "exact_mismatch_pixels": total - exact_equal,
        "exact_match_ratio": float(exact_equal / total) if total else 1.0,
        "reference_unique_values": [int(value) for value in np.unique(reference)],
        "candidate_unique_values": [int(value) for value in np.unique(candidate)],
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("h5py is required to compare Roach static-map assets") from exc

    reference_path = Path(args.reference).expanduser().resolve()
    candidate_path = Path(args.candidate).expanduser().resolve()
    with h5py.File(str(reference_path), "r") as reference, h5py.File(
        str(candidate_path), "r"
    ) as candidate:
        reference_info = _map_info(reference)
        candidate_info = _map_info(candidate)
        ref_selection, cand_selection, overlap_report = _aligned_overlap(
            reference_info,
            candidate_info,
            args.offset_tolerance_pixels,
        )
        overlap_pixels = int(np.prod(overlap_report["overlap_shape"]))
        if overlap_pixels > int(args.max_overlap_pixels):
            raise MemoryError(
                f"Aligned overlap has {overlap_pixels} pixels, above --max-overlap-pixels "
                f"{args.max_overlap_pixels}; this comparator is intended for small-Town parity."
            )

        layer_reports = {}
        for layer in REQUIRED_LAYERS:
            if layer not in reference or layer not in candidate:
                raise KeyError(f"Both assets must contain layer {layer!r}")
            ref_array = np.asarray(reference[layer][ref_selection], dtype=np.uint8)
            cand_array = np.asarray(candidate[layer][cand_selection], dtype=np.uint8)
            layer_reports[layer] = _layer_metrics(ref_array, cand_array)

    output = {
        "reference_path": str(reference_path),
        "candidate_path": str(candidate_path),
        "reference": {
            **reference_info,
            "offset": np.asarray(reference_info["offset"]).tolist(),
        },
        "candidate": {
            **candidate_info,
            "offset": np.asarray(candidate_info["offset"]).tolist(),
        },
        "alignment": overlap_report,
        "layers": layer_reports,
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
