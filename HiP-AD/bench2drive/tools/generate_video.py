#!/usr/bin/env python3

import argparse
import cv2
from pathlib import Path

from tqdm import tqdm


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def _image_files(directory: Path):
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _discover_image_directories(input_root: Path):
    direct_images = _image_files(input_root)
    if direct_images:
        return [(input_root.name, input_root, direct_images)]

    nested_images = input_root / "images"
    if nested_images.is_dir():
        images = _image_files(nested_images)
        if images:
            return [(input_root.name, nested_images, images)]

    discovered = []
    for child in sorted(path for path in input_root.iterdir() if path.is_dir()):
        image_directory = child / "images" if (child / "images").is_dir() else child
        images = _image_files(image_directory)
        if images:
            discovered.append((child.name, image_directory, images))
    return discovered


def create_video(images, output_video: Path, fps: float) -> int:
    first_frame = cv2.imread(str(images[0]))
    if first_frame is None:
        raise RuntimeError(f"failed to read first frame: {images[0]}")
    height, width = first_frame.shape[:2]

    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"failed to create video: {output_video}")

    try:
        for image_path in tqdm(images, desc=output_video.stem):
            frame = cv2.imread(str(image_path))
            if frame is None:
                raise RuntimeError(f"failed to read frame: {image_path}")
            if frame.shape[:2] != (height, width):
                raise RuntimeError(
                    f"frame size mismatch for {image_path}: "
                    f"expected {(width, height)}, got {(frame.shape[1], frame.shape[0])}"
                )
            writer.write(frame)
    finally:
        writer.release()
    return len(images)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create MP4 videos from CARLA evaluation image folders.")
    parser.add_argument("-f", "--input-root", required=True, help="Image folder or evaluation root.")
    parser.add_argument("-o", "--output-dir", help="Output directory; defaults to the input root.")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_root = Path(args.input_root).expanduser().resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(f"input root is not a directory: {input_root}")
    if args.fps <= 0:
        raise ValueError("--fps must be positive")

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else input_root
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    sources = _discover_image_directories(input_root)
    if not sources:
        raise RuntimeError(f"no JPG or PNG frames found beneath: {input_root}")

    outputs = [(name, images, output_dir / f"{name}.mp4") for name, _, images in sources]
    existing = [output for _, _, output in outputs if output.exists()]
    if existing and not args.overwrite:
        paths = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"output already exists; pass --overwrite to replace it: {paths}")

    for name, images, output_video in outputs:
        frame_count = create_video(images, output_video, args.fps)
        print(f"created {output_video} from {frame_count} frames ({name})", flush=True)


if __name__ == "__main__":
    main()
