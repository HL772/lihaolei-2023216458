#!/usr/bin/env python3
import argparse
import random
import shutil
import sys
from pathlib import Path
from typing import Optional

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def find_images(src_dir: Path) -> list[Path]:
    return sorted(
        [
            p
            for p in src_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        ]
    )


def list_dataset_images(dataset_dir: Path) -> list[Path]:
    images: list[Path] = []
    for split in ("train", "val"):
        img_dir = dataset_dir / "images" / split
        if not img_dir.exists():
            continue
        for img_path in img_dir.iterdir():
            if img_path.is_file() and img_path.suffix.lower() in IMAGE_EXTS:
                images.append(img_path)
    return images


def parse_index(stem: str) -> Optional[tuple[str, int, int]]:
    if "_" not in stem:
        return None
    prefix, num_str = stem.rsplit("_", 1)
    if not num_str.isdigit():
        return None
    return prefix, int(num_str), len(num_str)


def get_existing_info(dataset_dir: Path) -> tuple[int, int, Optional[str], int]:
    images = list_dataset_images(dataset_dir)
    if not images:
        return 0, 0, None, 0

    max_index = 0
    width = 0
    prefix: Optional[str] = None
    unmatched: list[Path] = []
    for img_path in images:
        parsed = parse_index(img_path.stem)
        if parsed is None:
            unmatched.append(img_path)
            continue
        img_prefix, idx, num_width = parsed
        if prefix is None:
            prefix = img_prefix
        elif img_prefix != prefix:
            raise SystemExit(
                f"Mixed filename prefixes found in dataset: '{prefix}' and '{img_prefix}'."
            )
        max_index = max(max_index, idx)
        width = max(width, num_width)

    if unmatched:
        sample = ", ".join(p.name for p in unmatched[:3])
        extra = "..." if len(unmatched) > 3 else ""
        raise SystemExit(
            f"Existing dataset images do not match the '{prefix or 'prefix'}_001' pattern: "
            f"{sample}{extra}"
        )
    if prefix is None:
        raise SystemExit("Existing dataset images do not match the expected naming pattern.")

    return max_index, width, prefix, len(images)


def write_data_yaml(out_dir: Path, class_name: str) -> None:
    content = "\n".join(
        [
            "path: .",
            "train: images/train",
            "val: images/val",
            f"names: [{class_name}]",
            "",
        ]
    )
    (out_dir / "data.yaml").write_text(content, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare a YOLO dataset directory from a folder of images."
    )
    parser.add_argument(
        "--source",
        default="photo",
        help="Source image folder (default: photo).",
    )
    parser.add_argument(
        "--output",
        default="yolo_dataset",
        help="Output dataset folder (default: yolo_dataset).",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train split ratio in [0,1] (default: 0.8).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splitting (default: 42).",
    )
    parser.add_argument(
        "--prefix",
        default="bike",
        help="Prefix for renamed images (default: bike).",
    )
    parser.add_argument(
        "--class-name",
        default="haluo_bike",
        help="Class name for data.yaml (default: haluo_bike).",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append images to an existing dataset instead of recreating it.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recreate output folder if it already exists.",
    )
    args = parser.parse_args()

    src_dir = Path(args.source)
    out_dir = Path(args.output)
    if not src_dir.exists():
        print(f"Source dir not found: {src_dir}", file=sys.stderr)
        return 1

    images = find_images(src_dir)
    if not images:
        print(f"No images found in: {src_dir}", file=sys.stderr)
        return 1

    if not (0.0 < args.train_ratio < 1.0):
        print("--train-ratio must be between 0 and 1.", file=sys.stderr)
        return 1

    if args.overwrite and args.append:
        print("--overwrite and --append cannot be used together.", file=sys.stderr)
        return 1

    out_nonempty = out_dir.exists() and any(out_dir.iterdir())
    if out_nonempty:
        if args.overwrite:
            shutil.rmtree(out_dir)
            out_nonempty = False
        elif not args.append:
            print(
                f"Output dir '{out_dir}' is not empty. Use --append to add images "
                "or --overwrite to recreate it.",
                file=sys.stderr,
            )
            return 1

    append_mode = args.append and out_nonempty
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)

    img_train_dir = out_dir / "images" / "train"
    img_val_dir = out_dir / "images" / "val"
    lbl_train_dir = out_dir / "labels" / "train"
    lbl_val_dir = out_dir / "labels" / "val"
    for d in (img_train_dir, img_val_dir, lbl_train_dir, lbl_val_dir):
        d.mkdir(parents=True, exist_ok=True)

    existing_max = 0
    existing_width = 0
    existing_prefix: Optional[str] = None
    existing_count = 0
    if append_mode:
        existing_max, existing_width, existing_prefix, existing_count = get_existing_info(
            out_dir
        )

    prefix = existing_prefix or args.prefix
    if existing_prefix is not None and args.prefix != existing_prefix:
        print(
            f"Using existing prefix '{existing_prefix}' to keep dataset naming consistent."
        )

    indices = list(range(len(images)))
    rng = random.Random(args.seed)
    rng.shuffle(indices)
    train_count = int(len(images) * args.train_ratio)
    if len(images) > 1:
        train_count = max(1, min(train_count, len(images) - 1))
    train_set = set(indices[:train_count])

    total_max = existing_max + len(images)
    width = max(existing_width, len(str(total_max)), 1)
    for i, img_path in enumerate(images):
        split = "train" if i in train_set else "val"
        out_img_dir = img_train_dir if split == "train" else img_val_dir
        out_lbl_dir = lbl_train_dir if split == "train" else lbl_val_dir
        index = existing_max + i + 1
        stem = f"{prefix}_{index:0{width}d}"
        new_name = f"{stem}{img_path.suffix.lower()}"
        shutil.copy2(img_path, out_img_dir / new_name)
        (out_lbl_dir / f"{stem}.txt").write_text("", encoding="utf-8")

    data_yaml = out_dir / "data.yaml"
    if not data_yaml.exists() or args.overwrite:
        write_data_yaml(out_dir, args.class_name)

    val_count = len(images) - train_count
    total_count = existing_count + len(images)
    print(f"Found {len(images)} new images in {src_dir}")
    print(f"Added -> Train: {train_count}  Val: {val_count}")
    print(f"Total images: {total_count}")
    print(f"Output: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
