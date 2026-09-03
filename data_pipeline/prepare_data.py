from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Optional

from PIL import Image


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

# This script is expected to be run from the repository root:
# cv-project3-anatomical-ldm/
DATASET_ROOT = Path("data/UPLOAD_FINAL/1_Experiment/standard_box/f0")

# We intentionally use only the development split (f0/train + f0/val).
# The strict holdout folder is never referenced by this script.
SPLITS = ("train", "val")

OUTPUT_ROOT = Path("processed_patches_severe")

VALID_TOOTH_CLASSES = {0, 1, 2}  # single-, double-, triple-root teeth

# Operational criterion for severe radiographic bone loss.
# We call these "severe RBL" cases, not automatically "Stage IV".
BONE_LOSS_THRESHOLD = 0.33

# Add a little context around the annotated tooth bounding box.
CROP_CONTEXT_SCALE = 1.25

OUTPUT_SIZE = 256

# YOLO-Pose keypoint order used by this dataset.
KP_CEJ_M = 0
KP_BL_M = 1
KP_RL_M = 2
KP_CEJ_D = 3
KP_BL_D = 4
KP_RL_D = 5
KP_RL_C = 6


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def euclidean_distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    """Return Euclidean distance between two 2D pixel coordinates."""
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def get_keypoint(
    parts: list[str],
    keypoint_index: int,
    img_w: int,
    img_h: int,
) -> Optional[tuple[float, float]]:
    """
    Read one YOLO-Pose keypoint.

    Label structure:
        class + bbox(xc, yc, w, h) + 11 * (x, y, visibility)

    Visibility 0 means the point is not usable.
    """
    base = 5 + keypoint_index * 3

    if base + 2 >= len(parts):
        return None

    x = float(parts[base])
    y = float(parts[base + 1])
    visibility = float(parts[base + 2])

    if visibility <= 0:
        return None

    # Defensive check for invalid normalized coordinates.
    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
        return None

    return x * img_w, y * img_h


def relative_bone_loss(
    cej: Optional[tuple[float, float]],
    bone_level: Optional[tuple[float, float]],
    root_apex: Optional[tuple[float, float]],
) -> Optional[float]:
    """
    Estimate relative radiographic bone loss:

        distance(CEJ, bone level) / distance(CEJ, root apex)

    Returns None when the required landmarks are unavailable.
    """
    if cej is None or bone_level is None or root_apex is None:
        return None

    root_length = euclidean_distance(cej, root_apex)

    if root_length <= 1e-8:
        return None

    bone_loss = euclidean_distance(cej, bone_level)
    return bone_loss / root_length


def find_image(images_dir: Path, stem: str) -> Optional[Path]:
    """Find the image matching a label stem, without assuming one extension."""
    for extension in (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"):
        candidate = images_dir / f"{stem}{extension}"
        if candidate.exists():
            return candidate
    return None


def crop_from_bbox(
    image: Image.Image,
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    context_scale: float = CROP_CONTEXT_SCALE,
) -> Optional[Image.Image]:
    """Crop a tooth patch from a normalized YOLO bounding box."""
    img_w, img_h = image.size

    cx = x_center * img_w
    cy = y_center * img_h
    crop_w = width * img_w * context_scale
    crop_h = height * img_h * context_scale

    x_min = max(0, int(round(cx - crop_w / 2)))
    y_min = max(0, int(round(cy - crop_h / 2)))
    x_max = min(img_w, int(round(cx + crop_w / 2)))
    y_max = min(img_h, int(round(cy + crop_h / 2)))

    if x_max <= x_min or y_max <= y_min:
        return None

    if (x_max - x_min) < 10 or (y_max - y_min) < 10:
        return None

    return image.crop((x_min, y_min, x_max, y_max))


# ---------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------

def process_split(split: str, metadata_writer: csv.DictWriter) -> dict[str, int]:
    split_root = DATASET_ROOT / split
    images_dir = split_root / "images"
    labels_dir = split_root / "labels"
    output_dir = OUTPUT_ROOT / split

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "label_files": 0,
        "tooth_annotations": 0,
        "valid_tooth_annotations": 0,
        "annotations_with_rbl": 0,
        "severe_cases": 0,
        "saved_patches": 0,
    }

    for label_path in sorted(labels_dir.glob("*.txt")):
        stats["label_files"] += 1

        image_path = find_image(images_dir, label_path.stem)
        if image_path is None:
            print(f"[WARNING] No image found for {label_path.name}")
            continue

        with Image.open(image_path) as loaded:
            image = loaded.convert("RGB")

        img_w, img_h = image.size

        with label_path.open("r", encoding="utf-8") as handle:
            lines = handle.readlines()

        for annotation_index, line in enumerate(lines):
            parts = line.strip().split()

            if not parts:
                continue

            stats["tooth_annotations"] += 1

            # Expected: class + 4 bbox values + 11 keypoints * 3 = 38.
            if len(parts) != 38:
                print(
                    f"[WARNING] {label_path.name}, row {annotation_index}: "
                    f"expected 38 values, found {len(parts)}. Skipped."
                )
                continue

            try:
                class_id = int(float(parts[0]))
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
            except ValueError:
                print(
                    f"[WARNING] Invalid numeric values in "
                    f"{label_path.name}, row {annotation_index}. Skipped."
                )
                continue

            if class_id not in VALID_TOOTH_CLASSES:
                continue

            stats["valid_tooth_annotations"] += 1

            cej_m = get_keypoint(parts, KP_CEJ_M, img_w, img_h)
            bl_m = get_keypoint(parts, KP_BL_M, img_w, img_h)
            rl_m = get_keypoint(parts, KP_RL_M, img_w, img_h)

            cej_d = get_keypoint(parts, KP_CEJ_D, img_w, img_h)
            bl_d = get_keypoint(parts, KP_BL_D, img_w, img_h)
            rl_d = get_keypoint(parts, KP_RL_D, img_w, img_h)

            rl_c = get_keypoint(parts, KP_RL_C, img_w, img_h)

            # If a side-specific root apex is unavailable, use the central
            # root apex as a fallback when it is annotated.
            mesial_apex = rl_m if rl_m is not None else rl_c
            distal_apex = rl_d if rl_d is not None else rl_c

            rbl_mesial = relative_bone_loss(cej_m, bl_m, mesial_apex)
            rbl_distal = relative_bone_loss(cej_d, bl_d, distal_apex)

            available_rbl = [
                value for value in (rbl_mesial, rbl_distal)
                if value is not None
            ]

            if not available_rbl:
                continue

            stats["annotations_with_rbl"] += 1

            max_rbl = max(available_rbl)

            if max_rbl <= BONE_LOSS_THRESHOLD:
                continue

            stats["severe_cases"] += 1

            patch = crop_from_bbox(
                image=image,
                x_center=x_center,
                y_center=y_center,
                width=width,
                height=height,
            )

            if patch is None:
                continue

            patch = patch.resize(
                (OUTPUT_SIZE, OUTPUT_SIZE),
                Image.Resampling.LANCZOS,
            )

            output_name = (
                f"{label_path.stem}"
                f"_tooth_{annotation_index}"
                f"_rbl_{max_rbl:.3f}.png"
            )
            output_path = output_dir / output_name
            patch.save(output_path)

            stats["saved_patches"] += 1

            metadata_writer.writerow(
                {
                    "split": split,
                    "source_image": image_path.name,
                    "label_file": label_path.name,
                    "annotation_index": annotation_index,
                    "tooth_class": class_id,
                    "rbl_mesial": (
                        f"{rbl_mesial:.6f}" if rbl_mesial is not None else ""
                    ),
                    "rbl_distal": (
                        f"{rbl_distal:.6f}" if rbl_distal is not None else ""
                    ),
                    "max_rbl": f"{max_rbl:.6f}",
                    "patch_file": str(output_path),
                }
            )

    return stats


def main() -> None:
    print("Preparing severe-RBL tooth patches")
    print(f"Dataset root: {DATASET_ROOT.resolve()}")
    print(f"Output root:  {OUTPUT_ROOT.resolve()}")
    print(f"RBL threshold: > {BONE_LOSS_THRESHOLD:.2f}")
    print()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    metadata_path = OUTPUT_ROOT / "metadata.csv"
    fieldnames = [
        "split",
        "source_image",
        "label_file",
        "annotation_index",
        "tooth_class",
        "rbl_mesial",
        "rbl_distal",
        "max_rbl",
        "patch_file",
    ]

    all_stats: dict[str, dict[str, int]] = {}

    with metadata_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for split in SPLITS:
            stats = process_split(split, writer)
            all_stats[split] = stats

    print("\nExtraction completed.")
    for split, stats in all_stats.items():
        print(f"\n[{split}]")
        for key, value in stats.items():
            print(f"  {key}: {value}")

    total_saved = sum(stats["saved_patches"] for stats in all_stats.values())
    print(f"\nTotal saved patches: {total_saved}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()