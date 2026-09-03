from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

from tooth_segmenter import SmallUNet


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

STANDARD_ROOT = Path(
    "data/UPLOAD_FINAL/1_Experiment/standard_box/f0"
)

HOLDOUT_ROOT = Path(
    "data/UPLOAD_FINAL/1_Experiment/holdout_test_standard_box"
)

SEVERE_ROOT = Path(
    "processed_patches_severe"
)

SEVERE_METADATA = SEVERE_ROOT / "metadata.csv"

SEGMENTER_CHECKPOINT = Path(
    "outputs/anatomical_guidance/tooth_segmenter/best_model.pt"
)

OUTPUT_ROOT = Path(
    "processed_patches_severe_guided"
)

CROP_CONTEXT_SCALE = 1.25
MODEL_SIZE = 256
OUTPUT_SIZE = 256
MASK_THRESHOLD = 0.5


# ---------------------------------------------------------------------
# Helper: standard case ID
# ---------------------------------------------------------------------

def parse_case_id(filename: str) -> str:
    """
    Examples:
        Image100.png   -> 100
        Image144_1.png -> 144
    """

    stem = Path(filename).stem

    match = re.fullmatch(
        r"Image(\d+)(?:_\d+)?",
        stem,
    )

    if match is None:
        raise ValueError(
            f"Cannot parse case ID from: {filename}"
        )

    return match.group(1)


# ---------------------------------------------------------------------
# Strict holdout IDs
# ---------------------------------------------------------------------

def collect_holdout_ids() -> set[str]:
    ids: set[str] = set()

    for path in HOLDOUT_ROOT.rglob("*"):

        if not path.is_file():
            continue

        if path.suffix.lower() not in {
            ".png",
            ".jpg",
            ".jpeg",
        }:
            continue

        ids.add(
            parse_case_id(path.name)
        )

    return ids


# ---------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------

def compute_crop_box(
    img_w: int,
    img_h: int,
    x_center: float,
    y_center: float,
    width: float,
    height: float,
) -> tuple[int, int, int, int]:
    """
    Same geometry used in prepare_data.py.
    """

    cx = x_center * img_w
    cy = y_center * img_h

    crop_w = (
        width
        * img_w
        * CROP_CONTEXT_SCALE
    )

    crop_h = (
        height
        * img_h
        * CROP_CONTEXT_SCALE
    )

    x_min = max(
        0,
        int(round(cx - crop_w / 2)),
    )

    y_min = max(
        0,
        int(round(cy - crop_h / 2)),
    )

    x_max = min(
        img_w,
        int(round(cx + crop_w / 2)),
    )

    y_max = min(
        img_h,
        int(round(cy + crop_h / 2)),
    )

    if x_max <= x_min or y_max <= y_min:
        raise ValueError(
            "Invalid crop box."
        )

    return (
        x_min,
        y_min,
        x_max,
        y_max,
    )


def bbox_from_yolo(
    img_w: int,
    img_h: int,
    x_center: float,
    y_center: float,
    width: float,
    height: float,
) -> tuple[int, int, int, int]:
    """
    Convert normalized YOLO bbox to pixel coordinates.
    """

    cx = x_center * img_w
    cy = y_center * img_h

    box_w = width * img_w
    box_h = height * img_h

    x_min = max(
        0,
        int(round(cx - box_w / 2)),
    )

    y_min = max(
        0,
        int(round(cy - box_h / 2)),
    )

    x_max = min(
        img_w,
        int(round(cx + box_w / 2)),
    )

    y_max = min(
        img_h,
        int(round(cy + box_h / 2)),
    )

    return (
        x_min,
        y_min,
        x_max,
        y_max,
    )


# ---------------------------------------------------------------------
# Pose label
# ---------------------------------------------------------------------

def load_pose_bbox(
    label_path: Path,
    annotation_index: int,
) -> tuple[float, float, float, float]:

    lines = label_path.read_text(
        encoding="utf-8"
    ).splitlines()

    if annotation_index >= len(lines):
        raise IndexError(
            f"{label_path}: annotation "
            f"{annotation_index} does not exist."
        )

    parts = lines[
        annotation_index
    ].split()

    if len(parts) != 38:
        raise ValueError(
            f"{label_path}, annotation "
            f"{annotation_index}: expected "
            f"38 values, got {len(parts)}."
        )

    return (
        float(parts[1]),
        float(parts[2]),
        float(parts[3]),
        float(parts[4]),
    )


# ---------------------------------------------------------------------
# Segmenter inference
# ---------------------------------------------------------------------

@torch.no_grad()
def predict_full_mask(
    model: torch.nn.Module,
    image: Image.Image,
    device: torch.device,
) -> np.ndarray:
    """
    Predict tooth/background mask and resize it
    back to the original image size.
    """

    original_w, original_h = image.size

    resized = TF.resize(
        image,
        [MODEL_SIZE, MODEL_SIZE],
        interpolation=InterpolationMode.BICUBIC,
    )

    tensor = (
        TF.to_tensor(resized)
        .unsqueeze(0)
        .to(device)
    )

    logits = model(tensor)

    probability = torch.sigmoid(
        logits
    )[0, 0]

    binary = (
        probability >= MASK_THRESHOLD
    ).to(torch.uint8)

    mask_256 = (
        binary.cpu().numpy()
        * 255
    ).astype(np.uint8)

    mask_image = Image.fromarray(
        mask_256,
        mode="L",
    )

    mask_original = mask_image.resize(
        (original_w, original_h),
        Image.Resampling.NEAREST,
    )

    return (
        np.asarray(mask_original) > 0
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    if not SEVERE_METADATA.exists():
        raise FileNotFoundError(
            f"Missing metadata: "
            f"{SEVERE_METADATA}"
        )

    if not SEGMENTER_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Missing segmenter checkpoint: "
            f"{SEGMENTER_CHECKPOINT}"
        )

    holdout_ids = collect_holdout_ids()

    print(
        "Strict holdout IDs:",
        len(holdout_ids),
    )

    # -------------------------------------------------------------
    # Load segmenter
    # -------------------------------------------------------------

    checkpoint = torch.load(
        SEGMENTER_CHECKPOINT,
        map_location=device,
    )

    model = SmallUNet().to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print(
        "Segmenter checkpoint epoch:",
        checkpoint["epoch"],
    )

    print(
        "Segmenter validation Dice:",
        checkpoint["val_dice"],
    )

    # -------------------------------------------------------------
    # Read severe metadata
    # -------------------------------------------------------------

    with SEVERE_METADATA.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:

        severe_rows = list(
            csv.DictReader(handle)
        )

    print(
        "Severe metadata rows:",
        len(severe_rows),
    )

    # -------------------------------------------------------------
    # Output folders
    # -------------------------------------------------------------

    for split in ("train", "val"):

        (
            OUTPUT_ROOT
            / split
            / "images"
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

        (
            OUTPUT_ROOT
            / split
            / "masks"
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

    output_metadata_path = (
        OUTPUT_ROOT
        / "metadata.csv"
    )

    output_fieldnames = [
        "split",
        "case_id",
        "source_image",
        "label_file",
        "annotation_index",
        "tooth_class",
        "rbl_mesial",
        "rbl_distal",
        "max_rbl",
        "image_file",
        "mask_file",
        "bbox_mask_fraction",
        "crop_x_min",
        "crop_y_min",
        "crop_x_max",
        "crop_y_max",
    ]

    written = 0
    skipped = 0

    # -------------------------------------------------------------
    # Process all severe teeth
    # -------------------------------------------------------------

    with output_metadata_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as output_handle:

        writer = csv.DictWriter(
            output_handle,
            fieldnames=output_fieldnames,
        )

        writer.writeheader()

        for index, row in enumerate(
            severe_rows
        ):

            split = row["split"]

            if split not in {
                "train",
                "val",
            }:
                raise ValueError(
                    f"Unexpected split: {split}"
                )

            source_image = row[
                "source_image"
            ]

            case_id = parse_case_id(
                source_image
            )

            # -----------------------------------------------------
            # Absolute anti-leakage protection
            # -----------------------------------------------------

            if case_id in holdout_ids:

                raise RuntimeError(
                    "STRICT HOLDOUT LEAKAGE DETECTED: "
                    f"{source_image}"
                )

            annotation_index = int(
                row["annotation_index"]
            )

            image_path = (
                STANDARD_ROOT
                / split
                / "images"
                / source_image
            )

            label_path = (
                STANDARD_ROOT
                / split
                / "labels"
                / row["label_file"]
            )

            if not image_path.exists():
                raise FileNotFoundError(
                    f"Missing image: {image_path}"
                )

            if not label_path.exists():
                raise FileNotFoundError(
                    f"Missing pose label: "
                    f"{label_path}"
                )

            # -----------------------------------------------------
            # Original radiograph
            # -----------------------------------------------------

            with Image.open(
                image_path
            ) as loaded:

                image = loaded.convert(
                    "RGB"
                )

            img_w, img_h = image.size

            # -----------------------------------------------------
            # Severe tooth bbox
            # -----------------------------------------------------

            (
                x_center,
                y_center,
                width,
                height,
            ) = load_pose_bbox(
                label_path,
                annotation_index,
            )

            tooth_bbox = bbox_from_yolo(
                img_w,
                img_h,
                x_center,
                y_center,
                width,
                height,
            )

            crop_box = compute_crop_box(
                img_w,
                img_h,
                x_center,
                y_center,
                width,
                height,
            )

            # -----------------------------------------------------
            # Segment all teeth
            # -----------------------------------------------------

            full_mask = predict_full_mask(
                model,
                image,
                device,
            )

            # -----------------------------------------------------
            # Keep only the segmentation inside
            # the severe tooth bounding box.
            # -----------------------------------------------------

            (
                bbox_x_min,
                bbox_y_min,
                bbox_x_max,
                bbox_y_max,
            ) = tooth_bbox

            target_mask = np.zeros_like(
                full_mask,
                dtype=np.uint8,
            )

            target_mask[
                bbox_y_min:bbox_y_max,
                bbox_x_min:bbox_x_max,
            ] = full_mask[
                bbox_y_min:bbox_y_max,
                bbox_x_min:bbox_x_max,
            ].astype(np.uint8)

            bbox_region = target_mask[
                bbox_y_min:bbox_y_max,
                bbox_x_min:bbox_x_max,
            ]

            bbox_mask_fraction = float(
                bbox_region.mean()
            )

            # Defensive warning, but do not silently remove cases.
            if bbox_mask_fraction < 0.10:

                print(
                    "[WARNING] Very low mask coverage: "
                    f"{source_image}, "
                    f"tooth={annotation_index}, "
                    f"fraction="
                    f"{bbox_mask_fraction:.3f}"
                )

            target_mask_image = (
                Image.fromarray(
                    target_mask * 255,
                    mode="L",
                )
            )

            # -----------------------------------------------------
            # Same crop for X-ray and mask
            # -----------------------------------------------------

            image_crop = image.crop(
                crop_box
            )

            mask_crop = (
                target_mask_image.crop(
                    crop_box
                )
            )

            image_crop = image_crop.resize(
                (
                    OUTPUT_SIZE,
                    OUTPUT_SIZE,
                ),
                Image.Resampling.LANCZOS,
            )

            mask_crop = mask_crop.resize(
                (
                    OUTPUT_SIZE,
                    OUTPUT_SIZE,
                ),
                Image.Resampling.NEAREST,
            )

            # Force binary mask after resizing.
            mask_array = np.asarray(
                mask_crop
            )

            mask_array = (
                mask_array >= 128
            ).astype(np.uint8) * 255

            mask_crop = Image.fromarray(
                mask_array,
                mode="L",
            )

            # -----------------------------------------------------
            # Use same basename as original severe patch
            # -----------------------------------------------------

            original_patch_path = Path(
                row["patch_file"]
            )

            output_name = (
                original_patch_path.name
            )

            image_output_path = (
                OUTPUT_ROOT
                / split
                / "images"
                / output_name
            )

            mask_output_path = (
                OUTPUT_ROOT
                / split
                / "masks"
                / output_name
            )

            image_crop.save(
                image_output_path
            )

            mask_crop.save(
                mask_output_path
            )

            (
                crop_x_min,
                crop_y_min,
                crop_x_max,
                crop_y_max,
            ) = crop_box

            writer.writerow(
                {
                    "split":
                        split,

                    "case_id":
                        case_id,

                    "source_image":
                        source_image,

                    "label_file":
                        row["label_file"],

                    "annotation_index":
                        annotation_index,

                    "tooth_class":
                        row["tooth_class"],

                    "rbl_mesial":
                        row["rbl_mesial"],

                    "rbl_distal":
                        row["rbl_distal"],

                    "max_rbl":
                        row["max_rbl"],

                    "image_file":
                        str(
                            image_output_path
                        ),

                    "mask_file":
                        str(
                            mask_output_path
                        ),

                    "bbox_mask_fraction":
                        f"{bbox_mask_fraction:.6f}",

                    "crop_x_min":
                        crop_x_min,

                    "crop_y_min":
                        crop_y_min,

                    "crop_x_max":
                        crop_x_max,

                    "crop_y_max":
                        crop_y_max,
                }
            )

            written += 1

            print(
                f"{index + 1:03d}/"
                f"{len(severe_rows):03d} | "
                f"{split:5s} | "
                f"{source_image} | "
                f"tooth="
                f"{annotation_index} | "
                f"mask="
                f"{bbox_mask_fraction:.3f}"
            )

    print()
    print("Guided dataset completed.")
    print(
        "Saved pairs:",
        written,
    )
    print(
        "Skipped:",
        skipped,
    )
    print(
        "Output:",
        OUTPUT_ROOT,
    )
    print(
        "Metadata:",
        output_metadata_path,
    )


if __name__ == "__main__":
    main()