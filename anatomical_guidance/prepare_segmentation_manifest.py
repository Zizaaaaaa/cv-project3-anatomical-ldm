from __future__ import annotations

import csv
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

AUX_ROOT = Path("data/UPLOAD_FINAL/2_Auxiliary_Segmentation")

STANDARD_ROOT = Path("data/UPLOAD_FINAL/1_Experiment/standard_box/f0")
HOLDOUT_ROOT = Path(
    "data/UPLOAD_FINAL/1_Experiment/holdout_test_standard_box"
)

OUTPUT_DIR = Path("outputs/anatomical_guidance")
OUTPUT_CSV = OUTPUT_DIR / "segmentation_manifest.csv"

SEED = 42

# Fraction of auxiliary-only IDs reserved for validating the tooth
# segmentation model.
AUX_ONLY_VAL_FRACTION = 0.15

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".JPG",
    ".JPEG",
    ".PNG",
}


# ---------------------------------------------------------------------
# ID parsing
# ---------------------------------------------------------------------

def parse_standard_case_id(stem: str) -> str | None:
    """
    Examples:
        Image100   -> 100
        Image144_1 -> 144

    The optional suffix belongs to the standard-box filename, not to
    the patient/case identifier used by the auxiliary dataset.
    """
    match = re.fullmatch(r"Image(\d+)(?:_\d+)?", stem)

    if match is None:
        return None

    return match.group(1)


def parse_auxiliary_name(stem: str) -> tuple[int, str] | None:
    """
    Examples:
        1_100 -> view_id=1, case_id=100
        6_163 -> view_id=6, case_id=163
    """
    match = re.fullmatch(r"([1-6])_(\d+)", stem)

    if match is None:
        return None

    view_id = int(match.group(1))
    case_id = match.group(2)

    return view_id, case_id


# ---------------------------------------------------------------------
# Standard split IDs
# ---------------------------------------------------------------------

def collect_standard_ids(directory: Path, recursive: bool = False) -> set[str]:
    ids: set[str] = set()

    iterator = directory.rglob("*") if recursive else directory.iterdir()

    for path in iterator:
        if not path.is_file():
            continue

        if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue

        case_id = parse_standard_case_id(path.stem)

        if case_id is not None:
            ids.add(case_id)

    return ids


# ---------------------------------------------------------------------
# Auxiliary dataset discovery
# ---------------------------------------------------------------------

def collect_auxiliary_samples() -> list[dict[str, str | int]]:
    samples: list[dict[str, str | int]] = []

    seen_stems: set[str] = set()

    for original_split in ("train", "val"):

        images_dir = AUX_ROOT / original_split / "images"
        labels_dir = AUX_ROOT / original_split / "labels"

        if not images_dir.exists():
            raise FileNotFoundError(
                f"Missing auxiliary images directory: {images_dir}"
            )

        if not labels_dir.exists():
            raise FileNotFoundError(
                f"Missing auxiliary labels directory: {labels_dir}"
            )

        for image_path in sorted(images_dir.iterdir()):

            if not image_path.is_file():
                continue

            if image_path.suffix not in IMAGE_EXTENSIONS:
                continue

            parsed = parse_auxiliary_name(image_path.stem)

            if parsed is None:
                raise ValueError(
                    f"Unexpected auxiliary filename: {image_path.name}"
                )

            view_id, case_id = parsed

            if image_path.stem in seen_stems:
                raise RuntimeError(
                    f"Duplicate auxiliary image stem: {image_path.stem}"
                )

            seen_stems.add(image_path.stem)

            label_path = labels_dir / f"{image_path.stem}.txt"

            if not label_path.exists():
                raise FileNotFoundError(
                    f"Missing segmentation label for {image_path.name}: "
                    f"{label_path}"
                )

            samples.append(
                {
                    "case_id": case_id,
                    "view_id": view_id,
                    "original_aux_split": original_split,
                    "image_path": str(image_path),
                    "label_path": str(label_path),
                }
            )

    return samples


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:

    random.seed(SEED)

    standard_train_ids = collect_standard_ids(
        STANDARD_ROOT / "train" / "images"
    )

    standard_val_ids = collect_standard_ids(
        STANDARD_ROOT / "val" / "images"
    )

    strict_holdout_ids = collect_standard_ids(
        HOLDOUT_ROOT,
        recursive=True,
    )

    # Defensive split checks.
    if standard_train_ids & standard_val_ids:
        raise RuntimeError(
            "Standard train and val contain overlapping case IDs."
        )

    if standard_train_ids & strict_holdout_ids:
        raise RuntimeError(
            "Standard train overlaps strict holdout."
        )

    if standard_val_ids & strict_holdout_ids:
        raise RuntimeError(
            "Standard val overlaps strict holdout."
        )

    samples = collect_auxiliary_samples()

    samples_by_case: dict[str, list[dict[str, str | int]]] = defaultdict(list)

    for sample in samples:
        samples_by_case[str(sample["case_id"])].append(sample)

    auxiliary_ids = set(samples_by_case)

    auxiliary_only_ids = (
        auxiliary_ids
        - standard_train_ids
        - standard_val_ids
        - strict_holdout_ids
    )

    # -------------------------------------------------------------
    # Create our own patient/case-level validation split.
    #
    # Important:
    # standard f0/val IDs are NOT used to train or tune the segmenter.
    # strict holdout IDs are NEVER used.
    # -------------------------------------------------------------

    shuffled_aux_only = sorted(auxiliary_only_ids)
    random.shuffle(shuffled_aux_only)

    num_seg_val_ids = max(
        1,
        round(len(shuffled_aux_only) * AUX_ONLY_VAL_FRACTION),
    )

    segmenter_val_ids = set(
        shuffled_aux_only[:num_seg_val_ids]
    )

    auxiliary_only_train_ids = set(
        shuffled_aux_only[num_seg_val_ids:]
    )

    segmenter_train_ids = (
        standard_train_ids
        | auxiliary_only_train_ids
    )

    # Extremely important anti-leakage checks.
    assert not (segmenter_train_ids & segmenter_val_ids)
    assert not (segmenter_train_ids & standard_val_ids)
    assert not (segmenter_train_ids & strict_holdout_ids)

    assert not (segmenter_val_ids & standard_val_ids)
    assert not (segmenter_val_ids & strict_holdout_ids)

    # -------------------------------------------------------------
    # Assign each image to its role.
    # -------------------------------------------------------------

    final_rows: list[dict[str, str | int]] = []

    for sample in samples:

        case_id = str(sample["case_id"])

        if case_id in strict_holdout_ids:
            role = "excluded_strict_holdout"

        elif case_id in standard_val_ids:
            role = "excluded_standard_val"

        elif case_id in segmenter_val_ids:
            role = "segmenter_val"

        elif case_id in segmenter_train_ids:
            role = "segmenter_train"

        else:
            raise RuntimeError(
                f"Case ID {case_id} was not assigned a role."
            )

        row = dict(sample)
        row["role"] = role

        final_rows.append(row)

    # -------------------------------------------------------------
    # Save manifest.
    # -------------------------------------------------------------

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "case_id",
        "view_id",
        "role",
        "original_aux_split",
        "image_path",
        "label_path",
    ]

    with OUTPUT_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(final_rows)

    # -------------------------------------------------------------
    # Report.
    # -------------------------------------------------------------

    role_image_counts = Counter(
        str(row["role"])
        for row in final_rows
    )

    role_case_ids: dict[str, set[str]] = defaultdict(set)

    for row in final_rows:
        role_case_ids[str(row["role"])].add(
            str(row["case_id"])
        )

    print("Segmentation manifest created.")
    print()

    print(f"Auxiliary images: {len(final_rows)}")
    print(f"Auxiliary unique case IDs: {len(auxiliary_ids)}")
    print()

    print(f"Standard train IDs: {len(standard_train_ids)}")
    print(f"Standard val IDs: {len(standard_val_ids)}")
    print(f"Strict holdout IDs: {len(strict_holdout_ids)}")
    print(f"Auxiliary-only IDs: {len(auxiliary_only_ids)}")
    print()

    print("Assigned roles:")

    for role in (
        "segmenter_train",
        "segmenter_val",
        "excluded_standard_val",
        "excluded_strict_holdout",
    ):

        print(
            f"  {role:27s} "
            f"cases={len(role_case_ids[role]):3d} "
            f"images={role_image_counts[role]:4d}"
        )

    print()

    print(
        "Train/val ID overlap:",
        len(
            role_case_ids["segmenter_train"]
            & role_case_ids["segmenter_val"]
        ),
    )

    print(
        "Segmenter train / standard-val overlap:",
        len(
            role_case_ids["segmenter_train"]
            & standard_val_ids
        ),
    )

    print(
        "Segmenter train / strict-holdout overlap:",
        len(
            role_case_ids["segmenter_train"]
            & strict_holdout_ids
        ),
    )

    print()

    print(f"Manifest: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()