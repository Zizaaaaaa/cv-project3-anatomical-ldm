from __future__ import annotations

import csv
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

import random

IMAGE_SIZE = 256


def rasterize_yolo_polygons(
    label_path: Path,
    image_width: int,
    image_height: int,
) -> Image.Image:
    """
    Convert YOLO segmentation polygons into one binary semantic mask.

    Output:
        background = 0
        tooth      = 255

    Each row of the YOLO label has the form:

        class x1 y1 x2 y2 x3 y3 ...

    Coordinates are normalized to [0, 1].
    """

    mask = Image.new(
        mode="L",
        size=(image_width, image_height),
        color=0,
    )

    draw = ImageDraw.Draw(mask)

    lines = label_path.read_text(
        encoding="utf-8"
    ).splitlines()

    for line_number, line in enumerate(lines, start=1):

        if not line.strip():
            continue

        parts = line.split()

        if len(parts) < 7:
            raise ValueError(
                f"{label_path}, line {line_number}: "
                f"too few values for a polygon."
            )

        # The first value is the class ID.
        coordinates = [
            float(value)
            for value in parts[1:]
        ]

        if len(coordinates) % 2 != 0:
            raise ValueError(
                f"{label_path}, line {line_number}: "
                f"odd number of polygon coordinates."
            )

        points = []

        for index in range(0, len(coordinates), 2):

            x_norm = coordinates[index]
            y_norm = coordinates[index + 1]

            if not (
                0.0 <= x_norm <= 1.0
                and 0.0 <= y_norm <= 1.0
            ):
                raise ValueError(
                    f"{label_path}, line {line_number}: "
                    f"coordinate outside [0, 1]."
                )

            x = x_norm * image_width
            y = y_norm * image_height

            points.append((x, y))

        if len(points) >= 3:
            draw.polygon(
                points,
                fill=255,
            )

    return mask


class ToothSegmentationDataset(Dataset):
    """
    Dataset for binary tooth segmentation.

    Each sample returns:

        image:
            FloatTensor [3, H, W], values in [0, 1]

        mask:
            FloatTensor [1, H, W], values exactly 0 or 1

        metadata:
            useful information for debugging
    """

    def __init__(
        self,
        manifest_path: str | Path,
        role: str,
        image_size: int = IMAGE_SIZE,
        augment: bool = False,
    ) -> None:

        self.manifest_path = Path(manifest_path)
        self.role = role
        self.image_size = image_size
        self.augment = augment

        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found: {self.manifest_path}"
            )

        with self.manifest_path.open(
            "r",
            encoding="utf-8",
            newline="",
        ) as handle:

            rows = list(
                csv.DictReader(handle)
            )

        self.rows = [
            row
            for row in rows
            if row["role"] == role
        ]

        if not self.rows:
            raise RuntimeError(
                f"No samples found with role '{role}'."
            )

        print(
            f"Loaded {len(self.rows)} samples "
            f"for role '{role}'."
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, object]:

        row = self.rows[index]

        image_path = Path(row["image_path"])
        label_path = Path(row["label_path"])

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        if not label_path.exists():
            raise FileNotFoundError(
                f"Label not found: {label_path}"
            )

        with Image.open(image_path) as loaded:
            image = loaded.convert("RGB")

        original_width, original_height = image.size

        mask = rasterize_yolo_polygons(
            label_path=label_path,
            image_width=original_width,
            image_height=original_height,
        )

        # -------------------------------------------------------------
        # Joint geometric augmentation
        # -------------------------------------------------------------

        if self.augment:

            # Horizontal flipping is reasonable here because the task is
            # generic tooth/background segmentation rather than tooth numbering.
            if random.random() < 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            # Mild affine augmentation.
            angle = random.uniform(-4.0, 4.0)

            translate_x = int(
                random.uniform(-0.03, 0.03)
                * image.width
            )

            translate_y = int(
                random.uniform(-0.03, 0.03)
                * image.height
            )

            scale = random.uniform(
                0.97,
                1.03,
            )

            image = TF.affine(
                image,
                angle=angle,
                translate=[translate_x, translate_y],
                scale=scale,
                shear=[0.0, 0.0],
                interpolation=InterpolationMode.BICUBIC,
                fill=0,
            )

            mask = TF.affine(
                mask,
                angle=angle,
                translate=[translate_x, translate_y],
                scale=scale,
                shear=[0.0, 0.0],
                interpolation=InterpolationMode.NEAREST,
                fill=0,
            )

        # Important:
        #
        # images use BICUBIC interpolation;
        # segmentation masks MUST use NEAREST interpolation.
        #
        # Using bicubic/bilinear on masks would create artificial
        # intermediate values around the polygon borders.
        image = TF.resize(
            image,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.BICUBIC,
        )

        mask = TF.resize(
            mask,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.NEAREST,
        )

        image_tensor = TF.to_tensor(image)

        mask_tensor = TF.to_tensor(mask)

        # Defensive binarization.
        mask_tensor = (
            mask_tensor >= 0.5
        ).float()

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "case_id": row["case_id"],
            "view_id": int(row["view_id"]),
            "image_path": str(image_path),
            "label_path": str(label_path),
        }