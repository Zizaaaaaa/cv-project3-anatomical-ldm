from __future__ import annotations

import random
from pathlib import Path

from PIL import Image
import torch
from torchvision.transforms import functional as TF
from torchvision.utils import make_grid, save_image


ROOT = Path("processed_patches_severe_guided")
OUTPUT_DIR = Path("outputs/anatomical_guidance/final_guided_dataset_check")

NUM_TRAIN = 8
NUM_VAL = 6

SEED = 42


def collect_pairs(split: str):
    images_dir = ROOT / split / "images"
    masks_dir = ROOT / split / "masks"

    image_files = sorted(images_dir.glob("*.png"))

    pairs = []

    for image_path in image_files:
        mask_path = masks_dir / image_path.name

        if not mask_path.exists():
            raise FileNotFoundError(
                f"Missing mask for {image_path.name}"
            )

        pairs.append((image_path, mask_path))

    return pairs


def load_pair(image_path: Path, mask_path: Path):
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")

    image_tensor = TF.to_tensor(image)
    mask_tensor = (TF.to_tensor(mask) >= 0.5).float()

    mask_rgb = mask_tensor.repeat(3, 1, 1)

    overlay = image_tensor.clone()
    overlay[0] = torch.maximum(
        overlay[0],
        mask_tensor[0],
    )

    return image_tensor, mask_rgb, overlay


def main():
    random.seed(SEED)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    selected = []

    for split, count in [
        ("train", NUM_TRAIN),
        ("val", NUM_VAL),
    ]:
        pairs = collect_pairs(split)

        chosen = random.sample(
            pairs,
            min(count, len(pairs)),
        )

        for pair in chosen:
            selected.append(
                (split, pair[0], pair[1])
            )

    images = []
    masks = []
    overlays = []

    for split, image_path, mask_path in selected:

        image, mask, overlay = load_pair(
            image_path,
            mask_path,
        )

        images.append(image)
        masks.append(mask)
        overlays.append(overlay)

        print(
            f"{split:5s} | "
            f"{image_path.name}"
        )

    save_image(
        make_grid(images, nrow=4),
        OUTPUT_DIR / "images.png",
    )

    save_image(
        make_grid(masks, nrow=4),
        OUTPUT_DIR / "masks.png",
    )

    save_image(
        make_grid(overlays, nrow=4),
        OUTPUT_DIR / "overlays.png",
    )

    print()
    print("Final guided dataset check completed.")
    print("Outputs:", OUTPUT_DIR)


if __name__ == "__main__":
    main()