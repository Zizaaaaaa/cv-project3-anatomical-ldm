from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torchvision.utils import make_grid, save_image

from segmentation_dataset import ToothSegmentationDataset


MANIFEST = Path(
    "outputs/anatomical_guidance/segmentation_manifest.csv"
)

OUTPUT_DIR = Path(
    "outputs/anatomical_guidance/dataset_checks"
)

NUM_EXAMPLES = 12


def main() -> None:

    dataset = ToothSegmentationDataset(
        manifest_path=MANIFEST,
        role="segmenter_train",
        image_size=256,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    images = []
    masks_rgb = []
    overlays = []

    print()
    print("Checking samples...")

    for index in range(
        min(NUM_EXAMPLES, len(dataset))
    ):

        sample = dataset[index]

        image = sample["image"]
        mask = sample["mask"]

        # Basic shape checks.
        if image.shape != (3, 256, 256):
            raise RuntimeError(
                f"Unexpected image shape: {image.shape}"
            )

        if mask.shape != (1, 256, 256):
            raise RuntimeError(
                f"Unexpected mask shape: {mask.shape}"
            )

        unique_values = torch.unique(mask)

        if not all(
            value.item() in (0.0, 1.0)
            for value in unique_values
        ):
            raise RuntimeError(
                f"Mask is not binary: {unique_values}"
            )

        mask_rgb = mask.repeat(3, 1, 1)

        # Simple overlay:
        #
        # retain the original X-ray and brighten the pixels
        # belonging to the tooth mask.
        overlay = image.clone()

        overlay[0] = torch.maximum(
            overlay[0],
            mask[0],
        )

        images.append(image)
        masks_rgb.append(mask_rgb)
        overlays.append(overlay)

        print(
            f"{index:02d} | "
            f"case={sample['case_id']} | "
            f"view={sample['view_id']} | "
            f"mask fraction="
            f"{mask.mean().item():.4f}"
        )

    image_grid = make_grid(
        images,
        nrow=4,
        padding=2,
    )

    mask_grid = make_grid(
        masks_rgb,
        nrow=4,
        padding=2,
    )

    overlay_grid = make_grid(
        overlays,
        nrow=4,
        padding=2,
    )

    save_image(
        image_grid,
        OUTPUT_DIR / "images.png",
    )

    save_image(
        mask_grid,
        OUTPUT_DIR / "masks.png",
    )

    save_image(
        overlay_grid,
        OUTPUT_DIR / "overlays.png",
    )

    print()
    print("Sanity check completed.")
    print(f"Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()