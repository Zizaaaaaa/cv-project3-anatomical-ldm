from __future__ import annotations

import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.utils import make_grid, save_image

from segmentation_dataset import ToothSegmentationDataset
from tooth_segmenter import (
    SmallUNet,
    dice_score,
    soft_dice_loss,
)


MANIFEST = Path(
    "outputs/anatomical_guidance/segmentation_manifest.csv"
)

OUTPUT_DIR = Path(
    "outputs/anatomical_guidance/overfit_test"
)

NUM_SAMPLES = 8
BATCH_SIZE = 4

NUM_EPOCHS = 100
LEARNING_RATE = 1e-3

SEED = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:

    set_seed(SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    full_dataset = ToothSegmentationDataset(
        manifest_path=MANIFEST,
        role="segmenter_train",
        image_size=256,
    )

    # Deliberately use only the first few images.
    #
    # The goal is NOT generalization.
    # We want the network to memorize them.
    dataset = Subset(
        full_dataset,
        range(NUM_SAMPLES),
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    model = SmallUNet().to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print(
        f"Overfitting on {NUM_SAMPLES} images..."
    )

    for epoch in range(
        1,
        NUM_EPOCHS + 1,
    ):

        model.train()

        total_loss = 0.0
        total_dice = 0.0
        total_samples = 0

        for batch in loader:

            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            logits = model(images)

            bce = F.binary_cross_entropy_with_logits(
                logits,
                masks,
            )

            dice_loss = soft_dice_loss(
                logits,
                masks,
            )

            loss = bce + dice_loss

            optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()

            optimizer.step()

            batch_size = images.shape[0]

            with torch.no_grad():
                dice = dice_score(
                    logits,
                    masks,
                )

            total_loss += (
                loss.item() * batch_size
            )

            total_dice += (
                dice.item() * batch_size
            )

            total_samples += batch_size

        epoch_loss = (
            total_loss / total_samples
        )

        epoch_dice = (
            total_dice / total_samples
        )

        if (
            epoch == 1
            or epoch % 10 == 0
        ):
            print(
                f"Epoch {epoch:03d} | "
                f"loss={epoch_loss:.4f} | "
                f"dice={epoch_dice:.4f}"
            )

    # -------------------------------------------------------------
    # Final visual sanity check
    # -------------------------------------------------------------

    model.eval()

    images_to_show = []
    masks_to_show = []
    predictions_to_show = []

    with torch.no_grad():

        for index in range(NUM_SAMPLES):

            sample = full_dataset[index]

            image = (
                sample["image"]
                .unsqueeze(0)
                .to(device)
            )

            mask = sample["mask"]

            logits = model(image)

            probability = torch.sigmoid(
                logits
            )

            prediction = (
                probability >= 0.5
            ).float()

            images_to_show.append(
                image.squeeze(0).cpu()
            )

            masks_to_show.append(
                mask.repeat(3, 1, 1)
            )

            predictions_to_show.append(
                prediction
                .squeeze(0)
                .repeat(3, 1, 1)
                .cpu()
            )

    save_image(
        make_grid(
            images_to_show,
            nrow=4,
        ),
        OUTPUT_DIR / "images.png",
    )

    save_image(
        make_grid(
            masks_to_show,
            nrow=4,
        ),
        OUTPUT_DIR / "ground_truth.png",
    )

    save_image(
        make_grid(
            predictions_to_show,
            nrow=4,
        ),
        OUTPUT_DIR / "predictions.png",
    )

    print()
    print("Overfit test completed.")
    print("Outputs:", OUTPUT_DIR)


if __name__ == "__main__":
    main()