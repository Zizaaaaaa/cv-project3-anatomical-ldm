from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from segmentation_dataset import ToothSegmentationDataset
from tooth_segmenter import (
    SmallUNet,
    dice_score,
    soft_dice_loss,
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

MANIFEST = Path(
    "outputs/anatomical_guidance/segmentation_manifest.csv"
)

OUTPUT_DIR = Path(
    "outputs/anatomical_guidance/tooth_segmenter"
)

CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
SAMPLE_DIR = OUTPUT_DIR / "samples"

IMAGE_SIZE = 256

BATCH_SIZE = 8
LEARNING_RATE = 3e-4

SEED = 42


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------

@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:

    model.eval()

    loss_sum = 0.0
    dice_sum = 0.0
    total_samples = 0

    for batch in loader:

        images = batch["image"].to(
            device,
            non_blocking=True,
        )

        masks = batch["mask"].to(
            device,
            non_blocking=True,
        )

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

        dice = dice_score(
            logits,
            masks,
        )

        batch_size = images.shape[0]

        loss_sum += (
            loss.item() * batch_size
        )

        dice_sum += (
            dice.item() * batch_size
        )

        total_samples += batch_size

    return (
        loss_sum / total_samples,
        dice_sum / total_samples,
    )


# ---------------------------------------------------------------------
# Prediction visualization
# ---------------------------------------------------------------------

@torch.no_grad()
def save_validation_examples(
    model: torch.nn.Module,
    dataset: ToothSegmentationDataset,
    device: torch.device,
    epoch: int,
    num_examples: int = 8,
) -> None:

    model.eval()

    images = []
    ground_truths = []
    predictions = []

    count = min(
        num_examples,
        len(dataset),
    )

    for index in range(count):

        sample = dataset[index]

        image = sample["image"]
        mask = sample["mask"]

        logits = model(
            image
            .unsqueeze(0)
            .to(device)
        )

        probability = torch.sigmoid(
            logits
        )

        prediction = (
            probability >= 0.5
        ).float()

        images.append(image)

        ground_truths.append(
            mask.repeat(3, 1, 1)
        )

        predictions.append(
            prediction
            .squeeze(0)
            .repeat(3, 1, 1)
            .cpu()
        )

    SAMPLE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_image(
        make_grid(
            images,
            nrow=4,
        ),
        SAMPLE_DIR
        / f"epoch_{epoch:03d}_images.png",
    )

    save_image(
        make_grid(
            ground_truths,
            nrow=4,
        ),
        SAMPLE_DIR
        / f"epoch_{epoch:03d}_ground_truth.png",
    )

    save_image(
        make_grid(
            predictions,
            nrow=4,
        ),
        SAMPLE_DIR
        / f"epoch_{epoch:03d}_predictions.png",
    )


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------

def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
    )

    args = parser.parse_args()

    set_seed(SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")

    # -------------------------------------------------------------
    # Datasets
    # -------------------------------------------------------------

    train_dataset = ToothSegmentationDataset(
        manifest_path=MANIFEST,
        role="segmenter_train",
        image_size=IMAGE_SIZE,
        augment=True,
    )

    val_dataset = ToothSegmentationDataset(
        manifest_path=MANIFEST,
        role="segmenter_val",
        image_size=IMAGE_SIZE,
        augment=False,
    )

    print(
        f"Training images:   {len(train_dataset)}"
    )

    print(
        f"Validation images: {len(val_dataset)}"
    )

    # -------------------------------------------------------------
    # DataLoaders
    # -------------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )

    # -------------------------------------------------------------
    # Model
    # -------------------------------------------------------------

    model = SmallUNet().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )

    # -------------------------------------------------------------
    # Output folders
    # -------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    SAMPLE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    losses_path = (
        OUTPUT_DIR / "training_log.csv"
    )

    with losses_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.writer(handle)

        writer.writerow(
            [
                "epoch",
                "train_loss",
                "train_dice",
                "val_loss",
                "val_dice",
            ]
        )

    best_val_dice = -math.inf

    print()
    print(
        f"Starting training for "
        f"{args.epochs} epochs..."
    )

    # -------------------------------------------------------------
    # Epoch loop
    # -------------------------------------------------------------

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        model.train()

        train_loss_sum = 0.0
        train_dice_sum = 0.0
        train_samples = 0

        progress = tqdm(
            train_loader,
            desc=(
                f"Epoch "
                f"{epoch}/{args.epochs}"
            ),
        )

        for batch in progress:

            images = batch["image"].to(
                device,
                non_blocking=True,
            )

            masks = batch["mask"].to(
                device,
                non_blocking=True,
            )

            logits = model(images)

            bce = (
                F.binary_cross_entropy_with_logits(
                    logits,
                    masks,
                )
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

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            batch_size = images.shape[0]

            with torch.no_grad():

                dice = dice_score(
                    logits,
                    masks,
                )

            train_loss_sum += (
                loss.item()
                * batch_size
            )

            train_dice_sum += (
                dice.item()
                * batch_size
            )

            train_samples += batch_size

            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                dice=f"{dice.item():.4f}",
            )

        train_loss = (
            train_loss_sum
            / train_samples
        )

        train_dice = (
            train_dice_sum
            / train_samples
        )

        # ---------------------------------------------------------
        # Validation
        # ---------------------------------------------------------

        val_loss, val_dice = validate(
            model=model,
            loader=val_loader,
            device=device,
        )

        print(
            f"Epoch {epoch:03d} | "
            f"train loss={train_loss:.4f} | "
            f"train dice={train_dice:.4f} | "
            f"val loss={val_loss:.4f} | "
            f"val dice={val_dice:.4f}"
        )

        # ---------------------------------------------------------
        # CSV logging
        # ---------------------------------------------------------

        with losses_path.open(
            "a",
            newline="",
            encoding="utf-8",
        ) as handle:

            writer = csv.writer(handle)

            writer.writerow(
                [
                    epoch,
                    f"{train_loss:.8f}",
                    f"{train_dice:.8f}",
                    f"{val_loss:.8f}",
                    f"{val_dice:.8f}",
                ]
            )

        # ---------------------------------------------------------
        # Best model
        # ---------------------------------------------------------

        if val_dice > best_val_dice:

            best_val_dice = val_dice

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "epoch":
                        epoch,

                    "val_dice":
                        val_dice,
                },
                OUTPUT_DIR / "best_model.pt",
            )

        # ---------------------------------------------------------
        # Periodic diagnostic predictions
        # ---------------------------------------------------------

        if (
            epoch == 1
            or epoch % 5 == 0
            or epoch == args.epochs
        ):

            save_validation_examples(
                model=model,
                dataset=val_dataset,
                device=device,
                epoch=epoch,
            )

        # ---------------------------------------------------------
        # Periodic checkpoint
        # ---------------------------------------------------------

        if epoch % 10 == 0:

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "optimizer_state_dict":
                        optimizer.state_dict(),

                    "epoch":
                        epoch,
                },
                CHECKPOINT_DIR
                / f"epoch_{epoch:03d}.pt",
            )

    print()
    print("Training complete.")

    print(
        f"Best validation Dice: "
        f"{best_val_dice:.4f}"
    )

    print(
        f"Outputs: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()