from __future__ import annotations

import csv
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from diffusers import AutoencoderKL, DDPMScheduler, UNet2DModel


TRAIN_DIR = Path("processed_patches_severe/train")
VAL_DIR = Path("processed_patches_severe/val")

OUTPUT_DIR = Path("outputs/ldm_baseline_v2")
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
SAMPLE_DIR = OUTPUT_DIR / "samples"

IMAGE_SIZE = 256
BATCH_SIZE = 4
NUM_EPOCHS = 200
LEARNING_RATE = 1e-4
NUM_TRAIN_TIMESTEPS = 1000
NUM_INFERENCE_STEPS = 250

SAVE_EVERY = 25
SAMPLE_EVERY = 25
NUM_SAMPLE_IMAGES = 4

SEED = 42
VAE_MODEL_ID = "stabilityai/sd-vae-ft-mse"


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class DentalPatchDataset(Dataset):
    def __init__(self, data_dir: Path, train: bool) -> None:
        self.image_files = sorted(
            p for p in data_dir.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )

        steps = [
            transforms.Resize(
                (IMAGE_SIZE, IMAGE_SIZE),
                interpolation=transforms.InterpolationMode.BICUBIC,
            )
        ]

        if train:
            # Mild geometric augmentation: enough to increase variation
            # without changing the clinical content too aggressively.
            steps += [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(
                    degrees=3,
                    translate=(0.03, 0.03),
                    scale=(0.97, 1.03),
                    fill=0,
                ),
            ]

        steps += [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
            ),
        ]

        self.transform = transforms.Compose(steps)

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, index: int) -> torch.Tensor:
        with Image.open(self.image_files[index]) as image:
            return self.transform(image.convert("RGB"))


def build_unet(latent_size: int, latent_channels: int) -> UNet2DModel:
    # Smaller than v1: the real training set contains only 83 patches.
    return UNet2DModel(
        sample_size=latent_size,
        in_channels=latent_channels,
        out_channels=latent_channels,
        layers_per_block=2,
        block_out_channels=(64, 128, 256, 256),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )


@torch.no_grad()
def encode_train(images, vae, scaling_factor):
    return vae.encode(images).latent_dist.sample() * scaling_factor


@torch.no_grad()
def encode_eval(images, vae, scaling_factor):
    # Deterministic encoding makes validation losses comparable across epochs.
    return vae.encode(images).latent_dist.mode() * scaling_factor


@torch.no_grad()
def decode_latents(latents, vae, scaling_factor):
    return vae.decode(latents / scaling_factor).sample.clamp(-1, 1)


@torch.no_grad()
def generate_samples(
    model,
    vae,
    scheduler,
    device,
    latent_shape,
    scaling_factor,
    epoch,
):
    model.eval()

    c, h, w = latent_shape
    generator = torch.Generator(device=device).manual_seed(SEED + 1000 + epoch)

    latents = torch.randn(
        (NUM_SAMPLE_IMAGES, c, h, w),
        generator=generator,
        device=device,
    )

    scheduler.set_timesteps(NUM_INFERENCE_STEPS, device=device)

    for t in tqdm(
        scheduler.timesteps,
        desc=f"Sampling epoch {epoch}",
        leave=False,
    ):
        t_batch = torch.full(
            (latents.shape[0],),
            int(t.item()),
            device=device,
            dtype=torch.long,
        )
        noise_pred = model(latents, t_batch, return_dict=False)[0]
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    images = decode_latents(latents, vae, scaling_factor)
    images = (images + 1.0) / 2.0

    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    save_image(
        make_grid(images.cpu(), nrow=2),
        SAMPLE_DIR / f"epoch_{epoch:03d}.png",
    )

    model.train()


def validate(
    model,
    val_loader,
    vae,
    scheduler,
    scaling_factor,
    device,
):
    model.eval()

    total_loss = 0.0
    total_examples = 0

    # Use the same validation noise and timesteps every epoch.
    generator = torch.Generator(device=device).manual_seed(SEED + 5000)

    with torch.no_grad():
        for clean_images in val_loader:
            clean_images = clean_images.to(device, non_blocking=True)
            latents = encode_eval(clean_images, vae, scaling_factor)

            noise = torch.randn(
                latents.shape,
                generator=generator,
                device=device,
                dtype=latents.dtype,
            )

            batch_size = latents.shape[0]
            timesteps = torch.randint(
                0,
                scheduler.config.num_train_timesteps,
                (batch_size,),
                generator=generator,
                device=device,
                dtype=torch.long,
            )

            noisy_latents = scheduler.add_noise(latents, noise, timesteps)
            noise_pred = model(noisy_latents, timesteps, return_dict=False)[0]
            loss = F.mse_loss(noise_pred, noise)

            total_loss += loss.item() * batch_size
            total_examples += batch_size

    return total_loss / total_examples


def main() -> None:
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not TRAIN_DIR.exists() or not VAL_DIR.exists():
        raise FileNotFoundError(
            "processed_patches_severe/train or val not found. "
            "Run data_pipeline/prepare_data.py first."
        )

    train_dataset = DentalPatchDataset(TRAIN_DIR, train=True)
    val_dataset = DentalPatchDataset(VAL_DIR, train=False)

    print(f"Device: {device}")
    print(f"Training patches: {len(train_dataset)}")
    print(f"Validation patches: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )

    print(f"Loading VAE: {VAE_MODEL_ID}")
    vae = AutoencoderKL.from_pretrained(VAE_MODEL_ID).to(device)
    vae.eval()

    for parameter in vae.parameters():
        parameter.requires_grad = False

    scaling_factor = float(vae.config.scaling_factor)
    latent_channels = int(vae.config.latent_channels)

    with torch.no_grad():
        dummy = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
        dummy_latent = vae.encode(dummy).latent_dist.mode()

    latent_h, latent_w = dummy_latent.shape[-2:]
    if latent_h != latent_w:
        raise RuntimeError(f"Expected square latent, got {latent_h}x{latent_w}")

    print(f"Latent shape: {latent_channels} x {latent_h} x {latent_w}")

    model = build_unet(latent_h, latent_channels).to(device)

    scheduler = DDPMScheduler(
        num_train_timesteps=NUM_TRAIN_TIMESTEPS,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon",
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    losses_path = OUTPUT_DIR / "losses.csv"
    with losses_path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss"])

    best_val_loss = math.inf

    print("\nStarting LDM baseline v2 training...")

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        train_loss_sum = 0.0
        train_examples = 0

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{NUM_EPOCHS}",
        )

        for clean_images in progress:
            clean_images = clean_images.to(device, non_blocking=True)

            with torch.no_grad():
                latents = encode_train(clean_images, vae, scaling_factor)

            noise = torch.randn_like(latents)

            batch_size = latents.shape[0]
            timesteps = torch.randint(
                0,
                scheduler.config.num_train_timesteps,
                (batch_size,),
                device=device,
                dtype=torch.long,
            )

            noisy_latents = scheduler.add_noise(latents, noise, timesteps)
            noise_pred = model(noisy_latents, timesteps, return_dict=False)[0]

            loss = F.mse_loss(noise_pred, noise)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss_sum += loss.item() * batch_size
            train_examples += batch_size
            progress.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = train_loss_sum / train_examples
        val_loss = validate(
            model,
            val_loader,
            vae,
            scheduler,
            scaling_factor,
            device,
        )

        print(
            f"Epoch {epoch:03d} | "
            f"train loss: {train_loss:.6f} | "
            f"val loss: {val_loss:.6f}"
        )

        with losses_path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [epoch, f"{train_loss:.8f}", f"{val_loss:.8f}"]
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_pretrained(OUTPUT_DIR / "best_unet")

        if epoch % SAVE_EVERY == 0:
            model.save_pretrained(CHECKPOINT_DIR / f"epoch_{epoch:03d}")

        if epoch % SAMPLE_EVERY == 0:
            generate_samples(
                model=model,
                vae=vae,
                scheduler=scheduler,
                device=device,
                latent_shape=(latent_channels, latent_h, latent_w),
                scaling_factor=scaling_factor,
                epoch=epoch,
            )

    model.save_pretrained(OUTPUT_DIR / "final_unet")

    print("\nTraining complete.")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()