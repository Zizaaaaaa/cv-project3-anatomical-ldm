from __future__ import annotations

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


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

TRAIN_DIR = Path("processed_patches_severe/train")
VAL_DIR = Path("processed_patches_severe/val")

OUTPUT_DIR = Path("outputs/ldm_baseline")
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
SAMPLE_DIR = OUTPUT_DIR / "samples"

IMAGE_SIZE = 256
BATCH_SIZE = 4
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
NUM_TRAIN_TIMESTEPS = 1000
NUM_INFERENCE_STEPS = 250

SAVE_EVERY = 10
SAMPLE_EVERY = 10
NUM_SAMPLE_IMAGES = 4

SEED = 42

# A pretrained VAE gives us a stable image <-> latent mapping.
# The diffusion U-Net itself is still trained from scratch on our dental patches.
VAE_MODEL_ID = "stabilityai/sd-vae-ft-mse"


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------

class DentalPatchDataset(Dataset):
    def __init__(self, data_dir: Path, image_size: int = 256) -> None:
        self.data_dir = data_dir
        self.image_files = sorted(
            path
            for path in data_dir.iterdir()
            if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )

        # The VAE expects RGB images normalized to [-1, 1].
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5],
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, index: int) -> torch.Tensor:
        image_path = self.image_files[index]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            return self.transform(image)


# ---------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------

def build_unet(latent_size: int, latent_channels: int) -> UNet2DModel:
    """
    Small unconditional U-Net for latent diffusion.

    The important difference from the previous DDPM is that this network
    operates on VAE latents, not directly on 256x256 RGB pixels.
    """
    return UNet2DModel(
        sample_size=latent_size,
        in_channels=latent_channels,
        out_channels=latent_channels,
        layers_per_block=2,
        block_out_channels=(128, 256, 256, 512),
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
def encode_images(
    images: torch.Tensor,
    vae: AutoencoderKL,
    scaling_factor: float,
) -> torch.Tensor:
    posterior = vae.encode(images).latent_dist
    latents = posterior.sample()
    return latents * scaling_factor


@torch.no_grad()
def decode_latents(
    latents: torch.Tensor,
    vae: AutoencoderKL,
    scaling_factor: float,
) -> torch.Tensor:
    images = vae.decode(latents / scaling_factor).sample
    return images.clamp(-1, 1)


@torch.no_grad()
def generate_samples(
    model: UNet2DModel,
    vae: AutoencoderKL,
    noise_scheduler: DDPMScheduler,
    device: torch.device,
    latent_shape: tuple[int, int, int],
    scaling_factor: float,
    epoch: int,
) -> None:
    model.eval()

    channels, height, width = latent_shape

    generator = torch.Generator(device=device)
    generator.manual_seed(SEED + epoch)

    latents = torch.randn(
        (NUM_SAMPLE_IMAGES, channels, height, width),
        generator=generator,
        device=device,
    )

    noise_scheduler.set_timesteps(NUM_INFERENCE_STEPS, device=device)

    for timestep in tqdm(
        noise_scheduler.timesteps,
        desc=f"Sampling epoch {epoch}",
        leave=False,
    ):
        timestep_batch = torch.full(
            (latents.shape[0],),
            int(timestep.item()),
            device=device,
            dtype=torch.long,
        )

        noise_pred = model(
            latents,
            timestep_batch,
            return_dict=False,
        )[0]

        latents = noise_scheduler.step(
            noise_pred,
            timestep,
            latents,
        ).prev_sample

    decoded = decode_latents(latents, vae, scaling_factor)
    decoded = (decoded + 1.0) / 2.0

    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    grid = make_grid(decoded.cpu(), nrow=2)
    save_image(grid, SAMPLE_DIR / f"epoch_{epoch:03d}.png")

    model.train()


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------

def main() -> None:
    set_seed(SEED)

    if not TRAIN_DIR.exists():
        raise FileNotFoundError(
            f"Training directory not found: {TRAIN_DIR.resolve()}"
        )

    if not VAL_DIR.exists():
        raise FileNotFoundError(
            f"Validation directory not found: {VAL_DIR.resolve()}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"Train directory: {TRAIN_DIR.resolve()}")
    print(f"Validation directory: {VAL_DIR.resolve()}")

    train_dataset = DentalPatchDataset(TRAIN_DIR, IMAGE_SIZE)
    val_dataset = DentalPatchDataset(VAL_DIR, IMAGE_SIZE)

    if len(train_dataset) == 0:
        raise RuntimeError("No training patches found.")

    if len(val_dataset) == 0:
        raise RuntimeError("No validation patches found.")

    print(f"Training patches: {len(train_dataset)}")
    print(f"Validation patches: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    print(f"\nLoading VAE: {VAE_MODEL_ID}")
    vae = AutoencoderKL.from_pretrained(VAE_MODEL_ID)
    vae.to(device)
    vae.eval()

    # We are not training the VAE in this baseline.
    for parameter in vae.parameters():
        parameter.requires_grad = False

    scaling_factor = float(vae.config.scaling_factor)
    latent_channels = int(vae.config.latent_channels)

    # Stable-Diffusion-style VAEs downsample spatial dimensions by 8.
    # We infer the actual latent size once instead of hard-coding 32x32.
    with torch.no_grad():
        dummy = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
        dummy_latent = vae.encode(dummy).latent_dist.mode()

    latent_height = dummy_latent.shape[-2]
    latent_width = dummy_latent.shape[-1]

    if latent_height != latent_width:
        raise RuntimeError(
            f"Expected square latents, got {latent_height}x{latent_width}."
        )

    latent_size = latent_height

    print(
        f"Latent shape: "
        f"{latent_channels} x {latent_height} x {latent_width}"
    )
    print(f"VAE scaling factor: {scaling_factor}")

    model = build_unet(latent_size, latent_channels).to(device)

    noise_scheduler = DDPMScheduler(
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

    best_val_loss = math.inf

    print("\nStarting latent diffusion training...")

    for epoch in range(1, NUM_EPOCHS + 1):
        # -------------------------
        # Training
        # -------------------------
        model.train()
        train_loss_sum = 0.0
        train_examples = 0

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{NUM_EPOCHS}",
        )

        for clean_images in progress:
            clean_images = clean_images.to(
                device,
                non_blocking=True,
            )

            with torch.no_grad():
                latents = encode_images(
                    clean_images,
                    vae,
                    scaling_factor,
                )

            noise = torch.randn_like(latents)

            batch_size = latents.shape[0]
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (batch_size,),
                device=device,
                dtype=torch.long,
            )

            noisy_latents = noise_scheduler.add_noise(
                latents,
                noise,
                timesteps,
            )

            noise_pred = model(
                noisy_latents,
                timesteps,
                return_dict=False,
            )[0]

            loss = F.mse_loss(noise_pred, noise)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss_sum += loss.item() * batch_size
            train_examples += batch_size

            progress.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = train_loss_sum / train_examples

        # -------------------------
        # Validation
        # -------------------------
        model.eval()
        val_loss_sum = 0.0
        val_examples = 0

        with torch.no_grad():
            for clean_images in val_loader:
                clean_images = clean_images.to(
                    device,
                    non_blocking=True,
                )

                latents = encode_images(
                    clean_images,
                    vae,
                    scaling_factor,
                )

                noise = torch.randn_like(latents)

                batch_size = latents.shape[0]
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (batch_size,),
                    device=device,
                    dtype=torch.long,
                )

                noisy_latents = noise_scheduler.add_noise(
                    latents,
                    noise,
                    timesteps,
                )

                noise_pred = model(
                    noisy_latents,
                    timesteps,
                    return_dict=False,
                )[0]

                loss = F.mse_loss(noise_pred, noise)

                val_loss_sum += loss.item() * batch_size
                val_examples += batch_size

        val_loss = val_loss_sum / val_examples

        print(
            f"Epoch {epoch:03d} | "
            f"train loss: {train_loss:.6f} | "
            f"val loss: {val_loss:.6f}"
        )

        # Keep a compact text log that can later be plotted for the report.
        with (OUTPUT_DIR / "losses.csv").open(
            "a",
            encoding="utf-8",
        ) as log_file:
            if epoch == 1:
                log_file.write("epoch,train_loss,val_loss\n")
            log_file.write(
                f"{epoch},{train_loss:.8f},{val_loss:.8f}\n"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_dir = OUTPUT_DIR / "best_unet"
            model.save_pretrained(best_dir)

        if epoch % SAVE_EVERY == 0:
            checkpoint_dir = CHECKPOINT_DIR / f"epoch_{epoch:03d}"
            model.save_pretrained(checkpoint_dir)

        if epoch % SAMPLE_EVERY == 0:
            generate_samples(
                model=model,
                vae=vae,
                noise_scheduler=noise_scheduler,
                device=device,
                latent_shape=(
                    latent_channels,
                    latent_height,
                    latent_width,
                ),
                scaling_factor=scaling_factor,
                epoch=epoch,
            )

    final_dir = OUTPUT_DIR / "final_unet"
    model.save_pretrained(final_dir)

    print("\nTraining complete.")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Final model: {final_dir}")
    print(f"Generated samples: {SAMPLE_DIR}")


if __name__ == "__main__":
    main()