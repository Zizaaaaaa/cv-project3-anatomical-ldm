from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import make_grid, save_image

from diffusers import AutoencoderKL


DATA_DIR = Path("processed_patches_severe/val")
OUTPUT_PATH = Path("outputs/ldm_baseline/vae_reconstruction_check.png")

IMAGE_SIZE = 256
NUM_IMAGES = 6
VAE_MODEL_ID = "stabilityai/sd-vae-ft-mse"


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_files = sorted(
        path
        for path in DATA_DIR.iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )[:NUM_IMAGES]

    if not image_files:
        raise RuntimeError(f"No images found in {DATA_DIR.resolve()}")

    transform = transforms.Compose(
        [
            transforms.Resize(
                (IMAGE_SIZE, IMAGE_SIZE),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
            ),
        ]
    )

    originals = []

    for image_path in image_files:
        with Image.open(image_path) as image:
            originals.append(transform(image.convert("RGB")))

    originals = torch.stack(originals).to(device)

    print(f"Device: {device}")
    print(f"Images checked: {len(image_files)}")
    print(f"Loading VAE: {VAE_MODEL_ID}")

    vae = AutoencoderKL.from_pretrained(VAE_MODEL_ID).to(device)
    vae.eval()

    with torch.no_grad():
        # mode() is deterministic and is useful here because we only want to
        # inspect how much anatomical information the VAE preserves.
        latents = vae.encode(originals).latent_dist.mode()
        reconstructions = vae.decode(latents).sample

    originals = ((originals + 1.0) / 2.0).clamp(0, 1)
    reconstructions = ((reconstructions + 1.0) / 2.0).clamp(0, 1)

    # Layout:
    # first row(s) = real patches
    # following row(s) = their VAE reconstructions, in the same order.
    comparison = torch.cat([originals, reconstructions], dim=0).cpu()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    grid = make_grid(comparison, nrow=len(image_files), padding=2)
    save_image(grid, OUTPUT_PATH)

    print(f"Saved comparison to: {OUTPUT_PATH}")
    print("Top row: original patches")
    print("Bottom row: VAE reconstructions")


if __name__ == "__main__":
    main()