import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as T
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler

# --- 1. Dataset Class ---
class GuidedSeverePatchDataset(Dataset):
    def __init__(self, root_dir, split="train"):
        self.img_dir = os.path.join(root_dir, split, "images")
        self.mask_dir = os.path.join(root_dir, split, "masks")
        
        if not os.path.exists(self.img_dir):
            raise FileNotFoundError(f"Directory images not found: {self.img_dir}")
        if not os.path.exists(self.mask_dir):
            raise FileNotFoundError(f"Directory masks not found: {self.mask_dir}")
            
        self.filenames = sorted([f for f in os.listdir(self.img_dir) if f.endswith('.png')])

        self.img_transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.5], [0.5])
        ])

        self.mask_transform = T.Compose([
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        img_path = os.path.join(self.img_dir, fname)
        mask_path = os.path.join(self.mask_dir, fname)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        img_tensor = self.img_transform(image)    # (3, 256, 256)
        mask_tensor = self.mask_transform(mask)  # (1, 256, 256)

        return img_tensor, mask_tensor

# --- 2. Training Function ---
def train_guided_ldm(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--> Device in use: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    train_dataset = GuidedSeverePatchDataset(args.dataset_path, split="train")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    print("--> Uploading pre-trained VAE...")
    vae = AutoencoderKL.from_pretrained(args.vae_pretrained, subfolder="vae")
    vae.eval()
    vae.requires_grad_(False)
    vae.to(device)

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    print("--> Beginning U-Net (5 channels of input: 4 latents + 1 mask)...")
    unet = UNet2DConditionModel(
        sample_size=32,
        in_channels=5,
        out_channels=4,
        layers_per_block=2,
        block_out_channels=(128, 256, 512, 512),
        down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
        cross_attention_dim=768 # Dummy encoder state dimension
    )
    unet.to(device)

    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.lr)

    print(f"--> Beginning Guided LDM Training ({args.epochs} epochs)...")
    unet.train()

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for step, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)

            with torch.no_grad():
                latents = vae.encode(images).latent_dist.sample() * 0.18215

            masks_32 = F.interpolate(masks, size=(32, 32), mode="nearest")

            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            unet_input = torch.cat([noisy_latents, masks_32], dim=1)

            dummy_encoder_hidden_states = torch.zeros(
                (latents.shape[0], 1, 768), device=device, dtype=unet.dtype
            )

            noise_pred = unet(unet_input, timesteps, encoder_hidden_states=dummy_encoder_hidden_states).sample
            loss = F.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoca [{epoch+1}/{args.epochs}] - Loss: {avg_loss:.6f}")

    unet.save_pretrained(args.output_dir)
    print(f"--> Model saved with success in: {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Guided LDM Training for Dental Patches")
    parser.add_argument("--dataset_path", type=str, default="processed_patches_severe_guided", help="Path to the guided dataset")
    parser.add_argument("--output_dir", type=str, default="outputs/guided_ldm_checkpoint", help="Local directory for saving the model")
    parser.add_argument("--epochs", type=int, default=80, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Dimension of batch")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--vae_pretrained", type=str, default="CompVis/stable-diffusion-v1-4", help="Pretrained VAE model")

    args = parser.parse_args()
    train_guided_ldm(args)