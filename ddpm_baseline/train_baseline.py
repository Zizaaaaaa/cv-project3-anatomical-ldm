import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from diffusers import UNet2DModel, DDPMScheduler
from tqdm import tqdm


class DentalPatchesDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.image_files = [f for f in os.listdir(data_dir) if f.endswith('.png')]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir, self.image_files[idx])
        image = Image.open(img_name).convert('RGB')       
        if self.transform:
            image = self.transform(image)
            
        return image



def main():
    data_dir = "processed_patches_stage4"
    batch_size = 4
    num_epochs = 50
    learning_rate = 1e-4
    image_size = 256
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # selenziona l'hardware
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    dataset = DentalPatchesDataset(data_dir, transform=transform)
    if len(dataset) == 0:
        print(f"No images found in {data_dir}.")
        return
        
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    model = UNet2DModel(
        sample_size=image_size,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(64, 128, 256, 512), 
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
    model.to(device)
    
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    print(f"Starting training on {device}...")
    
    # Training Loop
    # DDPM training: the model is trained to predict the noise added at a random timestep.
    for epoch in range(num_epochs):
        model.train()
        progress_bar = tqdm(total=len(dataloader), desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for step, batch in enumerate(dataloader):
            clean_images = batch.to(device)
            # Sample noise and timestep
            noise = torch.randn(clean_images.shape).to(device)
            bs = clean_images.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bs,), device=device).long()
            
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
            
            # Predict the noise residual
            noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
            loss = F.mse_loss(noise_pred, noise)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            progress_bar.update(1)
            progress_bar.set_postfix(loss=loss.item())
            
        #save checkpoint every N epochs
        if (epoch + 1) % 10 == 0:
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), f"checkpoints/unet_epoch_{epoch+1}.pt")
            
    print("Training complete! Final model weights saved.")
    # Save the final model
    os.makedirs("models", exist_ok=True)
    model.save_pretrained("models/baseline_ddpm_unet")

if __name__ == "__main__":
    main()
