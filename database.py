import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class PeriodontitisPatchDataset(Dataset):
    def __init__(self, patchesdir="processed_patches_stage4", img_size=256):
        self.patches_dir = patchesdir
        self.image_files = [
            f for f in os.listdir(patchesdir) 
            if f.endswith(('.jpg', '.png', '.jpeg'))
        ]
        
        # Scaling a [-1, 1]
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5), 
            transforms.ToTensor(),                 
            transforms.Normalize([0.5], [0.5])    
        ])

    def __len__(self):
        return len(self.image_files)
    def __getitem__(self, idx):
        img_path = os.path.join(self.patches_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB") #conversione RGB per Unet
        image_tensor = self.transform(image)
        return image_tensor