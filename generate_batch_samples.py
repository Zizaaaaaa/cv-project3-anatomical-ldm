import argparse
import os
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler

def generate_batch(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--> Dispositivo in uso: {device}")

    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint non trovato in: {args.checkpoint_path}. Addestra prima il modello o verifica il percorso.")

    print(f"--> Caricamento pesi addestrati da: {args.checkpoint_path}")
    
    vae = AutoencoderKL.from_pretrained(args.vae_pretrained, subfolder="vae").to(device)
    unet = UNet2DConditionModel.from_pretrained(args.checkpoint_path).to(device)
    
    vae.eval()
    unet.eval()

    scheduler = DDPMScheduler(num_train_timesteps=1000)

    # Identificazione di TUTTE le maschere (train e val) per avere più variabilità
    all_mask_files = []
    for split in ["train", "val"]:
        mdir = os.path.join(args.dataset_path, split, "masks")
        if os.path.exists(mdir):
            for f in os.listdir(mdir):
                if f.endswith('.png'):
                    all_mask_files.append(os.path.join(mdir, f))
                    
    if not all_mask_files:
        raise FileNotFoundError(f"Nessuna maschera trovata in {args.dataset_path}")

    os.makedirs(args.output_dir, exist_ok=True)

    transform_mask = T.Compose([T.ToTensor()])
    dummy_encoder_hidden_states = torch.zeros((1, 1, 768), device=device)

    import random
    print(f"--> Trovate {len(all_mask_files)} maschere reali totali.")
    print(f"--> Inizio generazione di {args.num_samples} radiografie sintetiche uniche...")

    for idx in range(args.num_samples):
        # Scegli una maschera a caso per ogni generazione
        mask_path = random.choice(all_mask_files)
        fname = os.path.basename(mask_path)
        
        mask_img = Image.open(mask_path).convert("L")
        
        # Aggiungiamo una piccola data augmentation anche in inferenza sulla maschera? 
        # (Opzionale: per ora teniamo la maschera reale per non alterare l'anatomia)
        mask_tensor = transform_mask(mask_img).unsqueeze(0).to(device)
        mask_32 = F.interpolate(mask_tensor, size=(32, 32), mode="nearest")

        # Rumore sempre nuovo!
        latents = torch.randn((1, 4, 32, 32), device=device)
        scheduler.set_timesteps(args.num_inference_steps)

        with torch.no_grad():
            for t in scheduler.timesteps:
                unet_input = torch.cat([latents, mask_32], dim=1)
                noise_pred = unet(unet_input, t, encoder_hidden_states=dummy_encoder_hidden_states).sample
                latents = scheduler.step(noise_pred, t, latents).prev_sample

            decoded = vae.decode(latents / 0.18215).sample
            decoded = (decoded / 2 + 0.5).clamp(0, 1)

        save_img = T.ToPILImage()(decoded.squeeze(0).cpu())
        
        # Nome univoco per non sovrascrivere!
        save_path = os.path.join(args.output_dir, f"synthetic_{idx:03d}_{fname}")
        save_img.save(save_path)

        print(f"[{idx+1}/{args.num_samples}] Salvata: {save_path}")

    print(f"\n Processo completato! Immagini salvate in: {args.output_dir}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generazione in batch di radiografie sintetiche tramite Guided LDM")
    parser.add_argument("--num_samples", type=int, default=20, help="Numero di immagini da generare")
    parser.add_argument("--checkpoint_path", type=str, default="outputs/guided_ldm_checkpoint", help="Percorso dei pesi U-Net addestrati")
    parser.add_argument("--dataset_path", type=str, default="processed_patches_severe_guided", help="Percorso del dataset contenente le maschere")
    parser.add_argument("--output_dir", type=str, default="outputs/generated_samples", help="Directory di destinazione delle immagini generate")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Numero di step di denoising")
    parser.add_argument("--vae_pretrained", type=str, default="CompVis/stable-diffusion-v1-4", help="Modello VAE HuggingFace")

    args = parser.parse_args()
    generate_batch(args)