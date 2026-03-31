import os
import glob
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.utils import save_image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from diffusion import DiffusionReverseProcess
from unet import Unet

# 32x32 is much better for U-Nets than 28x28 apparently?
transform = transforms.Compose([
    transforms.Resize((32, 32)), 
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
channels = 1
num_steps = 1000

fashion_ds = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
fashion_dl = DataLoader(fashion_ds, batch_size=64, shuffle=True) # Batch size 64 is safe for 6GB VRAM

# 2. Initialize the Model and the Reverse Process
model = Unet(im_channels=channels, t_emb_dim=128).to(device)
reverse_process = DiffusionReverseProcess(num_time_steps=num_steps).to(device)

checkpoint_files = glob.glob("fashion_unet_epoch_*.pth")
checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0])) 

if os.path.exists("fashion_unet_final.pth"):
    checkpoint_files.append("fashion_unet_final.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
channels = 1 
num_steps = 1000
num_samples_for_fid = 5000

fid = FrechetInceptionDistance(feature=2048).to(device)

print("Extracting features from Real Images...")
real_images, _ = next(iter(fashion_dl)) 
real_images = real_images[:num_samples_for_fid].to(device)

# FashionMNIST is 1-channel. Inception requires 3 channels. We duplicate the channels.
real_images_rgb = real_images.repeat(1, 3, 1, 1)

# FID expects images in bytes [0, 255] format 
real_images_bytes = ((real_images_rgb + 1.0) / 2.0 * 255).to(torch.uint8)
fid.update(real_images_bytes, real=True)

model = Unet(im_channels=channels, t_emb_dim=128).to(device)
reverse_process = DiffusionReverseProcess(num_time_steps=num_steps).to(device)

checkpoint_files = glob.glob("./ddpm/jashandeep/checkpoints_FMNIST/fashion_unet_epoch_*.pth")
checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
# FILTER: Keep only every 2th checkpoint
checkpoint_files = checkpoint_files[::2] 

if os.path.exists("./ddpm/jashandeep/checkpoints_FMNIST/fashion_unet_final.pth"):
    checkpoint_files.append("./ddpm/jashandeep/checkpoints_FMNIST/fashion_unet_final.pth")

fid_scores = {}

for ckpt_path in checkpoint_files:
    print(f"\nEvaluating {ckpt_path}...")
    
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval() 
    
    with torch.no_grad():
        chunk_size = 500 
        num_chunks = num_samples_for_fid // chunk_size
        
        print(f"Generating {num_samples_for_fid} samples in {num_chunks} chunks...")
        
        for chunk in range(num_chunks):
            xt = torch.randn(chunk_size, channels, 32, 32).to(device)
            for i in reversed(range(num_steps)):
                t = torch.full((chunk_size,), i, dtype=torch.long, device=device)
                noise_pred = model(xt, t)
                xt, _ = reverse_process.sample_prev_timestep(xt, noise_pred, t)
            fake_images_rgb = xt.repeat(1, 3, 1, 1)
            fake_images_bytes = ((fake_images_rgb.clamp(-1, 1) + 1.0) / 2.0 * 255).to(torch.uint8)
            fid.update(fake_images_bytes, real=False)
            torch.cuda.empty_cache() 
            print(f"  - Chunk {chunk + 1}/{num_chunks} complete.")
        score = fid.compute()
        print(f"FID Score for {ckpt_path}: {score.item():.2f}")
        fid_scores[ckpt_path] = score.item()
        fid.reset()
        fid.update(real_images_bytes, real=True)

