import os
import glob
import torch
import torch
from torchvision import transforms
from torchvision.utils import save_image

from diffusion import DiffusionReverseProcess
from unet import Unet

# 32x32 is much better for U-Nets than 28x28 apparently?
transform = transforms.Compose([
    transforms.Resize((32, 32)), 
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 1. Setup Device and Hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
channels = 1 # FashionMNIST is grayscale, so 1 channel
num_steps = 1000

# 2. Initialize the Model and the Reverse Process
model = Unet(im_channels=channels, t_emb_dim=128).to(device)
reverse_process = DiffusionReverseProcess(num_time_steps=num_steps).to(device)

# 3. Get a sorted list of all your epoch checkpoints
checkpoint_files = glob.glob("./ddpm/jashandeep/checkpoints_FMNIST/fashion_unet_epoch_*.pth")
checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0])) 

if os.path.exists("./ddpm/jashandeep/checkpoints_FMNIST/fashion_unet_final.pth"):
    checkpoint_files.append("./ddpm/jashandeep/checkpoints_FMNIST/fashion_unet_final.pth")

# 4. Loop through each checkpoint and generate
for ckpt_path in checkpoint_files:
    print(f"Loading weights from {ckpt_path}...")
    
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval() 
    
    with torch.no_grad():
        # A. Start with pure Gaussian noise (let's generate a batch of 16 images)
        batch_size = 16
        xt = torch.randn(batch_size, channels, 32, 32).to(device)
        
        # B. Gradually denoise the image (from t=999 down to t=0)
        for i in reversed(range(num_steps)):
            t = torch.full((batch_size,), i, dtype=torch.long, device=device)
            
            # The UNet predicts the noise added at this time step
            noise_pred = model(xt, t)
            
            # Step backwards using your Reverse Process math
            xt, _ = reverse_process.sample_prev_timestep(xt, noise_pred, t)
            
        # C. Denormalize images from [-1, 1] back to [0, 1] so they save correctly
        final_images = (xt.clamp(-1, 1) + 1.0) / 2.0
        
        # D. Save the 16 images as a 4x4 grid
        save_name = f"sample_{os.path.basename(ckpt_path).replace('.pth', '.png')}"
        save_image(final_images, save_name, nrow=4)
        print(f"Saved {save_name}\n")