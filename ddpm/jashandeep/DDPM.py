import torch
from torchvision.utils import save_image
from torchvision import transforms
import matplotlib.pyplot as plt

from diffusion import DiffusionReverseProcess
from unet import Unet

def generate_images(num_images=16):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Generating on: {device}')
    
    num_timesteps = 1000
    img_size = 32
    channels = 1
    
    model = Unet(im_channels=channels, t_emb_dim=128).to(device)
    model.load_state_dict(torch.load("./ddpm/jashandeep/checkpoints_MNIST/monster_unet_epoch_105.pth", map_location=device))
    model.eval() # Tell the model it's in "testing" mode, not training
    
    # 3. Initialize the Reverse Math Process
    reverse_process = DiffusionReverseProcess(num_time_steps=num_timesteps).to(device)
    
    print("Starting generation...")

    with torch.no_grad():
        x = torch.randn(num_images, channels, img_size, img_size).to(device)
        
        # Step B: Loop backwards from t=999 down to t=0
        for i in reversed(range(num_timesteps)):
            # Create a tensor of the current time step for the whole batch
            t = torch.full((num_images,), i, dtype=torch.long, device=device)
            
            # Ask the U-Net to guess the noise
            predicted_noise = model(x, t)
            
            # Use our reverse math to subtract the noise and step backwards
            x, _ = reverse_process.sample_prev_timestep(x, predicted_noise, t)
            
            # Print a little progress update so we know it hasn't crashed
            if i % 100 == 0:
                print(f"Step {i} complete...")
                
    print("Generation complete! Plotting...")
    x = x.cpu()
    x = (x + 1.0) / 2.0 
    x = torch.clamp(x, 0.0, 1.0)
    
    # Create a 4x4 grid to show the 16 images
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(x[i].squeeze(), cmap='gray')
        ax.axis('off')
        
    plt.tight_layout()
    plt.show()
    
generate_images()