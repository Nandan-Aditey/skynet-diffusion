import torch
from torchvision import transforms
import matplotlib.pyplot as plt

from diffusion import DDIMReverseProcess
from unet import Unet

def generate_images_ddim(num_images=16, ddim_steps=50):
    # 1. Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Generating on: {device} using DDIM!')
    
    num_timesteps = 1000
    img_size = 32
    channels = 1
    model = Unet(im_channels=channels, t_emb_dim=128).to(device)
    model.load_state_dict(torch.load("./ddpm/jashandeep/checkpoints_MNIST/monster_unet_epoch_105.pth", map_location=device, weights_only=True))
    model.eval() 
    
    reverse_process = DDIMReverseProcess(num_time_steps=num_timesteps).to(device)
    
    # Calculate the jump sizes (e.g., 1000 steps // 50 jumps = jump by 20 every loop)
    step_size = num_timesteps // ddim_steps
    time_steps = list(range(num_timesteps - 1, -1, -step_size))
    
    print(f"Starting generation in {ddim_steps} steps...")
    
    with torch.no_grad():
        x = torch.randn(num_images, channels, img_size, img_size).to(device)
        
        for i, current_t in enumerate(time_steps):
            
            t = torch.full((num_images,), current_t, dtype=torch.long, device=device)
        
            if i == len(time_steps) - 1:
                next_t = -1 # If we are on the last loop, jump to -1 (perfectly clean)
            else:
                next_t = time_steps[i + 1]
                
            t_prev = torch.full((num_images,), next_t, dtype=torch.long, device=device)
            predicted_noise = model(x, t)
            x = reverse_process.step(x, predicted_noise, t, t_prev)
            
            if (i + 1) % 10 == 0:
                print(f"Completed DDIM jump {i + 1}/{ddim_steps}...")
                
    print("Generation complete! Plotting...")
    x = x.cpu()
    x = (x + 1.0) / 2.0 
    x = torch.clamp(x, 0.0, 1.0)
    
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(x[i].squeeze(), cmap='gray')
        ax.axis('off')
        
    plt.tight_layout()
    plt.show()
    
generate_images_ddim()