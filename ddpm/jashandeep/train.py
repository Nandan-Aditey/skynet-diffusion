import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from diffusion import DiffusionForwardProcess
from unet import Unet

# 32x32 is much better for U-Nets than 28x28 apparently?
transform = transforms.Compose([
    transforms.Resize((32, 32)), 
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def train():
    batch_size = 64     #Safe for 6GB VRAM
    num_epochs = 10
    lr = 1e-4         
    num_timesteps = 1000
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Firing up the engines on: {device}')

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = Unet(im_channels=1, t_emb_dim=128).to(device)
    forward_process = DiffusionForwardProcess(num_time_steps=num_timesteps).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    #==============RESUME BLOCK=================
    resume_training = True               # Change to False to start from scratch!
    start_epoch = 0                      # Keeps our print statements accurate
    checkpoint_path = "./ddpm/jashandeep/checkpoints_FMNIST/fashion_unet_epoch_100.pth" # Which brain to load

    if resume_training:
        import os
        if os.path.exists(checkpoint_path):
            model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
            print(f"Successfully loaded checkpoint: {checkpoint_path}")
            
            start_epoch = int(checkpoint_path.split('_')[-1].split('.')[0])
            print(f"Resuming training from Epoch {start_epoch + 1}...")
        else:
            print(f"Could not find {checkpoint_path}. Starting from scratch!")

    for epoch in range(start_epoch, start_epoch + num_epochs):
        model.train()
        epoch_losses = []
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}") #cool progress bar
        
        for imgs, _ in pbar:
            imgs = imgs.to(device)
            
            # Step A: Generate random target noise and random time-steps
            actual_noise = torch.randn_like(imgs).to(device)
            t = torch.randint(0, num_timesteps, (imgs.shape[0],)).to(device)
            
            # Step B: Corrupt the images with the math we built
            noisy_imgs = forward_process.add_noise(imgs, actual_noise, t)
            
            # Step C: The U-Net makes its guess
            optimizer.zero_grad() # Clear old math
            predicted_noise = model(noisy_imgs, t)
            
            # Step D: See how wrong the U-Net was
            loss = criterion(predicted_noise, actual_noise)
            
            # Step E: Backpropagation (Learn from the mistake)
            loss.backward()
            optimizer.step()
            
            # Update progress bar with current loss
            epoch_losses.append(loss.item())
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
            
        # End of Epoch summary
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"End of Epoch {epoch+1} | Average Loss: {avg_loss:.4f}\n")

        if (epoch + 1) % 5 == 0:
            checkpoint_name = f"./ddpm/jashandeep/checkpoints_MNIST/monster_unet_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), checkpoint_name)
            print(f"Checkpoint saved: {checkpoint_name}\n")
        else:
            # Just print a normal newline for spacing
            print() 

    # 6. Save the final brain!
    torch.save(model.state_dict(), "monster.pth")
    print("Training Complete! Final model saved.")
train()
