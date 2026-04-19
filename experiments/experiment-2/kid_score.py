import torch
import time
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchmetrics.image.kid import KernelInceptionDistance

from ddpm.jashandeep.unet import Unet
from ddpm.jashandeep.diffusion import DiffusionReverseProcess, DDIMReverseProcess
from experiment6.ddim_ddpm_img_diff import sample_ddpm, sample_ddim

def run_speed_and_quality_test(num_images):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device}, generating {num_images} samples")

    # Safe batch limit for 6GB VRAM
    mini_batch_size = 256 

    # 1. Load Model
    model = Unet(im_channels=1, t_emb_dim=128).to(device)
    model.load_state_dict(torch.load("./ddpm/jashandeep/checkpoints_FMNIST/fashion_unet_final.pth", map_location=device, weights_only=True))
    model.eval()
    
    ddpm_process = DiffusionReverseProcess().to(device)
    ddim_process = DDIMReverseProcess().to(device)
    
    fixed_noise = torch.randn(num_images, 1, 32, 32).to(device)
    
    subset_val = min(50, num_images - 1)
    kid = KernelInceptionDistance(subset_size=subset_val).to(device)
    
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])
    dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=mini_batch_size, shuffle=True)
    
    real_img_batches = []
    real_processed = 0
    for imgs, _ in dataloader:
        if real_processed >= num_images:
            break
        current_bs = min(imgs.shape[0], num_images - real_processed)
        imgs = imgs[:current_bs].to(device)
        imgs_uint8 = (imgs.repeat(1, 3, 1, 1) * 255).to(torch.uint8)
        real_img_batches.append(imgs_uint8)
        real_processed += current_bs

    def populate_kid_with_reals():
        for b in real_img_batches:
            kid.update(b, real=True)

    ddim_step_counts = [10, 20, 50, 100, 500]
    results_time = {}
    results_kid = {}

    with torch.no_grad():
        print("Running DDPM (1000 steps)...")
        populate_kid_with_reals()
        total_ddpm_time = 0
        fake_processed = 0
        
        while fake_processed < num_images:
            current_bs = min(mini_batch_size, num_images - fake_processed)
            noise_chunk = fixed_noise[fake_processed:fake_processed + current_bs]
            
            torch.cuda.synchronize() 
            start_time = time.perf_counter()
            
            fake_imgs = sample_ddpm(model, ddpm_process, noise_chunk, device)
            
            torch.cuda.synchronize() 
            total_ddpm_time += (time.perf_counter() - start_time)
            
            fake_imgs = fake_imgs.to(device)
            fake_imgs_uint8 = (fake_imgs.repeat(1, 3, 1, 1) * 255).to(torch.uint8)
            kid.update(fake_imgs_uint8, real=False)
            
            fake_processed += current_bs

        results_time['DDPM (1000)'] = total_ddpm_time
        kid_mean, _ = kid.compute()
        results_kid['DDPM (1000)'] = kid_mean.item()
        
        print(f"DDPM (1000) -> Time: {total_ddpm_time:.5f}s | KID: {kid_mean.item():.5f}")
        kid.reset()

        #DDIM
        for steps in ddim_step_counts:
            print(f"Running DDIM ({steps} steps)...")
            populate_kid_with_reals()
            total_ddim_time = 0
            fake_processed = 0
            
            while fake_processed < num_images:
                current_bs = min(mini_batch_size, num_images - fake_processed)
                noise_chunk = fixed_noise[fake_processed:fake_processed + current_bs]
                
                torch.cuda.synchronize()
                start_time = time.perf_counter()
                
                fake_imgs = sample_ddim(model, ddim_process, noise_chunk, device, steps=steps)
                
                torch.cuda.synchronize()
                total_ddim_time += (time.perf_counter() - start_time)
                
                fake_imgs = fake_imgs.to(device)
                fake_imgs_uint8 = (fake_imgs.repeat(1, 3, 1, 1) * 255).to(torch.uint8)
                kid.update(fake_imgs_uint8, real=False)
                
                fake_processed += current_bs
                
            results_time[f'DDIM ({steps})'] = total_ddim_time
            kid_mean, _ = kid.compute()
            results_kid[f'DDIM ({steps})'] = kid_mean.item()
            
            print(f"DDIM ({steps})  -> Time: {total_ddim_time:.2f}s | KID: {kid_mean.item():.4f}")
            kid.reset()

    names = list(results_time.keys())
    times = list(results_time.values())
    kids = list(results_kid.values())
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    #Time
    bars1 = ax1.bar(names, times, color=['red'] + ['blue'] * len(ddim_step_counts))
    ax1.set_ylabel(f'Seconds to Generate {num_images} Images')
    ax1.set_title('Generation Speed (Lower is Better)')
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval + (max(times)*0.02), f'{yval:.2f}s', ha='center', va='bottom')

    bars2 = ax2.bar(names, kids, color=['red'] + ['blue'] * len(ddim_step_counts))
    ax2.set_ylabel('KID Score')
    ax2.set_title('Image Quality (Lower is Better)')
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, yval + (max(kids)*0.02), f'{yval:.4f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(f'./ddpm/jashandeep/kid_scores/time_kid_batch_size_{num_images}.png')
    plt.close()

num_images = [1024, 2048]
for num in num_images:
    print(f"\n{'='*40}\nStarting test for batch size: {num}\n{'='*40}")
    run_speed_and_quality_test(num)