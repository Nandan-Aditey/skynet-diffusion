import torch
import time
import matplotlib.pyplot as plt
from ddpm.jashandeep.unet import Unet
from ddpm.jashandeep.diffusion import DiffusionReverseProcess,DDIMReverseProcess
from experiment6.ddim_ddpm_img_diff import sample_ddpm, sample_ddim

def run_speed_test(num_images):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device}, generating {num_images} samples")

    model = Unet(im_channels=1, t_emb_dim=128).to(device)
    model.load_state_dict(torch.load("./ddpm/jashandeep/checkpoints_FMNIST/fashion_unet_final.pth", map_location=device, weights_only=True))
    model.eval()
    
    ddpm_process = DiffusionReverseProcess().to(device)
    ddim_process = DDIMReverseProcess().to(device)
    
    fixed_noise = torch.randn(num_images, 1, 32, 32).to(device)
    
    ddim_step_counts = [10, 20, 50, 100, 500]
    results = {}

    with torch.no_grad():
        print("Running DDPM (1000 steps)...")
        torch.cuda.synchronize() 
        start_time = time.perf_counter()
        
        _ = sample_ddpm(model, ddpm_process, fixed_noise.clone(), device)
        torch.cuda.synchronize() 
        ddpm_time = time.perf_counter() - start_time
        
        print(f"DDPM 1000 steps took: {ddpm_time:.2f} seconds")
        results['DDPM (1000)'] = ddpm_time

        # --- DDIM TESTS ---
        for steps in ddim_step_counts:
            print(f"Running DDIM ({steps} steps)...")
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            
            _ = sample_ddim(model, ddim_process, fixed_noise.clone(), device, steps=steps)
            
            torch.cuda.synchronize()
            ddim_time = time.perf_counter() - start_time
            
            print(f"DDIM {steps} steps took: {ddim_time:.2f} seconds")
            results[f'DDIM ({steps})'] = ddim_time

    names = list(results.keys())
    times = list(results.values())
    
    plt.figure(figsize=(10, 5))
    bars = plt.bar(names, times, color=['red'] + ['blue'] * len(ddim_step_counts))
    plt.ylabel(f'Seconds to Generate {num_images} Images')
    plt.title('DDPM vs DDIM Generation Speed')
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.1, f'{yval:.4f}s', ha='center', va='bottom')
        
    plt.tight_layout()
    plt.show()

num_images = [16,32,64,128,256]
for num in num_images:
    run_speed_test(num)