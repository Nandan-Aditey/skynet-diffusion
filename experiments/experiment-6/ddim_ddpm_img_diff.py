import torch                            #type: ignore
import matplotlib.pyplot as plt         #type: ignore
from ddpm.jashandeep.diffusion import DiffusionReverseProcess, DDIMReverseProcess
from ddpm.jashandeep.unet import Unet

def sample_ddpm(model, process, x, device):
    with torch.no_grad():
        for i in reversed(range(1000)):
            t = torch.full((x.shape[0],), i, dtype=torch.long, device=device)
            predicted_noise = model(x, t)
            x, _ = process.sample_prev_timestep(x, predicted_noise, t)
    return format_for_plot(x)

def sample_ddim(model, process, x, device, steps):
    step_size = 1000 // steps
    time_steps = list(range(999, -1, -step_size))
    
    with torch.no_grad():
        for i, current_t in enumerate(time_steps):
            t = torch.full((x.shape[0],), current_t, dtype=torch.long, device=device)
            next_t = -1 if i == len(time_steps) - 1 else time_steps[i + 1]
            t_prev = torch.full((x.shape[0],), next_t, dtype=torch.long, device=device)
            
            predicted_noise = model(x, t)
            x = process.step(x, predicted_noise, t, t_prev)
    return format_for_plot(x)

def format_for_plot(x):
    x = x.cpu()
    x = (x + 1.0) / 2.0 
    return torch.clamp(x, 0.0, 1.0)

def plot_showdown(d1, d2, i1, i2):
    fig, axes = plt.subplots(4, 8, figsize=(16, 8))
    row_labels = ["DDPM (Run 1)", "DDPM (Run 2)", "DDIM (Run 1)", "DDIM (Run 2)"]
    rows = [d1, d2, i1, i2]
    
    for row_idx, (row_data, label) in enumerate(zip(rows, row_labels)):
        for col_idx in range(8):
            ax = axes[row_idx, col_idx]
            ax.imshow(row_data[col_idx].squeeze(), cmap='gray')
            ax.axis('off')
            if col_idx == 0:
                ax.set_title(label, loc='left', fontsize=12, fontweight='bold', pad=10)
                
    plt.tight_layout()
    plt.show()

def run_diffusion_showdown(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Running Showdown on: {device}')
    
    model = Unet(im_channels=1, t_emb_dim=128).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval() 
    
    ddpm_process = DiffusionReverseProcess().to(device)
    ddim_process = DDIMReverseProcess().to(device)

    # Use 8 fixed noise samples
    fixed_noise = torch.randn(8, 1, 32, 32).to(device)
    
    print("Starting DDPM Run 1 (1000 steps)...")
    ddpm_results_1 = sample_ddpm(model, ddpm_process, fixed_noise.clone(), device)
    
    print("Starting DDPM Run 2 (1000 steps)...")
    ddpm_results_2 = sample_ddpm(model, ddpm_process, fixed_noise.clone(), device)
    
    print("Starting DDIM Run 1 (50 steps)...")
    ddim_results_1 = sample_ddim(model, ddim_process, fixed_noise.clone(), device, steps=1)
    
    print("Starting DDIM Run 2 (50 steps)...")
    ddim_results_2 = sample_ddim(model, ddim_process, fixed_noise.clone(), device, steps=1)    

    return ddpm_results_1, ddpm_results_2, ddim_results_1, ddim_results_2
    

ddpm_results_1, ddpm_results_2, ddim_results_1, ddim_results_2 = run_diffusion_showdown("./ddpm/jashandeep/checkpoints_MNIST/mnist_unet_epoch_20.pth")