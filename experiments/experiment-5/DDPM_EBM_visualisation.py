#ALL IN ONE CODE: EBM and DDPM training + generation & visualization
#this does NOT hv FID comparison yet, just for pretty looking plots purposes

import torch                            #type: ignore
import torch.nn as nn                   #type: ignore
import torch.optim as optim             #type: ignore
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm                   #type: ignore
import math 


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Running on: {device}")

#dataset: 5 pointed star
def get_star_galaxy(n_samples=5000):
    t = np.linspace(0, 2 * np.pi, n_samples)
    # 5-pointed star parametric equation (hypocycloid variation)
    r = 3 + np.cos(5 * t) 
    x = r * np.cos(t)
    y = r * np.sin(t)
    data = np.stack([x, y], axis=1)
    # noise
    noise = np.random.normal(0, 0.15, size=data.shape)
    return torch.tensor(data + noise, dtype=torch.float32)



#  EBM model 
class EnergyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 256),
            nn.Softplus(), # Smooth activation critical for 2nd order derivatives
            nn.Linear(256, 256),
            nn.Softplus(),
            nn.Linear(256, 1) # outputs SCALAR Energy
        )

    def forward(self, x):
        return self.net(x)

    def get_score(self, x):
        x.requires_grad_(True)
        energy = self.forward(x)
        score = -torch.autograd.grad(energy.sum(), x, create_graph=True)[0]
        return score

#  DDPM (MLP with timeembedding)
#note to self: this is a standard sinusoidal time embedding, 
#similar to positional encodings in transformers, but for time steps in diffusion models
class TimeEmbedding(nn.Module): 
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class DDPM_MLP(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            TimeEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mid_layer = nn.Sequential(
            nn.Linear(2 + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2) # Outputs VECTOR (predicted noise)
        )

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        x_input = torch.cat([x, t_emb], dim=-1)
        return self.mid_layer(x_input)



# Langevin Dynamics Sampling (for EBM)
def sample_langevin(model, n_samples=1000, steps=200, lr=0.01, return_traj=False):
    x = torch.randn(n_samples, 2).to(device) * 4 # Start wide
    traj = [x.detach().cpu().numpy()]
    
    for _ in range(steps):
        x.requires_grad_(True)
        energy = model(x)
        # score= -grad(Energy)
        grad = torch.autograd.grad(energy.sum(), x)[0]
        
        # langevinstep: x = x - (lr/2) * grad(E) + sqrt(lr) * noise
        noise = torch.randn_like(x)
        x = x.detach() - (lr/2) * grad + np.sqrt(lr) * noise
        if return_traj:
            traj.append(x.detach().cpu().numpy())
            
    if return_traj:
        return x.detach().cpu().numpy(), np.stack(traj, axis=1)
    return x.detach().cpu().numpy()

# DDPM reverse sampling
@torch.no_grad()
def sample_ddpm(model, betas, alphas, alphas_cumprod, n_samples=1000, n_steps=1000, return_traj=False):
    x = torch.randn(n_samples, 2).to(device) # Start from N(0,I)
    traj = [x.detach().cpu().numpy()]
    
    for i in reversed(range(n_steps)):
        t = torch.full((n_samples,), i, device=device, dtype=torch.float)
        
        # noise predicn
        pred_noise = model(x, t)
        
        alpha = alphas[i]
        alpha_cp = alphas_cumprod[i]
        beta = betas[i]
        
        #denoising step
        mean = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / torch.sqrt(1 - alpha_cp)) * pred_noise)
        
        # posterior noise (except at the very last step)
        if i > 0:
            noise = torch.randn_like(x)
            x = mean + torch.sqrt(beta) * noise
        else:
            x = mean
            
        if return_traj:
            traj.append(x.detach().cpu().numpy())
            
    if return_traj:
        return x.cpu().numpy(), np.stack(traj, axis=1)
    return x.cpu().numpy()

###TRAINING AND COMPARISON FUNCTION
def train_and_compare(epochs=3000, batch_size=1024, n_compare=1500, n_traj=5):
    # Setup Models
    ebm = EnergyNet().to(device)
    ebm_opt = optim.Adam(ebm.parameters(), lr=1e-3)
    # DSMs sigma
    dsm_sigma = 0.1 

    ddpm = DDPM_MLP().to(device)
    ddpm_opt = optim.Adam(ddpm.parameters(), lr=1e-3)
    # DDPM schedule
    n_ddpm_steps = 1000
    betas = torch.linspace(1e-4, 0.02, n_ddpm_steps).to(device)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    print("Training EBM and DDPM in parallel...")
    pbar = tqdm(range(epochs))
    for epoch in pbar:
        #ground truth
        batch = get_star_galaxy(batch_size).to(device)

        # EBM TRAINING using DSM
        # Perturb data with single-scale noise
        noise_ebm = torch.randn_like(batch) * dsm_sigma
        perturbed_batch = batch + noise_ebm
        perturbed_batch.requires_grad_(True)
        
        # Compute Score (negative grad of Energy)
        # We need retain_graph=True because we backward twice (once for EBM, once for DDPM)
        energy = ebm(perturbed_batch)
        score = -torch.autograd.grad(energy.sum(), perturbed_batch, create_graph=True, retain_graph=True)[0]
        
        # DSM Loss: match score to perturbation direction
        # Target score is -noise / sigma^2
        target_score = -noise_ebm / (dsm_sigma**2)
        ebm_loss = 0.5 * torch.norm(score - target_score, dim=-1).pow(2).mean()

        ebm_opt.zero_grad()
        ebm_loss.backward(retain_graph=True) # Retain graph for DDPM backward
        ebm_opt.step()

        # Train DDPM (via Noise Prediction) 
        # Sample random timesteps
        t_ddpm = torch.randint(0, n_ddpm_steps, (batch_size,), device=device).long()
        noise_ddpm = torch.randn_like(batch)
        
        # Add noise according to DDPM schedule
        a_cp = alphas_cumprod[t_ddpm][:, None]
        noisy_batch = torch.sqrt(a_cp) * batch + torch.sqrt(1 - a_cp) * noise_ddpm
        
        # Predict noise and compute MSE loss
        pred_noise = ddpm(noisy_batch, t_ddpm.float())
        ddpm_loss = nn.MSELoss()(pred_noise, noise_ddpm)
        
        ddpm_opt.zero_grad()
        ddpm_loss.backward()
        ddpm_opt.step()

        if epoch % 100 == 0:
            pbar.set_description(f"EBM loss: {ebm_loss.item():.4f} | DDPM loss: {ddpm_loss.item():.4f}")

    
    # Visualization Generation!!

    print("\nGenerating comparison plots...")
    plt.figure(figsize=(20, 12))
    plt.suptitle("EBM trajectories+comparison with DDPM", fontsize=22, fontweight='bold')
    
    # Grid for Heatmap&Quiver
    grid_res = 30
    x_range = np.linspace(-5, 5, grid_res)
    y_range = np.linspace(-5, 5, grid_res)
    xv, yv = np.meshgrid(x_range, y_range)
    grid_points = np.stack([xv.flatten(), yv.flatten()], axis=1)
    grid_tensor = torch.tensor(grid_points, dtype=torch.float32).to(device)

    # Calculate EBM Energy and Score on Grid
    ebm.eval()
    with torch.no_grad():
        energy_grid = ebm(grid_tensor).cpu().numpy().reshape(grid_res, grid_res)
    
    score_grid = ebm.get_score(grid_tensor).detach().cpu().numpy()
    score_u = score_grid[:, 0].reshape(grid_res, grid_res)
    score_v = score_grid[:, 1].reshape(grid_res, grid_res)

    # Subplot 1: EBM Energy Heatmap
    plt.subplot(2, 3, 1)
    plt.contourf(xv, yv, energy_grid, levels=50, cmap='magma_r')
    plt.colorbar(label='Energy E(x)')
    plt.title("1. EBM energy landscape", fontsize=14)
    plt.axis('equal')

    # Subplot 2: EBM Score Quiver (Gradient Field)
    plt.subplot(2, 3, 2)
    # Plot the grid arrows 
    plt.quiver(xv, yv, score_u, score_v, color='white', alpha=0.6, scale=150)
    plt.title("2. EBM score field ∇x log p(x)", fontsize=14)
    # Also overlay train data to confirm alignment
    train_data_vis = get_star_galaxy(500)
    plt.scatter(train_data_vis[:,0], train_data_vis[:,1], s=1, color='cyan', alpha=0.3)
    plt.axis('equal')
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    ax = plt.gca()
    ax.set_facecolor('#222222') # Dark background for white arrows

    # generate samples and trajectories
    print("Sampling EBM (Langevin)...")
    ebm_samples, ebm_trajs = sample_langevin(ebm, n_samples=n_compare, steps=250, return_traj=True)
    
    print("Sampling DDPM (Reverse Process)...")
    ddpm_samples, ddpm_trajs = sample_ddpm(ddpm, betas, alphas, alphas_cumprod, n_samples=n_compare, n_steps=n_ddpm_steps, return_traj=True)

    # Subplot 3: Langevin Trajectories (fir EBM)
    plt.subplot(2, 3, 3)
    for i in range(n_traj):
        plt.plot(ebm_trajs[i, :, 0], ebm_trajs[i, :, 1], alpha=0.6, linewidth=1, label=f'Traj {i+1}' if i==0 else "")
        plt.scatter(ebm_trajs[i, 0, 0], ebm_trajs[i, 0, 1], marker='o', color='red', s=20) # Start
        plt.scatter(ebm_trajs[i, -1, 0], ebm_trajs[i, -1, 1], marker='X', color='green', s=30) # End
    plt.title("3. EBM Langevin Trajectories", fontsize=14)
    plt.axis('equal')
    plt.xlim(-6, 6)
    plt.ylim(-6, 6)
    plt.grid(True, alpha=0.3)

    # Subplot 4: DDPM teverse trajectories
    plt.subplot(2, 3, 4)
    # Downsample DDPM trajectory for visualization (it has 1000 steps)
    vis_step = 20
    for i in range(n_traj):
        plt.plot(ddpm_trajs[i, ::vis_step, 0], ddpm_trajs[i, ::vis_step, 1], alpha=0.6, linewidth=1)
        plt.scatter(ddpm_trajs[i, 0, 0], ddpm_trajs[i, 0, 1], marker='o', color='red', s=20) # Start
        plt.scatter(ddpm_trajs[i, -1, 0], ddpm_trajs[i, -1, 1], marker='X', color='green', s=30) # End
    plt.title("4. DDPM Reverse Trajectories", fontsize=14)
    plt.axis('equal')
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.grid(True, alpha=0.3)

    # Final generated outputs compared

    # Subplot 5: ground truth data
    plt.subplot(2, 3, 5)
    train_data_final = get_star_galaxy(n_compare)
    plt.scatter(train_data_final[:, 0], train_data_final[:, 1], s=1, color='purple', alpha=0.5)
    plt.title("5. Training Data (Ground Truth)\n", fontsize=14)
    plt.axis('equal')
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)

    # Subplot 6: Comparing EBM vs DDPM Outputs
    plt.subplot(2, 3, 6)
    # Scatter plot both on top of each other with different colors
    plt.scatter(ebm_samples[:, 0], ebm_samples[:, 1], s=1, color='orange', alpha=0.5, label='EBM (Langevin)')
    plt.scatter(ddpm_samples[:, 0], ddpm_samples[:, 1], s=1, color='dodgerblue', alpha=0.3, label='DDPM (Diffusion)')
    plt.title("6. Final Generated Outputs\nEBM (Orange) vs DDPM (Blue)", fontsize=14)
    plt.axis('equal')
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
   
    plt.legend(loc='upper right', markerscale=10)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
 
 #change this to plt.show() the saving is for my UGCL machine
    save_path = '/home/ndurga/star_comparison.png' ##change this to local path
    plt.savefig(save_path, dpi=300)
    print(f"\nSuccess! Plot saved to: {save_path}")
    plt.show()
    print("Done.")

#main
if __name__ == "__main__":
    train_and_compare(epochs=3000, batch_size=1024)
    