import torch
import torch.nn as nn
import numpy as np
import matplotlib
# Use the 'Agg' backend for headless server environments
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.datasets import make_swiss_roll
import random

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- SEEDING BLOCK ---
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False

set_seed(6250)
# -------------------------

# ---------------------------------------------------------
# 1. Dataset Preparation
# ---------------------------------------------------------
def get_swiss_roll(n_samples=5000, noise=0.5):
    x, _ = make_swiss_roll(n_samples=n_samples, noise=noise)
    x = x[:, [0, 2]]
    # Normalize to zero mean and roughly unit variance
    x = (x - x.mean(axis=0)) / x.std(axis=0)
    return torch.tensor(x, dtype=torch.float32).to(device)

# ---------------------------------------------------------
# 2. Neural Network Architectures
# ---------------------------------------------------------
class MLPCore(nn.Module):
    """Shared core architecture for both networks"""
    def __init__(self, input_dim=2, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    def forward(self, x, t):
        t = t.view(-1, 1)
        xt = torch.cat([x, t], dim=1)
        return self.net(xt)

# Network 1: Predicts Noise (epsilon)
class EpsilonNet(MLPCore):
    pass 

# Network 2: Predicts Score directly
class ScoreNet(MLPCore):
    pass

# ---------------------------------------------------------
# 3. Diffusion Utilities
# ---------------------------------------------------------
T = 100
beta = torch.linspace(1e-4, 0.02, T).to(device)
alpha = 1.0 - beta
alpha_bar = torch.cumprod(alpha, dim=0)

sqrt_alpha_bar = torch.sqrt(alpha_bar)
sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar)

# ---------------------------------------------------------
# 4. Training Functions
# ---------------------------------------------------------
def train_ddpm(model, data, epochs=1200, batch_size=512, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(data), batch_size=batch_size, shuffle=True)
    
    print("Training DDPM (Epsilon) Network...")
    for epoch in range(epochs):
        for batch in loader:
            x_0 = batch[0]
            t = torch.randint(0, T, (x_0.shape[0],), device=device)
            eps = torch.randn_like(x_0)
            
            x_t = sqrt_alpha_bar[t].view(-1, 1) * x_0 + sqrt_one_minus_alpha_bar[t].view(-1, 1) * eps
            t_scaled = t.float() / T
            
            # Predict noise
            eps_theta = model(x_t, t_scaled)
            loss = nn.functional.mse_loss(eps_theta, eps)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def train_sde(model, data, epochs=1200, batch_size=512, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(data), batch_size=batch_size, shuffle=True)
    
    print("Training SDE (Score) Network...")
    for epoch in range(epochs):
        for batch in loader:
            x_0 = batch[0]
            t = torch.randint(0, T, (x_0.shape[0],), device=device)
            eps = torch.randn_like(x_0)
            
            x_t = sqrt_alpha_bar[t].view(-1, 1) * x_0 + sqrt_one_minus_alpha_bar[t].view(-1, 1) * eps
            t_scaled = t.float() / T
            
            # True score is -eps / sqrt(1 - alpha_bar_t)
            std = sqrt_one_minus_alpha_bar[t].view(-1, 1)
            target_score = -eps / std
            
            # Predict score
            score_theta = model(x_t, t_scaled)
            
            # Continuous-time score matching loss usually weights by variance (std^2)
            # to prevent loss from exploding at t=0
            weight = std ** 2
            loss = torch.mean(weight * (score_theta - target_score) ** 2)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

# ---------------------------------------------------------
# 5. Combined Sampling (Tracking Trajectories)
# ---------------------------------------------------------
@torch.no_grad()
def sample_and_track(eps_model, score_model, n_samples=1000):
    # Start from EXACT same noise
    x_ddpm = torch.randn(n_samples, 2, device=device)
    x_sde = x_ddpm.clone()
    
    traj_ddpm = [x_ddpm.clone()]
    traj_sde = [x_sde.clone()]
    
    for i in reversed(range(T)):
        t_tensor = torch.full((n_samples,), i, device=device)
        t_scaled = t_tensor.float() / T
        
        # Shared noise increment
        z = torch.randn_like(x_ddpm) if i > 0 else torch.zeros_like(x_ddpm)
        
        b_t = beta[i]
        a_t = alpha[i]
        a_bar_t = alpha_bar[i]
        
        # --- DDPM Ancestral Step (using EpsilonNet) ---
        eps_pred = eps_model(x_ddpm, t_scaled)
        x_ddpm = (1 / torch.sqrt(a_t)) * (x_ddpm - (b_t / torch.sqrt(1 - a_bar_t)) * eps_pred) + torch.sqrt(b_t) * z
        traj_ddpm.append(x_ddpm.clone())
        
        # --- SDE Euler-Maruyama Step (using ScoreNet) ---
        score_pred = score_model(x_sde, t_scaled)
        drift = -0.5 * b_t * x_sde - b_t * score_pred
        x_sde = x_sde - drift + torch.sqrt(b_t) * z
        traj_sde.append(x_sde.clone())
        
    return traj_ddpm, traj_sde

# ---------------------------------------------------------
# 6. Evaluation (MMD)
# ---------------------------------------------------------
def compute_mmd(x, y, sigma=1.0):
    xx = torch.cdist(x, x)**2
    yy = torch.cdist(y, y)**2
    xy = torch.cdist(x, y)**2

    k_xx = torch.exp(-xx / (2 * sigma**2)).mean()
    k_yy = torch.exp(-yy / (2 * sigma**2)).mean()
    k_xy = torch.exp(-xy / (2 * sigma**2)).mean()

    return (k_xx + k_yy - 2 * k_xy).item()

# ---------------------------------------------------------
# 7. Execution & Unified Visualization
# ---------------------------------------------------------
if __name__ == "__main__":
    n_samples = 2000
    data = get_swiss_roll(n_samples=5000, noise=0.5)
    true_samples = data[:n_samples] 
    
    # Init and Train Models
    eps_net = EpsilonNet().to(device)
    score_net = ScoreNet().to(device)
    
    train_ddpm(eps_net, data, epochs=1200, batch_size=512)
    train_sde(score_net, data, epochs=1200, batch_size=512)
    
    print("Sampling and tracking trajectories...")
    traj_ddpm, traj_sde = sample_and_track(eps_net, score_net, n_samples=n_samples)
    
    eval_steps = list(range(0, T + 1, 10))
    if T not in eval_steps:
        eval_steps.append(T)
        
    mmd_ddpm_true, mmd_sde_true, mmd_ddpm_sde = [], [], []
    
    print("Computing MMD over trajectories...")
    for step_idx in eval_steps:
        x_d = traj_ddpm[step_idx]
        x_s = traj_sde[step_idx]
        
        mmd_ddpm_true.append(compute_mmd(true_samples, x_d, sigma=1.0))
        mmd_sde_true.append(compute_mmd(true_samples, x_s, sigma=1.0))
        mmd_ddpm_sde.append(compute_mmd(x_d, x_s, sigma=1.0))

    # --- Unified Plotting ---
    print("\nGenerating comprehensive master plot...")
    fig, axes = plt.subplots(2, 2, figsize=(18, 16))
    
    # Convert required data to CPU numpy
    t_ddpm_np = np.stack([t.cpu().numpy() for t in traj_ddpm])
    t_sde_np = np.stack([t.cpu().numpy() for t in traj_sde])
    true_cpu = true_samples.cpu().numpy()
    
    # ---------------------------------------------------------
    # Subplot [0, 0]: Final Distributions
    # ---------------------------------------------------------
    axes[0, 0].scatter(true_cpu[:, 0], true_cpu[:, 1], alpha=0.3, s=5, c='gray', label='Ground Truth')
    axes[0, 0].scatter(t_ddpm_np[-1, :, 0], t_ddpm_np[-1, :, 1], alpha=0.3, s=5, c='blue', label='DDPM')
    axes[0, 0].scatter(t_sde_np[-1, :, 0], t_sde_np[-1, :, 1], alpha=0.3, s=5, c='red', label='SDE')
    axes[0, 0].set_title('Generated Distributions (t=0)', fontsize=14)
    axes[0, 0].legend()
    axes[0, 0].set_xlim(-3.5, 3.5)
    axes[0, 0].set_ylim(-3.5, 3.5)
    axes[0, 0].set_aspect('equal')
    axes[0, 0].grid(True, linestyle='--', alpha=0.6)

    # ---------------------------------------------------------
    # Subplot [0, 1]: MMD Trajectory Line Chart
    # ---------------------------------------------------------
    t_axis = [T - step for step in eval_steps]
    axes[0, 1].plot(t_axis, mmd_ddpm_true, marker='o', linewidth=2, color='blue', label='DDPM vs True')
    axes[0, 1].plot(t_axis, mmd_sde_true, marker='s', linewidth=2, color='red', label='SDE vs True')
    axes[0, 1].plot(t_axis, mmd_ddpm_sde, marker='^', linewidth=2, color='purple', label='DDPM vs SDE')
    axes[0, 1].invert_xaxis()
    axes[0, 1].set_xlabel('Timestep (t) [Pure Noise → Data]', fontsize=12)
    axes[0, 1].set_ylabel('MMD Score', fontsize=12)
    axes[0, 1].set_title('MMD Convergence during Reverse Sampling', fontsize=14)
    axes[0, 1].grid(True, linestyle='--', alpha=0.7)
    axes[0, 1].legend()

    # ---------------------------------------------------------
    # Subplot [1, 0]: MMD Bar Chart
    # ---------------------------------------------------------
    labels = ['DDPM vs True', 'SDE vs True', 'DDPM vs SDE']
    values = [mmd_ddpm_true[-1], mmd_sde_true[-1], mmd_ddpm_sde[-1]]
    colors = ['blue', 'red', 'purple']
    bars = axes[1, 0].bar(labels, values, color=colors, alpha=0.7)
    axes[1, 0].set_ylabel('MMD Score', fontsize=12)
    axes[1, 0].set_title('Final MMD Comparisons (t=0)', fontsize=14)
    axes[1, 0].grid(axis='y', linestyle='--', alpha=0.7)
    for bar, v in zip(bars, values):
        yval = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, yval + (max(values)*0.02), 
                 f"{v:.5f}", ha='center', va='bottom', fontsize=11, fontweight='bold')
    axes[1, 0].set_ylim(0, max(values) * 1.25)

    # ---------------------------------------------------------
    # Subplot [1, 1]: Combined Vector Trajectories
    # ---------------------------------------------------------
    # Select 25 random points to track to avoid visual clutter
    track_indices = np.random.choice(n_samples, 25, replace=False)
    arrow_interval = 8 # Draw an arrow every 8 steps
    
    axes[1, 1].set_title('Particle Trajectories (DDPM=Blue, SDE=Red)', fontsize=14)
    
    # Plot true data faintly in the background for context
    axes[1, 1].scatter(true_cpu[:, 0], true_cpu[:, 1], alpha=0.1, s=2, c='gray')
    
    for idx in track_indices:
        # DDPM Path
        xd = t_ddpm_np[:, idx, 0]
        yd = t_ddpm_np[:, idx, 1]
        axes[1, 1].plot(xd, yd, color='blue', alpha=0.4, linewidth=1)
        # DDPM Arrows
        Xd, Yd = xd[:-1:arrow_interval], yd[:-1:arrow_interval]
        Ud, Vd = xd[1::arrow_interval] - Xd, yd[1::arrow_interval] - Yd
        axes[1, 1].quiver(Xd, Yd, Ud, Vd, color='navy', angles='xy', scale_units='xy', scale=1, alpha=0.8, width=0.004)

        # SDE Path
        xs = t_sde_np[:, idx, 0]
        ys = t_sde_np[:, idx, 1]
        axes[1, 1].plot(xs, ys, color='red', alpha=0.4, linewidth=1)
        # SDE Arrows
        Xs, Ys = xs[:-1:arrow_interval], ys[:-1:arrow_interval]
        Us, Vs = xs[1::arrow_interval] - Xs, ys[1::arrow_interval] - Ys
        axes[1, 1].quiver(Xs, Ys, Us, Vs, color='darkred', angles='xy', scale_units='xy', scale=1, alpha=0.8, width=0.004)

    axes[1, 1].set_xlim(-3.5, 3.5)
    axes[1, 1].set_ylim(-3.5, 3.5)
    axes[1, 1].set_aspect('equal')
    axes[1, 1].grid(True, linestyle='--', alpha=0.6)

    # Save final composite image
    plt.tight_layout()
    output_filename = "comprehensive_diffusion_analysis.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Master plot successfully saved to {output_filename}")