import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
import os

# --- SETTINGS ---
# Change these three to point to your specific checkpoint
CHECKPOINT_PATH = "checkpoints/mnist_cosine_epoch_25.pth"
SCHEDULE_TYPE = "cosine"  # Must match the checkpoint's training
SEED = 7               # Change this to get different digits

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
timesteps = 300

# --- ARCHITECTURE (WideResUNet - Corrected) ---
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, time):
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class SelfAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
    def forward(self, x):
        b, c, h, w = x.shape
        x_flat = x.view(b, c, h * w).permute(0, 2, 1)
        x_norm = self.ln(x_flat)
        attn_out, _ = self.mha(x_norm, x_norm, x_norm)
        return (attn_out + x_flat).permute(0, 2, 1).view(b, c, h, w)

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    def forward(self, x, t):
        h = F.relu(self.bn1(self.conv1(x)))
        h = h + self.time_mlp(t)[:, :, None, None]
        h = self.bn2(self.conv2(h))
        return F.relu(h + self.shortcut(x))

class WideResUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_dim = 256
        self.time_mlp = nn.Sequential(SinusoidalPositionEmbeddings(self.time_dim), nn.Linear(self.time_dim, self.time_dim), nn.GELU())
        self.inc = nn.Conv2d(1, 64, 3, padding=1)
        self.down1 = ResBlock(64, 128, self.time_dim)
        self.down2 = ResBlock(128, 256, self.time_dim)
        self.mid1 = ResBlock(256, 256, self.time_dim)
        self.attn = SelfAttention(256)
        self.mid2 = ResBlock(256, 256, self.time_dim)
        self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2) 
        self.res_up1 = ResBlock(128 + 128, 128, self.time_dim) 
        self.up2 = nn.Conv2d(128, 64, 3, padding=1)           
        self.res_up2 = ResBlock(64 + 64, 64, self.time_dim)    
        self.outc = nn.Conv2d(64, 1, 1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        x1 = self.inc(x); x2 = self.down1(x1, t_emb)
        x3 = F.max_pool2d(x2, 2); x4 = self.down2(x3, t_emb)
        m = self.mid1(x4, t_emb); m = self.attn(m); m = self.mid2(m, t_emb)
        u1 = self.up1(m); u1 = torch.cat([u1, x2], 1); u1 = self.res_up1(u1, t_emb)
        u2 = self.up2(u1); u2 = torch.cat([u2, x1], 1); u2 = self.res_up2(u2, t_emb)
        return self.outc(u2)

# --- UTILS ---
def get_beta_schedule(name, timesteps):
    if name == "linear": return torch.linspace(0.0001, 0.02, timesteps)
    elif name == "cosine":
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + 0.008) / 1.008 * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.999)
    elif name == "exponential":
        return torch.exp(torch.linspace(math.log(0.0001), math.log(0.02), timesteps))
    return None

# --- GENERATION ---
def generate_single():
    torch.manual_seed(SEED)
    print(f"Generating single image using {SCHEDULE_TYPE} schedule...")
    
    betas = get_beta_schedule(SCHEDULE_TYPE, timesteps).to(device)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

    model = WideResUNet().to(device)
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"File not found: {CHECKPOINT_PATH}")
        return

    # Load weights
    state_dict = torch.load(CHECKPOINT_PATH, map_location=device)
    # Check if this is a checkpoint with metadata or just raw weights
    if 'model_state_dict' in state_dict:
        model.load_state_dict(state_dict['model_state_dict'])
    else:
        model.load_state_dict(state_dict)
    
    model.eval()

    with torch.no_grad():
        img = torch.randn((1, 1, 28, 28), device=device)
        for i in reversed(range(1, timesteps)):
            t = torch.full((1,), i, device=device, dtype=torch.long)
            pred_noise = model(img, t)
            
            alpha_t, beta_t = alphas[i], betas[i]
            sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alphas_cumprod[i]
            
            mean = (1 / torch.sqrt(alpha_t)) * (img - ((beta_t / sqrt_one_minus_alpha_cumprod_t) * pred_noise))
            if i > 1:
                img = mean + torch.sqrt(beta_t) * torch.randn_like(img)
            else:
                img = mean

    # Save output
    img = (img.clamp(-1, 1) + 1) / 2
    plt.imshow(img[0].cpu().squeeze(), cmap="gray")
    plt.title(f"Seed: {SEED} | {SCHEDULE_TYPE}")
    plt.axis("off")
    plt.savefig("single_output.png")
    print("Done! View single_output.png")

if __name__ == "__main__":
    generate_single()