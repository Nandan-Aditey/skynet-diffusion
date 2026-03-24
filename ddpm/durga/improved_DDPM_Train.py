
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import math
import os
import sys

# --- 1. CONFIGURATION ---
SCHEDULE_NAME = "linear" # "linear", "cosine", "exponential"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
timesteps = 300
batch_size = 128
epochs = 30 
lr = 2e-4
save_interval = 5 
print(f"Using device: {device}")
os.makedirs("checkpoints", exist_ok=True)

# --- 2. BETA SCHEDULES ---
def get_beta_schedule(name, timesteps):
    if name == "linear":
        return torch.linspace(0.0001, 0.02, timesteps)
    elif name == "cosine":
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + 0.008) / 1.008 * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.999)
    elif name == "exponential":
        return torch.exp(torch.linspace(math.log(0.0001), math.log(0.02), timesteps))
    else:
        raise ValueError(f"Unknown schedule: {name}")

betas = get_beta_schedule(SCHEDULE_NAME, timesteps).to(device)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

def get_index_from_list(vals, t, x_shape):
    batch_size = t.shape[0]
    out = vals.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

# --- 3. NETWORK MODULES ---
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
        
        # Initial projection
        self.inc = nn.Conv2d(1, 64, 3, padding=1)
        
        # Encoder stages
        self.down1 = ResBlock(64, 128, self.time_dim)   # 28x28
        self.down2 = ResBlock(128, 256, self.time_dim)  # 14x14 (after pool)
        
        # Bottleneck
        self.mid1 = ResBlock(256, 256, self.time_dim)
        self.attn = SelfAttention(256)
        self.mid2 = ResBlock(256, 256, self.time_dim)
        
        # Decoder stages
        self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2) # 14x14 -> 28x28
        self.res_up1 = ResBlock(128 + 128, 128, self.time_dim) # Concat with down1
        
        self.up2 = nn.Conv2d(128, 64, 3, padding=1)           # Stay at 28x28
        self.res_up2 = ResBlock(64 + 64, 64, self.time_dim)    # Concat with inc
        
        self.outc = nn.Conv2d(64, 1, 1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        
        # --- Encoder ---
        x1 = self.inc(x)             # [B, 64, 28, 28]
        x2 = self.down1(x1, t_emb)   # [B, 128, 28, 28]
        
        x3 = F.max_pool2d(x2, 2)     # [B, 128, 14, 14]
        x4 = self.down2(x3, t_emb)   # [B, 256, 14, 14]
        
        # --- Bottleneck ---
        m = self.mid1(x4, t_emb)     # [B, 256, 14, 14]
        m = self.attn(m)
        m = self.mid2(m, t_emb)
        
        # --- Decoder ---
        u1 = self.up1(m)             # [B, 128, 28, 28] (Upsampled 14 to 28)
        u1 = torch.cat([u1, x2], 1)  # Match: [B, 128+128, 28, 28]
        u1 = self.res_up1(u1, t_emb)
        
        u2 = self.up2(u1)            # [B, 64, 28, 28]
        u2 = torch.cat([u2, x1], 1)  # Match: [B, 64+64, 28, 28]
        u2 = self.res_up2(u2, t_emb)
        
        return self.outc(u2)

# --- 4. DATA & TRAINING ---
try:
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Lambda(lambda t: (t * 2) - 1)
    ])
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model = WideResUNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"Starting Training ({SCHEDULE_NAME} schedule)...")
    for epoch in range(epochs):
        epoch_loss = 0
        model.train()
        for i, (images, _) in enumerate(dataloader):
            images = images.to(device)
            optimizer.zero_grad()
            
            t = torch.randint(0, timesteps, (batch_size,), device=device).long()
            noise = torch.randn_like(images)
            
            x_noisy = get_index_from_list(sqrt_alphas_cumprod, t, images.shape) * images + \
                      get_index_from_list(sqrt_one_minus_alphas_cumprod, t, images.shape) * noise
            
            predicted_noise = model(x_noisy, t)
            loss = F.mse_loss(noise, predicted_noise)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch} | Loss: {epoch_loss/len(dataloader):.6f}")

        if (epoch + 1) % save_interval == 0:
            ckpt_path = f"checkpoints/mnist_{SCHEDULE_NAME}_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), ckpt_path)
            print(f"--- Saved: {ckpt_path}")

    torch.save(model.state_dict(), f"mnist_diffusion_{SCHEDULE_NAME}_final.pth")
    print("DONE.")

except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)