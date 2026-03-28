import os
import glob
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.utils import save_image
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class DiffusionForwardProcess(nn.Module):
    def __init__(self, num_time_steps=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()

        betas = torch.linspace(beta_start, beta_end, num_time_steps)
        alphas = 1 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        
        self.register_buffer('sqrt_alpha_bars', torch.sqrt(alpha_bars))
        self.register_buffer('sqrt_one_minus_alpha_bars', torch.sqrt(1 - alpha_bars))

    def add_noise(self, original, noise, t):
        sqrt_alpha_bar_t = self.sqrt_alpha_bars[t][:, None, None, None]
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bars[t][:, None, None, None]
        
        return (sqrt_alpha_bar_t * original) + (sqrt_one_minus_alpha_bar_t * noise)  #x_t


class DiffusionReverseProcess(nn.Module):
    def __init__(self, num_time_steps=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        
        betas = torch.linspace(beta_start, beta_end, num_time_steps)
        alphas = 1 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        
        #adding aplha_bar_0=1.0
        alpha_bars_prev = torch.cat([torch.tensor([1.0]), alpha_bars[:-1]])

        sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        sqrt_alpha_bars = torch.sqrt(alpha_bars)
        sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars)
        
        mean_coef = betas / sqrt_one_minus_alpha_bars
        
        #variance: beta_t * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)
        posterior_variance = betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)

        self.register_buffer('betas', betas)
        self.register_buffer('alpha_bars', alpha_bars)
        self.register_buffer('sqrt_recip_alphas', sqrt_recip_alphas)
        self.register_buffer('sqrt_alpha_bars', sqrt_alpha_bars)
        self.register_buffer('sqrt_one_minus_alpha_bars', sqrt_one_minus_alpha_bars)
        self.register_buffer('mean_coef', mean_coef)
        self.register_buffer('posterior_variance', posterior_variance)

    def _extract(self, a, t, x_shape):
        """Helper to get the right time-step value and reshape it for images"""
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, 1, 1, 1).to(t.device)

    def sample_prev_timestep(self, xt, noise_pred, t):
        #just extracting data and making sure the dimensions are okay
        sqrt_recip_alphas_t = self._extract(self.sqrt_recip_alphas, t, xt.shape)
        mean_coef_t = self._extract(self.mean_coef, t, xt.shape)
        posterior_variance_t = self._extract(self.posterior_variance, t, xt.shape)
        
        sqrt_alpha_bars_t = self._extract(self.sqrt_alpha_bars, t, xt.shape)
        sqrt_one_minus_alpha_bars_t = self._extract(self.sqrt_one_minus_alpha_bars, t, xt.shape)
        
        mean = sqrt_recip_alphas_t * (xt - mean_coef_t * noise_pred)
        
        x0 = (xt - sqrt_one_minus_alpha_bars_t * noise_pred) / sqrt_alpha_bars_t
        x0 = torch.clamp(x0, -1., 1.)

        if t[0] == 0:
            return mean, x0
        else:
            noise = torch.randn_like(xt)
            sigma = torch.sqrt(posterior_variance_t)
            return mean + sigma * noise, x0
        
#keep track of loss fn vs time, if n points, atleast have 4n patametres

# 32x32 is much better for U-Nets than 28x28 apparently?
transform = transforms.Compose([
    transforms.Resize((32, 32)), 
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

class TimeEmbedding(nn.Module):
    def __init__(self, n_out:int, t_emb_dim:int = 128):
        super().__init__()
        # A simple mini-network to process the time barcode
        self.te_block = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_emb_dim, n_out)
        )
        
    def forward(self, x):
        return self.te_block(x)
    
class NormActConv(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, num_groups:int = 8, kernel_size: int = 3):
        super().__init__()
        # GroupNorm is highly recommended for Diffusion models instead of BatchNorm
        self.g_norm = nn.GroupNorm(num_groups, in_channels)
        self.act = nn.SiLU()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size - 1)//2)
        
    def forward(self, x):
        x = self.g_norm(x)
        x = self.act(x)
        x = self.conv(x)
        return x

# 1. The Time Embedding Generator (The Mathematical Clock)
def get_time_embedding(time_steps: torch.Tensor, t_emb_dim: int) -> torch.Tensor:
    assert t_emb_dim % 2 == 0, "time embedding must be divisible by 2."
    
    factor = 2 * torch.arange(start=0, end=t_emb_dim//2, dtype=torch.float32, device=time_steps.device) / t_emb_dim
    factor = 10000 ** factor
    
    t_emb = time_steps[:, None]
    t_emb = t_emb / factor
    t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=1)
    return t_emb

class SelfAttentionBlock(nn.Module):
    def __init__(self, channels: int, num_groups: int = 8, num_heads: int = 4):
        super().__init__()
        self.g_norm = nn.GroupNorm(num_groups, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        
    def forward(self, x):
        batch_size, channels, h, w = x.shape
        # Flatten the image into a sequence of pixels for the attention mechanism
        x_flat = x.reshape(batch_size, channels, h * w).transpose(1, 2)
        x_norm = self.g_norm(x).reshape(batch_size, channels, h * w).transpose(1, 2)
        
        # Self-Attention calculates relationships between all pixels
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        
        # Add the attention output back to the original image (Skip Connection)
        out = x_flat + attn_out
        return out.transpose(1, 2).reshape(batch_size, channels, h, w)

# 3. Downward Block (Shrink image, inject time)
class DownC(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, t_emb_dim: int):
        super().__init__()
        self.cv1 = NormActConv(in_channels, out_channels)
        self.cv2 = NormActConv(out_channels, out_channels)
        self.t_emb_layers = TimeEmbedding(out_channels, t_emb_dim)
        # nn.MaxPool2d cuts the image height/width exactly in half
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2) 
        
    def forward(self, x, t):
        x = self.cv1(x)
        # Inject the time barcode by adding it to the image channels!
        x = x + self.t_emb_layers(t)[:, :, None, None] 
        x = self.cv2(x)
        return self.downsample(x)

# 4. Middle Block (Deepest thinking, inject time, use attention)
class MidC(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, t_emb_dim: int):
        super().__init__()
        self.cv1 = NormActConv(in_channels, out_channels)
        self.t_emb_layers = TimeEmbedding(out_channels, t_emb_dim)
        self.attn = SelfAttentionBlock(out_channels)
        self.cv2 = NormActConv(out_channels, out_channels)
        
    def forward(self, x, t):
        x = self.cv1(x)
        x = x + self.t_emb_layers(t)[:, :, None, None] # Inject time
        x = self.attn(x) # Look at global context
        x = self.cv2(x)
        return x

# 5. Upward Block (Expand image, glue skip connections, inject time)
class UpC(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, t_emb_dim: int):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # THE FIX: Input channels from below + Skip channels from across!
        self.cv1 = NormActConv(in_channels + out_channels, out_channels)
        
        self.cv2 = NormActConv(out_channels, out_channels)
        self.t_emb_layers = TimeEmbedding(out_channels, t_emb_dim)

        
    def forward(self, x, skip_x, t):
        x = self.upsample(x)
        # Glue the saved high-res details (skip_x) to the current image
        x = torch.cat([x, skip_x], dim=1) 
        
        x = self.cv1(x)
        x = x + self.t_emb_layers(t)[:, :, None, None] # Inject time
        x = self.cv2(x)
        return x
       
class Unet(nn.Module):
    def __init__(self, im_channels: int = 1, t_emb_dim: int = 128):
        super().__init__()
        self.im_channels = im_channels
        self.t_emb_dim = t_emb_dim
        
        # 1. Base network to process the time barcode
        self.t_proj = nn.Sequential(
            nn.Linear(t_emb_dim, t_emb_dim),
            nn.SiLU(),
            nn.Linear(t_emb_dim, t_emb_dim)
        )
        
        self.cv1 = nn.Conv2d(im_channels, 32, kernel_size=3, padding=1)
        
        self.downs = nn.ModuleList([
            DownC(32, 64, t_emb_dim),
            DownC(64, 128, t_emb_dim),
            DownC(128, 256, t_emb_dim)
        ])
        
        self.mids = nn.ModuleList([
            MidC(256, 256, t_emb_dim),
            MidC(256, 256, t_emb_dim)
        ])
        
        self.ups = nn.ModuleList([
            UpC(256, 128, t_emb_dim),
            UpC(128, 64, t_emb_dim),
            UpC(64, 32, t_emb_dim)
        ])
        
        self.cv2 = nn.Sequential(
            nn.GroupNorm(8, 32),
            nn.Conv2d(32, im_channels, kernel_size=3, padding=1)
        )

    def forward(self, x, t):
        out = self.cv1(x)
        
        # Generate the Time Barcode
        t_emb = get_time_embedding(t, self.t_emb_dim)
        t_emb = self.t_proj(t_emb)
        
        down_outs = []
        for down in self.downs:
            down_outs.append(out)
            out = down(out, t_emb)   
            
        for mid in self.mids:
            out = mid(out, t_emb)
            
        for up in self.ups:
            down_out = down_outs.pop()     
            out = up(out, down_out, t_emb)
            
        out = self.cv2(out)
        return out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
channels = 1
num_steps = 1000

fashion_ds = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
fashion_dl = DataLoader(fashion_ds, batch_size=64, shuffle=True) # Batch size 64 is safe for 6GB VRAM

# 2. Initialize the Model and the Reverse Process
model = Unet(im_channels=channels, t_emb_dim=128).to(device)
reverse_process = DiffusionReverseProcess(num_time_steps=num_steps).to(device)

checkpoint_files = glob.glob("fashion_unet_epoch_*.pth")
checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0])) 

if os.path.exists("fashion_unet_final.pth"):
    checkpoint_files.append("fashion_unet_final.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
channels = 1 
num_steps = 1000
num_samples_for_fid = 5000

fid = FrechetInceptionDistance(feature=2048).to(device)

print("Extracting features from Real Images...")
real_images, _ = next(iter(fashion_dl)) 
real_images = real_images[:num_samples_for_fid].to(device)

# FashionMNIST is 1-channel. Inception requires 3 channels. We duplicate the channels.
real_images_rgb = real_images.repeat(1, 3, 1, 1)

# FID expects images in bytes [0, 255] format 
real_images_bytes = ((real_images_rgb + 1.0) / 2.0 * 255).to(torch.uint8)
fid.update(real_images_bytes, real=True)

model = Unet(im_channels=channels, t_emb_dim=128).to(device)
reverse_process = DiffusionReverseProcess(num_time_steps=num_steps).to(device)

checkpoint_files = glob.glob("./ddpm/jashandeep/checkpoints_FMNIST/fashion_unet_epoch_*.pth")
checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
# FILTER: Keep only every 2th checkpoint
checkpoint_files = checkpoint_files[::2] 

if os.path.exists("./ddpm/jashandeep/checkpoints_FMNIST/fashion_unet_final.pth"):
    checkpoint_files.append("./ddpm/jashandeep/checkpoints_FMNIST/fashion_unet_final.pth")

fid_scores = {}

for ckpt_path in checkpoint_files:
    print(f"\nEvaluating {ckpt_path}...")
    
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval() 
    
    with torch.no_grad():
        chunk_size = 500 
        num_chunks = num_samples_for_fid // chunk_size
        
        print(f"Generating {num_samples_for_fid} samples in {num_chunks} chunks...")
        
        for chunk in range(num_chunks):
            xt = torch.randn(chunk_size, channels, 32, 32).to(device)
            for i in reversed(range(num_steps)):
                t = torch.full((chunk_size,), i, dtype=torch.long, device=device)
                noise_pred = model(xt, t)
                xt, _ = reverse_process.sample_prev_timestep(xt, noise_pred, t)
            fake_images_rgb = xt.repeat(1, 3, 1, 1)
            fake_images_bytes = ((fake_images_rgb.clamp(-1, 1) + 1.0) / 2.0 * 255).to(torch.uint8)
            fid.update(fake_images_bytes, real=False)
            torch.cuda.empty_cache() 
            print(f"  - Chunk {chunk + 1}/{num_chunks} complete.")
        score = fid.compute()
        print(f"FID Score for {ckpt_path}: {score.item():.2f}")
        fid_scores[ckpt_path] = score.item()
        fid.reset()
        fid.update(real_images_bytes, real=True)

