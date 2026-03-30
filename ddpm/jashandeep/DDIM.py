import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

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


class DDIMReverseProcess(nn.Module):
    def __init__(self, num_time_steps=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        # The base math is exactly the same!
        betas = torch.linspace(beta_start, beta_end, num_time_steps)
        alphas = 1 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        
        # We only need alpha_bars for DDIM
        self.register_buffer('alpha_bars', alpha_bars)

    def _extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, 1, 1, 1).to(t.device)

    def step(self, xt, noise_pred, t, t_prev):
        """
        Takes a massive leap backward from step 't' to step 't_prev'
        """
        # 1. Get alpha bars for current and previous step
        alpha_bar_t = self._extract(self.alpha_bars, t, xt.shape)
        
        # If t_prev is negative (we stepped past 0), the noise level is 1.0 (perfectly clean)
        if t_prev[0] < 0:
            alpha_bar_t_prev = torch.ones_like(alpha_bar_t) 
        else:
            alpha_bar_t_prev = self._extract(self.alpha_bars, t_prev, xt.shape)

        # 2. Predict the perfectly clean original image (x0)
        x0_pred = (xt - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)
        #x0_pred = torch.clamp(x0_pred, -1., 1.) # Keep colors from exploding

        # 3. Calculate the mathematical "direction" pointing to the previous step
        dir_xt = torch.sqrt(1 - alpha_bar_t_prev) * noise_pred

        # 4. Deterministically jump to the previous step! (Zero random noise added)
        x_prev = torch.sqrt(alpha_bar_t_prev) * x0_pred + dir_xt

        return x_prev
        
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
        self.te_block = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_emb_dim, n_out)
        )
        
    def forward(self, x):
        return self.te_block(x)
    
class NormActConv(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, num_groups:int = 8, kernel_size: int = 3):
        super().__init__()
        self.g_norm = nn.GroupNorm(num_groups, in_channels)
        self.act = nn.SiLU()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size - 1)//2)
        
    def forward(self, x):
        x = self.g_norm(x)
        x = self.act(x)
        x = self.conv(x)
        return x

#Time Embedding
def get_time_embedding(time_steps: torch.Tensor, t_emb_dim: int) -> torch.Tensor:
    """Transforms a time integer into a 128-dimensional barcode."""
    assert t_emb_dim % 2 == 0, "time embedding must be divisible by 2."
    
    factor = 2 * torch.arange(start=0, end=t_emb_dim//2, dtype=torch.float32, device=time_steps.device) / t_emb_dim
    factor = 10000 ** factor
    
    t_emb = time_steps[:, None]
    t_emb = t_emb / factor
    t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=1)
    return t_emb

#Self-Attention Block
class SelfAttentionBlock(nn.Module):
    def __init__(self, channels: int, num_groups: int = 8, num_heads: int = 4):
        super().__init__()
        self.g_norm = nn.GroupNorm(num_groups, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        
    def forward(self, x):
        batch_size, channels, h, w = x.shape
        x_flat = x.reshape(batch_size, channels, h * w).transpose(1, 2)
        x_norm = self.g_norm(x).reshape(batch_size, channels, h * w).transpose(1, 2)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        out = x_flat + attn_out
        return out.transpose(1, 2).reshape(batch_size, channels, h, w)

#Downward Block
class DownC(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, t_emb_dim: int):
        super().__init__()
        self.cv1 = NormActConv(in_channels, out_channels)
        self.cv2 = NormActConv(out_channels, out_channels)
        self.t_emb_layers = TimeEmbedding(out_channels, t_emb_dim)
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2) 
        
    def forward(self, x, t):
        x = self.cv1(x) #1x32x32 input
        x = x + self.t_emb_layers(t)[:, :, None, None] 
        x = self.cv2(x) 
        return self.downsample(x)

#Middle Block
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
        x = self.attn(x)
        x = self.cv2(x)
        return x

#Upward Block
class UpC(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, t_emb_dim: int):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.cv1 = NormActConv(in_channels + out_channels, out_channels)
        
        self.cv2 = NormActConv(out_channels, out_channels)
        self.t_emb_layers = TimeEmbedding(out_channels, t_emb_dim)
        
    def forward(self, x, skip_x, t):
        x = self.upsample(x)
        x = torch.cat([x, skip_x], dim=1)
        
        x = self.cv1(x)
        x = x + self.t_emb_layers(t)[:, :, None, None]
        x = self.cv2(x)
        return x
        
    def forward(self, x, skip_x, t):
        x = self.upsample(x)
        x = torch.cat([x, skip_x], dim=1)     
        x = self.cv1(x)
        x = x + self.t_emb_layers(t)[:, :, None, None]
        x = self.cv2(x)
        return x
       
class Unet(nn.Module):
    def __init__(self, im_channels: int = 1, t_emb_dim: int = 128):
        super().__init__()
        self.im_channels = im_channels
        self.t_emb_dim = t_emb_dim
        
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