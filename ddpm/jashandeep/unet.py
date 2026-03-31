import torch
import torch.nn as nn

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
        # Inject the time barcode
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
        x = self.attn(x) # Look at global context
        x = self.cv2(x)
        return x

#Upward Block
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
        
        #Base network to process the time barcode
        self.t_proj = nn.Sequential(
            nn.Linear(t_emb_dim, t_emb_dim),
            nn.SiLU(),
            nn.Linear(t_emb_dim, t_emb_dim)
        )     
        #Initial Image Scan
        self.cv1 = nn.Conv2d(im_channels, 32, kernel_size=3, padding=1)   
        #Shrinking the image, increasing channels
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
            
        # Output the predicted noise
        out = self.cv2(out)
        return out
    