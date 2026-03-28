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
    
def train():
    batch_size = 64     #Safe for 6GB VRAM
    num_epochs = 5
    lr = 1e-4         
    num_timesteps = 1000
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Firing up the engines on: {device}')

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = Unet(im_channels=1, t_emb_dim=128).to(device)
    forward_process = DiffusionForwardProcess(num_time_steps=num_timesteps).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}") #cool progress bar
        
        for imgs, _ in pbar:
            imgs = imgs.to(device)
            
            # Step A: Generate random target noise and random time-steps
            actual_noise = torch.randn_like(imgs).to(device)
            t = torch.randint(0, num_timesteps, (imgs.shape[0],)).to(device)
            
            # Step B: Corrupt the images with the math we built
            noisy_imgs = forward_process.add_noise(imgs, actual_noise, t)
            
            # Step C: The U-Net makes its guess
            optimizer.zero_grad() # Clear old math
            predicted_noise = model(noisy_imgs, t)
            
            # Step D: See how wrong the U-Net was
            loss = criterion(predicted_noise, actual_noise)
            
            # Step E: Backpropagation (Learn from the mistake)
            loss.backward()
            optimizer.step()
            
            # Update progress bar with current loss
            epoch_losses.append(loss.item())
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
            
        # End of Epoch summary
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"End of Epoch {epoch+1} | Average Loss: {avg_loss:.4f}\n")

        if (epoch + 1) % 5 == 0:
            checkpoint_name = f"./ddpm/jashandeep/checkpoints_FMNIST/fashion_mnist_unet_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), checkpoint_name)
            print(f"Checkpoint saved: {checkpoint_name}\n")
        else:
            # Just print a normal newline for spacing
            print() 

    # 6. Save the final brain!
    torch.save(model.state_dict(), "fashion_mnist_unet_final.pth")
    print("Training Complete! Final model saved.")
#train()

def generate_images(num_images=16):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Generating on: {device}')
    
    num_timesteps = 1000
    img_size = 32
    channels = 1
    
    model = Unet(im_channels=channels, t_emb_dim=128).to(device)
    model.load_state_dict(torch.load("./ddpm/jashandeep/checkpoints_MNIST/mnist_unet_final.pth", map_location=device))
    model.eval() # Tell the model it's in "testing" mode, not training
    
    # 3. Initialize the Reverse Math Process
    reverse_process = DiffusionReverseProcess(num_time_steps=num_timesteps).to(device)
    
    print("Starting generation...")

    with torch.no_grad():
        x = torch.randn(num_images, channels, img_size, img_size).to(device)
        
        # Step B: Loop backwards from t=999 down to t=0
        for i in reversed(range(num_timesteps)):
            # Create a tensor of the current time step for the whole batch
            t = torch.full((num_images,), i, dtype=torch.long, device=device)
            
            # Ask the U-Net to guess the noise
            predicted_noise = model(x, t)
            
            # Use our reverse math to subtract the noise and step backwards
            x, _ = reverse_process.sample_prev_timestep(x, predicted_noise, t)
            
            # Print a little progress update so we know it hasn't crashed
            if i % 100 == 0:
                print(f"Step {i} complete...")
                
    print("Generation complete! Plotting...")
    x = x.cpu()
    x = (x + 1.0) / 2.0 
    x = torch.clamp(x, 0.0, 1.0)
    
    # Create a 4x4 grid to show the 16 images
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(x[i].squeeze(), cmap='gray')
        ax.axis('off')
        
    plt.tight_layout()
    plt.show()
    
generate_images()