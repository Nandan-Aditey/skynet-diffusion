import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from tqdm import tqdm

# Ensure Mac GPU fallback for specialized operations
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# --- 1. RESIDUAL UNET COMPONENTS ---

class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, time_emb_dim):
        super().__init__()
        self.mlp = nn.Linear(time_emb_dim, out_c)
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.bn2 = nn.BatchNorm2d(out_c)
        
    def forward(self, x, t_emb):
        h = self.bn1(F.relu(self.conv1(x)))
        # Add time embedding
        time_emb = self.mlp(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = h + time_emb
        h = self.bn2(F.relu(self.conv2(h)))
        return h + (x if x.shape == h.shape else 0) # Residual connection

class UNet(nn.Module):
    def __init__(self, in_channels=3, time_emb_dim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # Simple Down/Up Sampling
        self.init_conv = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.down1 = ResidualBlock(64, 128, time_emb_dim)
        self.down2 = ResidualBlock(128, 256, time_emb_dim)
        self.up1 = ResidualBlock(256, 128, time_emb_dim)
        self.up2 = ResidualBlock(128, 64, time_emb_dim)
        self.out_conv = nn.Conv2d(64, in_channels, kernel_size=3, padding=1)

    def forward(self, x, t):
        t = t.float().view(-1, 1) / 1000.0 # Normalize time
        t_emb = self.time_mlp(t)
        
        x1 = self.init_conv(x)
        x2 = self.down1(x1, t_emb)
        x3 = F.max_pool2d(self.down2(x2, t_emb), 2)
        
        x4 = F.interpolate(x3, scale_factor=2, mode='bilinear')
        x5 = self.up1(x4, t_emb) + x2 # Skip connection
        x6 = self.up2(x5, t_emb) + x1 # Skip connection
        return self.out_conv(x6)

# --- 2. DIFFUSION SCHEDULER ---

class DDPMScheduler:
    def __init__(self, num_steps=1000, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.num_steps = num_steps
        self.betas = torch.linspace(beta_start, beta_end, num_steps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
    def add_noise(self, x_0, t):
        noise = torch.randn_like(x_0)
        sqrt_alpha_cumprod = self.alphas_cumprod[t].sqrt().view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod = (1 - self.alphas_cumprod[t]).sqrt().view(-1, 1, 1, 1)
        return sqrt_alpha_cumprod * x_0 + sqrt_one_minus_alpha_cumprod * noise, noise

# --- 3. THE WRAPPER & SAMPLER ---

class DDPM(nn.Module):
    def __init__(self, network, scheduler, device):
        super().__init__()
        self.network = network.to(device)
        self.scheduler = scheduler
        self.device = device

    def get_loss(self, x_0):
        t = torch.randint(0, self.scheduler.num_steps, (x_0.shape[0],), device=self.device)
        x_t, noise = self.scheduler.add_noise(x_0, t)
        predicted_noise = self.network(x_t, t)
        return F.mse_loss(predicted_noise, noise)

    @torch.no_grad()
    def sample(self, n_samples, size, channels=3):
        self.network.eval()
        x = torch.randn((n_samples, channels, size, size), device=self.device)
        for i in reversed(range(self.scheduler.num_steps)):
            t = torch.full((n_samples,), i, device=self.device, dtype=torch.long)
            pred_noise = self.network(x, t)
            
            alpha = self.scheduler.alphas[i]
            alpha_cumprod = self.scheduler.alphas_cumprod[i]
            beta = self.scheduler.betas[i]
            
            noise = torch.randn_like(x) if i > 0 else 0
            x = (1 / alpha.sqrt()) * (x - ((1 - alpha) / (1 - alpha_cumprod).sqrt()) * pred_noise) + beta.sqrt() * noise
        
        self.network.train()
        # Scale back to [0, 1] for visualization
        return (x.clamp(-1, 1) + 1) / 2

# --- 4. MAIN EXECUTION ---

CIFAR10_CLASSES = ['airplane','automobile','bird','cat','deer',
                   'dog','frog','horse','ship','truck']

def extract_cifar_to_files(root="./data/cifar_images"):
    """Download CIFAR-10 and save every image as an individual PNG file."""
    if os.path.isdir(root) and len(os.listdir(root)) == 10:
        print(f"CIFAR images already extracted to {root}/")
        return root

    print("Extracting CIFAR-10 to individual image files...")
    from torchvision.utils import save_image as _save
    raw = datasets.CIFAR10(root="./data", train=True, download=True,
                           transform=transforms.ToTensor())
    for cls in CIFAR10_CLASSES:
        os.makedirs(os.path.join(root, cls), exist_ok=True)

    counts = [0] * 10
    for img_tensor, label in tqdm(raw, desc="Saving images"):
        cls = CIFAR10_CLASSES[label]
        path = os.path.join(root, cls, f"{counts[label]:05d}.png")
        _save(img_tensor, path)
        counts[label] += 1

    print(f"Saved {sum(counts):,} images to {root}/")
    return root


def save_samples_individually(samples, epoch, out_dir="./samples"):
    """Save each generated sample as its own PNG file."""
    os.makedirs(out_dir, exist_ok=True)
    for i, img in enumerate(samples):
        path = os.path.join(out_dir, f"epoch_{epoch:03d}_sample_{i:02d}.png")
        utils.save_image(img, path)


def main():
    # Detect Mac GPU (MPS)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Settings
    IMG_SIZE = 32
    BATCH_SIZE = 64
    EPOCHS = 10

    # Extract CIFAR-10 into individual image files, then load with ImageFolder
    cifar_dir = extract_cifar_to_files()
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.ImageFolder(root=cifar_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"Loaded {len(dataset):,} individual images from {cifar_dir}/")

    # Init Model
    scheduler = DDPMScheduler(device=device)
    unet = UNet()
    model = DDPM(unet, scheduler, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Training
    for epoch in range(EPOCHS):
        pbar = tqdm(loader)
        for images, _ in pbar:
            images = images.to(device)
            optimizer.zero_grad()
            loss = model.get_loss(images)
            loss.backward()
            optimizer.step()
            pbar.set_description(f"Epoch {epoch} | Loss: {loss.item():.4f}")

        # Save each generated sample as its own file
        samples = model.sample(n_samples=16, size=IMG_SIZE)
        save_samples_individually(samples, epoch)
        print(f"Epoch {epoch}: saved 16 individual samples to ./samples/")

if __name__ == "__main__":
    main()
