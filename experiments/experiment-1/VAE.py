import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from tqdm import tqdm
import numpy as np
from torchvision.models import inception_v3, Inception_V3_Weights

# Ensure Mac GPU fallback for specialized operations
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'


class Encoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64,  kernel_size=4, stride=2, padding=1),  # 32 -> 16
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),           # 16 -> 8
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),          # 8 -> 4
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.fc_mu     = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)

    def forward(self, x):
        h = self.net(x)
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    def __init__(self, out_channels=3, latent_dim=128):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64,  kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        h = self.fc(z).view(-1, 256, 4, 4)
        return self.net(h)



class VAE(nn.Module):
    def __init__(self, in_channels=3, latent_dim=128, device="cpu"):
        super().__init__()
        self.encoder = Encoder(in_channels, latent_dim).to(device)
        self.decoder = Decoder(in_channels, latent_dim).to(device)
        self.latent_dim = latent_dim
        self.device = device

    def reparameterize(self, mu, logvar):
        std = (0.5 * logvar).exp()
        return mu + std * torch.randn_like(std)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

    def get_loss(self, x, beta=1.0):
        recon, mu, logvar = self(x)
        recon_loss = F.mse_loss(recon, x, reduction='sum') / x.shape[0]
        kl_loss    = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(1).mean()
        return recon_loss + beta * kl_loss, recon_loss, kl_loss

    @torch.no_grad()
    def sample(self, n_samples, size=None):
        self.eval()
        z = torch.randn(n_samples, self.latent_dim, device=self.device)
        imgs = self.decoder(z)
        self.train()
        return (imgs.clamp(-1, 1) + 1) / 2



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
    for img_tensor, label in tqdm(raw, desc="Saving images", file=sys.stdout):
        cls = CIFAR10_CLASSES[label]
        path = os.path.join(root, cls, f"{counts[label]:05d}.png")
        _save(img_tensor, path)
        counts[label] += 1

    print(f"Saved {sum(counts):,} images to {root}/")
    return root


def save_samples_individually(samples, epoch, out_dir="./samples_vae"):
    """Save each generated sample as its own PNG file."""
    os.makedirs(out_dir, exist_ok=True)
    for i, img in enumerate(samples):
        path = os.path.join(out_dir, f"epoch_{epoch:03d}_sample_{i:02d}.png")
        utils.save_image(img, path)


def compute_fid(model, loader, device, n_samples=1000, img_size=32):
    """Compute FID between real CIFAR-10 images and model samples.

    Strategy: run InceptionV3 feature extraction on the accelerator (CUDA or MPS)
    in float32, then compute the Fréchet distance on CPU via numpy/scipy (float64).
    This avoids the MPS float64 limitation while still using the GPU for the
    expensive part (InceptionV3 forward passes).
    For CUDA, the generative model is temporarily offloaded to CPU first so
    InceptionV3 has enough VRAM.
    """
    from scipy import linalg
    t0 = time.perf_counter()

    real_batches, real_count = [], 0
    for images, _ in loader:
        real_batches.append(((images.clamp(-1, 1) + 1) / 2 * 255).to(torch.uint8))
        real_count += images.shape[0]
        if real_count >= n_samples:
            break

    fake_batches, fake_count = [], 0
    while fake_count < n_samples:
        n = min(64, n_samples - fake_count)
        samples = model.sample(n_samples=n)
        fake_batches.append((samples * 255).to(torch.uint8).cpu())
        fake_count += n

    if device.type == "cuda":
        model.cpu()
        torch.cuda.empty_cache()

    inception = inception_v3(weights=Inception_V3_Weights.DEFAULT)
    inception.fc = nn.Identity()
    inception = inception.to(device).eval()

    def extract_features(batches):
        feats = []
        with torch.no_grad():
            for batch in batches:
                x = F.interpolate(batch.float().to(device) / 255.0,
                                  size=(299, 299), mode='bilinear', align_corners=False)
                x = (x - 0.5) / 0.5
                feats.append(inception(x).cpu().float())
        return torch.cat(feats).numpy()

    real_feats = extract_features(real_batches)
    fake_feats = extract_features(fake_batches)

    mu_r, sig_r = real_feats.mean(0), np.cov(real_feats, rowvar=False)
    mu_f, sig_f = fake_feats.mean(0), np.cov(fake_feats, rowvar=False)
    diff = mu_r - mu_f
    covmean, _ = linalg.sqrtm(sig_r @ sig_f, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    score = float(diff @ diff + np.trace(sig_r + sig_f - 2 * covmean))

    if device.type == "cuda":
        model.to(device)

    elapsed = time.perf_counter() - t0
    return score, elapsed



def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    IMG_SIZE   = 32
    BATCH_SIZE = 64
    EPOCHS     = 100
    LATENT_DIM = 128
    BETA       = 1.0

    cifar_dir = extract_cifar_to_files()
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.ImageFolder(root=cifar_dir, transform=transform)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"Loaded {len(dataset):,} individual images from {cifar_dir}/")

    model     = VAE(latent_dim=LATENT_DIM, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for epoch in range(EPOCHS):
        pbar = tqdm(loader, file=sys.stdout)
        for images, _ in pbar:
            images = images.to(device)
            optimizer.zero_grad()
            loss, recon_loss, kl_loss = model.get_loss(images, beta=BETA)
            loss.backward()
            optimizer.step()
            pbar.set_description(
                f"Epoch {epoch} | Loss: {loss.item():.4f} "
                f"(recon: {recon_loss.item():.4f}, kl: {kl_loss.item():.4f})"
            )

        samples = model.sample(n_samples=16)
        save_samples_individually(samples, epoch)

        fid_score, fid_time = compute_fid(model, loader, device, n_samples=1000, img_size=IMG_SIZE)
        print(f"Epoch {epoch}: saved 16 samples to ./samples_vae/ | FID: {fid_score:.2f} (computed in {fid_time:.1f}s)")

if __name__ == "__main__":
    main()
