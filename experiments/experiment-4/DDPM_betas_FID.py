import torch                                                    #type: ignore
import torch.nn as nn                                           #type: ignore                                        
import torch.nn.functional as F                                 #type: ignore
from torchvision import datasets, transforms, utils             #type: ignore
import os
import math
import shutil
from pytorch_fid.fid_score import calculate_fid_given_paths     #type: ignore

SCHEDULE_NAME = "linear" 
CHECKPOINT_STEPS = [5, 10, 15, 20, 25, 30] 
NUM_GEN_IMAGES = 500    
BATCH_SIZE = 50         
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
timesteps = 300

REAL_DIR = "fid_data/real_small"
GEN_BASE_DIR = "fid_data/generated_small"
os.makedirs(REAL_DIR, exist_ok=True)

# --- 2. ARCHITECTURE (The Missing Piece) ---

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
        x1 = self.inc(x)
        x2 = self.down1(x1, t_emb)
        x3 = F.max_pool2d(x2, 2)
        x4 = self.down2(x3, t_emb)
        m = self.mid1(x4, t_emb)
        m = self.attn(m)
        m = self.mid2(m, t_emb)
        u1 = self.up1(m)
        u1 = torch.cat([u1, x2], 1)
        u1 = self.res_up1(u1, t_emb)
        u2 = self.up2(u1)
        u2 = torch.cat([u2, x1], 1)
        u2 = self.res_up2(u2, t_emb)
        return self.outc(u2)


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
    
    return None

def prepare_real_images():
    if len(os.listdir(REAL_DIR)) >= NUM_GEN_IMAGES:
        return
    print(f"Extracting {NUM_GEN_IMAGES} real images...")
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor())
    for i in range(NUM_GEN_IMAGES):
        img, _ = dataset[i]
        utils.save_image(img, os.path.join(REAL_DIR, f"real_{i}.png"))

@torch.no_grad()
def generate_images(model, schedule_name, save_dir):
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)
    
    betas = get_beta_schedule(schedule_name, timesteps).to(device)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    
    model.eval()
    for b in range(NUM_GEN_IMAGES // BATCH_SIZE):
        img = torch.randn((BATCH_SIZE, 1, 28, 28), device=device)
        for i in reversed(range(1, timesteps)):
            t = torch.full((BATCH_SIZE,), i, device=device, dtype=torch.long)
            pred_noise = model(img, t)
            a_t, b_t, s_ot = alphas[i], betas[i], sqrt_one_minus_alphas_cumprod[i]
            mean = (1 / torch.sqrt(a_t)) * (img - ((b_t / s_ot) * pred_noise))
            img = mean + (torch.sqrt(b_t) * torch.randn_like(img) if i > 1 else 0)
        
        img = (img.clamp(-1, 1) + 1) / 2
        for j in range(BATCH_SIZE):
            idx = b * BATCH_SIZE + j
            utils.save_image(img[j], os.path.join(save_dir, f"gen_{idx}.png"))
        print(f"Generating: {((b+1)*BATCH_SIZE)}/{NUM_GEN_IMAGES}", end="\r")
    print()


if __name__ == "__main__":
    prepare_real_images()
    results = {}

    for epoch in CHECKPOINT_STEPS:
        possible_paths = [
            f"checkpoints/mnist_{SCHEDULE_NAME}_epoch_{epoch}.pth",
            f"mnist_diffusion_{SCHEDULE_NAME}_final.pth" if epoch >= 30 else "none"
        ]
        
        ckpt_path = next((p for p in possible_paths if os.path.exists(p)), None)
        
        if not ckpt_path:
            print(f"Epoch {epoch} not found, skipping...")
            continue
            
        print(f"\n--- Processing Epoch {epoch} ---")
        model = WideResUNet().to(device)
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state['model_state_dict'] if isinstance(state, dict) and 'model_state_dict' in state else state)
        
        gen_dir = os.path.join(GEN_BASE_DIR, f"epoch_{epoch}")
        generate_images(model, SCHEDULE_NAME, gen_dir)
        
        score = calculate_fid_given_paths([REAL_DIR, gen_dir], batch_size=BATCH_SIZE, device=device, dims=2048)
        results[epoch] = score
        print(f"Epoch {epoch} FID: {score:.2f}")

    print("\nFINAL RESULTS:")
    for ep, sc in sorted(results.items()):
        print(f"Epoch {ep}: {sc:.2f}")