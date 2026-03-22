import torch                                    #type: ignore
import torch.nn as nn                           #type: ignore
import math                                     #type: ignore
from torchvision import datasets, transforms    #type: ignore
from torch.utils.data import DataLoader         #type: ignore
from timm.utils import ModelEmaV3               #type: ignore
from tqdm import tqdm                           #type: ignore
import matplotlib.pyplot as plt                 #type: ignore
import numpy as np                              #type: ignore
from einops import rearrange                    #type: ignore
import random                                   #type: ignore
import os                                       #type: ignore
            

if torch.cuda.is_available():
    device = torch.device("cuda")
    # torch.set_float32_matmul_precision('high')
else:
    device = torch.device("cpu") 

print(f"Device: {device}")

class SinusoidalEmbeddings(torch.nn.Module):

    def __init__(self, time_steps:int, embedding_dimension: int):
        super().__init__()
        position = torch.arange(time_steps).unsqueeze(1).float()
        time_frequencies = torch.exp(torch.arange(0, embedding_dimension, 2).float() * -(math.log(10000.0) / embedding_dimension))

        embeddings = torch.zeros(time_steps, embedding_dimension, requires_grad = False)
        embeddings[:, 0::2] = torch.sin(position * time_frequencies)
        embeddings[:, 1::2] = torch.cos(position * time_frequencies)

        self.register_buffer("embeddings", embeddings)

    def forward(self, latent_var, timestep):
        sample_noise = self.embeddings[timestep]
        return sample_noise[:, :, None, None]


class ResBlock(torch.nn.Module):

    def __init__(self, in_channels: int, out_channels: int, embed_dim: int, num_groups: int, dropout_prob: float, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.ReLU = nn.ReLU(inplace = True)
        self.group_norm_1 = nn.GroupNorm(num_groups = num_groups, num_channels = in_channels)
        self.group_norm_2 = nn.GroupNorm(num_groups = num_groups, num_channels = out_channels)

        self.embedding_proj = nn.Conv2d(in_channels = embed_dim, out_channels = out_channels, kernel_size = 1)
        self.convolution_1 = nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, padding = padding)
        self.convolution_2 = nn.Conv2d(out_channels, out_channels, kernel_size = kernel_size, padding = padding)
        self.dropout = nn.Dropout(p = dropout_prob, inplace = True)
        self.residual_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, noisy_latent, embeddings):
        emb = self.embedding_proj(embeddings)
        denoise = self.convolution_1(self.ReLU(self.group_norm_1(noisy_latent)))
        denoise = denoise + emb
        denoise = self.dropout(denoise)
        denoise = self.convolution_2(self.ReLU(self.group_norm_2(denoise)))
        denoised_latent_variable = self.residual_proj(noisy_latent) + denoise
        return denoised_latent_variable
    

class Attention(nn.Module):

    def __init__(self, num_channels: int, num_heads: int, dropout_prob: float, num_groups: int = 32):

        super().__init__()

        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)
        
        self.proj1 = nn.Linear(num_channels, num_channels*3)
        self.proj2 = nn.Linear(num_channels, num_channels)

        self.num_heads = num_heads

        self.dropout_prob = dropout_prob

    def forward(self, x):
        
        residual = x 
        
        latent_var = self.group_norm(x)
        
        batch, channels, height, width = latent_var.shape
        latent_var = rearrange(latent_var, 'batch channels height width -> batch (height width) channels')
        latent_var = self.proj1(latent_var)
        
        latent_var = rearrange(latent_var, 'batch sequence_length (K num_heads feature_size) -> K batch num_heads sequence_length feature_size', K = 3, num_heads = self.num_heads)

        Query, Keys, Values = latent_var[0], latent_var[1], latent_var[2]
        latent_var = nn.functional.scaled_dot_product_attention(Query, Keys, Values, is_causal = False, dropout_p = self.dropout_prob)

        latent_var = rearrange(latent_var, 'batch num_heads sequence_length feature_size -> batch sequence_length (num_heads feature_size)')
        latent_var = self.proj2(latent_var)
        latent_var = rearrange(latent_var, 'batch (height width) feature_size -> batch feature_size height width', height = height, width = width)
        
        return residual + latent_var

class UnetLayer(nn.Module):

    def __init__(self, upscale: bool, attention: bool, num_groups: int, dropout_prob: float, num_heads: int, in_channels: int, out_channels: int, skip_channels: int, embed_dim: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.upscale = upscale
        
        if upscale:
            self.upsample = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1)
            res_in = in_channels + skip_channels
        else:
            res_in = in_channels

        self.ResBlock1 = ResBlock(res_in, out_channels, embed_dim, num_groups, dropout_prob, kernel_size, padding)
        self.ResBlock2 = ResBlock(out_channels, out_channels, embed_dim, num_groups, dropout_prob, kernel_size, padding)
        
        if not upscale:
            self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 2, padding = padding)
        
        if attention:
            self.attention_layer = Attention(out_channels, num_heads, dropout_prob, num_groups)

    def forward(self, latent_var, embeddings, skip_connection=None):
        if self.upscale:
            latent_var = self.upsample(latent_var)
            latent_var = torch.concat((latent_var, skip_connection), dim=1)

        latent_var = self.ResBlock1(latent_var, embeddings)
        if hasattr(self, 'attention_layer'):
            latent_var = self.attention_layer(latent_var)
        latent_var = self.ResBlock2(latent_var, embeddings)
        
        if not self.upscale:
            residual = latent_var
            latent_var = self.downsample(latent_var)
            return latent_var, residual
            
        return latent_var, None
    

class UNET(nn.Module):

    def __init__(self, Channels: list = [128, 256, 512, 512, 256, 128], Attentions: list = [False, True, False, False, False, True],
            Upscales: list = [False, False, False, True, True, True], num_groups: int = 32, dropout_prob: float = 0.1, num_heads: int = 8, 
            input_channels: int = 3, output_channels: int = 3, time_steps: int = 1000, kernel_size: int = 3, padding: int = 1):
        
        super().__init__()
        self.num_layers = len(Channels)
        self.shallow_conv = nn.Conv2d(input_channels, Channels[0], kernel_size = kernel_size, padding = padding)
        
        embed_dim = max(Channels)
        self.embeddings = SinusoidalEmbeddings(time_steps = time_steps, embedding_dimension = embed_dim)
        
        down_channels = []
        current_channels = Channels[0]

        for layer_index in range(self.num_layers):
            upscale = Upscales[layer_index]
            out_channels = Channels[layer_index]
            
            if upscale:
                skip_channels = down_channels.pop()
                in_channels = current_channels
            else:
                skip_channels = 0
                in_channels = current_channels
                down_channels.append(out_channels)
            
            layer = UnetLayer(upscale, Attentions[layer_index], num_groups, dropout_prob, num_heads, in_channels, out_channels, skip_channels, embed_dim, kernel_size, padding)
            setattr(self, f'Layer{layer_index + 1}', layer)
            current_channels = out_channels

        self.late_conv = nn.Conv2d(current_channels, current_channels//2, kernel_size = kernel_size, padding = padding)
        self.output_conv = nn.Conv2d(current_channels//2, output_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace = True)

    def forward(self, latent_var, timestep):
        latent_var = self.shallow_conv(latent_var)
        residuals = []
        embeddings = self.embeddings(latent_var, timestep)
        
        for layer_index in range(self.num_layers):
            layer = getattr(self, f'Layer{layer_index + 1}')
            
            if layer.upscale:
                skip = residuals.pop()
                latent_var, _ = layer(latent_var, embeddings, skip_connection=skip)
            else:
                latent_var, residual = layer(latent_var, embeddings)
                residuals.append(residual)
        
        return self.output_conv(self.relu(self.late_conv(latent_var)))
    

# class DDPM_Scheduler(nn.Module):

#     def __init__(self, num_time_steps: int = 1000, s: float = 0.008):
#         super().__init__()
        
#         steps = num_time_steps + 1
#         t = torch.linspace(0, num_time_steps, steps, dtype = torch.float64) / num_time_steps
        
#         alphas_prod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
#         alphas_prod = alphas_prod / alphas_prod[0]
        
#         betas = 1 - (alphas_prod[1:] / alphas_prod[:-1])
        
#         self.beta = torch.clip(betas, 0.0001, 0.02).float().to(device)
        
#         alpha = 1 - self.beta
#         self.alpha = torch.cumprod(alpha, dim = 0).requires_grad_(False).to(device)

#     def forward(self, t):
#         return self.beta[t], self.alpha[t]


class DDPM_Scheduler(nn.Module):

    def __init__(self, num_time_steps: int = 1000):
    
        super().__init__()

        self.beta = torch.linspace(1e-4, 0.02, num_time_steps, requires_grad = False).to(device)

        alpha = 1 - self.beta
        self.alpha = torch.cumprod(alpha, dim = 0).requires_grad_(False).to(device)

    def forward(self, t):

        return self.beta[t], self.alpha[t]

    

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def sample_images(model, scheduler, device, num_time_steps, epoch, save_dir="cifar-images"):
    os.makedirs(save_dir, exist_ok = True)
    model.eval()

    with torch.no_grad():
        for img_index in range(5):

            noise = torch.randn(1, 3, 32, 32).to(device)

            for timestep in reversed(range(0, num_time_steps)):
                t = torch.tensor([timestep], device=device)
                beta_t = scheduler.beta[t].view(1,1,1,1)
                alpha_bar_t = scheduler.alpha[t].view(1,1,1,1)
                alpha_t = 1 - beta_t

                noise = (1 / torch.sqrt(alpha_t)) * (noise - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * model(noise, t))

                if timestep > 0:
                    noise += torch.randn_like(noise) * torch.sqrt(beta_t)

            img = noise.squeeze(0).cpu()
            img = (img + 1) / 2
            img = torch.clamp(img, 0, 1)

            img = img.permute(1, 2, 0).numpy()
            plt.imsave(f"{save_dir}/epoch_{epoch}_num_{img_index+1}.png", img)

    model.train()


def train(batch_size: int = 128, num_time_steps: int = 1000, num_epochs: int = 200, seed: int = -1, ema_decay: float = 0.9999, lr: float = 2e-4, checkpoint_path: str = ""):
    set_seed(random.randint(0, 2**32-1)) if seed == -1 else set_seed(seed)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    train_dataset = datasets.CIFAR10(root='./data', train = True, download = True, transform = transform)
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, drop_last = True, num_workers = 4)

    scheduler = DDPM_Scheduler(num_time_steps = num_time_steps)
    model = UNET().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    ema = ModelEmaV3(model, decay=ema_decay)

    if checkpoint_path != "" and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['weights'])
        ema.load_state_dict(checkpoint['ema'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        print("No checkpoint found, training from scratch.")
    
    criterion = nn.MSELoss(reduction = 'mean')
    os.makedirs('cifar-checkpoints', exist_ok=True)

    for epoch in range(num_epochs):
        total_loss = 0
        
        for batch, (input_image, output_image) in enumerate(tqdm(train_loader, desc = f"Epoch {epoch+1}/{num_epochs}")):
            input_image = input_image.to(device)
            timestep = torch.randint(0, num_time_steps, (batch_size,), device = device)
            gaussian_noise = torch.randn_like(input_image, requires_grad = False)
            
            alpha = scheduler.alpha[timestep].view(batch_size, 1, 1, 1).to(device)
            noisy_image = (torch.sqrt(alpha)*input_image) + (torch.sqrt(1 - alpha)*gaussian_noise)
            
            optimizer.zero_grad()
            output = model(noisy_image, timestep)
            loss = criterion(output, gaussian_noise)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            ema.update(model)
        
        print(f'Epoch {epoch+1} | Loss {total_loss / len(train_loader):.5f}')

        if (epoch + 1) % 5 == 0:
            print(f"Generating sample at epoch {epoch+1}")
            sample_images(model, scheduler, device, num_time_steps, epoch+1)

        checkpoint = {'weights': model.state_dict(), 'optimizer': optimizer.state_dict(), 'ema': ema.state_dict()}
        torch.save(checkpoint, 'cifar-checkpoints/ddpm_checkpoint.pth')


def display_reverse(images: list):
    fig, axes = plt.subplots(1, 10, figsize=(10,1))
    
    for index, ax in enumerate(axes.flat):
        if index >= len(images):
            break
        
        image = images[index].squeeze(0)
        image = rearrange(image, 'c h w -> h w c')
        image = image.numpy()
        image = (image + 1) / 2
        image = np.clip(image, 0, 1)
        
        ax.imshow(image)
        ax.axis('off')
    
    plt.show()


def inference(checkpoint_path: str = "", num_time_steps: int = 1000, ema_decay: float = 0.9999):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model = UNET().to(device)
    model = torch.compile(model)
    model.load_state_dict(checkpoint['weights'])
    
    ema = ModelEmaV3(model, decay=ema_decay)
    ema.load_state_dict(checkpoint['ema'])
    
    scheduler = DDPM_Scheduler(num_time_steps=num_time_steps)
    times = [0, 15, 50, 100, 200, 300, 400, 550, 700, 999]

    with torch.no_grad():
        model = ema.module.eval()
        
        for index in range(10):
            images = []
            noise = torch.randn(1, 3, 32, 32).to(device)
        
            for timestep in reversed(range(1, num_time_steps)):
                timestep_list = torch.tensor([timestep], device = device)

                beta_t = scheduler.beta[timestep_list].view(1,1,1,1)
                alpha_bar_t = scheduler.alpha[timestep_list].view(1,1,1,1)
                alpha_t = 1 - beta_t

                noise = (1 / torch.sqrt(alpha_t)) * (noise - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * model(noise, timestep_list))
                
                if timestep in times:
                    images.append(noise.detach().cpu())
                
                epsilon_noise = torch.randn(1, 3, 32, 32).to(device)
                noise = noise + (epsilon_noise*torch.sqrt(beta_t))
        
            timestep_0 = torch.tensor([0], device = device)
            beta_0 = scheduler.beta[timestep_0].view(1,1,1,1)
            alpha_bar_0 = scheduler.alpha[timestep_0].view(1,1,1,1)
            alpha_0 = 1 - beta_0

            new_image = (1 / torch.sqrt(alpha_0)) * (noise - ((1 - alpha_0) / torch.sqrt(1 - alpha_bar_0)) * model(noise, timestep_0))
            images.append(new_image.detach().cpu())

            new_image_save = new_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
            new_image_save = (new_image_save + 1) / 2
            new_image_save = np.clip(new_image_save, 0, 1)

            os.makedirs("cifar-images", exist_ok=True)
            plt.imsave(f"cifar-images/sample_{index}.png", new_image_save)
            
            display_reverse(images)

def main():
    train(checkpoint_path = 'cifar-checkpoints/ddpm_checkpoint.pth', lr = 2e-4, num_epochs = 200, batch_size = 128)
    inference('cifar-checkpoints/ddpm_checkpoint.pth')

if __name__ == '__main__':
    main()