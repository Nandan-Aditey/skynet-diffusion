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
else:
    device = torch.device("cpu") 

print(f"Device: {device}")

class SinusoidalEmbeddings(torch.nn.Module):

    def __init__(self, time_steps:int, embedding_dimension: int):
        """
        Creates a lookup table that maps each timestep t ∈ [0, time_steps) to a vector of dimension `embedding_dimension`.
        The encoding uses sinusoidal functions at multiple frequencies, allowing the model to represent both coarse and fine 
        differences between timesteps.

        Args:
            time_steps (int)
            embedding_dimension (int)

        Attributes:
            embeddings (Tensor)
        """

        super().__init__()

        position = torch.arange(time_steps).unsqueeze(1).float()

        time_frequencies = torch.exp(torch.arange(0, embedding_dimension, 2).float() * -(math.log(10000.0) / embedding_dimension))

        embeddings = torch.zeros(time_steps, embedding_dimension, requires_grad = False)
        embeddings[:, 0::2] = torch.sin(position * time_frequencies)
        embeddings[:, 1::2] = torch.cos(position * time_frequencies)

        self.register_buffer("embeddings", embeddings)


    def forward(self, latent_var, timestep):
        """
        Given a batch of latent variables and their corresponding timesteps, this method returns the sinusoidal embeddings for
        those timesteps and reshapes them so they can be added to feature maps in a convolutional network.

        Args:
            latent_var (Tensor)
            timestep (Tensor)
        Returns:
            sample_noise (Tensor)
        """

        sample_noise = self.embeddings[timestep]

        return sample_noise[:, :, None, None]



class ResBlock(torch.nn.Module):

    def __init__(self, num_channels: int, num_groups: int, dropout_prob: float, kernel_size: int = 3, padding: int = 1):

        super().__init__()

        self.ReLU = nn.ReLU(inplace = True)
        
        self.group_norm_1 = nn.GroupNorm(num_groups = num_groups, num_channels = num_channels)
        self.group_norm_2 = nn.GroupNorm(num_groups = num_groups, num_channels = num_channels)

        self.convolution_1 = nn.Conv2d(num_channels, num_channels, kernel_size = kernel_size, padding = padding)
        self.convolution_2 = nn.Conv2d(num_channels, num_channels, kernel_size = kernel_size, padding = padding)

        self.dropout = nn.Dropout(p = dropout_prob, inplace = True)


    def forward(self, noisy_latent, embeddings):

        noisy_latent = noisy_latent + embeddings[:, :noisy_latent.shape[1], :, :]

        denoise = self.convolution_1(self.ReLU(self.group_norm_1(noisy_latent)))
        denoise = self.dropout(denoise)
        denoise = self.convolution_2(self.ReLU(self.group_norm_2(denoise)))

        denoised_latent_variable = noisy_latent + denoise
        
        return denoised_latent_variable
    


class Attention(nn.Module):

    def __init__(self, num_channels: int, num_heads:int , dropout_prob: float):

        super().__init__()

        self.proj1 = nn.Linear(num_channels, num_channels*3)
        self.proj2 = nn.Linear(num_channels, num_channels)
        
        self.num_heads = num_heads

        self.dropout_prob = dropout_prob


    def forward(self, latent_var):

        height, width = latent_var.shape[2:]

        latent_var = rearrange(latent_var, 'batch channels height width -> batch (height width) channels')
        latent_var = self.proj1(latent_var)
        latent_var = rearrange(latent_var, 'batch sequence_length (feature_size num_heads K) -> K batch num_heads sequence_length feature_size', K = 3, num_heads = self.num_heads)

        Query,Keys,Values = latent_var[0], latent_var[1], latent_var[2]
        latent_var = nn.functional.scaled_dot_product_attention(Query, Keys, Values, is_causal = False, dropout_p = self.dropout_prob)

        latent_var = rearrange(latent_var, 'batch num_heads (height width) feature_size -> batch height width (feature_size num_heads)', height = height, width = width)
        latent_var = self.proj2(latent_var)

        return rearrange(latent_var, 'batch height width feature_size -> batch feature_size height width')



class UnetLayer(nn.Module):

    def __init__(self, upscale: bool, attention: bool, num_groups: int, dropout_prob: float, num_heads: int, num_channels: int, kernel_size: int = 3, padding: int = 1):

        super().__init__()

        self.ResBlock1 = ResBlock(num_channels, num_groups, dropout_prob, kernel_size, padding)
        self.ResBlock2 = ResBlock(num_channels, num_groups, dropout_prob, kernel_size, padding)
        
        if upscale:
            self.conv = nn.ConvTranspose2d(num_channels, num_channels//2, kernel_size = 4, stride = 2, padding = padding)
        else:
            self.conv = nn.Conv2d(num_channels, num_channels*2, kernel_size = 3, stride = 2, padding = padding)
        
        if attention:
            self.attention_layer = Attention(num_channels, num_heads, dropout_prob)


    def forward(self, latent_var, embeddings):

        latent_var = self.ResBlock1(latent_var, embeddings)
        
        if hasattr(self, 'attention_layer'):
            latent_var = self.attention_layer(latent_var)
        
        latent_var = self.ResBlock2(latent_var, embeddings)
        
        return self.conv(latent_var), latent_var
    


class UNET(nn.Module):

    def __init__(self, Channels: list = [64, 128, 256, 512, 512, 384], Attentions: list = [False, True, False, False, False, True],
            Upscales: list = [False, False, False, True, True, True], num_groups: int = 32, dropout_prob: float = 0.1, num_heads: int = 8, 
            input_channels: int = 1, output_channels: int = 1, time_steps: int = 1000, kernel_size: int = 3, padding: int = 1):
        
        super().__init__()
        
        self.num_layers = len(Channels)
        self.shallow_conv = nn.Conv2d(input_channels, Channels[0], kernel_size = kernel_size, padding = padding)
        
        out_channels = (Channels[-1]//2) + Channels[0]
        
        self.late_conv = nn.Conv2d(out_channels, out_channels//2, kernel_size = kernel_size, padding = padding)
        self.output_conv = nn.Conv2d(out_channels//2, output_channels, kernel_size=1)
        
        self.relu = nn.ReLU(inplace = True)

        self.embeddings = SinusoidalEmbeddings(time_steps = time_steps, embedding_dimension = max(Channels))
        
        for layer_index in range(self.num_layers):

            layer = UnetLayer(Upscales[layer_index], Attentions[layer_index], num_groups, dropout_prob, num_heads, 
                              Channels[layer_index], kernel_size, padding)

            setattr(self, f'Layer{layer_index + 1}', layer)


    def forward(self, latent_var, timestep):

        latent_var = self.shallow_conv(latent_var)
        residuals = []

        embeddings = self.embeddings(latent_var, timestep)
        
        for layer_index in range(self.num_layers//2):

            layer = getattr(self, f'Layer{layer_index + 1}')
            embeddings = self.embeddings(latent_var, timestep)
            latent_var, residual = layer(latent_var, embeddings)

            residuals.append(residual)
        
        for layer_index in range(self.num_layers//2, self.num_layers):

            layer = getattr(self, f'Layer{layer_index + 1}')
            latent_var = torch.concat((layer(latent_var, embeddings)[0], residuals[self.num_layers - layer_index - 1]), dim = 1)
        
        return self.output_conv(self.relu(self.late_conv(latent_var)))
    


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


def sample_images(model, scheduler, device, num_time_steps, epoch, save_dir="mnist-images"):
    import os
    os.makedirs(save_dir, exist_ok=True)

    model.eval()

    with torch.no_grad():
        noise = torch.randn(1, 1, 32, 32).to(device)

        for timestep in reversed(range(1, num_time_steps)):
            t = torch.tensor([timestep], device=device)

            beta_t = scheduler.beta[t].view(1, 1, 1, 1)
            alpha_t = scheduler.alpha[t].view(1, 1, 1, 1)

            temp = beta_t / ((torch.sqrt(1 - alpha_t)) * (torch.sqrt(1 - beta_t)))

            noise = (1 / torch.sqrt(1 - beta_t)) * noise - (temp * model(noise, t))

            if timestep > 1:
                noise += torch.randn_like(noise) * torch.sqrt(beta_t)

        img = noise.squeeze(0).cpu()
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        plt.imsave(f"{save_dir}/epoch_{epoch}.png", img.squeeze(0).numpy(), cmap='gray')

    model.train()


def train(batch_size: int = 64, num_time_steps: int = 1000, num_epochs: int = 30, seed: int = -1, ema_decay: float = 0.9999, lr = 2e-5, checkpoint_path: str = ""):

    set_seed(random.randint(0, 2**32-1)) if seed == -1 else set_seed(seed)

    train_dataset = datasets.MNIST(root = './data', train = True, download = True,transform = transforms.ToTensor())
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, drop_last = True, num_workers = 4)

    scheduler = DDPM_Scheduler(num_time_steps = num_time_steps)
    model = UNET().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    ema = ModelEmaV3(model, decay=ema_decay)

    if checkpoint_path != "" and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['weights'])
        ema.load_state_dict(checkpoint['ema'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    else:
            print("No checkpoint found, training from scratch.")
    
    criterion = nn.MSELoss(reduction = 'mean')

    for epoch in range(num_epochs):

        total_loss = 0
        
        for batch, (input_image, output_image) in enumerate(tqdm(train_loader, desc = f"Epoch {epoch+1}/{num_epochs}")):

            input_image = input_image.to(device)
            input_image = nn.functional.pad(input_image, (2, 2, 2, 2))

            timestep = torch.randint(0, num_time_steps, (batch_size,), device = device)
            gaussian_noise = torch.randn_like(input_image, requires_grad = False)
            alpha = scheduler.alpha[timestep].view(batch_size, 1, 1, 1).to(device)
            noisy_image = (torch.sqrt(alpha)*input_image) + (torch.sqrt(1 - alpha)*gaussian_noise)
            
            output = model(noisy_image, timestep)

            optimizer.zero_grad()
            
            loss = criterion(output, gaussian_noise)
            total_loss += loss.item()
            loss.backward()
            
            optimizer.step()
            
            ema.update(model)
        
        print(f'Epoch {epoch+1} | Loss {total_loss / (60000/batch_size):.5f}')

        if (epoch + 1) % 5 == 0:
            print(f"Generating sample at epoch {epoch+1}")
            sample_images(ema.module, scheduler, device, num_time_steps, epoch+1)

        checkpoint = {'weights': model.state_dict(), 'optimizer': optimizer.state_dict(), 'ema': ema.state_dict()}
        
        torch.save(checkpoint, 'mnist-checkpoints/ddpm_checkpoint.pth')



def display_reverse(images: list):

    fig, axes = plt.subplots(1, 10, figsize=(10,1))
    
    for index, axes in enumerate(axes.flat):
        
        image = images[index].squeeze(0)
        image = rearrange(image, 'c h w -> h w c')
        image = image.numpy()
        
        axes.imshow(image)
        axes.axis('off')
    
    plt.show()



def inference(checkpoint_path: str = "", num_time_steps: int = 1000, ema_decay: float = 0.9999):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model = UNET().to(device)
    model.load_state_dict(checkpoint['weights'])
    
    ema = ModelEmaV3(model, decay=ema_decay)
    ema.load_state_dict(checkpoint['ema'])
    
    scheduler = DDPM_Scheduler(num_time_steps=num_time_steps)
    
    times = [0, 15, 50, 100, 200, 300, 400, 550, 700, 999]
    images = []

    model.eval()
    with torch.no_grad():
        for index in range(20):
            noise = torch.randn(1, 1, 32, 32).to(device)
        
            for timestep in reversed(range(1, num_time_steps)):
                timestep_tensor = torch.tensor([timestep], device=device)

                beta_t = scheduler.beta[timestep_tensor].view(1, 1, 1, 1)
                alpha_t = scheduler.alpha[timestep_tensor].view(1, 1, 1, 1)

                pred_noise = ema.module(noise, timestep_tensor)
                temp = beta_t / (torch.sqrt(1 - alpha_t) * torch.sqrt(1 - beta_t))
                noise = (1 / torch.sqrt(1 - beta_t)) * noise - (temp * pred_noise)
                
                if timestep > 1:
                    epsilon_noise = torch.randn_like(noise)
                    noise = noise + (epsilon_noise * torch.sqrt(beta_t))
                
                if timestep in times:
                    images.append(noise.detach().cpu())
        
            final_img = noise.detach().cpu().squeeze()
            final_img = (final_img - final_img.min()) / (final_img.max() - final_img.min() + 1e-8)
            
            plt.imsave(f"mnist-images/sample_{index}.png", final_img.numpy(), cmap='gray')
            
            images.append(noise.detach().cpu())
            display_reverse(images)
            
            images = []



def main():

    #train(checkpoint_path = 'mnist-checkpoints/ddpm_checkpoint.pth', lr = 2e-5, num_epochs = 75)
    inference('mnist-checkpoints/ddpm_checkpoint.pth')
    

if __name__ == '__main__':
    main()