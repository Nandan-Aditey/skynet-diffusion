import torch
import torch.nn as nn

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