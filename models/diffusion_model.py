"""
Simple DDPM model implementation.
Based on class notebook - minimal working version.
"""
import torch
import torch.nn as nn
import math


# ============================================================================
# TIME EMBEDDING
# ============================================================================

class SinusoidalPosEmb(nn.Module):
    """Sinusoidal position embedding for timesteps."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# ============================================================================
# RESIDUAL BLOCK
# ============================================================================

class ResBlock(nn.Module):
    """Residual block with time embedding."""
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU()
        
        # Shortcut if channels change
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x, time_emb):
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)
        
        # Add time embedding
        time_emb = self.time_mlp(time_emb)
        h = h + time_emb[:, :, None, None]
        
        h = self.norm2(h)
        h = self.act(h)
        h = self.conv2(h)
        
        return self.shortcut(x) + h


# ============================================================================
# SIMPLE UNET
# ============================================================================

class SimpleUNet(nn.Module):
    """
    Simple UNet for 32x32 RGB images.
    Based on class notebook architecture.
    
    Architecture:
    - 3-level encoder-decoder
    - Skip connections
    - Time embeddings injected at each ResBlock
    - ~1.5M parameters
    """
    def __init__(self, in_channels=3, model_channels=64):
        super().__init__()
        
        # Time embedding
        time_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(model_channels),
            nn.Linear(model_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        
        # Initial conv
        self.init_conv = nn.Conv2d(in_channels, model_channels, 3, padding=1)
        
        # Down: 32 -> 16
        self.down1_res = ResBlock(model_channels, model_channels, time_dim)
        self.down1_pool = nn.Conv2d(model_channels, model_channels, 3, stride=2, padding=1)
        
        # Down: 16 -> 8
        self.down2_res = ResBlock(model_channels, model_channels, time_dim)
        self.down2_pool = nn.Conv2d(model_channels, model_channels * 2, 3, stride=2, padding=1)
        
        # Middle: 8x8
        self.mid_res1 = ResBlock(model_channels * 2, model_channels * 2, time_dim)
        self.mid_res2 = ResBlock(model_channels * 2, model_channels * 2, time_dim)
        
        # Up: 8 -> 16
        self.up2_conv = nn.ConvTranspose2d(model_channels * 2, model_channels, 4, stride=2, padding=1)
        self.up2_res = ResBlock(model_channels * 2, model_channels, time_dim)  # After concat: 128 -> 64
        
        # Up: 16 -> 32
        self.up1_conv = nn.ConvTranspose2d(model_channels, model_channels, 4, stride=2, padding=1)
        self.up1_res = ResBlock(model_channels * 2, model_channels, time_dim)  # After concat: 128 -> 64
        
        # Output
        self.out_conv = nn.Conv2d(model_channels, in_channels, 3, padding=1)
    
    def forward(self, x, t):
        # Time embedding
        t_emb = self.time_embed(t)
        
        # Initial
        h_init = self.init_conv(x)
        
        # Down 1
        h1 = self.down1_res(h_init, t_emb)
        h1_pool = self.down1_pool(h1)
        
        # Down 2
        h2 = self.down2_res(h1_pool, t_emb)
        h2_pool = self.down2_pool(h2)
        
        # Middle
        h_mid = self.mid_res1(h2_pool, t_emb)
        h_mid = self.mid_res2(h_mid, t_emb)
        
        # Up 2
        h_up2 = self.up2_conv(h_mid)
        h_up2 = torch.cat([h_up2, h2], dim=1)
        h_up2 = self.up2_res(h_up2, t_emb)
        
        # Up 1
        h_up1 = self.up1_conv(h_up2)
        h_up1 = torch.cat([h_up1, h1], dim=1)
        h_up1 = self.up1_res(h_up1, t_emb)
        
        # Output
        return self.out_conv(h_up1)


# ============================================================================
# SCHEDULE FUNCTIONS
# ============================================================================

def get_linear_schedule(num_timesteps=1000, beta_start=0.0001, beta_end=0.02):
    """
    Linear beta schedule.
    Returns betas tensor of shape [num_timesteps].
    """
    return torch.linspace(beta_start, beta_end, num_timesteps)
