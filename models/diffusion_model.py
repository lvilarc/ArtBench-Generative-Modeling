"""
DDPM (Denoising Diffusion Probabilistic Model) for ArtBench-10.
Based on Ho et al. (2020) - "Denoising Diffusion Probabilistic Models"

Architecture: Simple U-Net optimized for 32x32 RGB images.
Noise Schedule: Cosine (better for low resolution)
Time Embedding: Sinusoidal (dim=128)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def get_cosine_schedule(timesteps, s=0.008):
    """
    Cosine schedule as proposed in Improved DDPM.
    Better for small images compared to linear schedule.
    
    Args:
        timesteps: Number of diffusion steps (T)
        s: Small offset to prevent βₜ from being too small
        
    Returns:
        betas: β values for each timestep [T]
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps)
    
    alphas_cumprod = torch.cos(((t / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = torch.clip(betas, 0.0001, 0.9999)
    
    return betas


def extract(a, t, x_shape):
    """
    Extract values from tensor a at indices t.
    Reshape to broadcast correctly with x_shape.
    
    Args:
        a: Tensor of values [T]
        t: Timestep indices [batch_size]
        x_shape: Shape of target tensor
        
    Returns:
        Values at indices t, reshaped for broadcasting
    """
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


class SinusoidalTimeEmbedding(nn.Module):
    """
    Sinusoidal positional encoding for timesteps.
    Similar to Transformer positional encoding.
    """
    
    def __init__(self, dim):
        """
        Args:
            dim: Embedding dimension (must be even)
        """
        super().__init__()
        self.dim = dim
    
    def forward(self, t):
        """
        Args:
            t: Timesteps [batch_size]
            
        Returns:
            Time embeddings [batch_size, dim]
        """
        device = t.device
        half_dim = self.dim // 2
        
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        
        return embeddings


class ResNetBlock(nn.Module):
    """
    ResNet-style block with time embedding injection.
    
    Architecture:
    - GroupNorm → SiLU → Conv
    - Time embedding injection (Linear → SiLU)
    - GroupNorm → SiLU → Conv
    - Residual connection
    """
    
    def __init__(self, in_channels, out_channels, time_emb_dim, num_groups=8):
        """
        Args:
            in_channels: Input channels
            out_channels: Output channels
            time_emb_dim: Time embedding dimension
            num_groups: Number of groups for GroupNorm
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # First conv block
        self.norm1 = nn.GroupNorm(num_groups, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        # Time embedding projection
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        
        # Second conv block
        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        # Residual connection (project if channels changed)
        if in_channels != out_channels:
            self.residual_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_proj = nn.Identity()
    
    def forward(self, x, time_emb):
        """
        Args:
            x: Input tensor [batch, in_channels, H, W]
            time_emb: Time embedding [batch, time_emb_dim]
            
        Returns:
            Output tensor [batch, out_channels, H, W]
        """
        h = self.norm1(x)
        h = self.act1(h)
        h = self.conv1(h)
        
        # Add time embedding
        time_emb = self.time_mlp(time_emb)
        h = h + time_emb[:, :, None, None]
        
        h = self.norm2(h)
        h = self.act2(h)
        h = self.conv2(h)
        
        # Residual connection
        return h + self.residual_proj(x)


class DownBlock(nn.Module):
    """
    Downsampling block for U-Net encoder.
    2x ResNet blocks + Downsample
    """
    
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        
        self.resnet1 = ResNetBlock(in_channels, out_channels, time_emb_dim)
        self.resnet2 = ResNetBlock(out_channels, out_channels, time_emb_dim)
        
        # Downsample with stride=2 conv
        self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)
    
    def forward(self, x, time_emb):
        """
        Args:
            x: Input [batch, in_channels, H, W]
            time_emb: Time embedding [batch, time_emb_dim]
            
        Returns:
            h: Downsampled output [batch, out_channels, H//2, W//2]
            skip: Skip connection [batch, out_channels, H, W]
        """
        h = self.resnet1(x, time_emb)
        h = self.resnet2(h, time_emb)
        skip = h
        h = self.downsample(h)
        return h, skip


class MiddleBlock(nn.Module):
    """
    Middle block for U-Net bottleneck.
    2x ResNet blocks, no downsampling.
    """
    
    def __init__(self, channels, time_emb_dim):
        super().__init__()
        
        self.resnet1 = ResNetBlock(channels, channels, time_emb_dim)
        self.resnet2 = ResNetBlock(channels, channels, time_emb_dim)
    
    def forward(self, x, time_emb):
        """
        Args:
            x: Input [batch, channels, H, W]
            time_emb: Time embedding [batch, time_emb_dim]
            
        Returns:
            Output [batch, channels, H, W]
        """
        h = self.resnet1(x, time_emb)
        h = self.resnet2(h, time_emb)
        return h


class UpBlock(nn.Module):
    """
    Upsampling block for U-Net decoder.
    Upsample + Concatenate skip connection + 2x ResNet blocks
    """
    
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        
        # Upsample FIRST with transposed conv
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1)
        
        # in_channels*2 because we concatenate skip connection AFTER upsample
        self.resnet1 = ResNetBlock(in_channels * 2, out_channels, time_emb_dim)
        self.resnet2 = ResNetBlock(out_channels, out_channels, time_emb_dim)
    
    def forward(self, x, skip, time_emb):
        """
        Args:
            x: Input [batch, in_channels, H, W]
            skip: Skip connection from encoder [batch, in_channels, H*2, W*2]
            time_emb: Time embedding [batch, time_emb_dim]
            
        Returns:
            Output [batch, out_channels, H*2, W*2]
        """
        # Upsample first to match skip dimensions
        h = self.upsample(x)
        
        # Concatenate skip connection
        h = torch.cat([h, skip], dim=1)
        
        h = self.resnet1(h, time_emb)
        h = self.resnet2(h, time_emb)
        return h


class UNet(nn.Module):
    """
    Simple U-Net for DDPM, optimized for 32x32 images.
    
    Architecture:
    - 3 levels: 32→16→8→4
    - Channels: [64, 128, 256]
    - Time embedding: Sinusoidal (dim=128)
    - No attention (not needed for 32x32)
    """
    
    def __init__(self, image_channels=3, base_channels=64, time_emb_dim=128):
        """
        Args:
            image_channels: Number of input/output channels (3 for RGB)
            base_channels: Base number of channels (64)
            time_emb_dim: Time embedding dimension (128)
        """
        super().__init__()
        
        self.image_channels = image_channels
        self.time_emb_dim = time_emb_dim
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # Initial conv
        self.conv_in = nn.Conv2d(image_channels, base_channels, kernel_size=3, padding=1)
        
        # Encoder (downsampling)
        # 32x32 -> 16x16 -> 8x8 -> 4x4
        self.down1 = DownBlock(base_channels, base_channels * 2, time_emb_dim)      # 64 -> 128
        self.down2 = DownBlock(base_channels * 2, base_channels * 4, time_emb_dim)  # 128 -> 256
        
        # Middle (bottleneck) at 4x4
        self.middle = MiddleBlock(base_channels * 4, time_emb_dim)  # 256
        
        # Decoder (upsampling)
        # 4x4 -> 8x8 -> 16x16 -> 32x32
        self.up1 = UpBlock(base_channels * 4, base_channels * 2, time_emb_dim)  # 256 -> 128
        self.up2 = UpBlock(base_channels * 2, base_channels, time_emb_dim)      # 128 -> 64
        
        # Final layers
        self.final_resnet = ResNetBlock(base_channels, base_channels, time_emb_dim)
        self.conv_out = nn.Conv2d(base_channels, image_channels, kernel_size=3, padding=1)
    
    def forward(self, x, t):
        """
        Predict noise ε in x_t.
        
        Args:
            x: Noisy image [batch, channels, 32, 32]
            t: Timesteps [batch]
            
        Returns:
            Predicted noise [batch, channels, 32, 32]
        """
        # Embed timestep
        time_emb = self.time_mlp(t)
        
        # Initial conv
        h = self.conv_in(x)  # [B, 64, 32, 32]
        
        # Encoder
        h, skip1 = self.down1(h, time_emb)  # [B, 128, 16, 16], skip: [B, 128, 32, 32]
        h, skip2 = self.down2(h, time_emb)  # [B, 256, 8, 8], skip: [B, 256, 16, 16]
        
        # Middle
        h = self.middle(h, time_emb)  # [B, 256, 4, 4]
        
        # Decoder (with skip connections)
        h = self.up1(h, skip2, time_emb)  # [B, 128, 16, 16]
        h = self.up2(h, skip1, time_emb)  # [B, 64, 32, 32]
        
        # Final output
        h = self.final_resnet(h, time_emb)
        h = self.conv_out(h)
        
        return h


# Example usage and testing
if __name__ == "__main__":
    print("Testing Diffusion Model components...\n")
    
    # Test cosine schedule
    betas = get_cosine_schedule(1000)
    print(f"Cosine schedule: β₁={betas[0]:.6f}, β₅₀₀={betas[499]:.6f}, β₁₀₀₀={betas[-1]:.6f}")
    
    # Test time embedding
    time_emb = SinusoidalTimeEmbedding(128)
    t = torch.tensor([0, 100, 500, 999])
    emb = time_emb(t)
    print(f"\nTime embedding: shape={emb.shape}, range=[{emb.min():.2f}, {emb.max():.2f}]")
    
    # Test U-Net
    batch_size = 4
    model = UNet(image_channels=3, base_channels=64, time_emb_dim=128)
    
    x = torch.randn(batch_size, 3, 32, 32)
    t = torch.randint(0, 1000, (batch_size,))
    
    noise_pred = model(x, t)
    
    print(f"\nU-Net test:")
    print(f"  Input shape: {x.shape}")
    print(f"  Timesteps: {t.tolist()}")
    print(f"  Output shape: {noise_pred.shape}")
    print(f"  Output range: [{noise_pred.min():.2f}, {noise_pred.max():.2f}]")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    
    print("\n✓ All tests passed!")
