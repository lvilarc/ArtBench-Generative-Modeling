"""
Convolutional Variational Autoencoder (ConvVAE) for ArtBench-10 image generation.
Simple but effective architecture for 32x32 RGB images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvVAE(nn.Module):
    """
    Convolutional VAE for 32x32 RGB images.
    
    Key features:
    - Deeper architecture with BatchNorm for stability
    - Moderate latent dimension (128) to capture art details
    - MSE reconstruction loss (appropriate for continuous RGB values)
    - Proper KL weighting to avoid posterior collapse
    """
    
    def __init__(self, latent_dim=128, image_channels=3):
        """
        Args:
            latent_dim: Dimension of latent space (default: 128)
            image_channels: Number of image channels (default: 3 for RGB)
        """
        super(ConvVAE, self).__init__()
        
        self.latent_dim = latent_dim
        self.image_channels = image_channels
        
        # Encoder: 32x32 -> 16x16 -> 8x8 -> 4x4 -> latent
        self.encoder = nn.Sequential(
            # 32x32 -> 16x16
            nn.Conv2d(image_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            # 16x16 -> 8x8
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            # 8x8 -> 4x4
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            # 4x4 -> 2x2
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
        )
        
        # Latent space: 512 * 2 * 2 = 2048
        self.fc_mu = nn.Linear(512 * 2 * 2, latent_dim)
        self.fc_logvar = nn.Linear(512 * 2 * 2, latent_dim)
        
        # Decoder: latent -> 2x2 -> 4x4 -> 8x8 -> 16x16 -> 32x32
        self.fc_decode = nn.Linear(latent_dim, 512 * 2 * 2)
        
        self.decoder = nn.Sequential(
            # 2x2 -> 4x4
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            # 4x4 -> 8x8
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            # 8x8 -> 16x16
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            # 16x16 -> 32x32
            nn.ConvTranspose2d(64, image_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # Output in [0, 1]
        )
    
    def encode(self, x):
        """
        Encode image to latent distribution parameters.
        
        Args:
            x: Input images [batch_size, channels, height, width]
            
        Returns:
            mu: Mean of latent distribution [batch_size, latent_dim]
            logvar: Log variance of latent distribution [batch_size, latent_dim]
        """
        h = self.encoder(x)
        h = h.view(h.size(0), -1)  # Flatten
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: z = mu + eps * std
        
        Args:
            mu: Mean [batch_size, latent_dim]
            logvar: Log variance [batch_size, latent_dim]
            
        Returns:
            z: Sampled latent vector [batch_size, latent_dim]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """
        Decode latent vector to image.
        
        Args:
            z: Latent vector [batch_size, latent_dim]
            
        Returns:
            x_recon: Reconstructed image [batch_size, channels, height, width] in [0,1]
        """
        h = self.fc_decode(z)
        h = h.view(h.size(0), 512, 2, 2)
        x_recon = self.decoder(h)
        return x_recon
    
    def forward(self, x):
        """
        Forward pass through VAE.
        
        Args:
            x: Input images [batch_size, channels, height, width]
            
        Returns:
            x_recon: Reconstructed images [batch_size, channels, height, width]
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar
    
    def sample(self, num_samples, device):
        """
        Generate new samples from prior N(0, I).
        
        Args:
            num_samples: Number of samples to generate
            device: Device to generate on
            
        Returns:
            samples: Generated images [num_samples, channels, height, width] in [0,1]
        """
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim, device=device)
            samples = self.decode(z)
        return samples


def vae_loss(x_recon, x, mu, logvar, beta=1.0):
    """
    VAE loss = Reconstruction Loss + β * KL Divergence
    
    Uses MSE for reconstruction (better than BCE for RGB images).
    
    Args:
        x_recon: Reconstructed images in [0,1]
        x: Original images in [0,1]
        mu: Mean of latent distribution
        logvar: Log variance of latent distribution
        beta: KL divergence weight (for annealing)
        
    Returns:
        loss: Total loss
        recon_loss: Reconstruction loss
        kl_loss: KL divergence loss
    """
    # Reconstruction loss (MSE per pixel, summed)
    recon_loss = F.mse_loss(x_recon, x, reduction='sum') / x.size(0)
    
    # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    
    # Total loss with beta weighting
    loss = recon_loss + beta * kl_loss
    
    return loss, recon_loss, kl_loss
