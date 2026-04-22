"""
Convolutional Variational Autoencoder (ConvVAE) for ArtBench-10 image generation.
Implements a VAE with convolutional layers for learning the latent distribution
of artwork images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvVAE(nn.Module):
    """
    Convolutional Variational Autoencoder for 32x32 images.
    
    Architecture:
    - Encoder: Conv layers -> Latent space (mean & logvar)
    - Decoder: Deconv layers -> Reconstructed image
    - Loss: Reconstruction (BCE) + KL divergence
    """
    
    def __init__(self, latent_dim=20, image_channels=3):
        """
        Initialize ConvVAE.
        
        Args:
            latent_dim: Dimension of the latent space (default: 20)
            image_channels: Number of image channels (default: 3 for RGB)
        """
        super(ConvVAE, self).__init__()
        
        self.latent_dim = latent_dim
        self.image_channels = image_channels
        
        # Encoder: 32x32 -> 16x16 -> 8x8 -> 4x4 -> flatten
        self.enc_conv1 = nn.Conv2d(image_channels, 32, kernel_size=4, stride=2, padding=1)  # 16x16
        self.enc_conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)                # 8x8
        self.enc_conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)               # 4x4
        self.enc_conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=0)              # 1x1
        
        # Flatten: 256 * 1 * 1 = 256
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        
        # Decoder: latent -> 4x4 -> 8x8 -> 16x16 -> 32x32
        self.fc_dec = nn.Linear(latent_dim, 256)
        
        self.dec_deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=1, padding=0)   # 4x4
        self.dec_deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)    # 8x8
        self.dec_deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)     # 16x16
        self.dec_deconv4 = nn.ConvTranspose2d(32, image_channels, kernel_size=4, stride=2, padding=1)  # 32x32
    
    def encode(self, x):
        """
        Encode image to latent space.
        
        Args:
            x: Input images [batch_size, channels, height, width]
            
        Returns:
            mu: Mean of latent distribution [batch_size, latent_dim]
            logvar: Log variance of latent distribution [batch_size, latent_dim]
        """
        h = F.relu(self.enc_conv1(x))
        h = F.relu(self.enc_conv2(h))
        h = F.relu(self.enc_conv3(h))
        h = F.relu(self.enc_conv4(h))
        h = h.view(h.size(0), -1)  # Flatten
        
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: sample from N(mu, std^2).
        
        Args:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
            
        Returns:
            z: Sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z):
        """
        Decode latent vector to image.
        
        Args:
            z: Latent vector [batch_size, latent_dim]
            
        Returns:
            x_recon: Reconstructed image [batch_size, channels, height, width]
        """
        h = F.relu(self.fc_dec(z))
        h = h.view(h.size(0), 256, 1, 1)  # Reshape
        
        h = F.relu(self.dec_deconv1(h))
        h = F.relu(self.dec_deconv2(h))
        h = F.relu(self.dec_deconv3(h))
        x_recon = self.dec_deconv4(h)  # Raw output for MSELoss
        return x_recon
    
    def forward(self, x):
        """
        Forward pass through VAE.
        
        Args:
            x: Input images [batch_size, channels, height, width]
            
        Returns:
            x_recon: Reconstructed images
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar
    
    def sample(self, num_samples, device):
        """
        Generate new samples by sampling from standard normal.
        
        Args:
            num_samples: Number of samples to generate
            device: Device to generate samples on
            
        Returns:
            samples: Generated images [num_samples, channels, height, width]
        """
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim, device=device)
            samples = self.decode(z)
        return samples


def vae_loss(x_recon, x, mu, logvar, beta=1.0):
    """
    Compute VAE loss = Reconstruction Loss + β * KL Divergence.
    
    Args:
        x_recon: Reconstructed images
        x: Original images
        mu: Mean of latent distribution
        logvar: Log variance of latent distribution
        beta: Weight for KL divergence term (default: 1.0)
        
    Returns:
        loss: Total VAE loss
        recon_loss: Reconstruction loss (MSE)
        kl_loss: KL divergence loss
    """
    # Reconstruction loss (Mean Squared Error)
    recon_loss = F.mse_loss(x_recon, x, reduction='mean')
    
    # KL divergence loss
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Total loss
    loss = recon_loss + beta * kl_loss
    
    return loss, recon_loss, kl_loss
