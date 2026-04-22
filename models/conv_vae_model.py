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
    - Decoder: Deconv layers -> Reconstructed image (outputs logits, not probabilities)
    - Loss: Reconstruction (BCE with logits) + KL divergence
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
        self.enc_conv1 = nn.Conv2d(image_channels, 32, kernel_size=4, stride=2, padding=1)    # 16x16
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
        Encode image to latent distribution parameters.
        
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
        h = h.view(h.size(0), -1)
        
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: sample z ~ N(mu, sigma^2) using noise.
        
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
        Decode latent vector into image logits (not normalized to [0,1]).
        
        Args:
            z: Latent vector [batch_size, latent_dim]
            
        Returns:
            x_recon: Reconstructed image logits [batch_size, channels, height, width]
        """
        h = F.relu(self.fc_dec(z))
        h = h.view(h.size(0), 256, 1, 1)
        
        h = F.relu(self.dec_deconv1(h))
        h = F.relu(self.dec_deconv2(h))
        h = F.relu(self.dec_deconv3(h))

        x_recon = self.dec_deconv4(h)
        return x_recon
    
    def forward(self, x):
        """
        Forward pass through VAE.
        
        Args:
            x: Input images [batch_size, channels, height, width]
            
        Returns:
            x_recon: Reconstructed image logits
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar
    
    def sample(self, num_samples, device):
        """
        Generate new samples from latent space.
        
        Note:
            Applies sigmoid to convert logits into valid pixel values [0,1].
        
        Args:
            num_samples: Number of samples to generate
            device: Device to generate samples on
            
        Returns:
            samples: Generated images [num_samples, channels, height, width]
        """
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim, device=device)
            logits = self.decode(z)
            samples = torch.sigmoid(logits)
        return samples


def vae_loss(x_recon, x, mu, logvar, beta=1.0):
    """
    Compute VAE loss = Reconstruction Loss + β * KL Divergence.
    
    Notes:
        - Uses BCEWithLogitsLoss, so x_recon must be logits (no sigmoid applied).
    
    Args:
        x_recon: Reconstructed image logits
        x: Original images (in [0,1])
        mu: Mean of latent distribution
        logvar: Log variance of latent distribution
        beta: Weight for KL divergence term
        
    Returns:
        loss: Total VAE loss
        recon_loss: Reconstruction loss (BCE with logits)
        kl_loss: KL divergence loss
    """
    recon_loss = F.binary_cross_entropy_with_logits(x_recon, x, reduction='mean')
    
    # KL divergence loss
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Total loss
    loss = recon_loss + beta * kl_loss
    
    return loss, recon_loss, kl_loss
