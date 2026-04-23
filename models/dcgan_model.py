"""
DCGAN (Deep Convolutional Generative Adversarial Network) for ArtBench-10.
Based on Radford et al. (2015) - "Unsupervised Representation Learning with 
Deep Convolutional Generative Adversarial Networks"

Architecture optimized for 32x32 RGB images.
"""

import torch
import torch.nn as nn


class Generator(nn.Module):
    """
    DCGAN Generator for 32x32 images.
    
    Takes a latent vector z and generates an image.
    Output range: [-1, 1] (tanh activation)
    """
    
    def __init__(self, z_dim=100, image_channels=3):
        """
        Initialize Generator.
        
        Args:
            z_dim: Dimension of latent space (default: 100)
            image_channels: Number of output channels (default: 3 for RGB)
        """
        super(Generator, self).__init__()
        
        self.z_dim = z_dim
        self.image_channels = image_channels
        
        # Project and reshape: z (100) -> (256, 4, 4)
        self.fc = nn.Linear(z_dim, 256 * 4 * 4)
        
        # Transposed convolutions to upsample
        self.main = nn.Sequential(
            # 256 x 4 x 4
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # 128 x 8 x 8
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # 64 x 16 x 16
            
            nn.ConvTranspose2d(64, image_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
            # 3 x 32 x 32, values in [-1, 1]
        )
    
    def forward(self, z):
        """
        Generate images from latent vectors.
        
        Args:
            z: Latent vectors [batch_size, z_dim]
            
        Returns:
            Generated images [batch_size, channels, 32, 32] in range [-1, 1]
        """
        # Project and reshape
        x = self.fc(z)
        x = x.view(x.size(0), 256, 4, 4)
        
        # Generate image
        x = self.main(x)
        return x


class Discriminator(nn.Module):
    """
    DCGAN Discriminator for 32x32 images.
    
    Classifies images as real or fake.
    Output range: [0, 1] (sigmoid activation)
    """
    
    def __init__(self, image_channels=3):
        """
        Initialize Discriminator.
        
        Args:
            image_channels: Number of input channels (default: 3 for RGB)
        """
        super(Discriminator, self).__init__()
        
        self.image_channels = image_channels
        
        self.main = nn.Sequential(
            # 3 x 32 x 32
            nn.Conv2d(image_channels, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 64 x 16 x 16
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # 128 x 8 x 8
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # 256 x 4 x 4
            
            nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=0, bias=False),
            # 1 x 1 x 1
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Classify images as real or fake.
        
        Args:
            x: Input images [batch_size, channels, 32, 32] in range [-1, 1]
            
        Returns:
            Probabilities [batch_size, 1] in range [0, 1]
        """
        x = self.main(x)
        x = x.view(x.size(0), -1)  # Flatten to [batch_size, 1]
        x = self.sigmoid(x)
        return x


def weights_init(m):
    """
    Initialize network weights according to DCGAN paper.
    
    Conv/ConvTranspose layers: mean=0.0, std=0.02
    BatchNorm layers: weight=1.0, bias=0.0
    
    Args:
        m: PyTorch module
    """
    classname = m.__class__.__name__
    
    if classname.find('Conv') != -1:
        # Initialize Conv and ConvTranspose layers
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    
    elif classname.find('BatchNorm') != -1:
        # Initialize BatchNorm layers
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# Example usage and testing
if __name__ == "__main__":
    # Test Generator
    z_dim = 100
    batch_size = 8
    
    generator = Generator(z_dim=z_dim)
    generator.apply(weights_init)
    
    z = torch.randn(batch_size, z_dim)
    fake_images = generator(z)
    
    print("Generator Test:")
    print(f"  Input shape: {z.shape}")
    print(f"  Output shape: {fake_images.shape}")
    print(f"  Output range: [{fake_images.min():.2f}, {fake_images.max():.2f}]")
    
    # Test Discriminator
    discriminator = Discriminator()
    discriminator.apply(weights_init)
    
    predictions = discriminator(fake_images)
    
    print("\nDiscriminator Test:")
    print(f"  Input shape: {fake_images.shape}")
    print(f"  Output shape: {predictions.shape}")
    print(f"  Output range: [{predictions.min():.2f}, {predictions.max():.2f}]")
    
    # Test with real images (simulated)
    real_images = torch.rand(batch_size, 3, 32, 32) * 2 - 1  # [-1, 1]
    predictions_real = discriminator(real_images)
    
    print("\nDiscriminator on 'real' images:")
    print(f"  Output range: [{predictions_real.min():.2f}, {predictions_real.max():.2f}]")
    
    print("\n✓ All tests passed!")
