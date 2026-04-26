"""
Test Convolutional VAE (Variational Autoencoder).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from models.conv_vae_model import ConvVAE
from training.conv_vae_trainer import ConvVAETrainer
from data import get_train_loader_from_csv, save_sample_grid

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("="*70)
print("CONVOLUTIONAL VAE TEST")
print("- Encoder-Decoder architecture with latent space")
print("- Loss: Reconstruction (MSE) + KL Divergence")
print("- Generates samples from learned latent distribution")
print("="*70)
print()

# Build model
print("Building ConvVAE...")
model = ConvVAE(latent_dim=128, image_channels=3).to(DEVICE)
num_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {num_params:,}")
print(f"Latent dimension: 128")
print()

# Build trainer
print("Building ConvVAETrainer...")
trainer = ConvVAETrainer(
    model, 
    device=DEVICE, 
    lr=1e-3,
    beta=0.5  # KL weight (will anneal from 0)
)
print("Trainer initialized (beta=0.5 with annealing)")
print()

# Load data
print("Loading dataset...")
train_loader = get_train_loader_from_csv()
print(f"Training batches: {len(train_loader)}")
print()

# Train 10 epochs (quick test)
print("Training 10 epochs...")
trainer.fit(train_loader, val_loader=train_loader, num_epochs=10)
print()

# Generate samples
print("Generating 36 samples from latent space...")
samples = trainer.sample(num_samples=36)
save_sample_grid(samples, "vae_test_samples.png", nrow=6)
print("Saved: vae_test_samples.png")
print()

print("="*70)
print("✓ VAE test complete!")
print("Check vae_test_samples.png for generated samples")
print("="*70)
