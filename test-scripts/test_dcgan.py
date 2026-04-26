"""
Test DCGAN (Deep Convolutional Generative Adversarial Network).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from models.dcgan_model import Generator, Discriminator
from training.dcgan_trainer import DCGANTrainer
from data import get_train_loader_from_csv, save_sample_grid

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("="*70)
print("DCGAN TEST")
print("- Generator vs Discriminator adversarial training")
print("- Generator: Noise -> Image")
print("- Discriminator: Image -> Real/Fake probability")
print("="*70)
print()

# Build models
print("Building Generator and Discriminator...")
generator = Generator(z_dim=100, image_channels=3).to(DEVICE)
discriminator = Discriminator(image_channels=3).to(DEVICE)

gen_params = sum(p.numel() for p in generator.parameters())
disc_params = sum(p.numel() for p in discriminator.parameters())
print(f"Generator parameters: {gen_params:,}")
print(f"Discriminator parameters: {disc_params:,}")
print(f"Total parameters: {gen_params + disc_params:,}")
print()

# Build trainer
print("Building DCGANTrainer...")
trainer = DCGANTrainer(
    generator=generator,
    discriminator=discriminator,
    device=DEVICE,
    lr=2e-4
)
print("Trainer initialized (lr=2e-4)")
print()

# Load data
print("Loading dataset...")
train_loader = get_train_loader_from_csv()
print(f"Training batches: {len(train_loader)}")
print()

# Train 5 epochs
print("Training 5 epochs...")
trainer.fit(train_loader, val_loader=train_loader, num_epochs=5)
print()

# Generate samples
print("Generating 36 samples from random noise...")
samples = trainer.sample(num_samples=36)
save_sample_grid(samples, "dcgan_test_samples.png", nrow=6)
print("Saved: dcgan_test_samples.png")
print()

print("="*70)
print("✓ DCGAN test complete!")
print("Check dcgan_test_samples.png for generated samples")
print("="*70)
