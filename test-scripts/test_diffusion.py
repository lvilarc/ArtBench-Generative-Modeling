"""
Test basic Diffusion Model (Linear schedule + DDPM sampling).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from models.diffusion_model import SimpleUNet
from training.diffusion_trainer import DiffusionTrainer
from data import get_train_loader_from_csv, save_sample_grid

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("="*70)
print("BASIC DIFFUSION MODEL TEST")
print("- Linear noise schedule (beta schedule)")
print("- DDPM sampling (Denoising Diffusion Probabilistic Model)")
print("- Gradually denoise from pure noise to image")
print("="*70)
print()

# Build model
print("Building SimpleUNet...")
model = SimpleUNet(in_channels=3, model_channels=64).to(DEVICE)
num_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {num_params:,}")
print()

# Build trainer
print("Building DiffusionTrainer...")
trainer = DiffusionTrainer(
    model, 
    device=DEVICE,
    num_timesteps=1000
)
print("Trainer initialized (linear schedule, 1000 timesteps)")
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
print("Generating 36 samples with DDPM (1000 denoising steps)...")
samples = trainer.sample(num_samples=36)
save_sample_grid(samples, "diffusion_test_samples.png", nrow=6)
print("Saved: diffusion_test_samples.png")
print()

print("="*70)
print("✓ Basic diffusion test complete!")
print("Check diffusion_test_samples.png for generated samples")
print("Compare with advanced_clean_ddim.png (cosine + DDIM)")
print("="*70)
