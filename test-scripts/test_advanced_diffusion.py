"""
Test advanced diffusion (NO EMA, only cosine + DDIM).
"""
import torch
from models.diffusion_model import SimpleUNet
from training.advanced_diffusion_trainer import AdvancedDiffusionTrainer
from data import get_train_loader_from_csv, save_sample_grid

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("="*70)
print("ADVANCED DIFFUSION - COSINE + DDIM")
print("- Cosine schedule (corrected clipping)")
print("- DDIM sampling (200 steps, 5x faster)")
print("- NO EMA")
print("="*70)
print()

# Build model
print("Building SimpleUNet...")
model = SimpleUNet(in_channels=3, model_channels=64).to(DEVICE)
num_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {num_params:,}")
print()

# Build advanced trainer
print("Building AdvancedDiffusionTrainer...")
trainer = AdvancedDiffusionTrainer(
    model, 
    device=DEVICE, 
    schedule='cosine',
    num_timesteps=1000
)
print("Trainer initialized (cosine schedule)")
print()

# Load data
print("Loading dataset...")
train_loader = get_train_loader_from_csv()
print(f"Training batches: {len(train_loader)}")
print()

# Train 5 epochs
print("Training 5 epochs...")
trainer.fit(train_loader, train_loader, num_epochs=5)
print()

# Test DDIM sampling (fast)
print("Generating 36 samples with DDIM (200 steps)...")
samples_ddim = trainer.sample(num_samples=36, method='ddim', ddim_steps=200, eta=0.0)
save_sample_grid(samples_ddim, "advanced_clean_ddim.png", nrow=6)
print("Saved: advanced_clean_ddim.png")
print()

# Test DDPM sampling for comparison (just 16)
print("Generating 16 samples with DDPM (1000 steps) for comparison...")
samples_ddpm = trainer.sample(num_samples=16, method='ddpm')
save_sample_grid(samples_ddpm, "advanced_clean_ddpm.png", nrow=4)
print("Saved: advanced_clean_ddpm.png")
print()

print("="*70)
print("✓ Advanced diffusion (cosine + DDIM) complete!")
print("Compare with organized_diffusion_test.png (linear + DDPM)")
print("="*70)
