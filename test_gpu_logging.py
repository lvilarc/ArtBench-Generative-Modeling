"""
Test with GPU and professional logging system.
Validates everything works before full training protocol.
"""

import numpy as np
import random
import torch
from models.conv_vae_model import ConvVAE
from models.dcgan_model import Generator, Discriminator, weights_init
from models.diffusion_model import UNet
from data import get_train_loader_from_csv, get_test_images_tensor
from training.conv_vae_trainer import ConvVAETrainer
from training.dcgan_trainer import DCGANTrainer
from training.diffusion_trainer import DiffusionTrainer
from utils.metrics import compute_metrics
from utils.logging import ExperimentLogger

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEST_SEED = 42
NUM_TEST_SAMPLES = 500  # More samples than quick test
NUM_EPOCHS = 2  # 2 epochs to see some improvement

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def test_model_with_logging(model_name):
    print(f"\n{'='*80}")
    print(f"Testing: {model_name.upper()} with GPU and Logging")
    print(f"{'='*80}\n")
    
    # Initialize logger
    logger = ExperimentLogger(model_name, mode="test")
    
    # Configuration
    config = {
        "model": model_name,
        "mode": "test",
        "num_epochs": NUM_EPOCHS,
        "num_seeds": 1,
        "seeds": [TEST_SEED],
        "device": str(DEVICE),
        "batch_size": 64,
        "num_samples": NUM_TEST_SAMPLES
    }
    logger.log_config(config)
    
    set_seed(TEST_SEED)
    
    # Load data
    print("Loading data...")
    train_loader = get_train_loader_from_csv()
    real = get_test_images_tensor(device=DEVICE)
    real = real[:NUM_TEST_SAMPLES]
    print(f"  Train loader: {len(train_loader)} batches")
    print(f"  Real samples: {real.shape}")
    
    # Build model
    print(f"\nBuilding {model_name} model...")
    if model_name == "vae":
        model = ConvVAE(latent_dim=20).to(DEVICE)
        trainer = ConvVAETrainer(model, DEVICE, lr=1e-3, beta=1.0)
    elif model_name == "gan":
        generator = Generator(z_dim=100).to(DEVICE)
        discriminator = Discriminator().to(DEVICE)
        generator.apply(weights_init)
        discriminator.apply(weights_init)
        trainer = DCGANTrainer(generator, discriminator, DEVICE, lr=2e-4)
        model = (generator, discriminator)
    elif model_name == "diffusion":
        model = UNet(image_channels=3, base_channels=64, time_emb_dim=128).to(DEVICE)
        trainer = DiffusionTrainer(model, DEVICE, lr=1e-4, timesteps=1000, use_ema=True)
    
    total_params = sum(p.numel() for p in (
        model.parameters() if model_name == "vae" else 
        (list(model[0].parameters()) + list(model[1].parameters()) if model_name == "gan" else 
        model.parameters())
    ))
    print(f"  Parameters: {total_params:,}")
    if model_name == "gan":
        print(f"  Device: {next(model[0].parameters()).device}")
    else:
        print(f"  Device: {next(model.parameters()).device}")
    
    # Log seed start
    logger.log_seed_start(TEST_SEED, 0, 1)
    
    # Train
    print(f"\nTraining {NUM_EPOCHS} epochs...")
    import time
    train_start = time.time()
    
    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()
        
        # Training epoch
        train_loss = 0.0
        for batch in train_loader:
            if model_name == "vae":
                metrics = trainer.train_step(batch)
                train_loss += metrics["loss"]
            elif model_name == "gan":
                metrics = trainer.train_step(batch)
                train_loss += metrics["loss_G"]
            elif model_name == "diffusion":
                metrics = trainer.train_step(batch)
                train_loss += metrics["loss"]
        
        epoch_time = time.time() - epoch_start
        avg_loss = train_loss / len(train_loader)
        
        epoch_metrics = {
            "loss": avg_loss,
            "time_seconds": epoch_time
        }
        logger.log_epoch(TEST_SEED, epoch + 1, NUM_EPOCHS, epoch_metrics)
    
    train_time = time.time() - train_start
    print(f"Training completed in {train_time:.2f}s ({train_time/60:.2f} min)")
    
    # Generate samples
    print(f"\nGenerating {NUM_TEST_SAMPLES} samples...")
    sample_start = time.time()
    with torch.no_grad():
        if model_name == "vae":
            samples = model.sample(NUM_TEST_SAMPLES, DEVICE)
        else:
            samples = trainer.sample(NUM_TEST_SAMPLES)
    sample_time = time.time() - sample_start
    print(f"  Generated in {sample_time:.2f}s ({NUM_TEST_SAMPLES/sample_time:.1f} samples/s)")
    print(f"  Shape: {samples.shape}")
    
    # Save sample grid
    sample_path = logger.get_sample_path(TEST_SEED)
    from data import save_sample_grid
    save_sample_grid(samples[:36], str(sample_path), nrow=6)
    print(f"  Grid saved: {sample_path}")
    
    # Calculate metrics
    print(f"\nCalculating metrics (FID/KID)...")
    metrics_start = time.time()
    fid, kid_mean, kid_std = compute_metrics(real, samples, use_cuda=torch.cuda.is_available())
    metrics_time = time.time() - metrics_start
    print(f"  Calculated in {metrics_time:.2f}s")
    
    # Log metrics
    seed_metrics = {
        "fid": fid,
        "kid_mean": kid_mean,
        "kid_std": kid_std,
        "training_time_seconds": train_time,
        "sampling_time_seconds": sample_time,
        "metrics_time_seconds": metrics_time
    }
    logger.log_seed_metrics(TEST_SEED, seed_metrics)
    
    # Final summary
    summary = {
        "model": model_name,
        "mode": "test",
        "num_seeds": 1,
        "num_epochs": NUM_EPOCHS,
        "device": str(DEVICE),
        "fid": {"mean": fid, "std": 0.0, "min": fid, "max": fid},
        "kid": {"mean": kid_mean, "std": 0.0, "min": kid_mean, "max": kid_mean},
        "total_time_seconds": train_time + sample_time + metrics_time
    }
    logger.log_final_results(summary)
    
    return True

def main():
    print("\n" + "#" * 80)
    print("# GPU + LOGGING TEST")
    print(f"# Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"# GPU: {torch.cuda.get_device_name(0)}")
    print(f"# Epochs: {NUM_EPOCHS}")
    print(f"# Samples: {NUM_TEST_SAMPLES}")
    print("#" * 80 + "\n")
    
    models_to_test = ["vae", "gan", "diffusion"]
    
    results = {}
    for model_name in models_to_test:
        try:
            success = test_model_with_logging(model_name)
            results[model_name] = "PASSED"
        except Exception as e:
            print(f"\nERROR in {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            results[model_name] = f"FAILED: {str(e)}"
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    for model_name, status in results.items():
        symbol = "✓" if "PASSED" in status else "✗"
        print(f"{symbol} {model_name.upper():12s}: {status}")
    print("=" * 80 + "\n")
    
    all_passed = all("PASSED" in status for status in results.values())
    
    if all_passed:
        print("All tests passed! Ready for full training protocol.")
        print("\nCheck experiment logs in: experiments/")
        print("\nTo run full protocol:")
        print("  python main.py --mode subset --model vae")
        print("  python main.py --mode subset --model gan")
        print("  python main.py --mode subset --model diffusion")
    else:
        print("Some tests failed. Fix errors before running full protocol.")

if __name__ == "__main__":
    main()
