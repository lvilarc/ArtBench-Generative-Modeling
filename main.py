import numpy as np
import random
import torch
import argparse
from models.conv_vae_model import ConvVAE, vae_loss
from models.dcgan_model import Generator, Discriminator, weights_init
from models.diffusion_model import SimpleUNet
from data import get_train_loader_from_csv, get_full_train_loader, get_test_images_tensor, save_sample_grid
from training.conv_vae_trainer import ConvVAETrainer
from training.dcgan_trainer import DCGANTrainer
from training.diffusion_trainer import DiffusionTrainer
from training.advanced_diffusion_trainer import AdvancedDiffusionTrainer
from utils.metrics import compute_metrics
from utils.logging import ExperimentLogger

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Full protocol: 10 seeds for statistical robustness
SEEDS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

# Reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_model(model_name):
    if model_name == "vae":
        return ConvVAE(latent_dim=20).to(DEVICE)

    elif model_name == "gan":
        generator = Generator(z_dim=100).to(DEVICE)
        discriminator = Discriminator().to(DEVICE)
        # Initialize weights according to DCGAN paper
        generator.apply(weights_init)
        discriminator.apply(weights_init)
        return (generator, discriminator)

    elif model_name == "diffusion":
        return SimpleUNet(in_channels=3, model_channels=64).to(DEVICE)
    
    elif model_name == "diffusion_advanced":
        return SimpleUNet(in_channels=3, model_channels=64).to(DEVICE)

    else:
        raise ValueError(f"Unknown model: {model_name}")
    

def build_trainer(model_name, model):
    if model_name == "vae":
        return ConvVAETrainer(model, DEVICE, lr=1e-3, beta=1.0)

    elif model_name == "gan":
        generator, discriminator = model
        return DCGANTrainer(generator, discriminator, DEVICE, lr=2e-4, label_smoothing=0.9)

    elif model_name == "diffusion":
        return DiffusionTrainer(model, DEVICE, num_timesteps=1000, beta_start=0.0001, beta_end=0.02)
    
    elif model_name == "diffusion_advanced":
        return AdvancedDiffusionTrainer(model, DEVICE, schedule='cosine', num_timesteps=1000)


def run_experiment(model_name, loader_fn, num_epochs, mode):
    # Initialize logger
    logger = ExperimentLogger(model_name, mode)
    
    # Log configuration
    config = {
        "model": model_name,
        "mode": mode,
        "num_epochs": num_epochs,
        "num_seeds": len(SEEDS),
        "seeds": SEEDS,
        "device": str(DEVICE),
        "batch_size": 64,
    }
    logger.log_config(config)
    
    all_results = []

    real = get_test_images_tensor(device=DEVICE)

    for seed_idx, seed in enumerate(SEEDS):
        logger.log_seed_start(seed, seed_idx, len(SEEDS))

        set_seed(seed)

        # Load data (FIXED subset or full)
        train_loader = loader_fn()

        model = build_model(model_name)
        trainer = build_trainer(model_name, model)

        trainer.fit(
            train_loader=train_loader,
            val_loader=train_loader,
            num_epochs=num_epochs
        )

        # Generate samples (different interfaces for VAE vs GAN vs Diffusion)
        with torch.no_grad():
            if model_name == "vae":
                samples = model.sample(5000, DEVICE)
            elif model_name == "gan":
                samples = trainer.sample(5000)
            elif model_name == "diffusion":
                # Generate in batches to avoid OOM on 4GB GPU (DDPM - 1000 steps)
                batch_size = 100
                all_samples = []
                for i in range(0, 5000, batch_size):
                    batch_samples = trainer.sample(min(batch_size, 5000 - i))
                    all_samples.append(batch_samples.cpu())
                    if (i // batch_size) % 10 == 0:
                        print(f"  Generated {i + batch_samples.shape[0]}/5000 samples...")
                samples = torch.cat(all_samples, dim=0).to(DEVICE)
            elif model_name == "diffusion_advanced":
                # Generate in batches with DDIM (200 steps - 5x faster)
                batch_size = 100
                all_samples = []
                for i in range(0, 5000, batch_size):
                    batch_samples = trainer.sample(min(batch_size, 5000 - i), method='ddim', ddim_steps=200)
                    all_samples.append(batch_samples.cpu())
                    if (i // batch_size) % 10 == 0:
                        print(f"  Generated {i + batch_samples.shape[0]}/5000 samples...")
                samples = torch.cat(all_samples, dim=0).to(DEVICE)
            else:
                raise NotImplementedError(f"Sampling not implemented for {model_name}")
        
        # Save sample grid with random selection (visual inspection)
        indices = torch.randperm(samples.size(0))[:36]  # Random 36 samples
        sample_path = logger.get_sample_path(seed)
        save_sample_grid(samples[indices], str(sample_path), nrow=6)
        print(f"  Saved sample grid: {sample_path}")

        fid, kid_mean, kid_std = compute_metrics(
            real,
            samples,
            use_cuda=torch.cuda.is_available()
        )

        seed_metrics = {
            "fid": fid,
            "kid_mean": kid_mean,
            "kid_std": kid_std
        }
        
        logger.log_seed_metrics(seed, seed_metrics)
        all_results.append(seed_metrics)

    # Aggregate results
    fids = [r["fid"] for r in all_results]
    kid_means = [r["kid_mean"] for r in all_results]

    summary = {
        "model": model_name,
        "mode": mode,
        "num_seeds": len(SEEDS),
        "fid": {
            "mean": float(np.mean(fids)),
            "std": float(np.std(fids)),
            "min": float(np.min(fids)),
            "max": float(np.max(fids))
        },
        "kid": {
            "mean": float(np.mean(kid_means)),
            "std": float(np.std(kid_means)),
            "min": float(np.min(kid_means)),
            "max": float(np.max(kid_means))
        }
    }
    
    logger.log_final_results(summary)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["subset", "full"],
        required=True,
        help="fixed subset (20%) or full dataset"
    )

    parser.add_argument(
        "--model",
        choices=["vae", "gan", "diffusion", "diffusion_advanced"],
        required=True,
        help="Model: vae, gan, diffusion (linear+DDPM), diffusion_advanced (cosine+DDIM)"
    )

    args = parser.parse_args()

    # Select dataset
    if args.mode == "subset":
        loader_fn = get_train_loader_from_csv
        # num_epochs = 30 # FINAL CONFIG
        num_epochs = 30

    elif args.mode == "full":
        loader_fn = get_full_train_loader
        num_epochs = 50  

    else:
        raise ValueError("Invalid mode")

    run_experiment(
        model_name=args.model,
        loader_fn=loader_fn,
        num_epochs=num_epochs,
        mode=args.mode
    )


if __name__ == "__main__":
    main()