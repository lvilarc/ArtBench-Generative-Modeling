import numpy as np
import random
import torch
import argparse
from models.conv_vae_model import ConvVAE, vae_loss
from models.dcgan_model import Generator, Discriminator, weights_init
from data import get_train_loader_from_csv, get_full_train_loader
from training.conv_vae_trainer import ConvVAETrainer
from training.dcgan_trainer import DCGANTrainer
from utils.metrics import compute_metrics

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        raise NotImplementedError("Diffusion not implemented yet")

    else:
        raise ValueError(f"Unknown model: {model_name}")
    

def build_trainer(model_name, model):
    if model_name == "vae":
        return ConvVAETrainer(model, DEVICE, lr=1e-3, beta=1.0)

    elif model_name == "gan":
        generator, discriminator = model
        return DCGANTrainer(generator, discriminator, DEVICE, lr=2e-4, label_smoothing=0.9)

    elif model_name == "diffusion":
        raise NotImplementedError


def run_experiment(model_name, loader_fn, num_epochs):
    all_results = []

    for seed in SEEDS:
        print(f"\n===== {model_name.upper()} | SEED {seed} =====")

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

        # Generate samples (different interfaces for VAE vs GAN)
        with torch.no_grad():
            if model_name == "vae":
                samples = model.sample(5000, DEVICE)
            elif model_name == "gan":
                samples = trainer.sample(5000)
            else:
                raise NotImplementedError(f"Sampling not implemented for {model_name}")
        
        real = sample_real_images(train_loader, 5000) # TODO: get 5000 images from the test set, fixed for all the experiments

        # TODO: save samples to disk

        fid, kid_mean, kid_std = compute_metrics(
            real,
            samples,
            use_cuda=torch.cuda.is_available()
        )

        all_results.append({
            "fid": fid,
            "kid_mean": kid_mean,
            "kid_std": kid_std
        })

    # Aggregate results
    fids = [r["fid"] for r in all_results]
    kid_means = [r["kid_mean"] for r in all_results]
    kid_stds = [r["kid_std"] for r in all_results] # TODO: what we do with this? Is it correct ?

    print("\n===== FINAL RESULTS =====")
    print(f"FID: {np.mean(fids):.4f} ± {np.std(fids):.4f}")
    print(f"KID: {np.mean(kid_means):.4f} ± {np.std(kid_means):.4f}")


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
        choices=["vae", "gan", "diffusion"],
        required=True
    )

    args = parser.parse_args()

    # Select dataset
    if args.mode == "subset":
        loader_fn = get_train_loader_from_csv
        num_epochs = 30

    elif args.mode == "full":
        loader_fn = get_full_train_loader
        num_epochs = 50  

    else:
        raise ValueError("Invalid mode")

    run_experiment(
        model_name=args.model,
        loader_fn=loader_fn,
        num_epochs=num_epochs
    )


if __name__ == "__main__":
    main()