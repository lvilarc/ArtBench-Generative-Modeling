"""
ConvVAE Trainer for ArtBench-10
Handles training loop for Variational Autoencoder (VAE).
"""

import torch
from models.conv_vae_model import vae_loss


class ConvVAETrainer:
    """
    Trainer for Convolutional Variational Autoencoder.

    Responsibilities:
    - Training loop
    - Validation loop
    - Loss optimization (Reconstruction + KL)
    """

    def __init__(self, model, device, lr=1e-3, beta=1.0):
        """
        Args:
            model: ConvVAE model
            device: torch device (cuda or cpu)
            lr: learning rate
            beta: KL divergence weight
        """
        self.model = model.to(device)
        self.device = device
        self.beta = beta

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def train_step(self, batch):
        """
        Single training step.

        Args:
            batch: tuple (images, labels optional)
        Returns:
            dict of losses
        """
        self.model.train()

        x = batch[0].to(self.device)

        x_recon, mu, logvar = self.model(x)

        loss, recon_loss, kl_loss = vae_loss(
            x_recon, x, mu, logvar, beta=self.beta
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "recon_loss": recon_loss.item(),
            "kl_loss": kl_loss.item()
        }

    def validate_step(self, batch):
        """
        Single validation step.
        """
        self.model.eval()

        with torch.no_grad():
            x = batch[0].to(self.device)

            x_recon, mu, logvar = self.model(x)

            loss, recon_loss, kl_loss = vae_loss(
                x_recon, x, mu, logvar, beta=self.beta
            )

        return {
            "loss": loss.item(),
            "recon_loss": recon_loss.item(),
            "kl_loss": kl_loss.item()
        }

    def fit(self, train_loader, val_loader=None, num_epochs=10):
        """
        Full training loop.
        """

        for epoch in range(num_epochs):

            train_stats = {"loss": 0.0, "recon_loss": 0.0, "kl_loss": 0.0}

            for batch in train_loader:
                metrics = self.train_step(batch)

                for k in train_stats:
                    train_stats[k] += metrics[k]

            n = len(train_loader)

            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(
                f"Train Loss: {train_stats['loss']/n:.4f} | "
                f"Recon: {train_stats['recon_loss']/n:.4f} | "
                f"KL: {train_stats['kl_loss']/n:.4f}"
            )

            if val_loader is not None:
                val_stats = {"loss": 0.0, "recon_loss": 0.0, "kl_loss": 0.0}

                for batch in val_loader:
                    metrics = self.validate_step(batch)

                    for k in val_stats:
                        val_stats[k] += metrics[k]

                n = len(val_loader)

                print(
                    f"Val Loss: {val_stats['loss']/n:.4f} | "
                    f"Recon: {val_stats['recon_loss']/n:.4f} | "
                    f"KL: {val_stats['kl_loss']/n:.4f}"
                )

    def encode(self, x):
        """Optional helper for latent extraction."""
        self.model.eval()
        with torch.no_grad():
            x = x.to(self.device)
            mu, logvar = self.model.encode(x)
        return mu, logvar

    def sample(self, num_samples):
        """
        Generate samples from latent space.
        """
        return self.model.sample(num_samples, self.device)
