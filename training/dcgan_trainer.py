"""
DCGAN Trainer for ArtBench-10
Handles adversarial training loop for Generator and Discriminator.
"""

import torch
import torch.nn as nn


class DCGANTrainer:
    """
    Trainer for Deep Convolutional Generative Adversarial Network.

    Responsibilities:
    - Adversarial training loop (alternating G and D)
    - Loss optimization (Binary Cross Entropy)
    - Sample generation
    """

    def __init__(self, generator, discriminator, device, lr=2e-4, label_smoothing=0.9):
        """
        Args:
            generator: Generator model
            discriminator: Discriminator model
            device: torch device (cuda or cpu)
            lr: learning rate for both G and D
            label_smoothing: smooth real labels (0.9 instead of 1.0) for stability
        """
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.device = device
        self.z_dim = generator.z_dim
        self.label_smoothing = label_smoothing

        # Separate optimizers for G and D
        self.optimizer_G = torch.optim.Adam(
            self.generator.parameters(), 
            lr=lr, 
            betas=(0.5, 0.999)
        )
        self.optimizer_D = torch.optim.Adam(
            self.discriminator.parameters(), 
            lr=lr, 
            betas=(0.5, 0.999)
        )

        # Loss function
        self.criterion = nn.BCELoss()

    def train_step(self, batch):
        """
        Single training step (one batch).

        Args:
            batch: tuple (images, labels optional)
        Returns:
            dict of losses and statistics
        """
        batch_size = batch[0].size(0)
        
        # Get real images and normalize to [-1, 1]
        real_images = batch[0].to(self.device)
        real_images = real_images * 2 - 1  # [0,1] -> [-1,1]
        
        # Create labels
        real_labels = torch.full((batch_size, 1), self.label_smoothing, device=self.device)
        fake_labels = torch.zeros(batch_size, 1, device=self.device)

        # ================== Train Discriminator ==================
        self.discriminator.train()
        self.optimizer_D.zero_grad()

        # Train with real images
        outputs_real = self.discriminator(real_images)
        loss_D_real = self.criterion(outputs_real, real_labels)

        # Train with fake images
        z = torch.randn(batch_size, self.z_dim, device=self.device)
        fake_images = self.generator(z)
        outputs_fake = self.discriminator(fake_images.detach())  # Detach to not train G
        loss_D_fake = self.criterion(outputs_fake, fake_labels)

        # Combined D loss
        loss_D = loss_D_real + loss_D_fake
        loss_D.backward()
        self.optimizer_D.step()

        # ================== Train Generator ==================
        self.generator.train()
        self.optimizer_G.zero_grad()

        # Generate new fake images
        z = torch.randn(batch_size, self.z_dim, device=self.device)
        fake_images = self.generator(z)

        # Train G to fool D (fake images should be classified as real)
        outputs_fake = self.discriminator(fake_images)
        loss_G = self.criterion(outputs_fake, torch.ones(batch_size, 1, device=self.device))

        loss_G.backward()
        self.optimizer_G.step()

        # Return statistics
        return {
            "loss_D": loss_D.item(),
            "loss_D_real": loss_D_real.item(),
            "loss_D_fake": loss_D_fake.item(),
            "loss_G": loss_G.item(),
            "D_x": outputs_real.mean().item(),  # D(real) should be ~1
            "D_G_z": outputs_fake.mean().item()  # D(fake) should be ~0.5 at equilibrium
        }

    def validate_step(self, batch):
        """
        Single validation step.
        """
        self.discriminator.eval()
        self.generator.eval()

        with torch.no_grad():
            batch_size = batch[0].size(0)
            
            # Get real images and normalize to [-1, 1]
            real_images = batch[0].to(self.device)
            real_images = real_images * 2 - 1
            
            # Create labels
            real_labels = torch.ones(batch_size, 1, device=self.device)
            fake_labels = torch.zeros(batch_size, 1, device=self.device)

            # Validate D with real
            outputs_real = self.discriminator(real_images)
            loss_D_real = self.criterion(outputs_real, real_labels)

            # Validate D with fake
            z = torch.randn(batch_size, self.z_dim, device=self.device)
            fake_images = self.generator(z)
            outputs_fake = self.discriminator(fake_images)
            loss_D_fake = self.criterion(outputs_fake, fake_labels)

            # Combined D loss
            loss_D = loss_D_real + loss_D_fake

            # Validate G
            loss_G = self.criterion(outputs_fake, torch.ones(batch_size, 1, device=self.device))

        return {
            "loss_D": loss_D.item(),
            "loss_D_real": loss_D_real.item(),
            "loss_D_fake": loss_D_fake.item(),
            "loss_G": loss_G.item(),
            "D_x": outputs_real.mean().item(),
            "D_G_z": outputs_fake.mean().item()
        }

    def fit(self, train_loader, val_loader=None, num_epochs=30):
        """
        Full training loop.
        """

        for epoch in range(num_epochs):

            train_stats = {
                "loss_D": 0.0,
                "loss_D_real": 0.0,
                "loss_D_fake": 0.0,
                "loss_G": 0.0,
                "D_x": 0.0,
                "D_G_z": 0.0
            }

            for batch in train_loader:
                metrics = self.train_step(batch)

                for k in train_stats:
                    train_stats[k] += metrics[k]

            n = len(train_loader)

            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(
                f"Train D_loss: {train_stats['loss_D']/n:.4f} "
                f"(real: {train_stats['loss_D_real']/n:.4f}, "
                f"fake: {train_stats['loss_D_fake']/n:.4f}) | "
                f"G_loss: {train_stats['loss_G']/n:.4f}"
            )
            print(
                f"D(x): {train_stats['D_x']/n:.4f} | "
                f"D(G(z)): {train_stats['D_G_z']/n:.4f}"
            )

            if val_loader is not None:
                val_stats = {
                    "loss_D": 0.0,
                    "loss_D_real": 0.0,
                    "loss_D_fake": 0.0,
                    "loss_G": 0.0,
                    "D_x": 0.0,
                    "D_G_z": 0.0
                }

                for batch in val_loader:
                    metrics = self.validate_step(batch)

                    for k in val_stats:
                        val_stats[k] += metrics[k]

                n = len(val_loader)

                print(
                    f"Val D_loss: {val_stats['loss_D']/n:.4f} "
                    f"(real: {val_stats['loss_D_real']/n:.4f}, "
                    f"fake: {val_stats['loss_D_fake']/n:.4f}) | "
                    f"G_loss: {val_stats['loss_G']/n:.4f}"
                )
                print(
                    f"D(x): {val_stats['D_x']/n:.4f} | "
                    f"D(G(z)): {val_stats['D_G_z']/n:.4f}"
                )

    def sample(self, num_samples):
        """
        Generate samples from latent space.
        
        Returns:
            Generated images in range [0, 1]
        """
        self.generator.eval()
        
        with torch.no_grad():
            z = torch.randn(num_samples, self.z_dim, device=self.device)
            fake_images = self.generator(z)
            
            # Convert from [-1, 1] to [0, 1]
            fake_images = (fake_images + 1) / 2
            
        return fake_images
