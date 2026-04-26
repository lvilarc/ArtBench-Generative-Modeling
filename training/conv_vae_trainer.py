"""
ConvVAE Trainer for ArtBench-10
Handles training with KL annealing for better convergence.
"""

import torch
from models.conv_vae_model import vae_loss


class ConvVAETrainer:
    """
    Trainer for Convolutional VAE with KL annealing.
    
    KL annealing gradually increases beta from 0 to target value,
    allowing the model to first learn good reconstructions before
    enforcing the prior constraint.
    """

    def __init__(self, model, device, lr=1e-3, beta=0.5):
        """
        Args:
            model: ConvVAE model
            device: torch device
            lr: learning rate
            beta: final KL weight (will anneal to this value)
        """
        self.model = model.to(device)
        self.device = device
        self.target_beta = beta
        self.current_beta = 0.0  # Start with 0 for annealing
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def train_step(self, batch, beta):
        """
        Single training step.
        
        Args:
            batch: tuple (images, labels optional)
            beta: current beta value for KL term
            
        Returns:
            dict of losses
        """
        self.model.train()
        
        x = batch[0].to(self.device)
        
        # Forward pass
        x_recon, mu, logvar = self.model(x)
        
        # Compute loss
        loss, recon_loss, kl_loss = vae_loss(x_recon, x, mu, logvar, beta=beta)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {
            "loss": loss.item(),
            "recon_loss": recon_loss.item(),
            "kl_loss": kl_loss.item()
        }

    def validate_step(self, batch, beta):
        """
        Single validation step.
        
        Args:
            batch: tuple (images, labels optional)
            beta: current beta value for KL term
            
        Returns:
            dict of losses
        """
        self.model.eval()
        
        with torch.no_grad():
            x = batch[0].to(self.device)
            
            x_recon, mu, logvar = self.model(x)
            
            loss, recon_loss, kl_loss = vae_loss(x_recon, x, mu, logvar, beta=beta)
        
        return {
            "loss": loss.item(),
            "recon_loss": recon_loss.item(),
            "kl_loss": kl_loss.item()
        }

    def fit(self, train_loader, val_loader=None, num_epochs=30):
        """
        Train the VAE with KL annealing.
        
        Beta increases linearly from 0 to target_beta over first half of training.
        
        Args:
            train_loader: DataLoader for training
            val_loader: DataLoader for validation (optional)
            num_epochs: Number of training epochs
        """
        # KL annealing schedule: increase beta linearly over first half of training
        anneal_epochs = num_epochs // 2
        
        for epoch in range(num_epochs):
            # Update beta (anneal over first half)
            if epoch < anneal_epochs:
                self.current_beta = self.target_beta * (epoch / anneal_epochs)
            else:
                self.current_beta = self.target_beta
            
            # Training
            train_losses = {"loss": 0, "recon_loss": 0, "kl_loss": 0}
            for batch in train_loader:
                batch_losses = self.train_step(batch, self.current_beta)
                for key in train_losses:
                    train_losses[key] += batch_losses[key]
            
            # Average training losses
            for key in train_losses:
                train_losses[key] /= len(train_loader)
            
            # Validation
            if val_loader is not None:
                val_losses = {"loss": 0, "recon_loss": 0, "kl_loss": 0}
                for batch in val_loader:
                    batch_losses = self.validate_step(batch, self.current_beta)
                    for key in val_losses:
                        val_losses[key] += batch_losses[key]
                
                # Average validation losses
                for key in val_losses:
                    val_losses[key] /= len(val_loader)
                
                print(f"\nEpoch {epoch+1}/{num_epochs} (beta={self.current_beta:.3f})")
                print(f"Train Loss: {train_losses['loss']:.4f} | "
                      f"Recon: {train_losses['recon_loss']:.4f} | "
                      f"KL: {train_losses['kl_loss']:.4f}")
                print(f"Val Loss: {val_losses['loss']:.4f} | "
                      f"Recon: {val_losses['recon_loss']:.4f} | "
                      f"KL: {val_losses['kl_loss']:.4f}")
            else:
                print(f"\nEpoch {epoch+1}/{num_epochs} (beta={self.current_beta:.3f})")
                print(f"Train Loss: {train_losses['loss']:.4f} | "
                      f"Recon: {train_losses['recon_loss']:.4f} | "
                      f"KL: {train_losses['kl_loss']:.4f}")

    def sample(self, num_samples):
        """
        Generate samples from the model.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            samples: Generated images [num_samples, channels, height, width]
        """
        return self.model.sample(num_samples, self.device)
