"""
Diffusion Trainer for ArtBench-10
Handles training loop for DDPM and sampling with DDIM.
"""

import torch
import torch.nn as nn
import copy
from models.diffusion_model import get_cosine_schedule, extract


class EMA:
    """
    Exponential Moving Average of model parameters.
    Helps stabilize training and improve sample quality.
    """
    
    def __init__(self, model, decay=0.9999):
        """
        Args:
            model: PyTorch model
            decay: EMA decay rate (default: 0.9999)
        """
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Update EMA parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        """Apply EMA parameters to model (for sampling)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        """Restore original parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


class DiffusionTrainer:
    """
    Trainer for Denoising Diffusion Probabilistic Model (DDPM).
    
    Supports:
    - DDPM training (predicting noise ε)
    - EMA for stable sampling
    - DDIM sampling (fast 50-step generation)
    """
    
    def __init__(self, model, device, lr=1e-4, timesteps=1000, use_ema=True, ema_decay=0.9999):
        """
        Args:
            model: UNet model
            device: torch device (cuda or cpu)
            lr: learning rate
            timesteps: number of diffusion steps (T)
            use_ema: whether to use EMA
            ema_decay: EMA decay rate
        """
        self.model = model.to(device)
        self.device = device
        self.timesteps = timesteps
        self.use_ema = use_ema
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        # EMA
        if use_ema:
            self.ema = EMA(self.model, decay=ema_decay)
        else:
            self.ema = None
        
        # Prepare noise schedule (cosine)
        betas = get_cosine_schedule(timesteps).to(device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=device), alphas_cumprod[:-1]])
        
        # Register as buffers (not parameters)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        
        # Precompute values for sampling
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod - 1))
    
    def register_buffer(self, name, tensor):
        """Helper to register buffers."""
        setattr(self, name, tensor)
    
    def q_sample(self, x_start, t, noise=None):
        """
        Forward diffusion: add noise to x_0 to get x_t.
        q(x_t | x_0) = N(x_t; √(ᾱ_t)·x_0, (1-ᾱ_t)·I)
        
        Args:
            x_start: Clean images [batch, channels, H, W]
            t: Timesteps [batch]
            noise: Optional noise tensor (if None, sample from N(0,I))
            
        Returns:
            Noisy images x_t
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_losses(self, x_start, t, noise=None):
        """
        Compute loss for denoising.
        L = MSE(ε, ε_θ(x_t, t))
        
        Args:
            x_start: Clean images [batch, channels, H, W]
            t: Timesteps [batch]
            noise: Optional noise tensor
            
        Returns:
            loss: MSE between true noise and predicted noise
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        # Add noise to get x_t
        x_noisy = self.q_sample(x_start, t, noise)
        
        # Predict noise
        predicted_noise = self.model(x_noisy, t)
        
        # MSE loss
        loss = nn.functional.mse_loss(predicted_noise, noise)
        
        return loss
    
    def train_step(self, batch):
        """
        Single training step.
        
        Args:
            batch: tuple (images, labels optional)
        Returns:
            dict of losses
        """
        self.model.train()
        
        # Get images and normalize to [-1, 1]
        x = batch[0].to(self.device)
        x = x * 2 - 1  # [0,1] -> [-1,1]
        
        batch_size = x.size(0)
        
        # Sample random timesteps
        t = torch.randint(0, self.timesteps, (batch_size,), device=self.device).long()
        
        # Compute loss
        loss = self.p_losses(x, t)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        self.optimizer.step()
        
        # Update EMA
        if self.use_ema:
            self.ema.update()
        
        return {"loss": loss.item()}
    
    def validate_step(self, batch):
        """
        Single validation step.
        """
        self.model.eval()
        
        with torch.no_grad():
            # Get images and normalize to [-1, 1]
            x = batch[0].to(self.device)
            x = x * 2 - 1
            
            batch_size = x.size(0)
            
            # Sample random timesteps
            t = torch.randint(0, self.timesteps, (batch_size,), device=self.device).long()
            
            # Compute loss
            loss = self.p_losses(x, t)
        
        return {"loss": loss.item()}
    
    def fit(self, train_loader, val_loader=None, num_epochs=30):
        """
        Full training loop.
        """
        for epoch in range(num_epochs):
            train_loss = 0.0
            
            for batch in train_loader:
                metrics = self.train_step(batch)
                train_loss += metrics["loss"]
            
            n = len(train_loader)
            
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"Train Loss: {train_loss/n:.6f}")
            
            if val_loader is not None:
                val_loss = 0.0
                
                for batch in val_loader:
                    metrics = self.validate_step(batch)
                    val_loss += metrics["loss"]
                
                n = len(val_loader)
                print(f"Val Loss: {val_loss/n:.6f}")
    
    @torch.no_grad()
    def sample_ddim(self, num_samples, ddim_steps=50):
        """
        Generate samples using DDIM (fast sampling).
        
        DDIM allows generating with fewer steps (e.g., 50 instead of 1000)
        while maintaining quality.
        
        Args:
            num_samples: Number of samples to generate
            ddim_steps: Number of DDIM steps (default: 50)
            
        Returns:
            Generated images in range [0, 1]
        """
        self.model.eval()
        
        # Use EMA model if available
        if self.use_ema:
            self.ema.apply_shadow()
        
        # Create DDIM timestep sequence
        # Sample evenly spaced timesteps
        c = self.timesteps // ddim_steps
        ddim_timesteps = torch.arange(0, self.timesteps, c, device=self.device).long()
        ddim_timesteps = torch.cat([ddim_timesteps, torch.tensor([self.timesteps - 1], device=self.device)])
        
        # Start from pure noise
        x = torch.randn(num_samples, 3, 32, 32, device=self.device)
        
        # Reverse process
        for i in reversed(range(len(ddim_timesteps))):
            t = ddim_timesteps[i]
            t_batch = torch.full((num_samples,), t, device=self.device, dtype=torch.long)
            
            # Predict noise
            predicted_noise = self.model(x, t_batch)
            
            # Get alpha values
            alpha_t = self.alphas_cumprod[t]
            
            if i > 0:
                alpha_t_prev = self.alphas_cumprod[ddim_timesteps[i - 1]]
            else:
                alpha_t_prev = torch.tensor(1.0, device=self.device)
            
            # Predict x_0
            pred_x0 = (x - torch.sqrt(1 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)
            pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)
            
            # Compute direction pointing to x_t
            dir_xt = torch.sqrt(1 - alpha_t_prev) * predicted_noise
            
            # Compute x_{t-1}
            x = torch.sqrt(alpha_t_prev) * pred_x0 + dir_xt
        
        # Restore original model if using EMA
        if self.use_ema:
            self.ema.restore()
        
        # Convert from [-1, 1] to [0, 1]
        x = (x + 1) / 2
        x = torch.clamp(x, 0.0, 1.0)
        
        return x
    
    def sample(self, num_samples):
        """
        Generate samples (wrapper that uses DDIM).
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            Generated images in range [0, 1]
        """
        return self.sample_ddim(num_samples, ddim_steps=50)
