"""
Simple DDPM trainer implementation.
Based on class notebook - minimal working version.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleDiffusion:
    """
    Simple DDPM schedule and sampling logic.
    Based on class notebook GaussianDiffusion.
    """
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, device='cuda'):
        self.num_timesteps = num_timesteps
        self.device = device
        
        # Linear schedule
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps).to(device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.]).to(device), self.alphas_cumprod[:-1]])
        
        # For forward diffusion q(x_t | x_0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        
        # For reverse sampling
        self.sqrt_recip_alphas = torch.sqrt(1. / self.alphas)
        
    def q_sample(self, x_0, t, noise=None):
        """Add noise to x_0 at timestep t."""
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alpha_prod = self._extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alpha_prod = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
        
        return sqrt_alpha_prod * x_0 + sqrt_one_minus_alpha_prod * noise
    
    @torch.no_grad()
    def p_sample(self, model, x, t, t_index):
        """Single reverse diffusion step."""
        # Predict noise
        predicted_noise = model(x, t)
        
        # Get coefficients
        betas_t = self._extract(self.betas, t, x.shape)
        sqrt_one_minus_alpha_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = self._extract(self.sqrt_recip_alphas, t, x.shape)
        
        # Compute mean
        model_mean = sqrt_recip_alphas_t * (x - betas_t * predicted_noise / sqrt_one_minus_alpha_cumprod_t)
        
        if t_index == 0:
            return model_mean
        else:
            # Add noise
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(betas_t) * noise
    
    @torch.no_grad()
    def p_sample_loop(self, model, shape):
        """Generate samples from noise."""
        model.eval()
        x = torch.randn(shape).to(self.device)
        
        for i in reversed(range(self.num_timesteps)):
            t = torch.full((shape[0],), i, dtype=torch.long, device=self.device)
            x = self.p_sample(model, x, t, i)
        
        return x
    
    def _extract(self, tensor, t, x_shape):
        """Extract values at timestep t and reshape for broadcasting."""
        out = tensor.gather(-1, t)
        return out.view(t.shape[0], *((1,) * (len(x_shape) - 1)))


# ============================================================================
# TRAINER CLASS
# ============================================================================

class DiffusionTrainer:
    """
    Simple DDPM trainer.
    Compatible with main.py experiment structure.
    """
    def __init__(self, model, device='cuda', num_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.model = model.to(device)
        self.device = device
        self.diffusion = SimpleDiffusion(num_timesteps, beta_start, beta_end, device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    def train_step(self, batch):
        """Single training step."""
        x = batch[0].to(self.device)
        x = x * 2 - 1  # Normalize to [-1, 1]
        
        batch_size = x.size(0)
        
        # Sample random timesteps
        t = torch.randint(0, self.diffusion.num_timesteps, (batch_size,), device=self.device).long()
        
        # Sample noise
        noise = torch.randn_like(x)
        
        # Add noise (forward diffusion)
        x_t = self.diffusion.q_sample(x, t, noise=noise)
        
        # Predict noise
        predicted_noise = self.model(x_t, t)
        
        # MSE loss
        loss = F.mse_loss(predicted_noise, noise)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        count = 0
        
        for batch in train_loader:
            loss = self.train_step(batch)
            total_loss += loss
            count += 1
        
        return total_loss / count
    
    @torch.no_grad()
    def sample(self, num_samples=36):
        """Generate samples using DDPM."""
        samples = self.diffusion.p_sample_loop(self.model, (num_samples, 3, 32, 32))
        
        # Convert from [-1, 1] to [0, 1]
        samples = (samples + 1) / 2
        samples = torch.clamp(samples, 0, 1)
        
        return samples
    
    def save_checkpoint(self, path):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    
    def load_checkpoint(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    def fit(self, train_loader, val_loader, num_epochs):
        """Train for multiple epochs. Compatible with main.py interface."""
        for epoch in range(num_epochs):
            avg_loss = self.train_epoch(train_loader)
            print(f"Epoch {epoch+1}/{num_epochs}: Loss = {avg_loss:.6f}")
