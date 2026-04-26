"""
Advanced DDPM trainer with:
- Cosine schedule
- DDIM sampling (accelerated)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# COSINE SCHEDULE
# ============================================================================

def get_cosine_schedule(num_timesteps=1000, s=0.008):
    """
    Cosine beta schedule from Improved DDPM paper.
    More stable than linear for high timesteps.
    
    IMPORTANT: Clipping max to 0.02 (same as linear schedule).
    """
    steps = num_timesteps + 1
    x = torch.linspace(0, num_timesteps, steps)
    alphas_cumprod = torch.cos(((x / num_timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    
    # CRITICAL FIX: Clip to reasonable range (not 0.9999!)
    return torch.clip(betas, 0.0001, 0.02)


# ============================================================================
# ADVANCED DIFFUSION WITH DDIM
# ============================================================================

class AdvancedDiffusion:
    """
    Advanced diffusion with cosine schedule and DDIM sampling.
    """
    def __init__(self, num_timesteps=1000, schedule='cosine', device='cuda'):
        self.num_timesteps = num_timesteps
        self.device = device
        
        # Get schedule
        if schedule == 'cosine':
            self.betas = get_cosine_schedule(num_timesteps).to(device)
        elif schedule == 'linear':
            self.betas = torch.linspace(0.0001, 0.02, num_timesteps).to(device)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")
        
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.]).to(device), self.alphas_cumprod[:-1]])
        
        # For forward diffusion q(x_t | x_0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        
        # For DDPM sampling
        self.sqrt_recip_alphas = torch.sqrt(1. / self.alphas)
        
        # For DDIM sampling
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod - 1)
    
    def q_sample(self, x_0, t, noise=None):
        """Add noise to x_0 at timestep t."""
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alpha_prod = self._extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alpha_prod = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
        
        return sqrt_alpha_prod * x_0 + sqrt_one_minus_alpha_prod * noise
    
    @torch.no_grad()
    def p_sample_ddpm(self, model, x, t, t_index):
        """Single DDPM reverse diffusion step (original, 1000 steps)."""
        predicted_noise = model(x, t)
        
        betas_t = self._extract(self.betas, t, x.shape)
        sqrt_one_minus_alpha_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = self._extract(self.sqrt_recip_alphas, t, x.shape)
        
        model_mean = sqrt_recip_alphas_t * (x - betas_t * predicted_noise / sqrt_one_minus_alpha_cumprod_t)
        
        if t_index == 0:
            return model_mean
        else:
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(betas_t) * noise
    
    @torch.no_grad()
    def p_sample_ddim(self, model, x, t, t_prev, eta=0.0):
        """
        Single DDIM reverse step.
        DDIM allows deterministic sampling with fewer steps.
        
        Args:
            eta: 0 for deterministic, 1 for stochastic (like DDPM)
        """
        # Predict noise
        predicted_noise = model(x, t)
        
        # Get alpha values
        alpha_cumprod_t = self._extract(self.alphas_cumprod, t, x.shape)
        alpha_cumprod_t_prev = self._extract(self.alphas_cumprod, t_prev, x.shape)
        
        # Predict x_0
        pred_x0 = (x - torch.sqrt(1 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_cumprod_t)
        pred_x0 = torch.clamp(pred_x0, -1, 1)
        
        # Direction pointing to x_t
        dir_xt = torch.sqrt(1 - alpha_cumprod_t_prev - eta**2 * (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_cumprod_t_prev)) * predicted_noise
        
        # DDIM step
        x_prev = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + dir_xt
        
        if eta > 0:
            noise = torch.randn_like(x)
            sigma = eta * torch.sqrt((1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_cumprod_t_prev))
            x_prev = x_prev + sigma * noise
        
        return x_prev
    
    @torch.no_grad()
    def sample_ddpm(self, model, shape):
        """Generate samples with DDPM (1000 steps)."""
        model.eval()
        x = torch.randn(shape).to(self.device)
        
        for i in reversed(range(self.num_timesteps)):
            t = torch.full((shape[0],), i, dtype=torch.long, device=self.device)
            x = self.p_sample_ddpm(model, x, t, i)
        
        return x
    
    @torch.no_grad()
    def sample_ddim(self, model, shape, ddim_steps=200, eta=0.0):
        """
        Generate samples with DDIM (accelerated, fewer steps).
        
        Args:
            ddim_steps: Number of sampling steps (e.g., 200 instead of 1000)
            eta: 0 for deterministic, 1 for stochastic
        """
        model.eval()
        x = torch.randn(shape).to(self.device)
        
        # Create timestep sequence
        c = self.num_timesteps // ddim_steps
        timesteps = list(range(0, self.num_timesteps, c))
        timesteps = timesteps[:ddim_steps]
        
        # Reverse sampling
        for i in reversed(range(len(timesteps))):
            t = torch.full((shape[0],), timesteps[i], dtype=torch.long, device=self.device)
            t_prev = torch.full((shape[0],), timesteps[i-1] if i > 0 else 0, dtype=torch.long, device=self.device)
            x = self.p_sample_ddim(model, x, t, t_prev, eta)
        
        return x
    
    def _extract(self, tensor, t, x_shape):
        """Extract values at timestep t and reshape for broadcasting."""
        out = tensor.gather(-1, t)
        return out.view(t.shape[0], *((1,) * (len(x_shape) - 1)))


# ============================================================================
# ADVANCED TRAINER
# ============================================================================

class AdvancedDiffusionTrainer:
    """
    Advanced DDPM trainer with cosine schedule and DDIM.
    """
    def __init__(self, model, device='cuda', schedule='cosine', num_timesteps=1000):
        self.model = model.to(device)
        self.device = device
        self.diffusion = AdvancedDiffusion(num_timesteps, schedule, device)
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
    def sample(self, num_samples=36, method='ddim', ddim_steps=200, eta=0.0):
        """
        Generate samples.
        
        Args:
            method: 'ddpm' (1000 steps) or 'ddim' (faster, e.g., 200 steps)
            ddim_steps: Number of steps for DDIM
            eta: DDIM stochasticity (0=deterministic)
        """
        # Sample
        if method == 'ddpm':
            samples = self.diffusion.sample_ddpm(self.model, (num_samples, 3, 32, 32))
        elif method == 'ddim':
            samples = self.diffusion.sample_ddim(self.model, (num_samples, 3, 32, 32), ddim_steps, eta)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Convert from [-1, 1] to [0, 1]
        samples = (samples + 1) / 2
        samples = torch.clamp(samples, 0, 1)
        
        return samples
    
    def fit(self, train_loader, val_loader, num_epochs):
        """Train for multiple epochs. Compatible with main.py interface."""
        for epoch in range(num_epochs):
            avg_loss = self.train_epoch(train_loader)
            print(f"Epoch {epoch+1}/{num_epochs}: Loss = {avg_loss:.6f}")
    
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
