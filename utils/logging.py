"""
Logging utilities for experiment tracking.
Saves detailed logs and per-seed metrics.
"""

from pathlib import Path
from datetime import datetime
import torch


class ExperimentLogger:
    """
    Simplified logging system for generative model experiments.
    Saves only a detailed log file with all metrics and results.
    """
    
    def __init__(self, model_name, mode, base_dir="experiments"):
        """
        Args:
            model_name: Name of the model (vae, gan, diffusion)
            mode: Training mode (subset, full)
            base_dir: Base directory for experiment logs
        """
        self.model_name = model_name
        self.mode = mode
        
        # Create experiment directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = Path(base_dir) / f"{model_name}_{mode}_{timestamp}"
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Create samples subdirectory
        self.samples_dir = self.exp_dir / "samples"
        self.samples_dir.mkdir(exist_ok=True)
        
        # Log file
        self.log_file = self.exp_dir / "experiment.log"
        
        self._log_header()
    
    def _log_header(self):
        """Write experiment header to log file."""
        with open(self.log_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(f"Experiment: {self.model_name.upper()} - {self.mode.upper()}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}\n")
            f.write("=" * 80 + "\n\n")
    
    def log_config(self, config):
        """Log experiment configuration to file."""
        msg = "Configuration:\n"
        for key, value in config.items():
            msg += f"  {key}: {value}\n"
        self._log_message(msg)
    
    def log_seed_start(self, seed, seed_idx, total_seeds):
        """Log start of training for a specific seed."""
        msg = f"\n[Seed {seed_idx+1}/{total_seeds}] Starting training with seed={seed}"
        self._log_message(msg)
        print(msg)
    
    def log_epoch(self, seed, epoch, total_epochs, metrics):
        """
        Log metrics for a specific epoch.
        
        Args:
            seed: Random seed
            epoch: Current epoch number
            total_epochs: Total number of epochs
            metrics: Dictionary with epoch metrics
        """
        msg = f"  Epoch {epoch}/{total_epochs}"
        for key, value in metrics.items():
            if isinstance(value, float):
                msg += f" | {key}: {value:.6f}"
            else:
                msg += f" | {key}: {value}"
        
        self._log_message(msg)
        if epoch % 5 == 0:  # Print every 5 epochs
            print(msg)
    
    def log_seed_metrics(self, seed, metrics):
        """
        Log metrics for a specific seed.
        
        Args:
            seed: Random seed
            metrics: Dictionary with FID, KID, etc.
        """
        # Log to file and console
        msg = f"\n[Seed {seed}] Results:"
        for key, value in metrics.items():
            if isinstance(value, float):
                msg += f"\n  {key}: {value:.6f}"
            else:
                msg += f"\n  {key}: {value}"
        self._log_message(msg)
        print(msg)
    
    def log_final_results(self, summary):
        """
        Log final aggregated results.
        
        Args:
            summary: Dictionary with mean, std, etc.
        """
        # Log summary to file and console
        msg = "\n" + "=" * 80 + "\n"
        msg += "FINAL RESULTS (Aggregated across all seeds)\n"
        msg += "=" * 80 + "\n"
        for key, value in summary.items():
            if isinstance(value, float):
                msg += f"{key}: {value:.6f}\n"
            elif isinstance(value, dict):
                msg += f"{key}:\n"
                for subkey, subval in value.items():
                    msg += f"  {subkey}: {subval:.6f}\n"
            else:
                msg += f"{key}: {value}\n"
        msg += "=" * 80 + "\n"
        
        self._log_message(msg)
        print(msg)
        
        print(f"\nResults saved to: {self.exp_dir}")
        print(f"  - Log: {self.log_file}")
    
    def _log_message(self, message):
        """Write message to log file."""
        with open(self.log_file, 'a') as f:
            f.write(message + "\n")
    
    def get_sample_path(self, seed):
        """Get path for saving sample images."""
        return self.samples_dir / f"seed_{seed}_samples.png"
