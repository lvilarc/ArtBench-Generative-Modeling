import torch
import torch.nn.functional as F
import numpy as np
from torchvision.models import inception_v3
from scipy import linalg


# =========================
# Inception Feature Extractor
# =========================
class InceptionFeatureExtractor(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.model = inception_v3(pretrained=True, transform_input=False)
        self.model.fc = torch.nn.Identity()  # remove classification head
        self.model.eval()
        self.model.to(device)
        self.device = device

    @torch.no_grad()
    def forward(self, x):
        # Resize to 299x299 as required by Inception
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        x = x.to(self.device)
        features = self.model(x)
        return features


# =========================
# Feature Extraction
# =========================
def get_features(images, model, batch_size=64):
    features = []

    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        feats = model(batch)
        features.append(feats.cpu().numpy())

    return np.concatenate(features, axis=0)


# =========================
# FID
# =========================
def compute_fid(fake_images, real_images, device="cuda"):
    """
    fake_images: Tensor [N, C, H, W] in [0,1]
    real_images: Tensor [N, C, H, W] in [0,1]
    """

    model = InceptionFeatureExtractor(device)

    fake_features = get_features(fake_images, model)
    real_features = get_features(real_images, model)

    mu1, sigma1 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)
    mu2, sigma2 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)

    diff = mu1 - mu2

    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)

    return float(fid)


# =========================
# KID (MMD with polynomial kernel)
# =========================
def polynomial_kernel(x, y):
    return (x @ y.T / x.shape[1] + 1) ** 3


def compute_kid(fake_images, real_images, device="cuda",
                num_subsets=50, subset_size=100):
    """
    Returns:
        mean, std
    """

    model = InceptionFeatureExtractor(device)

    fake_features = get_features(fake_images, model)
    real_features = get_features(real_images, model)

    m = subset_size
    scores = []

    for _ in range(num_subsets):
        idx_fake = np.random.choice(len(fake_features), m, replace=False)
        idx_real = np.random.choice(len(real_features), m, replace=False)

        x = fake_features[idx_fake]
        y = real_features[idx_real]

        k_xx = polynomial_kernel(x, x)
        k_yy = polynomial_kernel(y, y)
        k_xy = polynomial_kernel(x, y)

        # unbiased MMD
        np.fill_diagonal(k_xx, 0)
        np.fill_diagonal(k_yy, 0)

        mmd = (
            k_xx.sum() / (m * (m - 1)) +
            k_yy.sum() / (m * (m - 1)) -
            2 * k_xy.mean()
        )

        scores.append(mmd)

    return float(np.mean(scores)), float(np.std(scores))