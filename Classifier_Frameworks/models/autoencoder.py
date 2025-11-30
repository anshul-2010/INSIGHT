import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvAutoencoder(nn.Module):
    """Vanilla convolutional autoencoder used for reconstruction-driven forensics."""

    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.latent_dim = latent_dim
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, latent_dim),
        )
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, 64 * 8 * 8),
            nn.Unflatten(1, (64, 8, 8)),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.enc(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.dec(z)

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z


class AutoencoderDetector(nn.Module):
    """Classifier head that uses reconstruction error as an auxiliary forensic signal."""

    def __init__(self, latent_dim: int = 128, num_classes: int = 2):
        super().__init__()
        self.autoencoder = ConvAutoencoder(latent_dim=latent_dim)
        # Additional scalar for averaged reconstruction error
        self.head = nn.Sequential(
            nn.Linear(latent_dim + 1, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor):
        recon, z = self.autoencoder(x)
        # Mean absolute reconstruction error per sample as coarse anomaly descriptor
        recon_err = torch.mean(
            torch.abs(x - recon).flatten(start_dim=1), dim=1, keepdim=True
        )
        feat = torch.cat([z, recon_err], dim=1)
        logits = self.head(feat)
        return logits, feat, recon
