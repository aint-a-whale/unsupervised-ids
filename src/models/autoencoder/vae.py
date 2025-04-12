from typing import Callable, Literal, Optional, overload

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.autoencoder.base import BaseAutoencoder
from utils.training_callbacks import EarlyStopping


class VariationalAutoencoder(BaseAutoencoder):
    def __init__(
            self,
            input_dim: int = 71,
            hidden_layers: list[int] = [64, 32],
            latent_dim: int = 8,
            hidden_layers_decoder: Optional[list[int]] = None,
            dropout_rate: float = 0,
            kl_beta: float = 1.0,
            device: str = 'cuda' if torch.cuda.is_available() else 'cpu',  # Default to GPU if available, else CPU
    ):
        # The encoder outputs 2 * latent_dim for mean and log-variance
        self.dropout_rate = dropout_rate
        super().__init__(input_dim, hidden_layers, latent_dim * 2, hidden_layers_decoder, dropout_rate, device)
        self.latent_dim = latent_dim  # Update latent_dim to the actual latent dimension
        decoder_layers = [latent_dim] + (hidden_layers_decoder or hidden_layers[::-1]) + [input_dim]
        self.decoder = self._build_layer_sequence(decoder_layers).to(self.device)
        self.decoder.add_module('output_activation', nn.Sigmoid())

        self.kl_beta = kl_beta
        self.kl_loss: list[float] = []

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for VAE."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    @overload
    def forward(self, x: torch.Tensor, ret_latent: Literal[True] = True) -> tuple[torch.Tensor, ...]:
        ...

    @overload
    def forward(self, x: torch.Tensor, ret_latent: Literal[False] = False) -> torch.Tensor:
        ...

    def forward(self, x: torch.Tensor, ret_latent: bool = False):  # type: ignore
        x = x.to(self.device)  # Move input to the specified device
        enc = self.encoder(x)  # Encoder outputs concatenated mu and logvar
        mu, logvar = torch.chunk(enc, 2, dim=-1)  # Split into mu and logvar
        z = self.reparameterize(mu, logvar)  # Sample latent vector
        recon = self.decoder(z)  # Reconstructed input
        if ret_latent:
            return z, recon, mu, logvar  # Return latent, reconstruction, mu, and logvar
        else:
            return recon  # Return only reconstruction

    def vae_loss(self, recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """VAE loss function."""
        # Reconstruction loss (MSE)
        recon_loss = nn.MSELoss(reduction='sum')(recon_x, x)
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + self.kl_beta * kld_loss

    def fit(
        self,
        train_dl: DataLoader[tuple[torch.Tensor, ...]],
        val_dl: DataLoader[tuple[torch.Tensor, ...]],
        epochs: int = 50,
        lr: float = 0.001,
        early_stopping_delta: float = 0.0001,
        weight_decay: float = 0,
        beta_schedule: Optional[Callable[[int], float]] = None,
    ):
        early_stopping = EarlyStopping(min_delta=early_stopping_delta)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

        for epoch in range(epochs):
            if beta_schedule:
                self.beta = beta_schedule(epoch)

            running_loss = 0.0
            with tqdm(total=len(train_dl), desc=f"Epoch {epoch+1}/{epochs}") as pbar:
                for (inputs,) in train_dl:
                    inputs = inputs.to(self.device)  # Move inputs to the specified device
                    optimizer.zero_grad()
                    _, outputs, mu, logvar = self(inputs, ret_latent=True)
                    loss = self.vae_loss(outputs, inputs, mu, logvar)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    pbar.update()
                self.losses.append(running_loss / len(train_dl))
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                pbar.update()

                val_loss = self._validate_vae(val_dl)
                if early_stopping(epoch, val_loss):
                    break

    def _validate_vae(self, val_dl: DataLoader[tuple[torch.Tensor, ...]]) -> float:
        self.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, in val_dl:
                inputs = inputs.to(self.device)
                _, outputs, mu, logvar = self(inputs, ret_latent=True)
                loss = self.vae_loss(outputs, inputs, mu, logvar)
                val_loss += loss.item()
        return val_loss / len(val_dl)

    def detect_anomalies(
            self,
            data_loader: DataLoader,
            threshold_percentile: float = 99.5,
            use_kl: bool = True,
            use_recon: bool = True) -> torch.Tensor:
        recon_errors = []
        kl_divs = []
        with torch.no_grad():
            for inputs, in data_loader:
                inputs = inputs.to(self.device)
                _, outputs, mu, logvar = self(inputs, ret_latent=True)

                # Reconstruction error
                if use_recon:
                    recon_errors.append(torch.mean((outputs - inputs)**2, dim=1))

                # KL divergence
                if use_kl:
                    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
                    kl_divs.append(kl)

        # Combine scores
        scores = torch.cat(recon_errors + kl_divs) if (use_recon and use_kl) else torch.cat(recon_errors or kl_divs)
        threshold = np.percentile(scores.cpu().numpy(), threshold_percentile)
        return scores > torch.tensor(threshold)

    def compute_mahalanobis(self, data_loader: DataLoader) -> torch.Tensor:
        latents = self.predict_latent(data_loader)
        mean = latents.mean(axis=0)
        cov = np.cov(latents, rowvar=False)
        inv_cov = np.linalg.inv(cov + 1e-6 * np.eye(cov.shape[0]))
        delta = latents - mean
        dists = np.sqrt(np.einsum('ij,jk,ik->i', delta, inv_cov, delta))
        return torch.from_numpy(dists).float()
