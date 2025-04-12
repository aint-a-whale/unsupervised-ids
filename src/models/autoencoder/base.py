from typing import Literal, Optional, Self, overload
import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.logger import Logger
from utils.training_callbacks import EarlyStopping


class BaseAutoencoder(nn.Module):
    logger = Logger.instance()

    def __init__(
            self,
            input_dim: int = 71,
            hidden_layers: list[int] = [64, 32],
            latent_dim: int = 8,
            hidden_layers_decoder: Optional[list[int]] = None,
            dropout_rate: float = 0,
            device: str = 'cuda' if torch.cuda.is_available() else 'cpu',  # Default to GPU if available, else CPU
    ):
        super().__init__()
        encoder_layers = [input_dim] + hidden_layers + [latent_dim]
        decoder_layers = [latent_dim] + (hidden_layers_decoder or hidden_layers[::-1]) + [input_dim]

        self.device = device
        self.encoder = self._build_layer_sequence(encoder_layers).to(self.device)
        self.decoder = self._build_layer_sequence(decoder_layers).to(self.device)
        self.decoder.add_module('output_activation', nn.Sigmoid())
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.dropout_rate = dropout_rate
        self.losses: list[float] = []

    def _build_layer_sequence(self, layers: list[int]) -> nn.Sequential:
        """Helper function to create a sequence of layers."""
        seq = nn.Sequential()
        for i in range(len(layers) - 1):
            seq.append(nn.Linear(layers[i], layers[i + 1]))
            if i + 1 != len(layers) - 1:
                seq.append(nn.BatchNorm1d(layers[i + 1]))
                seq.append(nn.SiLU())
                if self.dropout_rate > 0:
                    seq.add_module(f'dropout_{i}', nn.Dropout(self.dropout_rate))
        return seq

    @overload
    def forward(self, x: torch.Tensor, ret_latent: Literal[True] = True) -> tuple[torch.Tensor, ...]:
        ...

    @overload
    def forward(self, x: torch.Tensor, ret_latent: Literal[False] = False) -> torch.Tensor:
        ...

    def forward(self, x: torch.Tensor, ret_latent: bool = False):
        x = x.to(self.device)
        enc = self.encoder(x)
        recon = self.decoder(enc)
        if ret_latent:
            return enc, recon
        else:
            return recon

    def predict_latent(self, x: DataLoader[tuple[torch.Tensor, ...]]) -> NDArray[np.float32]:
        self.eval()
        latent_representations = torch.zeros(size=(len(x.dataset), self.latent_dim))  # type: ignore
        with torch.no_grad():
            batch_size = x.batch_size or 1
            with tqdm(total=len(x.dataset)) as pbar:  # type: ignore
                for i, batch in enumerate(x):
                    inputs = batch[0].to(self.device)
                    output = self.encoder(inputs)
                    latent_representations[i * batch_size:batch_size * (i + 1)] = output.cpu()
                    pbar.update(batch_size)
        return latent_representations.numpy()

    def reconstruct(self, x: torch.Tensor | DataLoader[tuple[torch.Tensor, ...]], loss: bool = False) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            if isinstance(x, torch.Tensor):
                x = x.to(self.device)
                if len(tuple(x.shape)) == 1:
                    return self.forward(x=torch.unsqueeze(input=x, dim=0))[1]
                else:
                    return self.forward(x=x)[1]
            else:
                batch_size = x.batch_size or 1
                reconstructions = torch.zeros(size=(len(x.dataset), x.dataset[0][0].numel()))  # type: ignore
                with tqdm(total=len(x.dataset)) as pbar:  # type: ignore
                    for i, batch in enumerate(x):
                        inputs = batch[0].to(self.device)
                        output = self.forward(inputs)[1]
                        reconstructions[i * batch_size:batch_size * (i + 1)] = output.cpu()
                        pbar.update(batch_size)
                return reconstructions

    def fit(
        self,
        train_dl: DataLoader[tuple[torch.Tensor, ...]],
        val_dl: DataLoader[tuple[torch.Tensor, ...]],
        epochs: int = 50,
        lr: float = 0.001,
        early_stopping_delta: float = 0.0001,
    ):
        early_stopping = EarlyStopping(min_delta=early_stopping_delta)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = torch.nn.MSELoss()

        for epoch in range(epochs):
            with tqdm(total=len(train_dl), leave=False) as pbar:
                running_loss = 0.0
                for inputs, in train_dl:
                    inputs = inputs.to(self.device)
                    optimizer.zero_grad()
                    outputs = self(inputs)
                    loss = criterion(outputs, inputs)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    pbar.set_description(f'Epoch [{epoch}/{epochs}]')
                    pbar.set_postfix(train_loss=loss.item())
                    pbar.update()
                self.losses.append(running_loss)

                self.eval()
                with torch.no_grad():
                    val_running_loss = 0.0
                    for val_inputs, in val_dl:
                        val_inputs = val_inputs.to(self.device)
                        val_outputs = self(val_inputs)
                        val_running_loss += criterion(val_outputs, val_inputs).item()
                    val_running_loss_mean = val_running_loss / len(val_dl)

                if early_stopping(epoch, val_running_loss_mean):
                    break

    def save_weights(self, filepath: str) -> None:
        torch.save(self, filepath)

    @classmethod
    def load_weights(cls, filepath: str) -> Self:
        model = torch.load(filepath, weights_only=False)
        model.eval()  # Set the model to evaluation mode
        cls.logger.info(f"Weights loaded: input dimension: {model.input_dim}, latent dimension: {model.latent_dim}")
        return model

    def summary(self):
        summary = f"{self.__class__.__name__} Summary\n"
        summary += "------------------------\n"

        # Display encoder and decoder architectures
        summary += "Encoder:\n"
        for i, layer in enumerate(self.encoder):
            summary += f"  Layer {i}: {layer}\n"

        summary += "\nDecoder:\n"
        for i, layer in enumerate(self.decoder):
            summary += f"  Layer {i}: {layer}\n"

        # Count total and trainable parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # Display parameter information
        summary += "\n"
        summary += f"Total Parameters: {total_params:,}\n"
        summary += f"Trainable Parameters: {trainable_params:,}\n"
        print(summary)
