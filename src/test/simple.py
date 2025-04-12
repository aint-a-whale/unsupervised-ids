import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import time


class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, latent_dim))
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(), nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, input_dim))

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def generate_synthetic_data(num_samples, input_dim):
    """Generate random synthetic data"""
    return np.random.randn(num_samples, input_dim).astype(np.float32)


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Autoencoder Training')
    parser.add_argument('--device', type=str, default='cpu', choices=['cuda', 'cpu'], help='Device to run on: cuda or cpu')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')

    args = parser.parse_args()

    # Set device with safety check
    device = torch.device(args.device)
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU instead.")
        device = torch.device('cpu')

    # Configuration
    input_dim = 10
    latent_dim = 2
    num_samples = 100000

    # Generate synthetic data
    X = generate_synthetic_data(num_samples, input_dim)
    X_tensor = torch.from_numpy(X)

    # Create dataset and dataloader
    dataset = torch.utils.data.TensorDataset(X_tensor, X_tensor)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=0)

    # Initialize model, loss and optimizer
    model = Autoencoder(input_dim, latent_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    for epoch in range(1, args.epochs + 1):
        total_loss = 0
        for data in dataloader:
            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 1 == 0:
            print(f'Epoch {epoch}/{args.epochs}, Loss: {total_loss/len(dataloader):.4f}')

    print("Training complete!")


if __name__ == "__main__":
    s = time.time()
    main()
    print(time.time() - s)
