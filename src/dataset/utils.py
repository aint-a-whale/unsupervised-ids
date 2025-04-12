import torch
from torch.utils.data import TensorDataset


class LightningTorchDataset(TensorDataset):
    def __init__(self, dataset):
        self.dataset: torch.Tensor = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]
