import sys
from pathlib import Path

import sklearn
import sklearn.pipeline
import sklearn.preprocessing

sys.path.append(str(Path(__file__).parent.parent))
import os
import torch
from dataset.dataset import Dataset
from models.autoencoder.vae import VariationalAutoencoder
if __name__ == '__main__':
    torch.cuda.empty_cache()

    torch.multiprocessing.set_start_method('spawn', force=True)

    dataset_path = 'data/2018/combined/10000.parquet'
    dataset = Dataset.from_file(path=dataset_path, random_state=7)
    dataset.encode_labels()
    dataset.normalize(algorithm=sklearn.pipeline.Pipeline(
        steps=[
            ('qt', sklearn.preprocessing.QuantileTransformer()),
            ('minmax', sklearn.preprocessing.MinMaxScaler())
        ], verbose=False)
    )

    train_dataset, test_dataset = dataset.split(test_size=10)
    benign, attacks = train_dataset.label_partition(labels=[1], binary=True)

    benign_train, benign_test, benign_val = benign.split(val=True, test_size=40, val_size=10)

    benign_train_dl = benign_train.create_data_loader(include_label=False, batch_size=64, shuffle=True, num_workers=1)
    benign_val_dl = benign_val.create_data_loader(include_label=False, batch_size=2, shuffle=False)

    model = VariationalAutoencoder(hidden_layers=[128, 64, 32], latent_dim=8).to('cuda')
    model.fit(train_dl=benign_train_dl, val_dl=benign_val_dl, epochs=20)
