import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, QuantileTransformer
from torch.utils.data import DataLoader, TensorDataset

from utils.logger import Logger

logger = Logger.instance()


@dataclass(slots=True)
class Features:
    numeric: list[str] = field(
        init=False,
        default_factory=lambda: [
            'Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts', 'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 'Fwd Pkt Len Max',
            'Fwd Pkt Len Min', 'Fwd Pkt Len Mean', 'Fwd Pkt Len Std', 'Bwd Pkt Len Max', 'Bwd Pkt Len Min',
            'Bwd Pkt Len Mean', 'Bwd Pkt Len Std', 'Flow Byts/s', 'Flow Pkts/s', 'Flow IAT Mean', 'Flow IAT Std',
            'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Tot', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
            'Bwd IAT Tot', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd Header Len', 'Bwd Header Len',
            'Fwd Pkts/s', 'Bwd Pkts/s', 'Pkt Len Min', 'Pkt Len Max', 'Pkt Len Mean', 'Pkt Len Std', 'Pkt Len Var',
            'Down/Up Ratio', 'Pkt Size Avg', 'Fwd Seg Size Avg', 'Bwd Seg Size Avg', 'Fwd Act Data Pkts', 'Fwd Seg Size Min',
            'Active Mean', 'Active Std', 'Active Max', 'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min',
            'Subflow Fwd Pkts', 'Subflow Fwd Byts', 'Subflow Bwd Pkts', 'Subflow Bwd Byts', 'Init Fwd Win Byts',
            'Init Bwd Win Byts'
        ])

    boolean: list[str] = field(
        init=False,
        default_factory=lambda: [
            'Fwd PSH Flags', 'Fwd URG Flags', 'FIN Flag Cnt', 'SYN Flag Cnt', 'RST Flag Cnt', 'PSH Flag Cnt', 'ACK Flag Cnt',
            'URG Flag Cnt', 'CWE Flag Count', 'ECE Flag Cnt'
        ])

    categorical: list[str] = field(init=False, default_factory=lambda: ['Protocol'])
    exclude: list[str] = field(
        init=False,
        default_factory=lambda: [
            'Dst Port', 'Timestamp', 'Bwd PSH Flags', 'Bwd URG Flags', 'Fwd Byts/b Avg', 'Fwd Pkts/b Avg',
            'Fwd Blk Rate Avg', 'Bwd Byts/b Avg', 'Bwd Pkts/b Avg', 'Bwd Blk Rate Avg'
        ])


@dataclass
class Dataset:
    Xy: pd.DataFrame
    features: Features = field(init=False, default_factory=Features)

    transformer: Pipeline = field(
        init=False,
        default=Pipeline(
            steps=[
                ('quantile', QuantileTransformer(copy=False, output_distribution='uniform')),
                ('minmax', MinMaxScaler()),
            ],
            verbose=False,
        ),
    )
    label_encoder: LabelEncoder = field(init=False, default=LabelEncoder())

    normalized: bool = field(init=False, default=False)
    labels_encoded: bool = field(init=False, default=False)
    labels_converted_to_binary: bool = field(init=False, default=False)

    recurrent: bool = field(repr=False, default=False)
    name: str = ''
    random_state: Optional[int] = None
    device: str = field(default='cuda' if torch.cuda.is_available() else 'cpu')  # Add device parameter

    def __post_init__(self):
        self.dtypes = self.Xy.dtypes
        if self.recurrent is False:
            self.Xy.drop_duplicates(inplace=True)
            self.Xy.drop(columns=self.features.exclude, inplace=True)
            self.Xy[self.features.boolean] = self.Xy[self.features.boolean].astype('Int8')
            self.Xy = pd.get_dummies(self.Xy, columns=self.features.categorical)
            self.transformer.fit(self.Xy[self.features.numeric])
            self.label_encoder.fit(self.Xy.Label)
            self.labels_mapping = self.get_labels_mapping()
            self.Xy['Label_binary'] = self.Xy.Label.map(lambda x: 0 if x == 'Benign' else 1)

    @classmethod
    def from_file(cls, path: str, random_state=None, dtypes: dict[str, str] = {}, device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> 'Dataset':
        common_params = {'name': 'init', 'random_state': random_state, 'device': device}
        if path.endswith('parquet') or not dtypes:
            return Dataset(pd.read_parquet(path), **common_params)
        else:
            return Dataset(pd.read_csv(path, dtype=dtypes), **common_params)

    def __create_name(self, add_name: str) -> str:
        if self.name != '':
            if add_name is not None:
                name = self.name + '.' + add_name
            else:
                name = self.name + '.' + '[]'
        else:
            if add_name is not None:
                name = add_name
            else:
                name = '[]'
        return name

    def get_labels_mapping(self) -> dict[int, str]:
        return {i: cls for i, cls in enumerate(self.label_encoder.classes_)}

    def set_random_state(self, value: int) -> None:
        self.random_state = value

    @property
    def X(self) -> pd.DataFrame:
        return self.Xy.drop(columns=['Label', 'Label_binary'])

    @property
    def X_np(self) -> NDArray[np.float64]:
        return self.X.to_numpy(dtype=np.float64)

    @property
    def X_tensor(self) -> torch.Tensor:
        return torch.Tensor(self.X_np).to(self.device)  # Move tensor to the specified device

    @property
    def y(self) -> pd.Series:
        return self.Xy.Label

    @property
    def y_np(self) -> NDArray[np.float64]:
        return self.y.to_numpy(dtype=np.float64).reshape(-1, 1)

    @property
    def y_tensor(self) -> torch.Tensor:
        return torch.Tensor(self.y_np).to(self.device)  # Move tensor to the specified device

    @property
    def y_binary(self) -> pd.Series:
        return self.Xy.Label_binary

    @property
    def y_binary_np(self) -> NDArray[np.float64]:
        return self.y_binary.to_numpy(dtype=np.float64)

    @property
    def y_binary_tensor(self) -> torch.Tensor:
        return torch.Tensor(self.y_binary_np).to(self.device)  # Move tensor to the specified device

    @property
    def shape(self) -> tuple[int, int]:
        return self.Xy.shape

    def create_data_loader(
        self,
        include_label: bool = False,
        batch_size: int = 1,
        binary: bool = False,
        shuffle: bool = False,
        num_workers: int = 0,
    ) -> DataLoader[tuple[torch.Tensor, ...]]:
        if include_label is True:
            if binary is True:
                y = self.y_binary_tensor
            else:
                y = self.y_tensor
            dataset = TensorDataset(self.X_tensor, y)
        else:
            dataset = TensorDataset(self.X_tensor)
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )

    def _create_dataset(self, X: pd.DataFrame, y: pd.Series, name: str) -> 'Dataset':
        name = self.__create_name(add_name=name)
        dataset = Dataset(Xy=pd.concat([X, y], axis=1), recurrent=True, name=name, device=self.device)
        dataset.transformer = self.transformer
        dataset.normalized = self.normalized
        dataset.label_encoder = self.label_encoder
        dataset.labels_encoded = self.labels_encoded
        dataset.labels_mapping = self.labels_mapping
        return dataset

    def label_partition(self, labels: list[int] | list[str], binary: bool = False) -> tuple['Dataset', 'Dataset']:
        label_col = 'Label_binary' if binary else 'Label'

        if self.labels_encoded is True:
            if isinstance(labels[0], int) is True:
                labels_adj = labels
            else:
                labels_adj = [self.label_encoder.classes_.tolist().index(label) for label in labels]
        else:
            if isinstance(labels[0], str) is True:
                labels_adj = labels
            else:
                labels_adj = [self.label_encoder.classes_[np.array(labels)]]

        left = self.Xy[~self.Xy[label_col].isin(labels_adj)]
        right = self.Xy[self.Xy[label_col].isin(labels_adj)]

        return (
            self._create_dataset(
                X=left.drop(columns=['Label']), y=left.Label, name=f'special({",".join(map(str, left.Label.unique()))})'),
            self._create_dataset(
                X=right.drop(columns=['Label']), y=right.Label, name=f'special({",".join(map(str, right.Label.unique()))})'),
        )

    def split(self, test_size=15, val=False, val_size=10) -> tuple['Dataset', ...]:
        test_size = test_size / 100
        val_size = val_size / 100

        if val is True:
            test_val_size = test_size + val_size
        else:
            test_val_size = test_size

        X_train, X_temp, y_train, y_temp = train_test_split(
            pd.concat([self.X, self.Xy.Label_binary], axis=1),
            self.y,
            test_size=test_val_size,
            stratify=self.y,
            random_state=self.random_state,
        )

        if val is True:
            val_size = (1 / (test_val_size)) * val_size
            X_test, X_val, y_test, y_val = train_test_split(
                X_temp, y_temp, test_size=val_size, stratify=y_temp, random_state=self.random_state)

            return (
                self._create_dataset(X=X_train, y=y_train, name='train'),
                self._create_dataset(X=X_test, y=y_test, name='test'),
                self._create_dataset(X=X_val, y=y_val, name='val'),
            )
        else:
            return (
                self._create_dataset(X=X_train, y=y_train, name='train'),
                self._create_dataset(X=X_temp, y=y_temp, name='test'),
            )

    def downsample(self, n: int) -> 'Dataset':
        Xy = pd.concat([self.X, self.Xy.Label_binary, self.Xy.Label], axis=1).sample(n=n, random_state=self.random_state)
        return self._create_dataset(X=Xy.drop(columns=['Label']), y=Xy.Label, name=f'downsampled({n})')

    def _get_current_label(self, init_label: str | int) -> str | int:
        label = init_label
        if self.labels_encoded is True:
            if type(init_label) is not int:
                label = self.label_encoder.classes_.tolist().index(init_label)
        else:
            if type(init_label) is not str:
                label = self.label_encoder.classes_.tolist()[init_label]
        return label

    def make_major(
        self,
        major_class: int | str,
        percent: float,
        binary: bool = False,
    ) -> 'Dataset':
        label_col = 'Label_binary' if binary else 'Label'

        major_label = self._get_current_label(init_label=major_class)
        major_percent = percent / 100

        label_counts = self.Xy.Label.value_counts()
        label_samples = label_counts * 0

        num_major_label_samples = min(
            int(len(self.Xy) * major_percent),
            label_counts[major_label],
        )
        num_minor_label_samples = int(num_major_label_samples / major_percent - num_major_label_samples)  # x = y/p - y

        label_samples[major_label] = num_major_label_samples
        for label in label_counts.index:
            if label != major_label:
                label_samples[label] = min(
                    int(num_minor_label_samples / (self.label_encoder.classes_.size - 1)),
                    label_counts[label],
                )

        data_gb = self.Xy.groupby(by=label_col, as_index=False, group_keys=False)
        Xy = data_gb.apply(
            lambda x: x.
            sample(  # type: ignore
                n=min(label_samples[x.name], len(x)),  # type: ignore
                random_state=self.random_state,
            ))  # type: ignore

        return self._create_dataset(X=Xy.drop(columns=['Label']), y=Xy.Label, name=f'majored({major_label})')

    def normalize(self, algorithm: Optional[Pipeline] = None) -> None:
        if self.normalized is False:
            if algorithm is not None:
                self.transformer = algorithm
                self.transformer.fit(self.Xy[self.features.numeric])
            self.Xy[self.features.numeric] = self.transformer.transform(self.Xy[self.features.numeric])
            self.normalized = True
        else:
            logger.warning('already normalized')

    def unnormalize(self) -> None:
        if self.normalized is True:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=UserWarning)
                self.Xy[self.features.numeric] = self.transformer.inverse_transform(self.Xy[self.features.numeric])
            types_dict = dict(zip(self.features.numeric, [i.type for i in self.dtypes[self.features.numeric]]))
            self.Xy = self.Xy.astype(types_dict)
            self.normalized = False
        else:
            logger.warning('is unnormalized')

    def encode_labels(self) -> None:
        if self.labels_encoded is False:
            self.Xy.Label = self.label_encoder.transform(self.Xy.Label)
            self.labels_encoded = True
        else:
            logger.warning('already encoded')

    def decode_labels(self) -> None:
        if self.labels_encoded is True:
            self.Xy.Label = self.label_encoder.inverse_transform(self.Xy.Label)
            self.labels_encoded = False
        else:
            logger.warning('already decoded')

    def __repr__(self) -> str:
        label_counts = self.y.value_counts(sort=False)
        label_normalized_counts = self.y.value_counts(normalize=True, sort=False)

        label_combined_counts = pd.concat([label_counts, label_normalized_counts], axis=1)
        label_combined_counts.columns = ['count', 'perc']
        label_combined_counts.perc = label_combined_counts.perc.round(decimals=5) * 100

        if self.labels_encoded is True:
            label_combined_counts.index.rename('label_id', inplace=True)
            label_names = label_combined_counts.index.map(lambda x: self.label_encoder.classes_[x])
            label_combined_counts.insert(loc=0, column='label_name', value=label_names)
            label_combined_counts.set_index('label_name', append=True, inplace=True)

            binary_label_mapping = dict(
                pd.concat([self.y, self.y_binary], axis=1).groupby(by='Label', as_index=False).first().values)

        else:
            label_combined_counts.index.rename('label_name', inplace=True)
            label_indexes = label_combined_counts.index.map(lambda x: list(self.label_encoder.classes_).index(x))
            label_combined_counts.insert(loc=0, column='label_id', value=label_indexes)
            label_combined_counts.set_index('label_id', append=True, inplace=True)
            label_combined_counts = label_combined_counts.swaplevel()

            transformed_labels = pd.Series(self.label_encoder.transform(self.y), name=self.y.name)  # type: ignore

            binary_label_mapping = dict(
                pd.concat([transformed_labels, self.y_binary], axis=1).groupby(by='Label', as_index=False).first().values)

        label_idx = pd.Series([i[0] for i in label_combined_counts.index], index=label_combined_counts.index)
        binary_labels = label_idx.apply(lambda x: binary_label_mapping[x])

        label_combined_counts.insert(loc=2, column='binary_label', value=binary_labels, allow_duplicates=True)
        label_combined_counts.set_index('binary_label', append=True, inplace=True)

        label_combined_counts.sort_values(by='label_id', ascending=True, inplace=True)

        return (
            f'name: {self.name}\n'
            f"{self.Xy.head(3).to_markdown(index=False, headers='keys', tablefmt='psql')}\n"
            f'size: {self.Xy.shape[0]}\n'
            f'features: {self.X.shape[1]}\n'
            f'normalized: {self.normalized}\n'
            f'labels encoded: {self.labels_encoded}\n'
            f"{label_combined_counts.reset_index().to_markdown(index=False, headers='keys', tablefmt='psql')}")
