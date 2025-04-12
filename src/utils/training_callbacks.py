from utils.logger import Logger
from tqdm import tqdm


class EarlyStopping:
    '''Early stopping callback to stop training if validation loss doesn't improve.'''

    logger = Logger.instance()

    def __init__(self, patience=7, min_delta=0.001):
        '''
        Args:
            patience (int, optional): Number of epochs with no improvement to wait before stopping. Defaults to 7.
            min_delta (float, optional): Minimum change in monitored metric to qualify as improvement. Defaults to 0.001.
        '''
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = float('inf')
        self.wait_count = 0

    def _on_epoch_end(self, epoch: int, current_loss: float) -> bool:
        '''Called at the end of each epoch to check for early stopping.'''
        if current_loss < self.best_score + self.min_delta:
            self.best_score = current_loss
            self.wait_count = 0
        else:
            self.wait_count += 1
            if self.wait_count >= self.patience:
                self.logger.info(f'Early stopping triggered at epoch {epoch + 1}')
                return True
        return False

    def __call__(self, epoch: int, current_loss: float) -> bool:
        return self._on_epoch_end(epoch, current_loss)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass


class ProgressBar:
    def __init__(self, total_epochs: int, total_train_batches: int, total_val_batches: int) -> None:
        self.total_epochs = total_epochs
        self.total_train_batches = total_train_batches
        self.total_val_batches = total_val_batches
        self.epoch = 0

    def on_epoch_begin(self) -> None:
        self.epoch += 1
        print(f'Epoch {self.epoch}/{self.total_epochs}')

    def on_train_batch_begin(self) -> None:
        self.train_bar = tqdm(total=self.total_train_batches, desc='Training', leave=False)

    def on_train_batch_end(self) -> None:
        self.train_bar.update(1)

    def on_val_batch_begin(self) -> None:
        self.val_bar = tqdm(total=self.total_val_batches, desc='Validation', leave=False)

    def on_val_batch_end(self) -> None:
        self.val_bar.update(1)

    def on_epoch_end(self) -> None:
        self.train_bar.close()
        self.val_bar.close()
        print('')
