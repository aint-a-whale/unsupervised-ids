from contextlib import contextmanager
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.axes import Axes
from numpy.typing import NDArray

sns.set_style('darkgrid')


@dataclass
class DPI:
    target: int
    current: int = field(default=plt.rcParams['figure.dpi'], init=False)

    @contextmanager
    def quality(self):
        try:
            plt.rcParams['figure.dpi'] = self.target
            yield
        finally:
            plt.rcParams['figure.dpi'] = self.current


class Plot:
    def __init__(self, dpi: int = 100):
        self.dpi = DPI(target=dpi)

    def _line_array(self, ary: list[float] | NDArray[np.number]) -> None:
        with self.dpi.quality():
            x, y = zip(*enumerate(ary))
            fig, ax = plt.subplots()
            ax: Axes
            sns.lineplot(x=x, y=y, ax=ax)
            fig.tight_layout()
            plt.show()

    def loss(self, losses: list[float] | NDArray[np.number]) -> None:
        self._line_array(losses)
