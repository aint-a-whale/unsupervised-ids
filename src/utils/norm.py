import numpy as np
from numpy.typing import NDArray


class TanhNormalizer:
    """
    A class implementing a modified tanh-estimator for normalization.

    The normalization method is based on the approach proposed in "Efficient approach to
    Normalization of Multimodal Biometric Scores, 2011."

    Attributes:
    - mean_ (NDArray[np.float_]): Mean values for each column obtained during the fit step.
    - std_ (NDArray[np.float_]): Standard deviation values for each column obtained during the fit step.
    """
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, unnormalized_data: NDArray[np.int_ | np.float_]) -> None:
        """
        Calculate mean and standard deviation for each column of the input matrix.

        Parameters:
        - unnormalized_data (NDArray[np.int_ | np.float_]): Input matrix to calculate mean and std.
        """
        self.mean_ = np.mean(unnormalized_data, axis=0)
        self.std_ = np.std(unnormalized_data, axis=0)

    def transform(self, unnormalized_data: NDArray[np.int_ | np.float_]) -> NDArray[np.float_]:
        """
        Normalize each column of the input matrix using the hyperbolic tangent function.

        Parameters:
        - unnormalized_data (NDArray[np.int_ | np.float_]): Input matrix to be normalized.

        Returns:
        - NDArray[np.float_]: Normalized matrix where each column is scaled to the range [0, 1] using tanh.
        """
        if self.mean_ is None or self.std_ is None:
            raise ValueError("fit method must be called before transform")

        scaled_data = 0.5 * (np.tanh(0.01 * ((unnormalized_data - self.mean_) / self.std_)) + 1)
        return scaled_data

    def fit_transform(self, unnormalized_data: NDArray[np.int_ | np.float_]) -> NDArray[np.float_]:
        """
        Fit the normalization parameters and transform the input matrix in a single step.

        Parameters:
        - unnormalized_data (NDArray[np.int_ | np.float_]): Input matrix to be normalized.

        Returns:
        - NDArray[np.float_]: Normalized matrix where each column is scaled to the range [0, 1] using tanh.
        """
        self.fit(unnormalized_data)
        return self.transform(unnormalized_data)


'''Example usage:
input_matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

normalizer = TanhNormalizer()
normalized_matrix = normalizer.fit_transform(input_matrix)
'''
