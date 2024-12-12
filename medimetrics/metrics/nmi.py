from typing import Any

import numpy as np
from scipy.stats import entropy

from medimetrics.base import FullRefMetric


class NMI(FullRefMetric):
    def __init__(self) -> None:
        pass

    """
    Parameters:
    -----------
    image_true: np.array (H, W) or (H, W, D)
        Reference image
    image_test: np.array (H, W) or (H, W, D)
        Image to be evaluated against the reference image

    """

    def compute(self, image_true: np.ndarray, image_test: np.ndarray, bins: int = 100, **kwargs: Any) -> float:
        """Normalized Mutual Information, see:

        https://scikit-image.org/docs/stable/api/skimage.metrics.html#skimage.metrics.normalized_mutual_information
        """

        hist, bin_edges = np.histogramdd(
            [np.reshape(image_true, -1), np.reshape(image_test, -1)], bins=bins, density=True
        )

        H0 = entropy(np.sum(hist, axis=0))
        H1 = entropy(np.sum(hist, axis=1))
        H01 = entropy(np.reshape(hist, -1))

        return (H0 + H1) / H01
