from typing import Any

import numpy as np

from medimetrics.base import NonRefMetric


class MeanTotalVar(NonRefMetric):
    def __init__(self) -> None:
        pass

    """
    Parameters:
    -----------
    image: np.array (H, W)
        Reference image
    data_range:
        By default use joint maximum - joint minimum
    """

    def compute(self, image: np.ndarray, **kwargs: Any) -> float:
        xplus1 = np.roll(image, shift=1, axis=0)
        yplus1 = np.roll(image, shift=1, axis=1)

        variation = np.sqrt((xplus1 - image) ** 2 + (yplus1 - image) ** 2)

        mean_total_variation = variation[:-1, :-1].mean()

        return mean_total_variation
