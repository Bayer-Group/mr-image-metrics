from typing import Any

import numpy as np

from medimetrics.base import FullRefMetric


class MAE(FullRefMetric):
    def __init__(self) -> None:
        pass

    """
    Parameters:
    -----------
    image_true: np.array (H, W)
        Reference image
    image_test: np.array (H, W)
        Image to be evaluated against the reference image
    data_range:
        By default use joint maximum - joint minimum
    """

    def compute(self, image_true: np.ndarray, image_test: np.ndarray, **kwargs: Any) -> float:
        return np.mean(np.abs(image_true - image_test))
