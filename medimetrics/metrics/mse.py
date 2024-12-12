from typing import Any

import numpy as np

from medimetrics.base import FullRefMetric


class MSE(FullRefMetric):
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

    def compute(self, image_true: np.ndarray, image_test: np.ndarray, **kwargs: Any) -> float:
        return np.mean(np.power(image_true - image_test, 2))
