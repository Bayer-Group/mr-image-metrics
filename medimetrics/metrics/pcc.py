from typing import Any

import numpy as np

from medimetrics.base import FullRefMetric


class PCC(FullRefMetric):
    def __init__(self) -> None:
        pass

    """
    Parameters:
    -----------
    image_true: np.array (H, W)
        Reference image
    image_test: np.array (H, W)
        Image to be evaluated against the reference image

    """

    def compute(self, image_true: np.ndarray, image_test: np.ndarray, **kwargs: Any) -> float:
        return np.corrcoef(image_true.flatten(), image_test.flatten())[0, 1]
