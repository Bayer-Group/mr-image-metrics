from typing import Any

import numpy as np

from medimetrics.base import FullRefMetric


class NMSE(FullRefMetric):
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
        mse = np.power(image_true - image_test, 2).mean()

        # torch.std corrects the std dev by default, so do the same here!
        stddev = np.std(image_true, ddof=1)

        if stddev > 0:
            return mse / stddev
        else:
            return np.inf
