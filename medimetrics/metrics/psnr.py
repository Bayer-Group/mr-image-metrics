from typing import Any

import numpy as np

from medimetrics.base import FullRefMetric


class PSNR(FullRefMetric):
    def __init__(self) -> None:
        pass

    """
    Parameters:
    -----------
    image_true: np.array (H, W) or (H, W, D)
        Reference image
    image_test: np.array (H, W) or (H, W, D)
        Image to be evaluated against the reference image
    data_range:
        By default use joint maximum - joint minimum
    """

    def compute(self, image_true: np.ndarray, image_test: np.ndarray, data_range: float = None, **kwargs: Any) -> float:
        # If no data range is given, it is calculated from the range of the data
        if data_range is None:
            data_range = np.maximum(np.max(image_true), np.max(image_test)) - np.minimum(
                np.min(image_true), np.min(image_test)
            )

        mse = np.mean(np.power((image_true - image_test) / data_range, 2))
        # result = (20 * np.math.logn(10, data_range)) - (10* np.math.logn(10, (mse)
        # As the deviation was already normalized, the data_range is always 1.0
        # which turns the first term to 0.0

        if mse == 0:
            result = np.inf
        else:
            result = -10 * np.log10(mse)

        return result
