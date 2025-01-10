from typing import Any

import numpy as np
import torch
from piq import haarpsi

from medimetrics.base import FullRefMetric


class HaarPSI(FullRefMetric):
    def __init__(self) -> None:
        pass

    def compute(self, image_true: np.ndarray, image_test: np.ndarray, data_range: float = None, **kwargs: Any) -> float:
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

        # Only intensities > 0 allowed, shift minimum value to 0
        min_value = np.minimum(np.min(image_true), np.min(image_test))
        image_true = image_true - min_value
        image_test = image_test - min_value

        # If no data range is given, it is the maximum value of the two images
        if data_range is None:
            data_range = np.maximum(np.max(image_true), np.max(image_test))

        image_true_T = torch.Tensor(image_true.copy()).unsqueeze(0).unsqueeze(0)
        image_test_T = torch.Tensor(image_test.copy()).unsqueeze(0).unsqueeze(0)

        # Needs input (B, C, H, W)
        metric_score = haarpsi(image_true_T, image_test_T, data_range=data_range).item()

        return metric_score
