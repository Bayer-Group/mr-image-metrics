from typing import Any, Optional

import numpy as np

from medimetrics.base import FullRefMetric


class DICE(FullRefMetric):
    def __init__(self) -> None:
        pass

    """
    Parameters:
    -----------
    image_true: np.array (H, W)
        Reference image
    image_test: np.array (H, W)
        Image to be evaluated against the reference image
    label_idx:
        label index to select from segmentations
        if None, select all pixels > 0
    """

    def compute(
        self, image_true: np.ndarray, image_test: np.ndarray, label_idx: Optional[int] = None, **kwargs: Any
    ) -> float:
        # stabilize dice, when image_true and image_test are all zero
        epsilon = 0.00001

        # select labels from segmentations:
        if label_idx is None:
            binary_image_true = image_true > 0
            binary_image_test = image_test > 0
        else:
            binary_image_true = image_true == label_idx
            binary_image_test = image_test == label_idx

        double_intersection = 2.0 * np.count_nonzero(binary_image_true & binary_image_test) + epsilon

        dice = double_intersection / (
            np.count_nonzero(binary_image_true) + np.count_nonzero(binary_image_test) + epsilon
        )

        return dice
