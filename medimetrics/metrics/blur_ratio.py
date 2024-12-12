from typing import Any

import numpy as np

from medimetrics.base import NonRefMetric


class BlurRatio(NonRefMetric):
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
        # Compute gradients
        image_xplus1 = np.roll(image, 1, axis=0)
        image_xminus1 = np.roll(image, -1, axis=0)
        image_yplus1 = np.roll(image, 1, axis=1)
        image_yminus1 = np.roll(image, -1, axis=1)

        diff_x = np.abs(image_yminus1 - image_yplus1)
        diff_y = np.abs(image_xminus1 - image_xplus1)

        # Compute mean gradients:
        D_x_mean = diff_x[1:-1, :].mean()
        D_y_mean = diff_y[:, 1:-1].mean()

        local_max_edge_x = diff_x > np.maximum(np.roll(diff_x, -1, axis=1), np.roll(diff_x, 1, axis=1))
        local_max_edge_y = diff_y > np.maximum(np.roll(diff_y, -1, axis=0), np.roll(diff_y, 1, axis=0))

        edge_pixels = ((diff_x > D_x_mean) & (local_max_edge_x)) | ((diff_y > D_y_mean) & (local_max_edge_y))

        # average of neighboring pixels:
        a_x = (image_yminus1 + image_yplus1) / 2
        a_y = (image_xminus1 + image_xplus1) / 2

        eps = (image.max() - image.min()) / 10000

        # inverse blurriness:
        inverse_blurriness_x = (np.abs(image - a_x) + eps) / (a_x + eps)
        inverse_blurriness_y = (np.abs(image - a_y) + eps) / (a_y + eps)
        inverse_blurriness = np.maximum(inverse_blurriness_x, inverse_blurriness_y)
        blur_pixels = inverse_blurriness < 0.1

        # summarized values:
        blur_count = blur_pixels[1:-1, 1:-1].sum()
        edge_count = edge_pixels[1:-1, 1:-1].sum()

        # blur ratio:
        metric_value = blur_count / edge_count

        return metric_value


class MeanBlur(NonRefMetric):
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
        # Compute gradients
        image_xplus1 = np.roll(image, 1, axis=0)
        image_xminus1 = np.roll(image, -1, axis=0)
        image_yplus1 = np.roll(image, 1, axis=1)
        image_yminus1 = np.roll(image, -1, axis=1)

        # average of neighboring pixels:
        a_x = (image_yminus1 + image_yplus1) / 2
        a_y = (image_xminus1 + image_xplus1) / 2

        eps = (image.max() - image.min()) / 10000

        # inverse blurriness:
        inverse_blurriness_x = (np.abs(image - a_x) + eps) / (a_x + eps)
        inverse_blurriness_y = (np.abs(image - a_y) + eps) / (a_y + eps)
        inverse_blurriness = np.maximum(inverse_blurriness_x, inverse_blurriness_y)
        blur_pixels = inverse_blurriness < 0.1

        # summarized values:
        sum_blur = inverse_blurriness[1:-1, 1:-1].sum()
        blur_count = blur_pixels[1:-1, 1:-1].sum()

        # mean blur:
        metric_value = sum_blur / blur_count

        return metric_value
