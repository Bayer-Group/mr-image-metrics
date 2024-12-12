from typing import Any

import numpy as np
from scipy.ndimage import uniform_filter1d
from skimage.filters import sobel

from medimetrics.base import NonRefMetric


class BlurEffect(NonRefMetric):
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

    def compute(self, image: np.ndarray, h_size: int = 11, **kwargs: Any) -> float:
        """
        See: https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.blur_effect
        """

        shape = image.shape
        B = []

        slices = tuple([slice(2, s - 1) for s in shape])
        for ax in range(2):
            filt_im = uniform_filter1d(image, h_size, axis=ax)
            im_sharp = np.abs(sobel(image, axis=ax))
            im_blur = np.abs(sobel(filt_im, axis=ax))
            T = np.maximum(0, im_sharp - im_blur)
            M1 = np.sum(im_sharp[slices])
            M2 = np.sum(T[slices])
            B.append(np.abs(M1 - M2) / M1)

        return np.max(B)
