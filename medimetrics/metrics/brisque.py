from typing import Any

import numpy as np
from brisque import BRISQUE as BRISQUE_model

from medimetrics.base import NonRefMetric


class BRISQUE(NonRefMetric):
    def __init__(self) -> None:
        pass

    """
    Parameters:
    -----------
    image: np.array (H, W)
        Reference image
    """

    def compute(self, image: np.ndarray, **kwargs: Any) -> float:
        rgb_image = np.stack([image, image, image], axis=2)

        obj = BRISQUE_model(url=False)
        return obj.score(rgb_image)
