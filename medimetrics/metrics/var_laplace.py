from typing import Any

import numpy as np
from skimage.filters import laplace

from medimetrics.base import NonRefMetric


class VarLaplace(NonRefMetric):
    def __init__(self) -> None:
        pass

    def compute(self, image: np.ndarray, **kwargs: Any) -> float:
        """
        Parameters:
        -----------
        image: np.array (H, W)
            Reference image
        """

        return np.var(laplace(image))
