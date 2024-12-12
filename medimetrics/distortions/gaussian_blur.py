import numpy as np
from skimage.filters import gaussian

from medimetrics.base import Distortion


class GaussianBlur(Distortion):
    def __init__(self, max_strength: int = 5) -> None:
        super().__init__(max_strength)
        self.parameter_ranges = {
            "sigma": (0.2, 1.3),
        }

    def apply(self, image: np.ndarray) -> np.ndarray:
        return gaussian(image, sigma=self.parameters["sigma"])
