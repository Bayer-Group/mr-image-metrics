import numpy as np

from medimetrics.base import Distortion


class ShiftIntensity(Distortion):
    def __init__(self, max_strength: int = 5) -> None:
        super().__init__(max_strength)
        self.parameter_ranges = {
            "factor": (0.05, 0.25),
        }

    def apply(self, image: np.ndarray) -> np.ndarray:
        return image + self.parameters["factor"] * (image.max() - image.min())
