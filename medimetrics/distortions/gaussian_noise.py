import numpy as np

from medimetrics.base import Distortion


class GaussianNoise(Distortion):
    def __init__(self, max_strength: int = 5) -> None:
        super().__init__(max_strength)
        self.parameter_ranges = {
            "std": (0.1, 0.4),
        }

    def apply(self, image: np.ndarray) -> np.ndarray:
        # np.random.seed(1234)
        rng = np.random.default_rng(seed=1234)
        noise = rng.normal(loc=0, scale=1.0, size=image.shape)

        distorted_image = image.astype(np.float32) + noise.astype(np.float32) * (image.std() * self.parameters["std"])
        return distorted_image
