import numpy as np

from medimetrics.base import Distortion


def apply_log_gamma(image: np.ndarray, log_gamma: float) -> np.ndarray:
    # normalize to range 0.0 to 1.0, because this is how gamma transforms can be safely applied
    # and stay within original range
    image_min = image.min()
    image_max = image.max()
    norm_image = image - image_min
    if image_max > image_min:
        norm_image /= image_max - image_min

    distorted_image = np.power(norm_image, np.exp(log_gamma))

    # normalize back to original intensity range:
    if image_max > image_min:
        renorm_image = distorted_image * (image_max - image_min)
    renorm_image += image_min
    return renorm_image


class GammaLow(Distortion):
    def __init__(self, max_strength: int = 5) -> None:
        super().__init__(max_strength)
        self.parameter_ranges = {
            "log_gamma": (-0.01, -0.916),
        }

    def apply(self, image: np.ndarray) -> np.ndarray:
        return apply_log_gamma(image, self.parameters["log_gamma"])


class GammaHigh(Distortion):
    def __init__(self, max_strength: int = 5) -> None:
        super().__init__(max_strength)
        self.parameter_ranges = {
            "log_gamma": (0.095, 0.916),
        }

    def apply(self, image: np.ndarray) -> np.ndarray:
        return apply_log_gamma(image, self.parameters["log_gamma"])
