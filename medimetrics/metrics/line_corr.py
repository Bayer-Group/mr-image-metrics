from typing import Any, List

import numpy as np

from medimetrics.base import NonRefMetric


def corr(x: np.ndarray, y: np.ndarray) -> float:
    if (x == y).all():
        return 1.0
    elif np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    else:
        return np.corrcoef(x, y)[0, 1]


def get_corrcoefs(image: np.ndarray, distance: int = 1) -> List[float]:
    ces = []
    for x in range(0, image.shape[0] - distance):
        ces.append(corr(image[x, :], image[x + distance, :]))
    for y in range(0, image.shape[1] - distance):
        ces.append(corr(image[:, y], image[:, y + distance]))
    return ces


class MLC(NonRefMetric):
    def __init__(self) -> None:
        pass

    """
    Parameters:
    -----------
    image: np.array (H, W)
        Reference image
    """

    def compute(self, image: np.ndarray, **kwargs: Any) -> float:
        ces = get_corrcoefs(image, distance=1)
        return np.array(ces).mean()


class MSLC(NonRefMetric):
    def __init__(self) -> None:
        pass

    """
    Parameters:
    -----------
    image: np.array (H, W)
        Reference image
    """

    def compute(self, image: np.ndarray, **kwargs: Any) -> float:
        ces = get_corrcoefs(image, distance=image.shape[1] // 2)
        return np.array(ces).mean()
