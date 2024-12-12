from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

import numpy as np


class FullRefMetric(ABC):
    @abstractmethod
    def __init__(self) -> None:
        pass

    """
    Parameters:
    -----------
    image_true: np.ndarray (H, W)
        Reference image
    image_test: np.ndarray (H, W)
        Image to be evaluated against the reference image
    kwargs**:
        Any other optional keyword parameters
    """

    @abstractmethod
    def compute(self, image_true: np.ndarray, image_test: np.ndarray, **kwargs: Any) -> float:
        pass


class NonRefMetric(ABC):
    @abstractmethod
    def __init__(self) -> None:
        pass

    """
    Parameters:
    -----------
    image: np.array (H, W)
        Image to be evaluated
    kwargs**:
        Any other optional keyword parameters
    """

    @abstractmethod
    def compute(self, image: np.ndarray, **kwargs: Any) -> float:
        pass


class Distortion(ABC):
    def __init__(self, max_strength: int = 5) -> None:
        """
        Parameters:
        -----------
        max_strength:
            maximal strength level
        parameter_ranges: Dict[parameter_name, (min_value, max_value) ]
            Dictionary to store default parameter ranges
        """
        self.max_strength = max_strength
        self.parameter_ranges: Dict[str, Tuple[Any, Any]] = {}

    def __call__(self, image: np.ndarray, strength: int) -> np.ndarray:
        """
        Parameters:
        -----------
        image: np.ndarray (H, W)
            Image to be distorted
        strength:
            strength level to be distorted
        """
        if strength == 0:
            return image
        elif strength <= self.max_strength:
            self.parameters = self.get_parameters(strength)
            return self.apply(image)
        else:
            raise Exception(f"Selected strength {strength} is higher than maximum strength {self.max_strength}!")

    @abstractmethod
    def apply(self, image: np.ndarray) -> np.ndarray:
        """Distortions should be implemented here Parameters for selected
        strength are given in self.parameters: Dict[parameter_name,
        parameter_value]"""
        pass

    def get_parameters(self, strength: int) -> Dict[str, Any]:
        parameters: Dict[str, Any] = {}

        for p_name, (min_value, max_value) in self.parameter_ranges.items():
            # strength = 1 is min_value
            # strength = self.max_strength is max_value
            value = min_value + (max_value - min_value) * (strength - 1) / (self.max_strength - 1)
            if isinstance(min_value, int):
                parameters[p_name] = int(value)
            else:
                parameters[p_name] = value
        return parameters
