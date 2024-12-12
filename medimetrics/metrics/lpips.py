import warnings
from typing import Any

import lpips as lp
import numpy as np
import torch

from medimetrics.base import FullRefMetric


class LPIPS(FullRefMetric):
    def __init__(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            self.loss_fn_alex = lp.LPIPS(net="alex", verbose=False)  # best forward scores

    """
    Parameters:
    -----------
    image_true: np.array (H, W)
        Reference image
    image_test: np.array (H, W)
        Image to be evaluated against the reference image
    """

    def compute(self, image_true: np.ndarray, image_test: np.ndarray, **kwargs: Any) -> float:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.loss_fn_alex = self.loss_fn_alex.to(device)

        # convert to Tensor and move to gpu:
        T_image_true = torch.Tensor(image_true.copy()).to(device)
        T_image_test = torch.Tensor(image_test.copy()).to(device)

        T_image_true_RGB = torch.stack([T_image_true, T_image_true, T_image_true], dim=0).unsqueeze(0)
        T_image_test_RGB = torch.stack([T_image_test, T_image_test, T_image_test], dim=0).unsqueeze(0)

        # dims: (B, 3, X, Y)
        metric_value = self.loss_fn_alex(T_image_true_RGB, T_image_test_RGB).item()

        # move back to numpy/cpu
        # np_metric_image = metric_image.cpu().numpy()
        # np_metric_value = metric_value.detach().cpu().numpy()

        return metric_value
