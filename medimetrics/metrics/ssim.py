from typing import Any, List

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_erosion

from medimetrics.base import FullRefMetric


def gaussian_kernel(dims: int, kernel_size: int, sigma: float) -> torch.Tensor:
    """Creates a kernel with Gaussian weighting.

    Parameters:
    ----------
    dims:
        Currently only implemented for 2 dimensions (dims=2)
    kernel_size:
        side length in pixels of square shaped kernel
    sigma:
        standard deviation of Gaussian distribution
    """
    assert dims == 2
    dist = torch.arange(start=(1 - kernel_size) / 2, end=(1 + kernel_size) / 2, step=1)
    gauss = torch.exp(-torch.pow(dist / sigma, 2) / 2)
    gaussian_1d = (gauss / gauss.sum()).unsqueeze(dim=0)  # (1, kernel_size)

    kernel_2d = torch.matmul(gaussian_1d.t(), gaussian_1d)  # (kernel_size, 1) * (1, kernel_size)
    return kernel_2d.expand(1, 1, kernel_size, kernel_size)


class SSIM(FullRefMetric):
    def __init__(self) -> None:
        pass

    def compute(
        self,
        image_true: np.ndarray,
        image_test: np.ndarray,
        data_range: float = None,
        kernel_size: int = 11,
        k1: float = 0.01,
        k2: float = 0.03,
        sigma: float = 1.5,
        **kwargs: Any
    ) -> float:
        """
        Parameters:
        -----------
        image_true: np.array (H, W) or (H, W, D)
            Reference image
        image_test: np.array (H, W) or (H, W, D)
            Image to be evaluated against the reference image
        data_range:
            By default use joint maximum - joint minimum
        kernel_size:
            side length in pixels of square shaped kernel
        k1:
            constant parameter for SSIM calculation.
            Authors propose 0.01 for 8-bit integer images
        k2:
            constant parameter for SSIM calculation.
            Authors propose 0.03 for 8-bit integer images
        sigma:
            standard deviation of Gaussian distribution for weighting
            the kernel
        kwargs:
            place holder for keyword parameters

        """

        # If no data range is given, it is calculated from the range of the data, restricted to the passed or full mask
        if data_range is None:
            data_range = np.maximum(np.max(image_true), np.max(image_test)) - np.minimum(
                np.min(image_true), np.min(image_test)
            )

        # derive constants:
        c1 = pow(k1 * data_range, 2)
        c2 = pow(k2 * data_range, 2)

        # check if gpu is available:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        kernel = gaussian_kernel(2, kernel_size, sigma).to(device)

        # convert to Tensors
        pred = torch.Tensor(np.pad(image_test.copy(), pad_width=kernel_size // 2, mode="reflect"))
        target = torch.Tensor(np.pad(image_true.copy(), pad_width=kernel_size // 2, mode="reflect"))
        input_list = torch.stack(
            [
                pred,
                target,
                pred * pred,
                target * target,
                pred * target,
            ],
            dim=0,
        )  # (5 * B, H, W)

        # add channel dimension and move to gpu, if available
        input_list = input_list.unsqueeze(1).to(device)
        outputs = F.conv2d(input_list, kernel, padding="valid")

        # calculate ssim_map
        mu_pred_sq = outputs[0, 0, ...].pow(2)
        mu_target_sq = outputs[1, 0, ...].pow(2)
        mu_pred_target = outputs[0, 0, ...] * outputs[1, 0, ...]

        sigma_pred_sq = outputs[2, 0, ...] - mu_pred_sq
        sigma_target_sq = outputs[3, 0, ...] - mu_target_sq
        sigma_pred_target = outputs[4, 0, ...] - mu_pred_target

        dtype = torch.float64
        upper = 2 * sigma_pred_target.to(dtype) + c2
        lower = (sigma_pred_sq + sigma_target_sq).to(dtype) + c2

        ssim_map = ((2 * mu_pred_target + c1) * upper) / ((mu_pred_sq + mu_target_sq + c1) * lower)

        return np.mean(ssim_map.squeeze().cpu().numpy())


class MSSSIM(FullRefMetric):
    def __init__(self) -> None:
        pass

    def compute(
        self,
        image_true: np.ndarray,
        image_test: np.ndarray,
        data_range: float = None,
        kernel_size: int = 11,
        k1: float = 0.01,
        k2: float = 0.03,
        sigma: float = 1.5,
        betas: List[float] = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333],
        **kwargs: Any
    ) -> float:
        """
            Parameters:
            -----------
            image_true: np.array (H, W) or (H, W, D)
                Reference image
            image_test: np.array (H, W) or (H, W, D)
                Image to be evaluated against the reference image
            data_range:
                By default use joint maximum - joint minimum
            kernel_size:
                side length in pixels of square shaped kernel
            k1:
                constant parameter for SSIM calculation.
                Authors propose 0.01 for 8-bit integer images
            k2:
                constant parameter for SSIM calculation.
                Authors propose 0.03 for 8-bit integer images
            sigma:
                standard deviation of Gaussian distribution for weighting
                the kernel
            betas:

        sigma: float = 1.5,
            kwargs:
                place holder for keyword parameters

        """

        # If no data range is given, it is calculated from the range of the data, restricted to the passed or full mask
        data_range_was_None = False
        if data_range is None:
            data_range = np.maximum(np.max(image_true), np.max(image_test)) - np.minimum(
                np.min(image_true), np.min(image_test)
            )
            data_range_was_None = True

        # Check if image size is large enough for number of downsampling steps:
        while (
            True in [image_test.shape[d] < 2 ** len(betas) for d in range(2, len(image_test.shape))]
            or True
            in [
                image_test.shape[d] < (kernel_size - 1) * max(1, (len(betas) - 1)) ** 2
                for d in range(2, len(image_test.shape))
            ]
            and len(betas) > 2
        ):
            # Reducing betas from {betas} to {betas[0:-2]} to accommodate too small image size: {preds.shape}")
            betas = betas[0:-2]

        # check if gpu is available:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        kernel = gaussian_kernel(2, kernel_size, sigma).to(device)

        pad = kernel_size // 2

        # convert to Tensors, add channel dimension
        preds = torch.Tensor(image_test.copy()).unsqueeze(0)
        target = torch.Tensor(image_true.copy()).unsqueeze(0)

        mcs_list: List[np.ndarray] = []

        for _i in range(len(betas)):
            # set default data range if not set
            if data_range_was_None:
                data_range = torch.max(torch.cat((preds, target))) - torch.min(torch.cat((preds, target)))

            # derive constants:
            c1 = pow(k1 * data_range, 2)
            c2 = pow(k2 * data_range, 2)

            # padding:
            preds_padded = F.pad(preds, (pad, pad, pad, pad), mode="reflect")
            target_padded = F.pad(target, (pad, pad, pad, pad), mode="reflect")

            input_list = torch.stack(
                [
                    preds_padded,
                    target_padded,
                    preds_padded * preds_padded,
                    target_padded * target_padded,
                    preds_padded * target_padded,
                ],
                dim=0,
            )  # (5 * B, H, W)

            # add channel dimension and move to gpu, if available
            input_list = input_list.to(device)
            outputs = F.conv2d(input_list, kernel)

            # calculate ssim_map
            mu_pred_sq = outputs[0, 0, ...].pow(2)
            mu_target_sq = outputs[1, 0, ...].pow(2)
            mu_pred_target = outputs[0, 0, ...] * outputs[1, 0, ...]

            sigma_pred_sq = outputs[2, 0, ...] - mu_pred_sq
            sigma_target_sq = outputs[3, 0, ...] - mu_target_sq
            sigma_pred_target = outputs[4, 0, ...] - mu_pred_target

            dtype = torch.float64
            upper = 2 * sigma_pred_target.to(dtype) + c2
            lower = (sigma_pred_sq + sigma_target_sq).to(dtype) + c2

            ssim = ((2 * mu_pred_target + c1) * upper) / ((mu_pred_sq + mu_target_sq + c1) * lower)

            contrast_sensitivity = upper / lower

            ssim_cropped = ssim[..., pad:-pad, pad:-pad]
            cs_cropped = contrast_sensitivity[..., pad:-pad, pad:-pad]

            mcs_list.append(cs_cropped.mean())

            preds = F.avg_pool2d(preds, (2, 2))
            target = F.avg_pool2d(target, (2, 2))

        mcs_list[-1] = ssim_cropped.mean()

        # stack different scales into channel dimension:
        mcs_stack = torch.stack(mcs_list)

        betas_T = torch.Tensor(betas).to(device).view(-1, 1)
        mcs_weighted = mcs_stack.view(len(betas), -1) ** betas_T
        msssim = torch.prod(mcs_weighted, axis=0)

        # calculate the mean over all (unmasked) pixel locations
        return torch.mean(msssim).cpu().item()
