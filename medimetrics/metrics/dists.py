import os
from typing import Any, List

import numpy as np
import pyrootutils
import torch
import torch.nn.functional as F
from torchvision import models

from medimetrics.base import FullRefMetric

root = pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, dotenv=True)

class L2pooling(torch.nn.Module):
    def __init__(self, filter_size: int = 5, stride: int = 2, channels: int = None, pad_off: int = 0):
        super().__init__()
        self.padding = (filter_size - 2) // 2
        self.stride = stride
        self.channels = channels
        a = np.hanning(filter_size)[1:-1]
        # a = torch.hann_window(5,periodic=False)
        g = torch.Tensor(a[:, None] * a[None, :])
        g = g / torch.sum(g)
        self.register_buffer("filter", g[None, None, :, :].repeat((self.channels, 1, 1, 1)))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = input**2
        out = F.conv2d(input, self.filter, stride=self.stride, padding=self.padding, groups=input.shape[1])
        return (out + 1e-12).sqrt()


class DISTNet(torch.nn.Module):
    """Refer to https://github.com/dingkeyan93/DISTS."""

    def __init__(self, channels: int = 3, load_weights: bool = True):
        assert channels == 3
        super().__init__()
        vgg_pretrained_features = models.vgg16(weights="IMAGENET1K_V1").features
        self.stage1 = torch.nn.Sequential()
        self.stage2 = torch.nn.Sequential()
        self.stage3 = torch.nn.Sequential()
        self.stage4 = torch.nn.Sequential()
        self.stage5 = torch.nn.Sequential()
        for x in range(0, 4):
            self.stage1.add_module(str(x), vgg_pretrained_features[x])
        self.stage2.add_module(str(4), L2pooling(channels=64))
        for x in range(5, 9):
            self.stage2.add_module(str(x), vgg_pretrained_features[x])
        self.stage3.add_module(str(9), L2pooling(channels=128))
        for x in range(10, 16):
            self.stage3.add_module(str(x), vgg_pretrained_features[x])
        self.stage4.add_module(str(16), L2pooling(channels=256))
        for x in range(17, 23):
            self.stage4.add_module(str(x), vgg_pretrained_features[x])
        self.stage5.add_module(str(23), L2pooling(channels=512))
        for x in range(24, 30):
            self.stage5.add_module(str(x), vgg_pretrained_features[x])

        for param in self.parameters():
            param.requires_grad = False

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1))

        self.chns = [3, 64, 128, 256, 512, 512]
        self.register_parameter("alpha", torch.nn.Parameter(torch.randn(1, sum(self.chns), 1, 1)))
        self.register_parameter("beta", torch.nn.Parameter(torch.randn(1, sum(self.chns), 1, 1)))
        self.alpha.data.normal_(0.1, 0.01)
        self.beta.data.normal_(0.1, 0.01)

        weights = torch.load(os.path.join(root, "medimetrics/metrics/DISTS.pt"))
        self.alpha.data = weights["alpha"]
        self.beta.data = weights["beta"]

    def forward_once(self, x: np.ndarray) -> List[torch.Tensor]:
        h = (x - self.mean) / self.std
        h = self.stage1(h)
        h_relu1_2 = h
        h = self.stage2(h)
        h_relu2_2 = h
        h = self.stage3(h)
        h_relu3_3 = h
        h = self.stage4(h)
        h_relu4_3 = h
        h = self.stage5(h)
        h_relu5_3 = h
        return [x, h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3]

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert x.shape == y.shape

        with torch.no_grad():
            feats0 = self.forward_once(x)
            feats1 = self.forward_once(y)

            dist1 = torch.Tensor([[[[0]]]]).to(x.device)
            dist2 = torch.Tensor([[[[0]]]]).to(x.device)
            c1 = 1e-6
            c2 = 1e-6
            w_sum = self.alpha.sum() + self.beta.sum()
            alpha = torch.split(self.alpha / w_sum, self.chns, dim=1)
            beta = torch.split(self.beta / w_sum, self.chns, dim=1)

            for k in range(len(self.chns)):
                x_mean = feats0[k].mean([2, 3], keepdim=True)
                y_mean = feats1[k].mean([2, 3], keepdim=True)
                S1 = (2 * x_mean * y_mean + c1) / (x_mean**2 + y_mean**2 + c1)
                dist1 = dist1 + (alpha[k] * S1).sum(1, keepdim=True)

                x_var = ((feats0[k] - x_mean) ** 2).mean([2, 3], keepdim=True)
                y_var = ((feats1[k] - y_mean) ** 2).mean([2, 3], keepdim=True)
                xy_cov = (feats0[k] * feats1[k]).mean([2, 3], keepdim=True) - x_mean * y_mean
                S2 = (2 * xy_cov + c2) / (x_var + y_var + c2)
                dist2 = dist2 + (beta[k] * S2).sum(1, keepdim=True)

            score = 1 - (dist1 + dist2).squeeze()
            return score


class DISTS(FullRefMetric):
    def __init__(self) -> None:
        self.dists_network = DISTNet()

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
        image_true_T = torch.Tensor(image_true.copy()).unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)
        image_test_T = torch.Tensor(image_test.copy()).unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)

        self.dists_network = self.dists_network.to(device)

        score = self.dists_network(image_true_T, image_test_T).cpu().numpy()

        return score.mean()
