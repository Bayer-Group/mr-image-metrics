import math
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from medimetrics.base import FullRefMetric


class CWSSIM(FullRefMetric):
    def __init__(self) -> None:
        self.pyramid_initialized = False

    def init_pyramid(
        self, size_length: int = 256, K: int = 8, N: int = 4, device: torch.device = torch.device("cpu")
    ) -> None:
        if not self.pyramid_initialized:
            self.pyramid_initialized = True
            self.K = K
            self.N = N
            self.hilb = True
            self.includeHF = True

            size = (size_length, size_length // 2 + 1)

            self.hl0 = self.HL0_matrix(size).unsqueeze(0).unsqueeze(0).unsqueeze(0).to(device)

            self.le = []
            self.b = []
            self.s = []

            self.indF = [self.freq_shift(size[0], True, device)]
            self.indB = [self.freq_shift(size[0], False, device)]

            for n in range(N):
                le_m = self.L_matrix_cropped(size).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0).to(device)
                b_m = self.B_matrix(K, size).unsqueeze(0).unsqueeze(0).unsqueeze(0).to(device)
                s_m = self.S_matrix(K, size).unsqueeze(0).unsqueeze(0).unsqueeze(0).to(device)

                self.le.append(le_m.div_(4))
                self.b.append(b_m)
                self.s.append(s_m)

                size = (le_m.size(-2), le_m.size(-1))

                self.indF.append(self.freq_shift(size[0], True, device))
                self.indB.append(self.freq_shift(size[0], False, device))

    def compute(self, image_true: np.ndarray, image_test: np.ndarray, **kwargs: Any) -> float:
        """Computes the complex-weighted ssim metric value and the metric
        image.

        Parameters:
        -----------
        image_true: np.array (H, W)
            Reference image
        image_test: np.array (H, W)
            Image to be evaluated against the reference image
        data_range:
            By default use joint maximum - joint minimum

        Taken from https://github.com/dingkeyan93/IQA-optimization/blob/master/IQA_pytorch/CW_SSIM.py

        This is a pytorch implementation of Complex-Wavelet
        Structural SIMilarity (CW-SSIM) index.

        M. P. Sampat, Z. Wang, S. Gupta, A. C. Bovik, M. K. Markey.
        "Complex Wavelet Structural Similarity: A New Image Similarity Index",
        IEEE Transactions on Image Processing, 18(11), 2385-401, 2009.

        Matlab version:
        https://www.mathworks.com/matlabcentral/fileexchange/43017-complex-wavelet-structural-similarity-index-cw-ssim
        """

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        ori = 8
        level = 4
        channels = 1
        eps = 1e-12

        # pad to next potence of 2, minimum 64
        max_target_dim = 2 ** np.maximum(np.ceil(np.log2(np.array(image_true.shape).max())), 6).astype(np.uint16)
        target_shape = np.ones_like(np.array(image_true.shape)) * max_target_dim

        if np.any(target_shape != np.array(image_true.shape)):
            pad_before = (target_shape - np.array(image_true.shape)) // 2
            pad_after = target_shape - np.array(image_true.shape) - pad_before
            image_true = np.pad(image_true, list(zip(pad_before, pad_after)), mode="constant")
            image_test = np.pad(image_test, list(zip(pad_before, pad_after)), mode="constant")

        size_length = image_true.shape[0]
        self.init_pyramid(size_length, K=ori, N=level, device=device)
        win7 = (torch.ones(channels, 1, 7, 7) / (7 * 7)).to(device)
        s = size_length / 2 ** (level - 1)

        w = self.fspecial_gauss(s - 7 + 1, s / 4, 1).to(device)

        image_true_T = torch.Tensor(image_true).unsqueeze(0).repeat(1, 1, 1, 1).to(device)
        image_test_T = torch.Tensor(image_test).unsqueeze(0).repeat(1, 1, 1, 1).to(device)

        image_true_T = image_true_T * 255
        image_test_T = image_test_T * 255

        cw_x = self.pyramid(image_true_T)
        cw_y = self.pyramid(image_test_T)

        bandind = level
        band_cssim = []

        for i in range(ori):
            band1 = cw_x[bandind][:, :, :, i, :, :]
            band2 = cw_y[bandind][:, :, :, i, :, :]
            corr = self.conj(band1, band2)
            corr_band = self.conv2d_complex(corr, win7, groups=channels)
            varr = (self.abs(band1, eps)) ** 2 + (self.abs(band2, eps)) ** 2
            varr_band = F.conv2d(varr, win7, stride=1, padding=0, groups=channels)
            cssim_map = (2 * self.abs(corr_band, eps**2) + eps) / (varr_band + eps)
            band_cssim.append((cssim_map * w.repeat(cssim_map.shape[0], 1, 1, 1)).sum([2, 3]).mean(1))

        cwssim = torch.stack(band_cssim, dim=1).mean(1).item()

        return cwssim

    def abs(self, x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        return torch.sqrt(x[:, 0, ...] ** 2 + x[:, 1, ...] ** 2 + eps)

    def conj(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        a = x[:, 0, ...]
        b = x[:, 1, ...]
        c = y[:, 0, ...]
        d = -y[:, 1, ...]
        return torch.stack((a * c - b * d, b * c + a * d), dim=1)

    def conv2d_complex(self, x: torch.Tensor, win: torch.Tensor, groups: int = 1) -> torch.Tensor:
        real = F.conv2d(x[:, 0, ...], win, groups=groups)  # - F.conv2d(x[:,1], win, groups = groups)
        imaginary = F.conv2d(x[:, 1, ...], win, groups=groups)  # + F.conv2d(x[:,0], win, groups = groups)
        return torch.stack((real, imaginary), dim=1)

    def L(self, r: float) -> float:
        if r <= math.pi / 4:
            return 2
        elif r >= math.pi / 2:
            return 0
        else:
            return 2 * math.cos(math.pi / 2 * math.log(4 * r / math.pi) / math.log(2))

    def H(self, r: float) -> float:
        if r <= math.pi / 4:
            return 0
        elif r >= math.pi / 2:
            return 1
        else:
            return math.cos(math.pi / 2 * math.log(2 * r / math.pi) / math.log(2))

    def G(self, t: float, k: float, K: int) -> float:
        t0 = math.pi * k / K
        aK = 2 ** (K - 1) * math.factorial(K - 1) / math.sqrt(K * math.factorial(2 * (K - 1)))

        if (t - t0) > (math.pi / 2):
            return self.G(t - math.pi, k, K)
        elif (t - t0) < (-math.pi / 2):
            return self.G(t + math.pi, k, K)
        else:
            return aK * (math.cos(t - t0)) ** (K - 1)

    def S(self, t: float, k: float, K: float) -> float:
        t0 = math.pi * k / K
        dt = np.abs(t - t0)

        if dt < math.pi / 2:
            return 1
        elif dt == math.pi / 2:
            return 0
        else:
            return -1

    def L0(self, r: float) -> float:
        return self.L(r / 2) / 2

    def H0(self, r: float) -> float:
        return self.H(r / 2)

    def polar_map(self, s: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.linspace(0, math.pi, s[1]).view(1, s[1]).expand(s)
        if s[0] % 2 == 0:
            y = torch.linspace(-math.pi, math.pi, s[0] + 1).narrow(0, 1, s[0])
        else:
            y = torch.linspace(-math.pi, math.pi, s[0])
        y = y.view(s[0], 1).expand(s).mul(-1)

        r = (x**2 + y**2).sqrt()
        t = torch.atan2(y, x)
        return r, t

    def S_matrix(self, K: int, s: Tuple[int, int]) -> torch.Tensor:
        _, t = self.polar_map(s)
        sm = torch.Tensor(K, s[0], s[1])
        for k in range(K):
            for i in range(s[0]):
                for j in range(s[1]):
                    sm[k][i][j] = self.S(t[i][j], k, K)
        return sm

    def G_matrix(self, K: int, s: Tuple[int, int]) -> torch.Tensor:
        _, t = self.polar_map(s)
        g = torch.Tensor(K, s[0], s[1])
        for k in range(K):
            for i in range(s[0]):
                for j in range(s[1]):
                    g[k][i][j] = self.G(t[i][j], k, K)
        return g

    def B_matrix(self, K: int, s: Tuple[int, int]) -> torch.Tensor:
        g = self.G_matrix(K, s)
        r, _ = self.polar_map(s)
        h = r.apply_(self.H).unsqueeze(0)
        return h * g

    def L_matrix(self, s: Tuple[int, int]) -> torch.Tensor:
        r, _ = self.polar_map(s)
        return r.apply_(self.L)

    def LB_matrix(self, K: int, s: Tuple[int, int]) -> torch.Tensor:
        le = self.L_matrix(s).unsqueeze(0)
        b = self.B_matrix(K, s)
        return torch.cat((le, b), 0)

    def HL0_matrix(self, s: Tuple[int, int]) -> torch.Tensor:
        r, _ = self.polar_map(s)
        h = r.clone().apply_(self.H0).view(1, s[0], s[1])
        le = r.clone().apply_(self.L0).view(1, s[0], s[1])
        return torch.cat((h, le), 0)

    def central_crop(self, x: torch.Tensor) -> torch.Tensor:
        ns = [x.size(-2) // 2, x.size(-1) // 2 + 1]
        return x.narrow(-2, ns[1] - 1, ns[0]).narrow(-1, 0, ns[1])

    def cropped_size(self, s: Tuple[int, int]) -> List:
        return [s[0] // 2, s[1] // 2 + 1]

    def L_matrix_cropped(self, s: Tuple[int, int]) -> torch.Tensor:
        le = self.L_matrix(s)
        ns = self.cropped_size(s)
        return le.narrow(0, ns[1] - 1, ns[0]).narrow(1, 0, ns[1])

    def freq_shift(self, imgSize: int, fwd: bool, device: torch.device) -> torch.Tensor:
        ind = torch.LongTensor(imgSize).to(device)
        sgn = 1
        if fwd:
            sgn = -1
        for i in range(imgSize):
            ind[i] = (i + sgn * ((imgSize - 1) // 2)) % imgSize

        return torch.Tensor(np.array(ind)).to(torch.long)

    ##########
    def sp5_filters(self) -> Dict[str, np.ndarray]:
        filters = {}
        filters["harmonics"] = np.array([1, 3, 5])
        filters["mtx"] = np.array(
            [
                [0.3333, 0.2887, 0.1667, 0.0000, -0.1667, -0.2887],
                [0.0000, 0.1667, 0.2887, 0.3333, 0.2887, 0.1667],
                [0.3333, -0.0000, -0.3333, -0.0000, 0.3333, -0.0000],
                [0.0000, 0.3333, 0.0000, -0.3333, 0.0000, 0.3333],
                [0.3333, -0.2887, 0.1667, -0.0000, -0.1667, 0.2887],
                [-0.0000, 0.1667, -0.2887, 0.3333, -0.2887, 0.1667],
            ]
        )
        filters["hi0filt"] = np.array(
            [
                [
                    -0.00033429,
                    -0.00113093,
                    -0.00171484,
                    -0.00133542,
                    -0.00080639,
                    -0.00133542,
                    -0.00171484,
                    -0.00113093,
                    -0.00033429,
                ],
                [
                    -0.00113093,
                    -0.00350017,
                    -0.00243812,
                    0.00631653,
                    0.01261227,
                    0.00631653,
                    -0.00243812,
                    -0.00350017,
                    -0.00113093,
                ],
                [
                    -0.00171484,
                    -0.00243812,
                    -0.00290081,
                    -0.00673482,
                    -0.00981051,
                    -0.00673482,
                    -0.00290081,
                    -0.00243812,
                    -0.00171484,
                ],
                [
                    -0.00133542,
                    0.00631653,
                    -0.00673482,
                    -0.07027679,
                    -0.11435863,
                    -0.07027679,
                    -0.00673482,
                    0.00631653,
                    -0.00133542,
                ],
                [
                    -0.00080639,
                    0.01261227,
                    -0.00981051,
                    -0.11435863,
                    0.81380200,
                    -0.11435863,
                    -0.00981051,
                    0.01261227,
                    -0.00080639,
                ],
                [
                    -0.00133542,
                    0.00631653,
                    -0.00673482,
                    -0.07027679,
                    -0.11435863,
                    -0.07027679,
                    -0.00673482,
                    0.00631653,
                    -0.00133542,
                ],
                [
                    -0.00171484,
                    -0.00243812,
                    -0.00290081,
                    -0.00673482,
                    -0.00981051,
                    -0.00673482,
                    -0.00290081,
                    -0.00243812,
                    -0.00171484,
                ],
                [
                    -0.00113093,
                    -0.00350017,
                    -0.00243812,
                    0.00631653,
                    0.01261227,
                    0.00631653,
                    -0.00243812,
                    -0.00350017,
                    -0.00113093,
                ],
                [
                    -0.00033429,
                    -0.00113093,
                    -0.00171484,
                    -0.00133542,
                    -0.00080639,
                    -0.00133542,
                    -0.00171484,
                    -0.00113093,
                    -0.00033429,
                ],
            ]
        )
        filters["lo0filt"] = np.array(
            [
                [0.00341614, -0.01551246, -0.03848215, -0.01551246, 0.00341614],
                [-0.01551246, 0.05586982, 0.15925570, 0.05586982, -0.01551246],
                [-0.03848215, 0.15925570, 0.40304148, 0.15925570, -0.03848215],
                [-0.01551246, 0.05586982, 0.15925570, 0.05586982, -0.01551246],
                [0.00341614, -0.01551246, -0.03848215, -0.01551246, 0.00341614],
            ]
        )
        filters["lofilt"] = 2 * np.array(
            [
                [
                    0.00085404,
                    -0.00244917,
                    -0.00387812,
                    -0.00944432,
                    -0.00962054,
                    -0.00944432,
                    -0.00387812,
                    -0.00244917,
                    0.00085404,
                ],
                [
                    -0.00244917,
                    -0.00523281,
                    -0.00661117,
                    0.00410600,
                    0.01002988,
                    0.00410600,
                    -0.00661117,
                    -0.00523281,
                    -0.00244917,
                ],
                [
                    -0.00387812,
                    -0.00661117,
                    0.01396746,
                    0.03277038,
                    0.03981393,
                    0.03277038,
                    0.01396746,
                    -0.00661117,
                    -0.00387812,
                ],
                [
                    -0.00944432,
                    0.00410600,
                    0.03277038,
                    0.06426333,
                    0.08169618,
                    0.06426333,
                    0.03277038,
                    0.00410600,
                    -0.00944432,
                ],
                [
                    -0.00962054,
                    0.01002988,
                    0.03981393,
                    0.08169618,
                    0.10096540,
                    0.08169618,
                    0.03981393,
                    0.01002988,
                    -0.00962054,
                ],
                [
                    -0.00944432,
                    0.00410600,
                    0.03277038,
                    0.06426333,
                    0.08169618,
                    0.06426333,
                    0.03277038,
                    0.00410600,
                    -0.00944432,
                ],
                [
                    -0.00387812,
                    -0.00661117,
                    0.01396746,
                    0.03277038,
                    0.03981393,
                    0.03277038,
                    0.01396746,
                    -0.00661117,
                    -0.00387812,
                ],
                [
                    -0.00244917,
                    -0.00523281,
                    -0.00661117,
                    0.00410600,
                    0.01002988,
                    0.00410600,
                    -0.00661117,
                    -0.00523281,
                    -0.00244917,
                ],
                [
                    0.00085404,
                    -0.00244917,
                    -0.00387812,
                    -0.00944432,
                    -0.00962054,
                    -0.00944432,
                    -0.00387812,
                    -0.00244917,
                    0.00085404,
                ],
            ]
        )
        filters["bfilts"] = np.array(
            [
                [
                    0.00277643,
                    0.00496194,
                    0.01026699,
                    0.01455399,
                    0.01026699,
                    0.00496194,
                    0.00277643,
                    -0.00986904,
                    -0.00893064,
                    0.01189859,
                    0.02755155,
                    0.01189859,
                    -0.00893064,
                    -0.00986904,
                    -0.01021852,
                    -0.03075356,
                    -0.08226445,
                    -0.11732297,
                    -0.08226445,
                    -0.03075356,
                    -0.01021852,
                    0.00000000,
                    0.00000000,
                    0.00000000,
                    0.00000000,
                    0.00000000,
                    0.00000000,
                    0.00000000,
                    0.01021852,
                    0.03075356,
                    0.08226445,
                    0.11732297,
                    0.08226445,
                    0.03075356,
                    0.01021852,
                    0.00986904,
                    0.00893064,
                    -0.01189859,
                    -0.02755155,
                    -0.01189859,
                    0.00893064,
                    0.00986904,
                    -0.00277643,
                    -0.00496194,
                    -0.01026699,
                    -0.01455399,
                    -0.01026699,
                    -0.00496194,
                    -0.00277643,
                ],
                [
                    -0.00343249,
                    -0.00640815,
                    -0.00073141,
                    0.01124321,
                    0.00182078,
                    0.00285723,
                    0.01166982,
                    -0.00358461,
                    -0.01977507,
                    -0.04084211,
                    -0.00228219,
                    0.03930573,
                    0.01161195,
                    0.00128000,
                    0.01047717,
                    0.01486305,
                    -0.04819057,
                    -0.12227230,
                    -0.05394139,
                    0.00853965,
                    -0.00459034,
                    0.00790407,
                    0.04435647,
                    0.09454202,
                    -0.00000000,
                    -0.09454202,
                    -0.04435647,
                    -0.00790407,
                    0.00459034,
                    -0.00853965,
                    0.05394139,
                    0.12227230,
                    0.04819057,
                    -0.01486305,
                    -0.01047717,
                    -0.00128000,
                    -0.01161195,
                    -0.03930573,
                    0.00228219,
                    0.04084211,
                    0.01977507,
                    0.00358461,
                    -0.01166982,
                    -0.00285723,
                    -0.00182078,
                    -0.01124321,
                    0.00073141,
                    0.00640815,
                    0.00343249,
                ],
                [
                    0.00343249,
                    0.00358461,
                    -0.01047717,
                    -0.00790407,
                    -0.00459034,
                    0.00128000,
                    0.01166982,
                    0.00640815,
                    0.01977507,
                    -0.01486305,
                    -0.04435647,
                    0.00853965,
                    0.01161195,
                    0.00285723,
                    0.00073141,
                    0.04084211,
                    0.04819057,
                    -0.09454202,
                    -0.05394139,
                    0.03930573,
                    0.00182078,
                    -0.01124321,
                    0.00228219,
                    0.12227230,
                    -0.00000000,
                    -0.12227230,
                    -0.00228219,
                    0.01124321,
                    -0.00182078,
                    -0.03930573,
                    0.05394139,
                    0.09454202,
                    -0.04819057,
                    -0.04084211,
                    -0.00073141,
                    -0.00285723,
                    -0.01161195,
                    -0.00853965,
                    0.04435647,
                    0.01486305,
                    -0.01977507,
                    -0.00640815,
                    -0.01166982,
                    -0.00128000,
                    0.00459034,
                    0.00790407,
                    0.01047717,
                    -0.00358461,
                    -0.00343249,
                ],
                [
                    -0.00277643,
                    0.00986904,
                    0.01021852,
                    -0.00000000,
                    -0.01021852,
                    -0.00986904,
                    0.00277643,
                    -0.00496194,
                    0.00893064,
                    0.03075356,
                    -0.00000000,
                    -0.03075356,
                    -0.00893064,
                    0.00496194,
                    -0.01026699,
                    -0.01189859,
                    0.08226445,
                    -0.00000000,
                    -0.08226445,
                    0.01189859,
                    0.01026699,
                    -0.01455399,
                    -0.02755155,
                    0.11732297,
                    -0.00000000,
                    -0.11732297,
                    0.02755155,
                    0.01455399,
                    -0.01026699,
                    -0.01189859,
                    0.08226445,
                    -0.00000000,
                    -0.08226445,
                    0.01189859,
                    0.01026699,
                    -0.00496194,
                    0.00893064,
                    0.03075356,
                    -0.00000000,
                    -0.03075356,
                    -0.00893064,
                    0.00496194,
                    -0.00277643,
                    0.00986904,
                    0.01021852,
                    -0.00000000,
                    -0.01021852,
                    -0.00986904,
                    0.00277643,
                ],
                [
                    -0.01166982,
                    -0.00128000,
                    0.00459034,
                    0.00790407,
                    0.01047717,
                    -0.00358461,
                    -0.00343249,
                    -0.00285723,
                    -0.01161195,
                    -0.00853965,
                    0.04435647,
                    0.01486305,
                    -0.01977507,
                    -0.00640815,
                    -0.00182078,
                    -0.03930573,
                    0.05394139,
                    0.09454202,
                    -0.04819057,
                    -0.04084211,
                    -0.00073141,
                    -0.01124321,
                    0.00228219,
                    0.12227230,
                    -0.00000000,
                    -0.12227230,
                    -0.00228219,
                    0.01124321,
                    0.00073141,
                    0.04084211,
                    0.04819057,
                    -0.09454202,
                    -0.05394139,
                    0.03930573,
                    0.00182078,
                    0.00640815,
                    0.01977507,
                    -0.01486305,
                    -0.04435647,
                    0.00853965,
                    0.01161195,
                    0.00285723,
                    0.00343249,
                    0.00358461,
                    -0.01047717,
                    -0.00790407,
                    -0.00459034,
                    0.00128000,
                    0.01166982,
                ],
                [
                    -0.01166982,
                    -0.00285723,
                    -0.00182078,
                    -0.01124321,
                    0.00073141,
                    0.00640815,
                    0.00343249,
                    -0.00128000,
                    -0.01161195,
                    -0.03930573,
                    0.00228219,
                    0.04084211,
                    0.01977507,
                    0.00358461,
                    0.00459034,
                    -0.00853965,
                    0.05394139,
                    0.12227230,
                    0.04819057,
                    -0.01486305,
                    -0.01047717,
                    0.00790407,
                    0.04435647,
                    0.09454202,
                    -0.00000000,
                    -0.09454202,
                    -0.04435647,
                    -0.00790407,
                    0.01047717,
                    0.01486305,
                    -0.04819057,
                    -0.12227230,
                    -0.05394139,
                    0.00853965,
                    -0.00459034,
                    -0.00358461,
                    -0.01977507,
                    -0.04084211,
                    -0.00228219,
                    0.03930573,
                    0.01161195,
                    0.00128000,
                    -0.00343249,
                    -0.00640815,
                    -0.00073141,
                    0.01124321,
                    0.00182078,
                    0.00285723,
                    0.01166982,
                ],
            ]
        ).T
        return filters

    def fspecial_gauss(self, size: int, sigma: float, channels: int) -> torch.Tensor:
        # Function to mimic the 'fspecial' gaussian MATLAB function
        x, y = np.mgrid[-size // 2 + 1 : size // 2 + 1, -size // 2 + 1 : size // 2 + 1]
        g = np.exp(-((x**2 + y**2) / (2.0 * sigma**2)))
        g = torch.from_numpy(g / g.sum()).float().unsqueeze(0).unsqueeze(0)
        return g.repeat(channels, 1, 1, 1)

    # pass through steerable pyramid
    def pyramid(self, x: torch.Tensor) -> torch.Tensor:
        fftfull = torch.view_as_real(torch.fft.rfft2(x))
        xreal = fftfull[..., 0]
        xim = fftfull[..., 1]

        x = torch.cat((xreal.unsqueeze(1), xim.unsqueeze(1)), 1).unsqueeze(-3)
        x = torch.index_select(x, -2, self.indF[0])

        x = self.hl0 * x
        h0f = x.select(-3, 0).unsqueeze(-3)
        l0f = x.select(-3, 1).unsqueeze(-3)
        lf = l0f

        output = []

        for n in range(self.N):
            bf = self.b[n] * lf
            lf = self.le[n] * self.central_crop(lf)
            if self.hilb:
                hbf = self.s[n] * torch.cat((bf.narrow(1, 1, 1), -bf.narrow(1, 0, 1)), 1)
                bf = torch.cat((bf, hbf), -3)
            if self.includeHF and n == 0:
                bf = torch.cat((h0f, bf), -3)

            output.append(bf)

        output.append(lf)

        for n in range(len(output)):
            output[n] = torch.index_select(output[n], -2, self.indB[n])
            sig_size = [output[n].shape[-2], (output[n].shape[-1] - 1) * 2]

            output[n] = torch.stack((output[n].select(1, 0), output[n].select(1, 1)), -1)
            output[n] = torch.fft.irfft2(torch.view_as_complex(output[n]), s=sig_size)

        if self.includeHF:
            output.insert(0, output[0].narrow(-3, 0, 1))
            output[1] = output[1].narrow(-3, 1, output[1].size(-3) - 1)

        for n in range(len(output)):
            if self.hilb:
                if ((not self.includeHF) or 0 < n) and n < len(output) - 1:
                    nfeat = output[n].size(-3) // 2
                    o1 = output[n].narrow(-3, 0, nfeat).unsqueeze(1)
                    o2 = -output[n].narrow(-3, nfeat, nfeat).unsqueeze(1)
                    output[n] = torch.cat((o2, o1), 1)
                else:
                    output[n] = output[n].unsqueeze(1)

        for n in range(1, len(output)):
            output[n] = output[n] * (2 ** (n - 1))
        return output
