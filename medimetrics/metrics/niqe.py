import math
import os
from typing import Any, List, Optional, Tuple

import numpy as np
import pyrootutils
from PIL import Image
from scipy import special
from scipy.io import loadmat
from scipy.linalg import pinv
from scipy.ndimage import correlate1d

root = pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, dotenv=True)

from medimetrics.base import NonRefMetric


class NIQE(NonRefMetric):
    def __init__(self) -> None:
        self.gamma_range = np.arange(0.2, 10, 0.001)
        a = special.gamma(2.0 / self.gamma_range)
        a *= a
        b = special.gamma(1.0 / self.gamma_range)
        c = special.gamma(3.0 / self.gamma_range)
        self.prec_gammas = a / (b * c)

    def compute(self, image: np.ndarray, patch_size: int = 96, **kwargs: Any) -> float:
        """Taken from: https://github.com/guptapraful/niqe.

        Paper:
        A. Mittal, R. Soundararajan and A. C. Bovik,
        "Making a “Completely Blind” Image Quality Analyzer,"
        IEEE Signal Processing Letters, vol. 20, no. 3, pp. 209-212, March 2013,

        Parameters:
        -----------
        image: np.ndarray
            Input image

        patch_size:
            patch size used for feature extraction
        """

        # load trained parameters:
        params = loadmat(os.path.join(root, "medimetrics", "metrics", "niqe.mat"))
        pop_mu = np.ravel(params["pop_mu"])
        pop_cov = params["pop_cov"]

        feats = self.get_patches(image, patch_size)
        sample_mu = np.mean(feats, axis=0)
        sample_cov = np.cov(feats.T)

        X = sample_mu - pop_mu
        covmat = (pop_cov + sample_cov) / 2.0
        pinvmat = pinv(covmat)
        metric_value = np.sqrt(np.dot(np.dot(X, pinvmat), X))

        return metric_value

    def aggd_features(self, imdata: np.ndarray) -> Tuple[float, float, float, float, float, float]:
        # flatten imdata
        imdata.shape = (len(imdata.flat),)
        imdata2 = imdata * imdata
        left_data = imdata2[imdata < 0]
        right_data = imdata2[imdata >= 0]
        left_mean_sqrt = 0
        right_mean_sqrt = 0
        if len(left_data) > 0:
            left_mean_sqrt = np.sqrt(np.average(left_data))
        if len(right_data) > 0:
            right_mean_sqrt = np.sqrt(np.average(right_data))

        if right_mean_sqrt != 0:
            gamma_hat = left_mean_sqrt / right_mean_sqrt
        else:
            gamma_hat = np.inf
        # solve r-hat norm

        imdata2_mean = np.mean(imdata2)
        if imdata2_mean != 0:
            r_hat = (np.average(np.abs(imdata)) ** 2) / (np.average(imdata2))
        else:
            r_hat = np.inf
        rhat_norm = r_hat * (((math.pow(gamma_hat, 3) + 1) * (gamma_hat + 1)) / math.pow(math.pow(gamma_hat, 2) + 1, 2))

        # solve alpha by guessing values that minimize ro
        pos = np.argmin((self.prec_gammas - rhat_norm) ** 2)
        alpha = self.gamma_range[pos]

        gam1 = special.gamma(1.0 / alpha)
        gam2 = special.gamma(2.0 / alpha)
        gam3 = special.gamma(3.0 / alpha)

        aggdratio = np.sqrt(gam1) / np.sqrt(gam3)
        bl = aggdratio * left_mean_sqrt
        br = aggdratio * right_mean_sqrt

        # mean parameter
        N = (br - bl) * (gam2 / gam1)  # *aggdratio
        return (alpha, N, bl, br, left_mean_sqrt, right_mean_sqrt)

    def ggd_features(self, imdata: np.ndarray) -> Tuple[float, float]:
        nr_gam = 1 / self.prec_gammas
        sigma_sq = np.var(imdata)
        E = np.mean(np.abs(imdata))
        rho = sigma_sq / E**2
        pos = np.argmin(np.abs(nr_gam - rho))
        return self.gamma_range[pos], sigma_sq

    def paired_product(self, new_im: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        shift1 = np.roll(new_im.copy(), 1, axis=1)
        shift2 = np.roll(new_im.copy(), 1, axis=0)
        shift3 = np.roll(np.roll(new_im.copy(), 1, axis=0), 1, axis=1)
        shift4 = np.roll(np.roll(new_im.copy(), 1, axis=0), -1, axis=1)

        H_img = shift1 * new_im
        V_img = shift2 * new_im
        D1_img = shift3 * new_im
        D2_img = shift4 * new_im

        return (H_img, V_img, D1_img, D2_img)

    def gen_gauss_window(self, lw: float, sigma: float) -> List[float]:
        sd = np.float32(sigma)
        lw = int(lw)
        weights = [0.0] * (2 * lw + 1)
        weights[lw] = 1.0
        sum = 1.0
        sd *= sd
        for ii in range(1, lw + 1):
            tmp = np.exp(-0.5 * np.float32(ii * ii) / sd)
            weights[lw + ii] = tmp
            weights[lw - ii] = tmp
            sum += 2.0 * tmp
        for ii in range(2 * lw + 1):
            weights[ii] /= sum
        return weights

    def compute_image_mscn_transform(
        self, image: np.ndarray, C: int = 1, avg_window: Optional[List[float]] = None, extend_mode: str = "constant"
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if avg_window is None:
            avg_window = self.gen_gauss_window(3, 7.0 / 6.0)
        assert len(np.shape(image)) == 2
        h, w = np.shape(image)
        mu_image = np.zeros((h, w), dtype=np.float32)
        var_image = np.zeros((h, w), dtype=np.float32)
        image = np.array(image).astype("float32")
        correlate1d(image, avg_window, 0, mu_image, mode=extend_mode)
        correlate1d(mu_image, avg_window, 1, mu_image, mode=extend_mode)
        correlate1d(image**2, avg_window, 0, var_image, mode=extend_mode)
        correlate1d(var_image, avg_window, 1, var_image, mode=extend_mode)
        var_image = np.sqrt(np.abs(var_image - mu_image**2))
        return (image - mu_image) / (var_image + C), var_image, mu_image

    def niqe_extract_subband_feats(self, mscncoefs: np.ndarray) -> np.ndarray:
        alpha_m, N, bl, br, lsq, rsq = self.aggd_features(mscncoefs.copy())
        pps1, pps2, pps3, pps4 = self.paired_product(mscncoefs)
        alpha1, N1, bl1, br1, lsq1, rsq1 = self.aggd_features(pps1)
        alpha2, N2, bl2, br2, lsq2, rsq2 = self.aggd_features(pps2)
        alpha3, N3, bl3, br3, lsq3, rsq3 = self.aggd_features(pps3)
        alpha4, N4, bl4, br4, lsq4, rsq4 = self.aggd_features(pps4)
        return np.array(
            [
                alpha_m,
                (bl + br) / 2.0,
                alpha1,
                N1,
                bl1,
                br1,  # (V)
                alpha2,
                N2,
                bl2,
                br2,  # (H)
                alpha3,
                N3,
                bl3,
                bl3,  # (D1)
                alpha4,
                N4,
                bl4,
                bl4,  # (D2)
            ]
        )

    def extract_on_patches(self, img: np.ndarray, patch_size: float) -> np.ndarray:
        h, w = img.shape
        patch_size = int(patch_size)

        patches = []
        for j in range(0, h - patch_size + 1, patch_size):
            for i in range(0, w - patch_size + 1, patch_size):
                patch = img[j : j + patch_size, i : i + patch_size]
                patches.append(patch)

        patch_array = np.array(patches)

        patch_features = []
        for p in patch_array:
            patch_features.append(self.niqe_extract_subband_feats(p))
        patch_feature_array = np.array(patch_features)

        return patch_feature_array

    def get_patches(self, img: np.ndarray, patch_size: int) -> np.ndarray:
        h, w = np.shape(img)

        # ensure that the patch divides evenly into img
        hoffset = h % patch_size
        woffset = w % patch_size

        if hoffset > 0:
            img = img[:-hoffset, :]
        if woffset > 0:
            img = img[:, :-woffset]

        img = img.astype(np.float32)
        img_pil = Image.fromarray(img)
        w, h = img_pil.size
        img2 = np.array(img_pil.resize((int(w / 2), int(h / 2)), resample=Image.BICUBIC))

        mscn1, var, mu = self.compute_image_mscn_transform(img)
        mscn1 = mscn1.astype(np.float32)

        mscn2, _, _ = self.compute_image_mscn_transform(img2)
        mscn2 = mscn2.astype(np.float32)

        feats_lvl1 = self.extract_on_patches(mscn1, patch_size)
        feats_lvl2 = self.extract_on_patches(mscn2, patch_size / 2)

        feats = np.hstack((feats_lvl1, feats_lvl2))  # feats_lvl3))

        return feats
