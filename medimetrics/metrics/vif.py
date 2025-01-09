from typing import Any, List, Tuple

import numpy as np
import torch
from numpy.linalg import inv

# import pyPyrTools as ppt
# from pyPyrTools.corrDn import corrDn
from pyrtools.pyramids import SteerablePyramidSpace
from pyrtools.pyramids.c.wrapper import corrDn

from medimetrics.base import FullRefMetric


def vifvec(imref_batch: np.ndarray, imdist_batch: np.ndarray) -> np.ndarray:
    M = 3
    subbands = [4, 7, 10, 13, 16, 19, 22, 25]
    sigma_nsq = 0.4

    batch_num = 1
    if imref_batch.ndim >= 3:
        batch_num = imref_batch.shape[0]

    vif = np.zeros(
        [
            batch_num,
        ]
    )

    for a in range(batch_num):
        if batch_num > 1:
            imref = imref_batch[a, :, :]
            imdist = imdist_batch[a, :, :]
        else:
            imref = imref_batch
            imdist = imdist_batch

        # Wavelet Decomposition
        # pyr = ppt.Spyr(imref, 4, 'sp5Filters', 'reflect1')
        org = SteerablePyramidSpace(imref, height=4, order=5, edge_type="reflect1")
        # org = pyr.pyr[::-1]     #reverse list

        # pyr = ppt.Spyr(imdist, 4, 'sp5Filters', 'reflect1')
        dist = SteerablePyramidSpace(imdist, height=4, order=5, edge_type="reflect1")
        # dist = pyr.pyr[::-1]

        # Calculate parameters of the distortion channel

        tol = 1e-15  # tolerance for zero variance
        g_all = []
        vv_all = []

        for i in range(len(subbands)):
            sub = subbands[i]

            # size of window used in distortion channel estimation
            lev = int(np.ceil((sub - 1) / 6))
            winsize = int(2**lev + 1)
            win = np.ones([winsize, winsize])

            # y = org.[sub-1]
            y = org.pyr_coeffs[(lev, sub)]
            # yn = dist[sub-1]
            yn = dist.pyr_coeffs[(lev, sub)]

            # force subband to be a multiple of M
            newsize = [int(np.floor(y.shape[0] / M) * M), int(np.floor(y.shape[1] / M) * M)]
            y = y[: newsize[0], : newsize[1]]
            yn = yn[: newsize[0], : newsize[1]]

            # correlation with downsampling
            winstep = (M, M)
            winstart = (int(np.floor(M / 2)), int(np.floor(M / 2)))
            winstop = (y.shape[0] - int(np.ceil(M / 2)) + 1, y.shape[1] - int(np.ceil(M / 2)) + 1)

            # mean
            mean_x = corrDn(y, win / np.sum(win), "reflect1", winstep, winstart, winstop)
            mean_y = corrDn(yn, win / np.sum(win), "reflect1", winstep, winstart, winstop)

            # covariance
            cov_xy = corrDn(np.multiply(y, yn), win, "reflect1", winstep, winstart, winstop) - np.sum(
                win
            ) * np.multiply(mean_x, mean_y)

            # variance
            ss_x = corrDn(np.multiply(y, y), win, "reflect1", winstep, winstart, winstop) - np.sum(win) * np.multiply(
                mean_x, mean_x
            )
            ss_y = corrDn(np.multiply(yn, yn), win, "reflect1", winstep, winstart, winstop) - np.sum(win) * np.multiply(
                mean_y, mean_y
            )

            ss_x[np.where(ss_x < 0)] = 0
            ss_y[np.where(ss_y < 0)] = 0

            # Regression
            g = np.divide(cov_xy, (ss_x + tol))

            vv = (ss_y - np.multiply(g, cov_xy)) / (np.sum(win))

            g[np.where(ss_x < tol)] = 0
            vv[np.where(ss_x < tol)] = ss_y[np.where(ss_x < tol)]
            ss_x[np.where(ss_x < tol)] = 0

            g[np.where(ss_y < tol)] = 0
            vv[np.where(ss_y < tol)] = 0

            g[np.where(g < 0)] = 0
            vv[np.where(g < 0)] = ss_y[np.where(g < 0)]

            vv[np.where(vv <= tol)] = tol

            g_all.append(g)
            vv_all.append(vv)

        # calculate the parameters of reference
        ssarr, larr, cuarr = refparams_vecgsm(org, subbands, M)

        num = np.zeros([1, len(subbands)])
        den = np.zeros([1, len(subbands)])

        for i in range(len(subbands)):
            sub = subbands[i]
            g = g_all[i]
            vv = vv_all[i]
            ss = ssarr[i]
            lam = larr[i]
            # cu = cuarr[i]

            # neigvals = len(lam)
            # lev = math.ceil((sub - 1)/6)
            lev = np.ceil((sub - 1) / 6)
            winsize = 2**lev + 1
            offset = (winsize - 1) / 2
            # offset = math.ceil(offset/M)
            offset = np.ceil(offset / M)

            g = g[offset : g.shape[0] - offset, offset : g.shape[1] - offset]
            vv = vv[offset : vv.shape[0] - offset, offset : vv.shape[1] - offset]
            ss = ss[offset : ss.shape[0] - offset, offset : ss.shape[1] - offset]

            temp1, temp2 = 0, 0
            rt = []
            for j in range(len(lam)):
                temp1 += np.sum(
                    np.log2(1 + np.divide(np.multiply(np.multiply(g, g), ss) * lam[j], vv + sigma_nsq))
                )  # distorted image information
                temp2 += np.sum(np.log2(1 + np.divide(ss * lam[j], sigma_nsq)))  # reference image information
                rt.append(np.sum(np.log(1 + np.divide(ss * lam[j], sigma_nsq))))

            num[0, i] = temp1
            den[0, i] = temp2

        vif[a] = np.sum(num) / np.sum(den)
    print(vif)
    return vif


def refparams_vecgsm(
    org: List[torch.Tensor], subbands: List[torch.Tensor], M: int
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    # This function calculates the parameters of the reference image
    # l_arr = np.zeros([subbands[-1],M**2])
    l_arr, ssarr, cu_arr = [], [], []
    for i in range(len(subbands)):
        sub = subbands[i]
        y = org[sub - 1]

        # sizey = (math.floor(y.shape[0]/M)*M, math.floor(y.shape[1]/M)*M)
        sizey = (np.floor(y.shape[0] / M) * M, np.floor(y.shape[1] / M) * M)
        y = y[: sizey[0], : sizey[1]]

        # Collect MxM blocks, rearrange into M^2 dimensional vector
        temp_list = []
        for j in range(M):
            for k in range(M):
                temp_list.append(y[k : y.shape[0] - M + k + 1, j : y.shape[1] - M + j + 1].T.reshape(-1))

        temp = np.asarray(temp_list)
        mcu = np.mean(temp, axis=1).reshape(temp.shape[0], 1)
        mean_sub = temp - np.repeat(mcu, temp.shape[1], axis=1)
        cu = mean_sub @ mean_sub.T / temp.shape[1]
        # Calculate S field, non-overlapping blocks
        temp_list = []
        for j in range(M):
            for k in range(M):
                temp_list.append(y[k::M, j::M].T.reshape(-1))

        temp = np.asarray(temp)
        ss = inv(cu) @ temp
        ss = np.sum(np.multiply(ss, temp), axis=0) / (M**2)
        ss = ss.reshape(int(sizey[1] / M), int(sizey[0] / M)).T

        d, _ = np.linalg.eig(cu)
        l_arr.append(d)
        ssarr.append(ss)
        cu_arr.append(cu)

    return ssarr, l_arr, cu_arr


class VIF(FullRefMetric):
    def __init__(self) -> None:
        pass

    def compute(self, image_true: np.ndarray, image_test: np.ndarray, **kwargs: Any) -> float:
        """
        Parameters:
        -----------
        image_true: np.array (H, W)
            Reference image
        image_test: np.array (H, W)
            Image to be evaluated against the reference image
        """

        # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        image_true_T = torch.Tensor(image_true.copy())
        image_test_T = torch.Tensor(image_test.copy())

        return vifvec(image_true_T, image_test_T)[0]


if __name__ == "__main__":
    from medimetrics.distortions import GaussianBlur

    # create test images:
    image_true = np.random.rand(240, 240)
    vif = VIF()
    blur = GaussianBlur(10)
    for s in range(10):
        image_test = blur(image_true, s)

        print(s, ": ", vif.compute(image_true, image_test))