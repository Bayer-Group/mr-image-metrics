from typing import Dict, Tuple, Type

import numpy as np
from pytest import approx

from medimetrics.base import Distortion, FullRefMetric, NonRefMetric
from medimetrics.distortions import (
    BiasField,
    ElasticDeformation,
    GammaHigh,
    GammaLow,
    GaussianBlur,
    GaussianNoise,
    Ghosting,
    ReplaceArtifact,
    ShiftIntensity,
    Stripes,
    Translation,
)
from medimetrics.metrics import (
    BRISQUE,
    CWSSIM,
    DICE,
    DISTS,
    DSS,
    FSIM,
    GMSD,
    IWSSIM,
    LPIPS,
    MAE,
    MDSI,
    MLC,
    MSE,
    MSLC,
    MSSSIM,
    NIQE,
    NMI,
    NMSE,
    PCC,
    PSNR,
    SSIM,
    VIF,
    VSI,
    BlurCPBD,
    BlurEffect,
    BlurJNB,
    BlurRatio,
    BlurWidths,
    HaarPSI,
    MeanBlur,
    MeanTotalVar,
    VarLaplace,
)


def test_all_metrics() -> None:
    full_ref_metrics: Dict[Type[FullRefMetric], float] = {
        # Testing:
        DSS: 0.0,
        FSIM: 0.0,
        GMSD: 0.0,
        HaarPSI: 0.3799610137939,
        IWSSIM: 0.0,
        MDSI: 0.0,
        VIF: 0.0,
        VSI: 0.0,
        #
        SSIM: 0.0921228,
        MSSSIM: 0.1670256858,
        CWSSIM: 0.54867559671,
        PSNR: 7.586984038,
        MSE: 875072500.0,
        MAE: 23379.537,
        NMSE: 43265.363,
        LPIPS: 0.93677061,
        DISTS: 0.29267913,
        NMI: 1.03907630,
        PCC: 0.4285577,
    }

    non_ref_metrics: Dict[Type[NonRefMetric], float] = {
        BlurWidths: 3.434210777,
        BlurJNB: 38.511730,
        BlurCPBD: 0.7203690,
        MeanTotalVar: 5091.3994,
        VarLaplace: 124426790.0,
        NIQE: 34.750538,
        BRISQUE: 36.781788,
        MeanBlur: 0.15132087,
        BlurEffect: 0.194720,
        BlurRatio: 2.1108674,
        MLC: 0.8189621684,
        MSLC: 0.46369055353,
    }
    seg_metrics: Dict[Type[FullRefMetric], float] = {DICE: 0.9925010017752876}

    # create test images:
    np.random.seed(1234)
    image_true = np.random.randint(0, np.iinfo(np.uint16).max, size=(240, 240)).astype(np.float32)
    blur = GaussianBlur(5)
    gamma = GammaLow(5)
    trans = Translation(5)
    image_test = blur(gamma(trans(image_true, 3), 5), 3)

    coords = [(x, y) for x in list(range(10)) + list(range(230, 240)) for y in list(range(10)) + list(range(230, 240))]
    image_true[coords] = 0
    image_test[coords] = 0

    # create segmentations:
    image_true_seg = image_true > 1000
    image_test_seg = image_test > 1000

    for full_ref_metric, result_value in full_ref_metrics.items():
        # check results on reference images:
        FR = full_ref_metric()
        result = FR.compute(image_true, image_test)
        print(FR.__class__.__name__, ": ", result, ",")
        assert result == approx(result_value, rel=1e-5)

    for non_ref_metric, result_value in non_ref_metrics.items():
        # check results on reference images:
        NR = non_ref_metric()
        result = NR.compute(image_test)
        print(NR.__class__.__name__, ": ", result, ",")
        assert result == approx(result_value, rel=1e-5)

    for seg_metric, result_value in seg_metrics.items():
        # check results on reference images:
        SM = seg_metric()
        result = SM.compute(image_test_seg, image_true_seg)
        print(SM.__class__.__name__, ": ", result)
        assert result == approx(result_value, rel=1e-5)


def test_all_distortions() -> None:
    # create test image:
    np.random.seed(1234)
    image_true = np.random.randint(0, np.iinfo(np.uint16).max, size=(240, 240)).astype(np.float32)
    blur = GaussianBlur(5)
    gamma = GammaLow(5)
    trans = Translation(5)
    image_test = blur(gamma(trans(image_true, 3), 5), 3)

    coords = [(x, y) for x in list(range(10)) + list(range(230, 240)) for y in list(range(10)) + list(range(230, 240))]
    image_test[coords] = 0

    distortions: Dict[Type[Distortion], Tuple[Type[FullRefMetric], float]] = {
        BiasField: (MSE, 2651162400.0),
        Stripes: (MSE, 571848806.9771347),
        Ghosting: (MSE, 891331.2441304654),
        GaussianNoise: (MSE, 42687096.0),
        GaussianBlur: (MSE, 16478158.0),
        GammaLow: (MSE, 64989510.0),
        GammaHigh: (MSE, 233713710.0),
        ShiftIntensity: (MSE, 313777950.0),
        ReplaceArtifact: (MSE, 20263744.0),
        Translation: (MSE, 567287200.0),
        ElasticDeformation: (MSE, 70412070.0),
    }

    for distortion, result in distortions.items():
        D = distortion(5)
        M = result[0]()
        np.random.seed(42)
        diff = M.compute(image_test, D(image_test, 5))
        print(D.__class__.__name__, ": ", diff, ",")
        assert diff == approx(result[1], rel=1e-5)


if __name__ == "__main__":
    test_all_metrics()
    test_all_distortions()
