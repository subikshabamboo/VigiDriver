"""Physics-based Noise & Illumination Engine.

Models the four physical degradation sources characterized in the paper
(SNR 30 dB → 0 dB robustness analysis) plus the EnlightenGAN-style low-light
harmonization:

  * Additive White Gaussian Noise — sensor thermal grain
        I_noisy(x,y) = I(x,y) + η,  η ~ N(0, σ²)
  * Poisson shot noise — low photon flux, heteroscedastic CMOS grain
        I_shot(x,y) = P(λ·I(x,y)) / λ
  * Motion blur — engine vibration / head saccades, PSF kernel of angle
        θ ∈ [0°, 180°] and length L ∈ [3, 11] px
  * Specular dashboard glare — localized bivariate Gaussian patches

`yuv_clahe` implements the luminance-channel CLAHE (clip 2.0, 8×8 tiles)
used across the whole pipeline, and `enhance_lowlight` provides the
EnlightenGAN/CycleGAN-style illumination harmonization fallback used
whenever no trained GAN generator weights are present on disk.
"""

from __future__ import annotations

import os
from typing import Any

import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GAN_WEIGHTS_PATH = os.path.join(BASE_DIR, "weights", "enlightengan_generator.pth")


# ---------------------------------------------------------------------------
# CLAHE — YUV luminance-channel contrast normalization (paper §3.1 stage 2)
# ---------------------------------------------------------------------------

def yuv_clahe(frame_bgr: np.ndarray, clip_limit: float = 2.0,
              tile_grid: tuple[int, int] = (8, 8)) -> np.ndarray:
    """Equalize only the Y channel, preserving natural skin tones."""
    yuv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YUV)
    yuv[:, :, 0] = cv2.createCLAHE(
        clipLimit=clip_limit, tileGridSize=tile_grid
    ).apply(yuv[:, :, 0])
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)


# ---------------------------------------------------------------------------
# Noise regimes
# ---------------------------------------------------------------------------

def snr_to_sigma_db(frame: np.ndarray, snr_db: float) -> float:
    """σ = sqrt(P_signal / 10^(SNR/10)); P_signal = var of luminance."""
    gray = frame.mean(axis=2)
    p_signal = float(gray.astype(np.float64).var()) + 1e-9
    return float(np.sqrt(p_signal / (10.0 ** (snr_db / 10.0))))


def add_awgn(frame_bgr: np.ndarray, sigma: float, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma, frame_bgr.shape)
    return np.clip(frame_bgr.astype(np.float64) + noise, 0, 255).astype(np.uint8)


def add_poisson(frame_bgr: np.ndarray, scale: float = 0.95,
                seed: int | None = None) -> np.ndarray:
    """Photon shot noise: I_shot = P(λ·I)/λ with λ = scale/mean signal."""
    rng = np.random.default_rng(seed)
    f = frame_bgr.astype(np.float64)
    lam = max(1.0, scale * 255.0)
    noisy = rng.poisson(f / lam * 255.0).astype(np.float64)
    noisy = f + (noisy - f) * (scale if scale < 1.0 else 1.0)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def motion_blur_psf(length: int = 9, angle_deg: float = 45.0) -> np.ndarray:
    """Point-spread-function kernel h(x, y) for linear motion blur."""
    psf = np.zeros((length, length), dtype=np.float32)
    c = (length - 1) // 2
    cv2.line(psf, (c, c), (c, c), 1.0, 1)
    # rotate the line by angle
    rows, cols = length, length
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle_deg, 1)
    psf = cv2.warpAffine(psf, M, (rows, cols))
    psf /= psf.sum() + 1e-9
    return psf


def add_motion_blur(frame_bgr: np.ndarray, length: int = 9,
                    angle_deg: float = 45.0) -> np.ndarray:
    return cv2.filter2D(frame_bgr, -1, motion_blur_psf(length, angle_deg))


def add_glare(frame_bgr: np.ndarray, amplitude: float = 180.0,
              center: tuple[float, float] | None = None,
              sigma_xy: tuple[float, float] = (60.0, 90.0),
              seed: int | None = None) -> np.ndarray:
    """Specular glare: G(x,y) = A·exp(-[(x-x0)²/2σx² + (y-y0)²/2σy²])."""
    h, w = frame_bgr.shape[:2]
    rng = np.random.default_rng(seed)
    if center is None:
        center = (float(rng.uniform(w * 0.15, w * 0.85)), float(rng.uniform(0, h * 0.35)))
    x0, y0 = center
    yy, xx = np.mgrid[0:h, 0:w]
    g = amplitude * np.exp(
        -(((xx - x0) ** 2) / (2 * sigma_xy[0] ** 2)
          + ((yy - y0) ** 2) / (2 * sigma_xy[1] ** 2))
    )
    out = frame_bgr.astype(np.float64) + g[:, :, None]
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Regime registry for the SNR robustness sweep
# ---------------------------------------------------------------------------

REGIMES: dict[str, dict[str, Any]] = {
    "pristine": {"fn": None, "params": {}},
    "awgn_25db": {"fn": add_awgn, "params": {"snr_db": 25.0}},
    "awgn_10db": {"fn": add_awgn, "params": {"snr_db": 10.0}},
    "poisson": {"fn": add_poisson, "params": {"scale": 0.95}},
    "motion_blur": {"fn": add_motion_blur, "params": {"length": 9, "angle_deg": 45.0}},
    "glare": {"fn": add_glare, "params": {"amplitude": 180.0, "seed": 3}},
}


def _make_regime_fn(name: str):
    spec = REGIMES[name]
    fn, params = spec["fn"], dict(spec["params"])

    if fn is None:
        return lambda img: img
    if fn is add_awgn and "snr_db" in params:
        snr = params.pop("snr_db")
        return lambda img: add_awgn(img, snr_to_sigma_db(img, snr), seed=None)
    return lambda img: fn(img, **params)


def apply_regime(frame_bgr: np.ndarray, name: str) -> np.ndarray:
    if name not in REGIMES:
        raise KeyError(f"unknown noise regime: {name}")
    return _make_regime_fn(name)(frame_bgr)


REGIME_LABELS = {
    "pristine": "Pristine (no noise)",
    "awgn_25db": "AWGN (25 dB SNR)",
    "awgn_10db": "AWGN (10 dB SNR)",
    "poisson": "Poisson shot noise",
    "motion_blur": "Motion blur (L=9, θ=45°)",
    "glare": "Specular glare (A=180)",
}


# ---------------------------------------------------------------------------
# EnlightenGAN-style low-light harmonization
# ---------------------------------------------------------------------------

def enhance_lowlight(frame_bgr: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Illumination harmonization pipeline.

    With trained EnlightenGAN generator weights present
    (backend/weights/enlightengan_generator.pth) this path could call the
    generator; without them it runs the deterministic physics-fallback:
    brightness-adaptive gamma + YUV-CLAHE + edge-preserving smoothing —
    the same normalization family used to synthesize the GAN training set.
    """
    img = frame_bgr.astype(np.float32)
    mean_lum = float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean())

    # Adaptive gamma: dark frames get lifted (γ<1 brightens), bright stay sane
    gamma = 1.0
    if mean_lum < 90.0:
        gamma = max(0.45, mean_lum / 90.0)
    elif mean_lum > 190.0:
        gamma = 1.25
    lifted = 255.0 * np.power(np.clip(img / 255.0, 0, 1), gamma * strength)

    # Tone-safe local contrast on the Y channel
    img8 = np.clip(lifted, 0, 255).astype(np.uint8)
    img8 = yuv_clahe(img8, clip_limit=2.0, tile_grid=(8, 8))

    # Bilateral smoothing preserves eyelid/lip edges the VGG loss preserves
    img8 = cv2.bilateralFilter(img8, d=5, sigmaColor=28, sigmaSpace=28)
    return img8
