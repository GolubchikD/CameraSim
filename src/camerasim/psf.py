"""Standalone PSF kernel generators. All return numpy arrays summing to 1.

These can be passed to :class:`camerasim.detector.DetectorModel` via
``psf_kernel`` to replace the built-in Gaussian blur with a more realistic
optical model.
"""

from __future__ import annotations

from math import factorial

import numpy as np
from scipy.special import j1


def _resolve_size(size: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(size, int):
        return size, size
    h, w = size
    return int(h), int(w)


def _center_crop_or_pad(arr: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    H, W = shape
    h, w = arr.shape
    out = np.zeros((H, W), dtype=arr.dtype)
    # Copy the centred overlap region.
    src_r0 = max(0, (h - H) // 2)
    src_c0 = max(0, (w - W) // 2)
    dst_r0 = max(0, (H - h) // 2)
    dst_c0 = max(0, (W - w) // 2)
    rh = min(H, h) - max(0, dst_r0 + min(H, h) - H)
    cw = min(W, w) - max(0, dst_c0 + min(W, w) - W)
    rh = min(H - dst_r0, h - src_r0)
    cw = min(W - dst_c0, w - src_c0)
    out[dst_r0:dst_r0 + rh, dst_c0:dst_c0 + cw] = arr[
        src_r0:src_r0 + rh, src_c0:src_c0 + cw
    ]
    return out


def gaussian(
    size: int | tuple[int, int],
    sigma_px: float | tuple[float, float],
) -> np.ndarray:
    """2-D Gaussian PSF kernel, normalised to sum=1.

    Parameters
    ----------
    size : int or (H, W)
        Kernel grid size in pixels.
    sigma_px : float or (sy, sx)
        1-sigma in pixels along each axis.
    """
    H, W = _resolve_size(size)
    if np.isscalar(sigma_px):
        sy = sx = float(sigma_px)
    else:
        sy, sx = float(sigma_px[0]), float(sigma_px[1])

    cy = (H - 1) / 2.0
    cx = (W - 1) / 2.0
    if sy <= 0.0 or sx <= 0.0:
        k = np.zeros((H, W), dtype=np.float64)
        k[int(round(cy)), int(round(cx))] = 1.0
        return k

    y, x = np.mgrid[:H, :W].astype(np.float64)
    g = np.exp(-0.5 * ((y - cy) / sy) ** 2 - 0.5 * ((x - cx) / sx) ** 2)
    return g / g.sum()


def airy(
    size: int | tuple[int, int],
    fnumber: float,
    wavelength_m: float,
    pixel_pitch_m: float,
) -> np.ndarray:
    """Airy-disk PSF for a circular aperture, normalised to sum=1.

    Parameters
    ----------
    size : int or (H, W)
        Kernel grid size in pixels.
    fnumber : float
        Working f-number ``f/D``.
    wavelength_m : float
        Wavelength in metres.
    pixel_pitch_m : float
        Detector pixel pitch in metres.

    Notes
    -----
    The Airy intensity is ``[2 J1(x) / x]^2`` with
    ``x = pi * r / (lambda * F#)``, where ``r`` is the radial distance in
    the image plane in metres. The first zero falls at
    ``r = 1.22 * lambda * F#``.
    """
    H, W = _resolve_size(size)
    cy = (H - 1) / 2.0
    cx = (W - 1) / 2.0
    y, x = np.mgrid[:H, :W].astype(np.float64)
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2) * pixel_pitch_m
    arg = np.pi * r / (wavelength_m * fnumber)
    with np.errstate(divide="ignore", invalid="ignore"):
        psf = np.where(arg == 0, 1.0, (2.0 * j1(arg) / arg) ** 2)
    s = psf.sum()
    if s > 0:
        psf = psf / s
    return psf


def _zernike_radial(n: int, m: int, rho: np.ndarray) -> np.ndarray:
    """Zernike radial polynomial R_n^|m|(rho)."""
    m = abs(m)
    if (n - m) % 2 != 0:
        return np.zeros_like(rho)
    out = np.zeros_like(rho)
    for k in range((n - m) // 2 + 1):
        coeff = (
            (-1) ** k
            * factorial(n - k)
            / (
                factorial(k)
                * factorial((n + m) // 2 - k)
                * factorial((n - m) // 2 - k)
            )
        )
        out = out + coeff * rho ** (n - 2 * k)
    return out


def _zernike_term(n: int, m: int, rho: np.ndarray, theta: np.ndarray) -> np.ndarray:
    R = _zernike_radial(n, m, rho)
    if m > 0:
        return R * np.cos(m * theta)
    if m < 0:
        return R * np.sin(-m * theta)
    return R


def zernike(
    size: int | tuple[int, int],
    coeffs: dict[tuple[int, int], float] | list[tuple[int, int, float]],
    pupil_grid_px: int = 128,
) -> np.ndarray:
    """PSF from Zernike pupil aberrations, normalised to sum=1.

    Parameters
    ----------
    size : int or (H, W)
        Output PSF crop size in pixels.
    coeffs : dict ``{(n, m): amplitude_waves}`` or list of ``(n, m, amp)``
        Zernike coefficients in OSA/ANSI ``(n, m)`` convention with
        ``-n <= m <= n`` and ``(n - m) even``. Amplitude is in waves
        (1.0 == 2*pi radians of phase). Positive ``m`` -> cos(m*theta);
        negative ``m`` -> sin(|m|*theta); ``m = 0`` -> rotationally
        symmetric.
    pupil_grid_px : int
        Pupil grid resolution. Larger = better-resolved PSF, slower FFT.
    """
    H, W = _resolve_size(size)
    N = int(pupil_grid_px)

    y, x = np.mgrid[:N, :N].astype(np.float64)
    cy = (N - 1) / 2.0
    cx = (N - 1) / 2.0
    R = N / 2.0
    rho = np.sqrt((y - cy) ** 2 + (x - cx) ** 2) / R
    theta = np.arctan2(y - cy, x - cx)
    pupil_mask = (rho <= 1.0).astype(np.float64)

    phase = np.zeros((N, N), dtype=np.float64)
    items = coeffs.items() if isinstance(coeffs, dict) else (
        ((n, m), a) for n, m, a in coeffs
    )
    for (n, m), amp in items:
        if amp == 0.0:
            continue
        phase = phase + 2.0 * np.pi * amp * _zernike_term(n, m, rho, theta)
    phase = phase * pupil_mask

    field = pupil_mask * np.exp(1j * phase)
    spec = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field)))
    psf = np.abs(spec) ** 2

    if (H, W) != (N, N):
        psf = _center_crop_or_pad(psf, (H, W))

    s = psf.sum()
    if s > 0:
        psf = psf / s
    return psf
