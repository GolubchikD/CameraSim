"""Tests for :mod:`camerasim.psf` kernel generators."""

from __future__ import annotations

import numpy as np
import pytest

from camerasim import psf


# --- gaussian ---------------------------------------------------------------


def test_gaussian_sums_to_one():
    k = psf.gaussian(15, sigma_px=1.5)
    assert k.shape == (15, 15)
    np.testing.assert_allclose(k.sum(), 1.0, atol=1e-12)


def test_gaussian_symmetric():
    k = psf.gaussian(15, sigma_px=1.5)
    np.testing.assert_allclose(k, k.T, atol=1e-12)
    np.testing.assert_allclose(k, k[::-1, ::-1], atol=1e-12)


def test_gaussian_peak_at_center():
    k = psf.gaussian(11, sigma_px=1.0)
    H, W = k.shape
    assert np.unravel_index(np.argmax(k), k.shape) == (H // 2, W // 2)


def test_gaussian_anisotropic():
    """Different sigma per axis -> asymmetric kernel."""
    k = psf.gaussian(21, sigma_px=(2.0, 0.5))
    # Wider along axis 0 than axis 1: column slice through centre is
    # broader than row slice.
    col_fwhm = np.sum(k[:, 10] > k[:, 10].max() / 2)
    row_fwhm = np.sum(k[10, :] > k[10, :].max() / 2)
    assert col_fwhm > row_fwhm


def test_gaussian_zero_sigma_is_delta():
    k = psf.gaussian(11, sigma_px=0.0)
    assert k.sum() == 1.0
    assert k[5, 5] == 1.0


# --- airy -------------------------------------------------------------------


def test_airy_sums_to_one():
    k = psf.airy(51, fnumber=8.0, wavelength_m=550e-9, pixel_pitch_m=2e-6)
    np.testing.assert_allclose(k.sum(), 1.0, atol=1e-12)


def test_airy_non_negative_and_peaked():
    k = psf.airy(51, fnumber=8.0, wavelength_m=550e-9, pixel_pitch_m=2e-6)
    H, W = k.shape
    assert (k >= 0).all()
    assert np.unravel_index(np.argmax(k), k.shape) == (H // 2, W // 2)


def test_airy_first_zero_near_theoretical():
    """First dark ring at r = 1.22 * lambda * F# in image-plane metres."""
    fnumber, wl, pp = 8.0, 550e-9, 2e-6
    size = 101
    k = psf.airy(size, fnumber=fnumber, wavelength_m=wl, pixel_pitch_m=pp)
    centre = size // 2
    radial = k[centre, centre:]
    # Find first local minimum.
    diffs = np.diff(radial)
    first_min_idx = np.argmax(diffs > 0)  # transition from decreasing to increasing
    r_first_min = first_min_idx * pp
    r_theory = 1.22 * wl * fnumber
    # Tolerance: a couple of pixels.
    assert abs(r_first_min - r_theory) < 3 * pp


# --- zernike ---------------------------------------------------------------


def test_zernike_no_aberration_concentrated():
    """Zero coefficients -> diffraction-limited PSF: sums to 1 and peaks at centre."""
    k = psf.zernike(64, coeffs={}, pupil_grid_px=128)
    H, W = k.shape
    np.testing.assert_allclose(k.sum(), 1.0, atol=1e-10)
    assert np.unravel_index(np.argmax(k), k.shape) == (H // 2, W // 2)


def test_zernike_defocus_broadens_psf():
    """Adding defocus reduces the central peak fraction."""
    k0 = psf.zernike(64, coeffs={}, pupil_grid_px=128)
    k1 = psf.zernike(64, coeffs={(2, 0): 1.0}, pupil_grid_px=128)
    assert k1.max() < k0.max()
    np.testing.assert_allclose(k1.sum(), 1.0, atol=1e-10)


def test_zernike_accepts_list_form():
    k_dict = psf.zernike(32, coeffs={(2, 0): 0.5}, pupil_grid_px=128)
    k_list = psf.zernike(32, coeffs=[(2, 0, 0.5)], pupil_grid_px=128)
    np.testing.assert_allclose(k_dict, k_list, atol=1e-12)


def test_zernike_invalid_pair_zero_contribution():
    """(n, m) with (n-m) odd is invalid and contributes zero phase."""
    k_ref = psf.zernike(32, coeffs={}, pupil_grid_px=64)
    k_inv = psf.zernike(32, coeffs={(2, 1): 5.0}, pupil_grid_px=64)
    np.testing.assert_allclose(k_ref, k_inv, atol=1e-12)
