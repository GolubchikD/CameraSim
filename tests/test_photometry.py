"""Tests for :func:`camerasim.electrons_from_intensity`."""

from __future__ import annotations

import numpy as np
import pytest

from camerasim import electrons_from_intensity


def _intensity():
    y, x = np.mgrid[:16, :16]
    return np.exp(-((y - 7.5) ** 2 + (x - 7.5) ** 2) / 8.0)


# --- shape / dtype ----------------------------------------------------------


def test_shape_preserved():
    intensity = _intensity()
    e = electrons_from_intensity(
        intensity, qe=0.5, exposure_s=1e-3, pixel_area=(5e-6) ** 2,
        source_power=1e9,
    )
    assert e.shape == intensity.shape


def test_non_negative():
    e = electrons_from_intensity(
        _intensity(), qe=0.5, exposure_s=1e-3, pixel_area=(5e-6) ** 2,
        source_power=1e9,
    )
    assert (e >= 0).all()


# --- conservation -----------------------------------------------------------


def test_total_electrons_matches_qe_exposure_power():
    """Sum equals qe * exposure_s * source_power (photons normalised to source)."""
    e = electrons_from_intensity(
        _intensity(), qe=0.6, exposure_s=2e-3, pixel_area=(5e-6) ** 2,
        source_power=5e8,
    )
    expected = 0.6 * 2e-3 * 5e8
    np.testing.assert_allclose(e.sum(), expected, rtol=1e-12)


def test_pixel_area_cancels():
    """pixel_area should not change the result (normalisation absorbs it)."""
    i = _intensity()
    a = electrons_from_intensity(i, 0.5, 1e-3, 1.0, 1e9)
    b = electrons_from_intensity(i, 0.5, 1e-3, 1e-10, 1e9)
    np.testing.assert_allclose(a, b, rtol=1e-12)


def test_linear_in_source_power():
    i = _intensity()
    a = electrons_from_intensity(i, 0.5, 1e-3, 1e-12, 1e9)
    b = electrons_from_intensity(i, 0.5, 1e-3, 1e-12, 2e9)
    np.testing.assert_allclose(b, 2 * a, rtol=1e-12)


def test_linear_in_exposure_and_qe():
    i = _intensity()
    a = electrons_from_intensity(i, 0.5, 1e-3, 1e-12, 1e9)
    b = electrons_from_intensity(i, 1.0, 2e-3, 1e-12, 1e9)
    np.testing.assert_allclose(b, 4 * a, rtol=1e-12)


# --- validation -------------------------------------------------------------


def test_zero_intensity_raises():
    with pytest.raises(ValueError, match="identically zero"):
        electrons_from_intensity(
            np.zeros((8, 8)), 0.5, 1e-3, 1e-12, 1e9,
        )


def test_negative_intensity_raises():
    bad = np.ones((8, 8))
    bad[0, 0] = -0.1
    with pytest.raises(ValueError, match="non-negative"):
        electrons_from_intensity(bad, 0.5, 1e-3, 1e-12, 1e9)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(qe=-0.1),
        dict(qe=1.1),
        dict(exposure_s=0.0),
        dict(exposure_s=-1.0),
        dict(pixel_area=0.0),
        dict(pixel_area=-1.0),
        dict(source_power=-1.0),
    ],
)
def test_invalid_scalar_args_raise(kwargs):
    base = dict(
        qe=0.5, exposure_s=1e-3, pixel_area=1e-12, source_power=1e9,
    )
    base.update(kwargs)
    with pytest.raises(ValueError):
        electrons_from_intensity(_intensity(), **base)


# --- torch bridge -----------------------------------------------------------


def test_torch_in_torch_out():
    torch = pytest.importorskip("torch")
    i = torch.from_numpy(_intensity())
    e = electrons_from_intensity(i, 0.5, 1e-3, 1e-12, 1e9)
    assert isinstance(e, torch.Tensor)
    assert e.shape == i.shape
