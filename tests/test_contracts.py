"""Tests for data contracts (FrameMeta, RawFrame, assert_detector_headroom)."""

import pytest
import torch

from camerasim.contracts import (
    FrameMeta,
    RawFrame,
    assert_detector_headroom,
)


def test_frame_meta_creation():
    """FrameMeta can be created with required fields."""
    meta = FrameMeta(
        frame_id=42,
        exposure_s=1e-3,
        t_trigger=0.0,
        t_exposure_start=0.0,
        t_exposure_end=1e-3,
        t_readout_end=1.2e-3,
        dm_settled=True,
        extra={"key": "value"},
    )
    assert meta.frame_id == 42
    assert meta.exposure_s == 1e-3
    assert meta.dm_settled is True
    assert meta.extra["key"] == "value"


def test_raw_frame_creation():
    """RawFrame can be created with pixels, meta, and adu_max."""
    pixels = torch.full((32, 32), 100.0)
    meta = FrameMeta(
        frame_id=0,
        exposure_s=1e-3,
        t_trigger=0.0,
        t_exposure_start=0.0,
        t_exposure_end=1e-3,
        t_readout_end=1.2e-3,
    )
    frame = RawFrame(pixels=pixels, meta=meta, adu_max=1023)

    assert frame.pixels.shape == (32, 32)
    assert frame.adu_max == 1023
    assert frame.meta.frame_id == 0


def test_raw_frame_invalid_pixels_shape():
    """RawFrame raises ValueError if pixels is not 2-D."""
    meta = FrameMeta(
        frame_id=0,
        exposure_s=1e-3,
        t_trigger=0.0,
        t_exposure_start=0.0,
        t_exposure_end=1e-3,
        t_readout_end=1.2e-3,
    )
    with pytest.raises(ValueError, match="RawFrame.pixels must be 2-D"):
        RawFrame(pixels=torch.zeros(32), meta=meta, adu_max=1023)


def test_raw_frame_saturation_fraction():
    """RawFrame.saturation_fraction computes the fraction of pixels at adu_max."""
    pixels = torch.full((32, 32), 100.0)
    # Set 10% of pixels to adu_max
    pixels[:10, :] = 1023.0
    meta = FrameMeta(
        frame_id=0,
        exposure_s=1e-3,
        t_trigger=0.0,
        t_exposure_start=0.0,
        t_exposure_end=1e-3,
        t_readout_end=1.2e-3,
    )
    frame = RawFrame(pixels=pixels, meta=meta, adu_max=1023)

    sat_frac = frame.saturation_fraction
    expected = 10 * 32 / (32 * 32)  # 10 rows out of 32 rows
    assert abs(sat_frac - expected) < 1e-6, f"Expected {expected}, got {sat_frac}"


def test_assert_detector_headroom_passes_for_raw_frame():
    """assert_detector_headroom passes when peak is below max_fill."""
    pixels = torch.full((32, 32), 500.0)  # Peak at 500 ADU
    meta = FrameMeta(
        frame_id=0,
        exposure_s=1e-3,
        t_trigger=0.0,
        t_exposure_start=0.0,
        t_exposure_end=1e-3,
        t_readout_end=1.2e-3,
    )
    frame = RawFrame(pixels=pixels, meta=meta, adu_max=1023)

    # Peak fill: 500 / 1023 ≈ 0.489 < 0.8
    fill = assert_detector_headroom(frame, max_fill=0.8)
    assert 0.48 < fill < 0.49


def test_assert_detector_headroom_raises_for_raw_frame():
    """assert_detector_headroom raises when peak is at or above max_fill."""
    pixels = torch.full((32, 32), 900.0)  # Peak at 900 ADU
    meta = FrameMeta(
        frame_id=0,
        exposure_s=1e-3,
        t_trigger=0.0,
        t_exposure_start=0.0,
        t_exposure_end=1e-3,
        t_readout_end=1.2e-3,
    )
    frame = RawFrame(pixels=pixels, meta=meta, adu_max=1023)

    # Peak fill: 900 / 1023 ≈ 0.88 >= 0.8
    with pytest.raises(AssertionError, match="detector saturating"):
        assert_detector_headroom(frame, max_fill=0.8)


def test_assert_detector_headroom_bare_tensor():
    """assert_detector_headroom works with a bare tensor when adu_max is provided."""
    pixels = torch.full((32, 32), 300.0)

    # Should pass
    fill = assert_detector_headroom(pixels, max_fill=0.8, adu_max=1023)
    assert 0.29 < fill < 0.30

    # Should raise if peak is too high
    pixels_high = torch.full((32, 32), 900.0)
    with pytest.raises(AssertionError, match="detector saturating"):
        assert_detector_headroom(pixels_high, max_fill=0.8, adu_max=1023)


def test_assert_detector_headroom_bare_tensor_requires_adu_max():
    """assert_detector_headroom raises ValueError if adu_max is not provided for a bare tensor."""
    pixels = torch.full((32, 32), 500.0)

    with pytest.raises(ValueError, match="assert_detector_headroom needs adu_max"):
        assert_detector_headroom(pixels, max_fill=0.8)


def test_assert_detector_headroom_invalid_max_fill():
    """assert_detector_headroom raises ValueError if max_fill is out of range."""
    pixels = torch.full((32, 32), 100.0)

    with pytest.raises(ValueError, match="max_fill must be in"):
        assert_detector_headroom(pixels, max_fill=0.0, adu_max=1023)

    with pytest.raises(ValueError, match="max_fill must be in"):
        assert_detector_headroom(pixels, max_fill=1.5, adu_max=1023)
