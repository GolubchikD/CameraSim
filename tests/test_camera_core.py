"""Tests for the Camera exposure core."""

import warnings

import pytest
import torch

from camerasim.camera import Camera, CameraConfig
from camerasim.clock import VirtualClock
from camerasim.contracts import DetectorSaturationWarning, assert_detector_headroom


def test_camera_zero_photons_yields_bias():
    """Exposing a zero-photon map yields a frame whose mean ≈ bias (offset_adu)."""
    config = CameraConfig(
        shape=(64, 64),
        qe=0.6,
        gain_e_per_adu=2.0,
        read_noise_e=1.0,
        dark_current_e_s=10.0,
        offset_adu=100.0,
        bit_depth=14,
        readout_s=200e-6,
    )
    clock = VirtualClock()
    camera = Camera(config, clock, seed=0)

    photons = torch.zeros(64, 64)
    frame = camera.expose(photons, dm_settled=True)

    # With zero photons, we get only dark current + bias + read noise
    # Expected electrons: 0 + 10.0 * 1e-3 = 0.01 e-
    # Expected ADU: 0.01 / 2.0 + 100.0 ≈ 100.005
    # The mean should be close to bias (offset_adu)
    mean_adu = float(frame.pixels.mean())
    assert 99.0 < mean_adu < 101.0, f"Expected mean ≈ 100, got {mean_adu}"


def test_camera_positive_photons_tracked():
    """Exposing a positive flat photon map yields mean ADU that tracks qe*photons/gain + bias."""
    config = CameraConfig(
        shape=(64, 64),
        qe=0.6,
        gain_e_per_adu=2.0,
        read_noise_e=1.0,
        dark_current_e_s=10.0,
        offset_adu=100.0,
        bit_depth=14,
        readout_s=200e-6,
    )
    clock = VirtualClock()
    camera = Camera(config, clock, seed=0)
    camera.set_exposure(1e-3)  # 1 ms

    # Photons per pixel: 1000
    # Expected electrons: qe * photons + dark_current * exposure
    #                   = 0.6 * 1000 + 10.0 * 1e-3 = 600.01 e-
    # Expected ADU: 600.01 / 2.0 + 100.0 ≈ 400.0
    photons = torch.full((64, 64), 1000.0)
    frame = camera.expose(photons, dm_settled=True)

    mean_adu = float(frame.pixels.mean())
    expected_adu = (0.6 * 1000.0 + 10.0 * 1e-3) / 2.0 + 100.0
    # Allow ±10 ADU for shot/read noise across the mean
    assert abs(mean_adu - expected_adu) < 10.0, f"Expected mean ≈ {expected_adu}, got {mean_adu}"


def test_camera_frame_id_and_reseed():
    """reseed/frame_id gives distinct but reproducible frames.

    Two Cameras with the same seed produce identical frame sequences;
    frame N differs from frame N+1.
    """
    config = CameraConfig(shape=(32, 32), qe=0.6, gain_e_per_adu=2.0, read_noise_e=5.0)
    clock1 = VirtualClock()
    clock2 = VirtualClock()
    camera1 = Camera(config, clock1, seed=42)
    camera2 = Camera(config, clock2, seed=42)

    photons = torch.full((32, 32), 500.0)

    # Same seed -> same first frame
    frame1_a = camera1.expose(photons)
    frame1_b = camera2.expose(photons)
    assert torch.allclose(frame1_a.pixels, frame1_b.pixels), "Same seed should produce identical frames"

    # Second frame differs from first (different frame_id -> different noise)
    frame2_a = camera1.expose(photons)
    assert not torch.allclose(frame1_a.pixels, frame2_a.pixels), "Consecutive frames should differ"

    # But camera2's second frame matches camera1's second frame (reproducible)
    frame2_b = camera2.expose(photons)
    assert torch.allclose(frame2_a.pixels, frame2_b.pixels), "Same seed sequence should be reproducible"


def test_camera_saturation_telemetry():
    """Saturation telemetry present in meta.extra."""
    config = CameraConfig(shape=(32, 32), qe=0.6, gain_e_per_adu=2.0, bit_depth=10)
    clock = VirtualClock()
    camera = Camera(config, clock, seed=0)

    # Low photon level -> no saturation
    photons_low = torch.full((32, 32), 100.0)
    frame = camera.expose(photons_low)
    assert "saturation_fraction" in frame.meta.extra
    assert "roi_saturation_fraction" in frame.meta.extra
    assert frame.meta.extra["saturation_fraction"] < 0.01


def test_camera_saturation_warning():
    """A photon level that rails triggers DetectorSaturationWarning."""
    config = CameraConfig(
        shape=(32, 32),
        qe=0.6,
        gain_e_per_adu=2.0,
        bit_depth=10,  # adu_max = 1023
        read_noise_e=1.0,
    )
    clock = VirtualClock()
    camera = Camera(config, clock, seed=0, saturation_warn_fraction=1e-3)

    # High photon level that saturates the detector
    # To saturate: adu_max = 1023
    # (qe * photons) / gain + bias ≈ 1023
    # photons ≈ (1023 - 100) * 2.0 / 0.6 ≈ 3077
    # Use higher to ensure saturation
    photons_high = torch.full((32, 32), 5000.0)

    with pytest.warns(DetectorSaturationWarning, match="illuminated pixels at the ADC rail"):
        frame = camera.expose(photons_high)

    # Verify some pixels are actually saturated
    assert frame.saturation_fraction > 0.0


def test_camera_saturation_fraction_property():
    """RawFrame.saturation_fraction property works."""
    config = CameraConfig(shape=(16, 16), qe=0.6, gain_e_per_adu=2.0, bit_depth=8)
    clock = VirtualClock()
    camera = Camera(config, clock, seed=0, saturation_warn_fraction=1.0)  # Silence warning

    # Moderate photons -> some saturation
    photons = torch.full((16, 16), 2000.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DetectorSaturationWarning)
        frame = camera.expose(photons)

    sat_frac = frame.saturation_fraction
    assert 0.0 <= sat_frac <= 1.0, f"saturation_fraction must be in [0, 1]; got {sat_frac}"


def test_camera_set_flux_and_grab_dark():
    """set_flux(0) + grab with shutter closed yields ~bias (dark grab)."""
    config = CameraConfig(
        shape=(32, 32),
        qe=0.6,
        gain_e_per_adu=2.0,
        read_noise_e=1.0,
        dark_current_e_s=10.0,
        offset_adu=100.0,
        bit_depth=14,
    )
    clock = VirtualClock()
    camera = Camera(config, clock, seed=0)
    camera.set_flux(0.0)  # No flat source
    camera.set_shutter(False)  # Shutter closed
    camera.set_exposure(1e-3)

    frame = camera.grab()
    mean_adu = float(frame.pixels.mean())
    # With shutter closed and no flux: only dark current + bias
    # Expected: (10.0 * 1e-3) / 2.0 + 100.0 ≈ 100.005
    assert 99.0 < mean_adu < 101.0, f"Expected mean ≈ 100, got {mean_adu}"


def test_camera_set_flux_and_grab_flat():
    """set_flux(rate) + grab with shutter open yields signal tracking flux."""
    config = CameraConfig(
        shape=(32, 32),
        qe=0.6,
        gain_e_per_adu=2.0,
        read_noise_e=1.0,
        dark_current_e_s=10.0,
        offset_adu=100.0,
        bit_depth=14,
    )
    clock = VirtualClock()
    camera = Camera(config, clock, seed=0)
    camera.set_exposure(1e-3)  # 1 ms
    # Flux: 1e6 photons/pixel/s -> 1000 photons/pixel in 1 ms
    camera.set_flux(1e6)
    camera.set_shutter(True)

    frame = camera.grab()
    mean_adu = float(frame.pixels.mean())
    # Expected: (0.6 * 1000 + 10.0 * 1e-3) / 2.0 + 100.0 ≈ 400.0
    expected_adu = (0.6 * 1000.0 + 10.0 * 1e-3) / 2.0 + 100.0
    assert abs(mean_adu - expected_adu) < 10.0, f"Expected mean ≈ {expected_adu}, got {mean_adu}"


def test_camera_set_flux_validates_input():
    """set_flux raises on negative flux."""
    config = CameraConfig(shape=(16, 16))
    clock = VirtualClock()
    camera = Camera(config, clock, seed=0)

    with pytest.raises(ValueError, match="photons_per_pixel_s must be >= 0"):
        camera.set_flux(-1.0)



def test_assert_detector_headroom_passes():
    """assert_detector_headroom passes when under the threshold."""
    config = CameraConfig(shape=(16, 16), qe=0.6, gain_e_per_adu=2.0, bit_depth=10)
    clock = VirtualClock()
    camera = Camera(config, clock, seed=0)

    photons = torch.full((16, 16), 100.0)
    frame = camera.expose(photons)

    # Should not raise (low photon level -> well below adu_max)
    fill = assert_detector_headroom(frame, max_fill=0.8)
    assert 0.0 < fill < 0.8


def test_assert_detector_headroom_raises():
    """assert_detector_headroom raises when at or above the threshold."""
    config = CameraConfig(shape=(16, 16), qe=0.6, gain_e_per_adu=2.0, bit_depth=8)
    clock = VirtualClock()
    camera = Camera(config, clock, seed=0, saturation_warn_fraction=1.0)  # Silence warning

    # High photons to saturate
    photons = torch.full((16, 16), 5000.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DetectorSaturationWarning)
        frame = camera.expose(photons)

    # Should raise because pixels are at or near adu_max
    with pytest.raises(AssertionError, match="detector saturating"):
        assert_detector_headroom(frame, max_fill=0.8)


def test_camera_timing_advances():
    """Camera.expose() advances the clock through exposure + readout."""
    config = CameraConfig(shape=(16, 16), readout_s=500e-6)
    clock = VirtualClock(t0=0.0)
    camera = Camera(config, clock, seed=0)
    camera.set_exposure(2e-3)  # 2 ms exposure

    t0 = clock.now()
    photons = torch.zeros(16, 16)
    frame = camera.expose(photons)
    t1 = clock.now()

    # Clock should have advanced by exposure + readout
    expected_dt = 2e-3 + 500e-6
    assert abs((t1 - t0) - expected_dt) < 1e-9, f"Expected dt={expected_dt}, got {t1 - t0}"

    # Check metadata timing
    assert frame.meta.t_trigger == t0
    assert frame.meta.t_exposure_start == t0
    assert abs(frame.meta.t_exposure_end - (t0 + 2e-3)) < 1e-9
    assert abs(frame.meta.t_readout_end - t1) < 1e-9


def test_camera_shutter_closed():
    """Camera.expose() with shutter closed yields bias-only frames (no photons)."""
    config = CameraConfig(
        shape=(32, 32),
        qe=0.6,
        gain_e_per_adu=2.0,
        read_noise_e=1.0,
        dark_current_e_s=10.0,
        offset_adu=100.0,
    )
    clock = VirtualClock()
    camera = Camera(config, clock, seed=0)
    camera.set_shutter(False)  # Close shutter

    # Even with high photons, shutter closed -> only dark + bias
    photons = torch.full((32, 32), 5000.0)
    frame = camera.expose(photons)

    mean_adu = float(frame.pixels.mean())
    # Expected: dark + bias + read noise, no photons
    # dark electrons: 10.0 * 1e-3 = 0.01 e-
    # ADU: 0.01 / 2.0 + 100.0 ≈ 100
    assert 99.0 < mean_adu < 101.0, f"Closed shutter should yield bias ≈ 100, got {mean_adu}"


def test_camera_hot_pixels():
    """Hot pixels carry extra dark current."""
    hot_sites = ((5, 5), (10, 10))
    config = CameraConfig(
        shape=(32, 32),
        qe=0.6,
        gain_e_per_adu=2.0,
        read_noise_e=0.1,  # Low read noise for clearer signal
        dark_current_e_s=10.0,
        hot_pixel_dark_e_s=5e4,
        offset_adu=100.0,
        hot_pixels=hot_sites,
    )
    clock = VirtualClock()
    camera = Camera(config, clock, seed=0)
    camera.set_exposure(1e-3)

    photons = torch.zeros(32, 32)
    frame = camera.expose(photons)

    # Hot pixels should be significantly brighter than the background
    y0, x0 = hot_sites[0]
    bg_mean = float(frame.pixels[0:3, 0:3].mean())
    hot_val = float(frame.pixels[y0, x0])

    # Hot pixel extra: 5e4 * 1e-3 = 50 e- -> 50 / 2.0 = 25 ADU above background
    assert hot_val > bg_mean + 20.0, f"Hot pixel should be brighter; hot={hot_val}, bg={bg_mean}"
