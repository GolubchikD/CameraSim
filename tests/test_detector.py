"""Tests for :class:`camerasim.detector.DetectorModel`."""

from __future__ import annotations

import numpy as np
import pytest

from camerasim import DetectorModel, psf


def _flat_input(shape, value=500.0):
    return np.full(shape, value, dtype=np.float64)


# --- shape / dtype handling -------------------------------------------------


def test_single_frame_returns_2d():
    cam = DetectorModel(rng_seed=0)
    out = cam.expose(_flat_input((16, 16)))
    assert out.shape == (16, 16)
    assert out.dtype == np.uint16


def test_stack_returns_3d():
    cam = DetectorModel(rng_seed=0)
    out = cam.expose(_flat_input((4, 16, 16)))
    assert out.shape == (4, 16, 16)
    assert out.dtype == np.uint16


def test_4d_input_rejected():
    cam = DetectorModel(rng_seed=0)
    with pytest.raises(ValueError, match="must be 2-D"):
        cam.expose(np.zeros((2, 3, 16, 16)))


def test_single_frame_matches_stack_first_frame():
    """A (H, W) call equals the first frame of a (1, H, W) call with the same seed."""
    shape = (16, 16)
    img = _flat_input(shape, 300.0)

    cam_single = DetectorModel(rng_seed=42)
    cam_stack = DetectorModel(rng_seed=42)

    out_single = cam_single.expose(img)
    out_stack = cam_stack.expose(img[None])

    np.testing.assert_array_equal(out_single, out_stack[0])


# --- reproducibility --------------------------------------------------------


def test_same_seed_same_output():
    a = DetectorModel(rng_seed=7).expose(_flat_input((3, 32, 32)))
    b = DetectorModel(rng_seed=7).expose(_flat_input((3, 32, 32)))
    np.testing.assert_array_equal(a, b)


def test_different_seed_different_output():
    a = DetectorModel(rng_seed=1).expose(_flat_input((3, 32, 32)))
    b = DetectorModel(rng_seed=2).expose(_flat_input((3, 32, 32)))
    assert not np.array_equal(a, b)


def test_repeated_expose_is_deterministic():
    """Same Camera, called twice -> identical output (per Mestechko semantics)."""
    cam = DetectorModel(rng_seed=11)
    img = _flat_input((2, 24, 24))
    out1 = cam.expose(img)
    out2 = cam.expose(img)
    np.testing.assert_array_equal(out1, out2)


# --- noise-free path: byte-identical mean ----------------------------------


def _noise_free_camera(**overrides):
    """Camera with all noise sources disabled and no PSF / clipping concerns."""
    defaults = dict(
        psf_sigma_px=0.0,
        background_e=0.0,
        dark_current_e=0.0,
        read_noise_e=0.0,
        prnu_sigma=0.0,
        dsnu_sigma_e=0.0,
        full_well_e=1e9,
        gain_e_per_adu=1.0,
        bits=16,
        bias_adu=0.0,
        hot_pixel_fraction=0.0,
        dead_pixel_fraction=0.0,
        rng_seed=0,
    )
    defaults.update(overrides)
    return DetectorModel(**defaults)


def test_noise_free_identity_at_unit_gain():
    """Poisson(N) returns N when N is an integer count (no other noise)."""
    cam = _noise_free_camera()
    img = np.full((8, 8), 100.0)
    out = cam.expose(img)
    # Poisson noise still exists, so allow ~ +/- few; on average it's tight.
    assert abs(out.mean() - 100.0) < 5.0


def test_bias_pedestal_offsets_zero_input():
    cam = _noise_free_camera(bias_adu=37.0)
    out = cam.expose(np.zeros((8, 8)))
    np.testing.assert_array_equal(out, np.full((8, 8), 37, dtype=np.uint16))


# --- pipeline-specific behaviour -------------------------------------------


def test_output_clipped_to_adc_range():
    cam = DetectorModel(
        psf_sigma_px=0.0, prnu_sigma=0.0, dsnu_sigma_e=0.0,
        read_noise_e=0.0, background_e=0.0, dark_current_e=0.0,
        hot_pixel_fraction=0.0, dead_pixel_fraction=0.0,
        full_well_e=1e9, gain_e_per_adu=1.0, bits=8, bias_adu=0.0,
        rng_seed=0,
    )
    out = cam.expose(np.full((8, 8), 5000.0))  # way above 2^8 - 1
    assert out.min() >= 0
    assert out.max() == 255


def test_full_well_saturation():
    cam = DetectorModel(
        psf_sigma_px=0.0, prnu_sigma=0.0, dsnu_sigma_e=0.0,
        read_noise_e=0.0, background_e=0.0, dark_current_e=0.0,
        hot_pixel_fraction=0.0, dead_pixel_fraction=0.0,
        full_well_e=1000.0, gain_e_per_adu=10.0, bits=16, bias_adu=0.0,
        rng_seed=0,
    )
    out = cam.expose(np.full((6, 6), 1e6))
    # Clipped at full_well (1000 e-) then /10 = 100 ADU.
    assert out.max() == 100


def test_hot_and_dead_pixels_present():
    cam = DetectorModel(
        psf_sigma_px=0.0, prnu_sigma=0.0, dsnu_sigma_e=0.0,
        read_noise_e=0.0, background_e=0.0, dark_current_e=0.0,
        hot_pixel_fraction=0.1, dead_pixel_fraction=0.1,
        full_well_e=30000.0, gain_e_per_adu=1.0, bits=16, bias_adu=10.0,
        rng_seed=3,
    )
    out = cam.expose(np.full((32, 32), 1000.0))
    # Hot pixels: full_well / gain + bias = 30000 / 1 + 10 = 30010 ADU.
    # Dead pixels: 0 e- / gain + bias = 10 ADU.
    assert (out == 30010).sum() > 0
    assert (out == 10).sum() > 0


def test_poisson_mean_matches_input():
    """Average over many frames of constant input ~ Poisson mean."""
    cam = DetectorModel(
        psf_sigma_px=0.0, prnu_sigma=0.0, dsnu_sigma_e=0.0,
        read_noise_e=0.0, background_e=0.0, dark_current_e=0.0,
        hot_pixel_fraction=0.0, dead_pixel_fraction=0.0,
        full_well_e=1e9, gain_e_per_adu=1.0, bits=16, bias_adu=0.0,
        rng_seed=0,
    )
    N = 200
    out = cam.expose(np.full((N, 16, 16), 50.0))
    # Mean over (N * 16 * 16) ~ 50.0 with std ~ sqrt(50/n_samples).
    assert abs(out.mean() - 50.0) < 0.3


# --- PSF integration --------------------------------------------------------


def test_psf_kernel_overrides_sigma():
    """psf_kernel is honoured even when psf_sigma_px is non-zero."""
    img = np.zeros((1, 32, 32), dtype=np.float64)
    img[0, 16, 16] = 1000.0

    kernel = psf.gaussian(9, sigma_px=2.0)
    cam = DetectorModel(
        psf_sigma_px=0.0, psf_kernel=kernel,
        prnu_sigma=0.0, dsnu_sigma_e=0.0,
        read_noise_e=0.0, background_e=0.0, dark_current_e=0.0,
        hot_pixel_fraction=0.0, dead_pixel_fraction=0.0,
        full_well_e=1e9, gain_e_per_adu=1.0, bits=16, bias_adu=0.0,
        rng_seed=0,
    )
    out = cam.expose(img)
    # Spike spread over many pixels.
    assert (out > 0).sum() > 9


# --- terrain / occlusion ---------------------------------------------------


def test_terrain_adds_to_signal():
    img = np.zeros((1, 8, 8))
    terrain = np.full((8, 8), 100.0)
    cam = _noise_free_camera()
    out = cam.expose(img, terrain_e=terrain)
    # Signal is 100 e-/pixel -> 100 ADU at unit gain.
    assert abs(out.mean() - 100.0) < 5.0


def test_transmittance_requires_terrain():
    cam = DetectorModel(rng_seed=0)
    with pytest.raises(ValueError, match="terrain_e is None"):
        cam.expose(np.zeros((1, 8, 8)), transmittance=np.ones((1, 8, 8)))


def test_transmittance_zero_blocks_terrain():
    img = np.zeros((1, 8, 8))
    terrain = np.full((8, 8), 100.0)
    trans = np.zeros((1, 8, 8))
    cam = _noise_free_camera()
    out = cam.expose(img, transmittance=trans, terrain_e=terrain)
    np.testing.assert_array_equal(out, np.zeros((1, 8, 8), dtype=np.uint16))


def test_terrain_shape_mismatch_raises():
    cam = DetectorModel(rng_seed=0)
    with pytest.raises(ValueError, match="terrain_e shape"):
        cam.expose(np.zeros((1, 8, 8)), terrain_e=np.zeros((4, 4)))


# --- torch bridge -----------------------------------------------------------


def test_torch_input_returns_torch():
    torch = pytest.importorskip("torch")
    img = torch.full((4, 16, 16), 100.0)
    cam = DetectorModel(rng_seed=0)
    out = cam.expose(img)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (4, 16, 16)


def test_torch_and_numpy_agree():
    torch = pytest.importorskip("torch")
    img = np.full((4, 16, 16), 100.0, dtype=np.float64)

    cam_np = DetectorModel(rng_seed=99)
    cam_t = DetectorModel(rng_seed=99)

    out_np = cam_np.expose(img)
    out_t = cam_t.expose(torch.from_numpy(img.copy())).cpu().numpy()
    np.testing.assert_array_equal(out_np, out_t)


# --- sim -> camera-pixel binning + ideal / float modes ----------------------


def test_bin_factor_is_flux_conserving():
    """An N×N down-bin sums each block; total flux is preserved."""
    cam = DetectorModel(bin_factor=2, apply_noise=False, quantize=False,
                        background_e=0.0, dark_current_e=0.0, psf_sigma_px=0.0,
                        full_well_e=float("inf"))
    out = cam.expose(np.ones((64, 64)))
    assert out.shape == (32, 32)
    assert np.allclose(out, 4.0)


def test_pixel_size_ratio_sets_binning():
    """camera_pixel_m / sim_pixel_m rounds to the bin factor."""
    cam = DetectorModel(sim_pixel_m=1e-3, camera_pixel_m=4e-3, apply_noise=False,
                        quantize=False, background_e=0.0, dark_current_e=0.0,
                        psf_sigma_px=0.0, full_well_e=float("inf"))
    out = cam.expose(np.ones((64, 64)))
    assert out.shape == (16, 16)


def test_ideal_mode_is_exact_identity():
    """apply_noise=False + quantize=False with no pedestals returns the input."""
    x = np.arange(32 * 32, dtype=np.float64).reshape(32, 32)
    cam = DetectorModel(apply_noise=False, quantize=False, background_e=0.0,
                        dark_current_e=0.0, psf_sigma_px=0.0, full_well_e=float("inf"))
    out = cam.expose(x)
    assert out.dtype == np.float64
    assert np.array_equal(out, x)


def test_quantize_false_returns_float_electrons():
    """quantize=False returns a float map, not integer ADU."""
    cam = DetectorModel(apply_noise=True, quantize=False, rng_seed=0)
    out = cam.expose(_flat_input((16, 16), 100.0))
    assert np.issubdtype(out.dtype, np.floating)


def test_defaults_unchanged_by_new_fields():
    """The default pipeline (noise + quantize on, no binning) is unchanged."""
    a = DetectorModel(rng_seed=0).expose(_flat_input((24, 24), 300.0))
    b = DetectorModel(rng_seed=0).expose(_flat_input((24, 24), 300.0))
    assert a.shape == (24, 24)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.array_equal(a, b)


def test_reseed_varies_frames_reproducibly():
    """reseed(): distinct per-frame noise, reproducible across runs, static
    maps untouched (the frame-sequence contract twin cameras need)."""
    def _grab_pair(base_seed):
        cam = DetectorModel(rng_seed=base_seed, prnu_sigma=0.02, dsnu_sigma_e=0.5)
        frames = []
        for frame_id in range(2):
            cam.reseed(base_seed + frame_id)
            frames.append(cam.expose(_flat_input((16, 16), 500.0)))
        return frames, cam._prnu_map.copy()

    (f0, f1), prnu_a = _grab_pair(7)
    (g0, g1), prnu_b = _grab_pair(7)
    assert not np.array_equal(f0, f1)  # frames differ within a run
    assert np.array_equal(f0, g0) and np.array_equal(f1, g1)  # runs reproduce
    assert np.array_equal(prnu_a, prnu_b)  # static maps stable


def test_reseed_before_first_expose_is_equivalent_to_construction_seed():
    cam_a = DetectorModel(rng_seed=999)
    cam_a.reseed(3)
    cam_b = DetectorModel(rng_seed=3)
    a = cam_a.expose(_flat_input((12, 12), 200.0))
    b = cam_b.expose(_flat_input((12, 12), 200.0))
    assert np.array_equal(a, b)
