"""DetectorModel: full imaging-chain port of Mestechko's ``Camera``.

Pipeline (per frame, vectorised over the stack):

    optional terrain emission (occluded by per-frame transmittance)
      -> PSF (Gaussian sigma or pluggable kernel)
      -> + sky background + dark current  [e-]
      -> * PRNU frozen map
      -> Poisson shot noise
      -> + DSNU frozen offset             [e-]
      -> + Gaussian read noise            [e- RMS]
      -> hot/dead pixel clamps
      -> full-well clip
      -> / gain + bias, round, clip to [0, 2^bits - 1]

Generalisations vs. Mestechko's ``Camera``:
  * accepts ``(H, W)`` single-frame OR ``(N, H, W)`` stacked input;
  * transparently accepts and returns torch tensors;
  * optional ``psf_kernel`` overrides the built-in Gaussian blur.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve

from . import _backend


def _bin_frames(f: np.ndarray, factor: int) -> np.ndarray:
    """Flux-conserving N×N down-bin of a ``(N, H, W)`` stack (sum per block).

    The frame is cropped to a whole multiple of ``factor`` before binning.
    """
    if factor <= 1:
        return f
    n, h, w = f.shape
    h2, w2 = (h // factor) * factor, (w // factor) * factor
    if h2 == 0 or w2 == 0:
        return f
    f = f[:, :h2, :w2]
    return f.reshape(n, h2 // factor, factor, w2 // factor, factor).sum(axis=(2, 4))


@dataclass
class DetectorModel:
    """Camera / detector model. All noise sources are independent.

    Parameters mirror Mestechko's ``Camera`` so existing callers can swap in
    with no changes. See module docstring for pipeline order.

    Parameters
    ----------
    psf_sigma_px : float
        Gaussian PSF 1-sigma in pixels. 0 disables built-in blur. Ignored
        when ``psf_kernel`` is given.
    psf_kernel : np.ndarray or None
        Optional 2-D PSF kernel (will be normalised to sum=1 internally).
        When set, overrides ``psf_sigma_px`` and is convolved with each
        frame via :func:`scipy.signal.fftconvolve`.
    background_e, dark_current_e : float
        Sky pedestal and mean dark charge per pixel per frame, in electrons.
    read_noise_e : float
        Read-noise RMS in electrons (Gaussian, zero-mean).
    prnu_sigma : float
        Std-dev of multiplicative pixel gain (e.g. 0.02 = 2 %). Frozen.
    dsnu_sigma_e : float
        Std-dev of additive dark offset per pixel, in electrons. Frozen.
    full_well_e : float
        Saturation level [e-]; signal is clipped here before gain.
    gain_e_per_adu : float
        Conversion gain. ADU = electrons / gain.
    bits : int
        ADC resolution. Output is integer in ``[0, 2**bits - 1]``.
    bias_adu : float
        Pedestal added after gain conversion so negative read-noise
        excursions don't clip at 0 ADU.
    hot_pixel_fraction, dead_pixel_fraction : float
        Fraction of pixels with stuck readout. Hot pixels are clamped to
        full-well in the electron domain (saturate at the ADC ceiling).
        Dead pixels read ``bias_adu`` after the bias step.
    sim_pixel_m, camera_pixel_m : float or None
        Simulation-grid pitch and physical detector pixel pitch, in metres.
        When ``camera_pixel_m > sim_pixel_m`` the input frame is binned
        (flux-conserving N×N sum) by ``round(camera_pixel_m / sim_pixel_m)``
        as the FIRST step, before the detector chain — so a fine wave-optics
        grid is integrated onto the coarser physical detector pixel. The
        detector maps (PRNU/DSNU/bad pixels) are built at the binned size.
    bin_factor : int
        Explicit N×N bin factor; overrides the pixel-size ratio when > 1.
    apply_noise : bool
        When False, skip ALL stochastic / fixed-pattern steps (PRNU, Poisson
        shot, DSNU, read noise, hot/dead pixels): a deterministic "ideal"
        sensor. Used by closed-loop wavefront sensors that want binning
        without injecting noise into the control loop.
    quantize : bool
        When False, skip the gain + bias + ADC step and return the float
        electron map instead of integer ADU.
    rng_seed : int or None
        Seed for ALL stochastic processes. Internally split into two
        independent streams: one for the static detector maps
        (PRNU, DSNU, hot/dead, drawn once and cached) and one re-seeded
        on every :meth:`expose` call so repeated calls produce identical
        output. ``None`` for true random.
    """

    psf_sigma_px: float = 0.7
    psf_kernel: np.ndarray | None = None
    background_e: float = 20.0
    dark_current_e: float = 1.0
    read_noise_e: float = 1.5
    prnu_sigma: float = 0.02
    dsnu_sigma_e: float = 0.5
    full_well_e: float = 30_000.0
    gain_e_per_adu: float = 4.0
    bits: int = 8
    bias_adu: float = 25.0
    hot_pixel_fraction: float = 1e-4
    dead_pixel_fraction: float = 1e-4
    sim_pixel_m: float | None = None
    camera_pixel_m: float | None = None
    bin_factor: int = 1
    apply_noise: bool = True
    quantize: bool = True
    rng_seed: int | None = 0

    _prnu_map: np.ndarray | None = field(default=None, init=False, repr=False)
    _dsnu_map: np.ndarray | None = field(default=None, init=False, repr=False)
    _hot_mask: np.ndarray | None = field(default=None, init=False, repr=False)
    _dead_mask: np.ndarray | None = field(default=None, init=False, repr=False)
    _frame_seed: np.random.SeedSequence | None = field(
        default=None, init=False, repr=False
    )

    # ---- sim -> camera-pixel binning --------------------------------------

    def _binning_factor(self) -> int:
        """Integer N×N bin factor from ``bin_factor`` or the pixel-size ratio."""
        if self.bin_factor and self.bin_factor > 1:
            return int(self.bin_factor)
        if (
            self.sim_pixel_m
            and self.camera_pixel_m
            and self.camera_pixel_m > self.sim_pixel_m
        ):
            return max(1, int(round(self.camera_pixel_m / self.sim_pixel_m)))
        return 1

    # ---- detector-map setup ------------------------------------------------

    def _ensure_maps(self, shape: tuple[int, int]) -> None:
        """Build PRNU, DSNU, and bad-pixel maps once for a given frame shape."""
        if self._prnu_map is not None and self._prnu_map.shape == shape:
            return
        map_seed, frame_seed = np.random.SeedSequence(self.rng_seed).spawn(2)
        map_rng = np.random.default_rng(map_seed)
        self._frame_seed = frame_seed

        self._prnu_map = (
            map_rng.normal(1.0, self.prnu_sigma, size=shape)
            if self.prnu_sigma > 0
            else np.ones(shape)
        )
        self._dsnu_map = (
            map_rng.normal(0.0, self.dsnu_sigma_e, size=shape)
            if self.dsnu_sigma_e > 0
            else np.zeros(shape)
        )

        n_pix = shape[0] * shape[1]
        flat_idx = map_rng.permutation(n_pix)
        n_hot = int(self.hot_pixel_fraction * n_pix)
        n_dead = int(self.dead_pixel_fraction * n_pix)
        hot_idx = flat_idx[:n_hot]
        dead_idx = flat_idx[n_hot:n_hot + n_dead]

        hot = np.zeros(n_pix, dtype=bool)
        dead = np.zeros(n_pix, dtype=bool)
        hot[hot_idx] = True
        dead[dead_idx] = True
        self._hot_mask = hot.reshape(shape)
        self._dead_mask = dead.reshape(shape)

    # ---- PSF ---------------------------------------------------------------

    def _apply_psf(self, f: np.ndarray) -> np.ndarray:
        """Apply either the kernel-based PSF or the Gaussian shortcut, in-place semantics."""
        if self.psf_kernel is not None:
            kernel = np.asarray(self.psf_kernel, dtype=np.float64)
            s = float(kernel.sum())
            if s > 0:
                kernel = kernel / s
            # fftconvolve does not broadcast over a leading axis; loop per frame.
            out = np.empty_like(f)
            for k in range(f.shape[0]):
                out[k] = fftconvolve(f[k], kernel, mode="same")
            return out
        if self.psf_sigma_px > 0:
            return gaussian_filter(
                f,
                sigma=(0, self.psf_sigma_px, self.psf_sigma_px),
                mode="constant",
            )
        return f

    # ---- main pipeline -----------------------------------------------------

    def expose(
        self,
        scene_frames: Any,
        transmittance: Any | None = None,
        terrain_e: Any | None = None,
    ) -> Any:
        """Apply the detector chain to a single frame or a stack of frames.

        Parameters
        ----------
        scene_frames : array, ``(H, W)`` or ``(N, H, W)``
            Ideal electron-rate frames (target emission). Accepts numpy
            ndarray or torch tensor; output matches the input framework
            and device.
        transmittance : array or None
            Per-frame, per-pixel fraction of terrain light reaching the
            optics. Must match ``scene_frames`` shape. Required iff
            ``terrain_e`` is given.
        terrain_e : array ``(H, W)`` or None
            Static per-pixel mean electron rate of the imaged terrain.
            Enters BEFORE the PSF, multiplied by ``transmittance``.

        Returns
        -------
        adu : same shape as ``scene_frames``
            Quantised ADU output. ``uint16`` for ``bits <= 16``, else
            ``uint32`` on the numpy path; on the torch path uint dtypes
            are widened to the next signed int type for portability with
            older torch builds.
        """
        scene_arr, meta = _backend.to_numpy(scene_frames)

        if scene_arr.ndim == 2:
            single = True
            scene_arr = scene_arr[None]
        elif scene_arr.ndim == 3:
            single = False
        else:
            raise ValueError(
                f"scene_frames must be 2-D (H, W) or 3-D (N, H, W); got "
                f"{scene_arr.ndim}-D shape {scene_arr.shape}."
            )

        # -1. Sim -> camera-pixel binning (flux-conserving). Done first so the
        # whole chain (maps, PSF, noise) runs at the physical detector size.
        bin_factor = self._binning_factor()
        if bin_factor > 1:
            scene_arr = _bin_frames(scene_arr.astype(np.float64, copy=False), bin_factor)

        frame_shape = scene_arr.shape[1:]
        self._ensure_maps(frame_shape)
        # Fresh per-call RNG -- repeated expose() calls on the same Camera
        # with the same seed produce identical output.
        rng = np.random.default_rng(self._frame_seed)

        f = scene_arr.astype(np.float64, copy=True)

        # 0. Terrain emission (optionally occluded). Both inputs may be torch.
        if terrain_e is not None:
            terrain_arr, _ = _backend.to_numpy(terrain_e)
            terrain_arr = terrain_arr.astype(np.float64)
            if terrain_arr.shape != frame_shape:
                raise ValueError(
                    f"terrain_e shape {terrain_arr.shape} does not match "
                    f"frame shape {frame_shape}."
                )
            if transmittance is None:
                f = f + terrain_arr
            else:
                trans_arr, _ = _backend.to_numpy(transmittance)
                trans_arr = trans_arr.astype(np.float64)
                if trans_arr.ndim == 2:
                    trans_arr = trans_arr[None]
                if trans_arr.shape != f.shape:
                    raise ValueError(
                        f"transmittance shape {trans_arr.shape} must match "
                        f"scene_frames {f.shape}."
                    )
                f = f + terrain_arr * trans_arr
        elif transmittance is not None:
            raise ValueError(
                "transmittance was given but terrain_e is None; "
                "occlusion has nothing to occlude."
            )

        # 1. Optical PSF on source signal only.
        f = self._apply_psf(f)

        # 2-3. Sky background + mean dark current.
        f = f + self.background_e + self.dark_current_e

        # 4-8. Fixed-pattern + stochastic detector effects. Skipped wholesale in
        # the deterministic "ideal" mode (apply_noise=False) so a closed-loop
        # WFS gets binning without noise injected into the control loop.
        if self.apply_noise:
            # 4. PRNU multiplies the full incident charge.
            f = f * self._prnu_map
            # 5. Poisson shot noise on the total mean.
            f = rng.poisson(np.maximum(f, 0.0)).astype(np.float64)
            # 6. DSNU: additive frozen per-pixel offset.
            f = f + self._dsnu_map
            # 7. Read noise (Gaussian, IID per pixel per frame).
            if self.read_noise_e > 0:
                f = f + rng.normal(0.0, self.read_noise_e, size=f.shape)
            # 8. Hot/dead pixel clamps (electron domain).
            f = np.where(self._hot_mask, self.full_well_e, f)
            f = np.where(self._dead_mask, 0.0, f)

        # 9. Full-well clip.
        f = np.minimum(f, self.full_well_e)

        # 10. Gain + bias + ADC quantisation. Skipped when quantize=False, which
        # returns the float electron map (what a wavefront reconstructor wants).
        if not self.quantize:
            return _backend.from_numpy(f[0] if single else f, meta)

        adu = np.round(f / self.gain_e_per_adu + self.bias_adu)
        adu = np.clip(adu, 0, 2 ** self.bits - 1)
        adu = adu.astype(np.uint16 if self.bits <= 16 else np.uint32)

        if single:
            adu = adu[0]
        return _backend.from_numpy(adu, meta)
