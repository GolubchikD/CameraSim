"""Camera exposure core: photon map -> detector ADU with timing.

This is the reusable detector+timing+telemetry engine that can be imported by
any wavefront-sensing camera implementation (phase-diversity, Shack-Hartmann,
etc.) without dragging in scene/optics/DM specifics. The caller supplies a
mean-photon map; the camera applies detector physics and advances the clock.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import torch

from camerasim.clock import Clock
from camerasim.contracts import DetectorSaturationWarning, FrameMeta, RawFrame
from camerasim.detector import DetectorModel

__all__ = ["CameraConfig", "Camera"]


@dataclass
class CameraConfig:
    """Detector configuration (InGaAs-like defaults)."""

    shape: tuple[int, int] = (512, 640)
    qe: float = 0.6
    gain_e_per_adu: float = 2.0
    read_noise_e: float = 18.0
    dark_current_e_s: float = 200.0
    offset_adu: float = 100.0
    bit_depth: int = 14
    readout_s: float = 200e-6
    #: Injected hot-pixel sites ``((y, x), ...)`` — a *fault*, not geometry:
    #: these pixels carry ``hot_pixel_dark_e_s`` on top of the uniform
    #: ``dark_current_e_s``, in **both** shutter states ("hot pixels don't
    #: care about light"). Ground truth for bad-pixel masking. Default
    #: empty — byte-identical to pre-hot-pixel behavior.
    hot_pixels: tuple[tuple[int, int], ...] = ()
    #: Extra dark-current rate (e-/s) at each ``hot_pixels`` site.
    hot_pixel_dark_e_s: float = 5e4

    def __post_init__(self) -> None:
        if not (0.0 < self.qe <= 1.0):
            raise ValueError(f"qe must be in (0, 1]; got {self.qe}")
        if self.gain_e_per_adu <= 0.0:
            raise ValueError(f"gain_e_per_adu must be > 0; got {self.gain_e_per_adu}")
        if self.bit_depth < 8 or self.bit_depth > 16:
            raise ValueError(f"bit_depth must be in [8, 16]; got {self.bit_depth}")
        if self.hot_pixel_dark_e_s < 0.0:
            raise ValueError(f"hot_pixel_dark_e_s must be >= 0; got {self.hot_pixel_dark_e_s}")
        for y, x in self.hot_pixels:
            if not (0 <= y < self.shape[0] and 0 <= x < self.shape[1]):
                raise ValueError(f"hot_pixels site ({y}, {x}) outside detector shape {self.shape}")

    @property
    def adu_max(self) -> int:
        return 2**self.bit_depth - 1


class Camera:
    """Camera exposure core: photon map -> detector ADU with timing.

    Args:
        config: Detector configuration.
        clock: Shared bench clock.
        source_photons_per_s: Photon rate reaching the detector (before QE);
            this is a bookkeeping default — callers pass explicit photon maps
            to :meth:`expose`.
        seed: RNG seed for shot/read noise.
        saturation_warn_fraction: Fraction of *illuminated* pixels (expected
            photons > 0) allowed at the ADC rail before :meth:`expose` raises a
            :class:`DetectorSaturationWarning` (default 0.1%). Both the
            frame-wide and signal-scoped rail fractions are stamped into
            ``FrameMeta.extra`` every exposure regardless; set this to ``1.0``
            to silence the warning only.
    """

    def __init__(
        self,
        config: CameraConfig,
        clock: Clock,
        *,
        source_photons_per_s: float = 1e9,
        seed: int = 0,
        saturation_warn_fraction: float = 1e-3,
    ) -> None:
        self.config = config
        self._clock = clock
        if source_photons_per_s <= 0.0:
            raise ValueError(f"source_photons_per_s must be > 0; got {source_photons_per_s}")
        self.source_photons_per_s = float(source_photons_per_s)
        if not (0.0 <= saturation_warn_fraction <= 1.0):
            raise ValueError(
                f"saturation_warn_fraction must be in [0, 1]; got {saturation_warn_fraction}"
            )
        self.saturation_warn_fraction = float(saturation_warn_fraction)

        # Detector chain: qe + dark (uniform and hot-pixel) stay on this side
        # folded into the mean electron map (Poisson of the total mean is
        # identical either way); DetectorModel contributes shot + read noise,
        # gain, bias (= offset_adu), and the bit-depth ADU clamp. full_well is
        # +inf because this config clamps in the ADU domain only (bit-depth).
        # PRNU/DSNU/hot-fraction are zeroed: the twin's fault model injects
        # specific hot-pixel SITES (calibration ground truth) rather than
        # random fractions.
        self._seed = int(seed)
        self._detector_model = DetectorModel(
            psf_sigma_px=0.0,
            background_e=0.0,
            dark_current_e=0.0,
            read_noise_e=config.read_noise_e,
            prnu_sigma=0.0,
            dsnu_sigma_e=0.0,
            full_well_e=float("inf"),
            gain_e_per_adu=config.gain_e_per_adu,
            bits=config.bit_depth,
            bias_adu=config.offset_adu,
            hot_pixel_fraction=0.0,
            dead_pixel_fraction=0.0,
            apply_noise=True,
            quantize=True,
            rng_seed=self._seed,
        )
        self._exposure_s = 1e-3
        self._shutter_open = True
        self._frame_id = 0
        self._flat_photons_per_pixel_s = 0.0

    # -- Properties and setters ----------------------------------------------

    @property
    def shape(self) -> tuple[int, int]:
        return self.config.shape

    @property
    def adu_max(self) -> int:
        return self.config.adu_max

    @property
    def exposure_s(self) -> float:
        return self._exposure_s

    def set_exposure(self, exposure_s: float) -> None:
        if exposure_s <= 0.0:
            raise ValueError(f"exposure_s must be > 0; got {exposure_s}")
        self._exposure_s = float(exposure_s)

    @property
    def shutter_open(self) -> bool:
        return self._shutter_open

    def set_shutter(self, open_: bool) -> None:
        self._shutter_open = bool(open_)

    def set_flux(self, photons_per_pixel_s: float) -> None:
        """Set the built-in flat-source photon rate per pixel (photons/pixel/s).
        
        Args:
            photons_per_pixel_s: Uniform photon rate per pixel reaching the detector.
        """
        if photons_per_pixel_s < 0.0:
            raise ValueError(f"photons_per_pixel_s must be >= 0; got {photons_per_pixel_s}")
        self._flat_photons_per_pixel_s = float(photons_per_pixel_s)

    # -- Main exposure core --------------------------------------------------

    def grab(self, *, dm_settled: bool = True) -> RawFrame:
        """Grab one frame using the built-in flat source.
        
        Builds a mean-photon map from the flat source (if shutter is open)
        and calls :meth:`expose`.
        
        Args:
            dm_settled: Whether the DM (if any) was settled for the whole
                exposure window. Stamped into ``FrameMeta``.
        
        Returns:
            A :class:`RawFrame` with quantized ADU, full timing telemetry,
            and saturation fractions in ``meta.extra``.
        """
        if self._shutter_open:
            photons = torch.full(
                self.config.shape, 
                self._flat_photons_per_pixel_s * self._exposure_s,
                dtype=torch.float32
            )
        else:
            photons = torch.zeros(self.config.shape, dtype=torch.float32)
        return self.expose(photons, dm_settled=dm_settled)

    def expose(self, photons: torch.Tensor, *, dm_settled: bool = True) -> RawFrame:
        """Trigger one exposure, wait for readout, return the frame.

        Args:
            photons: Mean photon map ``(H, W)`` reaching the detector
                (before QE). The caller supplies this — no scene/optics
                modeling happens here.
            dm_settled: Whether the DM (if any) was settled for the whole
                exposure window. Stamped into ``FrameMeta``.

        Returns:
            A :class:`RawFrame` with quantized ADU, full timing telemetry,
            and saturation fractions in ``meta.extra``.
        """
        if photons.dim() != 2:
            raise ValueError(f"photons must be 2-D (H, W); got shape {tuple(photons.shape)}")
        if photons.shape != self.config.shape:
            raise ValueError(
                f"photons shape {tuple(photons.shape)} does not match detector shape {self.config.shape}"
            )

        cfg = self.config
        t_trigger = self._clock.now()
        t_exp_start = t_trigger
        self._clock.sleep(self._exposure_s)
        t_exp_end = self._clock.now()
        self._clock.sleep(cfg.readout_s)
        t_readout_end = self._clock.now()

        # Build mean electron map: qe*photons + uniform dark + hot-pixel extra
        if not self._shutter_open:
            photons = torch.zeros_like(photons)
        electrons_mean = cfg.qe * photons + cfg.dark_current_e_s * self._exposure_s
        if cfg.hot_pixels:
            # Hot pixels carry extra dark current regardless of shutter state
            hot_extra = torch.zeros(cfg.shape)
            ys, xs = zip(*cfg.hot_pixels)
            hot_extra[list(ys), list(xs)] = cfg.hot_pixel_dark_e_s * self._exposure_s
            electrons_mean = electrons_mean + hot_extra

        # Distinct, reproducible noise per frame: derive a per-frame seed and
        # reset ONLY the model's frame stream.
        self._detector_model.reseed(self._seed * 1_000_003 + self._frame_id)
        adu = self._detector_model.expose(electrons_mean).to(torch.float32)

        # Saturation telemetry: never discard the clip. The upstream
        # DetectorModel clamps at the ADC ceiling (full_well_e=+inf here, so
        # the ADU rail is the only rail); we measure the rail-pixel fraction
        # back off the returned frame. Two numbers, both stamped: whole-frame
        # (for logging parity with RawFrame.saturation_fraction) and, more
        # honestly, over the *signal footprint* (expected-photons > 0) so a
        # few railed spot cores are not diluted by the dark background. The
        # signal-scoped fraction drives the warning.
        at_rail = adu >= cfg.adu_max
        frame_sat = float(at_rail.to(torch.float32).mean())
        signal_mask = photons > 0.0
        n_signal = int(signal_mask.sum())
        roi_sat = (
            float((at_rail & signal_mask).to(torch.float32).sum()) / n_signal
            if n_signal > 0
            else 0.0
        )

        meta = FrameMeta(
            frame_id=self._frame_id,
            exposure_s=self._exposure_s,
            t_trigger=t_trigger,
            t_exposure_start=t_exp_start,
            t_exposure_end=t_exp_end,
            t_readout_end=t_readout_end,
            dm_settled=dm_settled,
            extra={"saturation_fraction": frame_sat, "roi_saturation_fraction": roi_sat},
        )
        if roi_sat > self.saturation_warn_fraction:
            warnings.warn(
                f"frame {self._frame_id}: {roi_sat:.2%} of illuminated pixels at the ADC rail "
                f"(adu_max={cfg.adu_max}), above the {self.saturation_warn_fraction:.2%} "
                f"tolerance -- saturated frames are nonlinear; lower the exposure/flux "
                f"before using this for scoring or calibration",
                DetectorSaturationWarning,
                stacklevel=2,
            )
        self._frame_id += 1
        return RawFrame(pixels=adu.to(torch.float32), meta=meta, adu_max=cfg.adu_max)
