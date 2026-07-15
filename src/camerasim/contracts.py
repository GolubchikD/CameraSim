"""Data contracts for camera acquisition (torch-based, vendor-neutral).

Portable camera frame/telemetry types that can be imported by any consumer
without dragging in physics models or wavefront-sensing specifics. This is the
wire format between the acquisition side (camera, clock) and the processing
side (calibration, preprocessing, RTC loops).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

__all__ = [
    "FrameMeta",
    "RawFrame",
    "DetectorSaturationWarning",
    "assert_detector_headroom",
]


class DetectorSaturationWarning(UserWarning):
    """A frame carried more pixels at the ADC rail than the configured
    tolerance. Its own category so consumers can escalate it selectively:
    calibration routines — where a saturated frame poisons gains/darks —
    can promote it to an error with
    ``warnings.simplefilter("error", DetectorSaturationWarning)`` while a
    live loop merely logs it."""


@dataclass
class FrameMeta:
    """Acquisition metadata for one camera frame.

    All times come from a shared clock. ``dm_settled`` records
    whether the DM reported settled for the *entire* exposure window.
    """

    frame_id: int
    exposure_s: float
    t_trigger: float
    t_exposure_start: float
    t_exposure_end: float
    t_readout_end: float
    dm_settled: bool = True
    extra: dict = field(default_factory=dict)


@dataclass
class RawFrame:
    """One detector frame in quantized ADU.

    Attributes:
        pixels: ``(H, W)`` float32 of quantized ADU values in
            ``[0, adu_max]`` (uint16 payload on a real wire).
        meta: :class:`FrameMeta`.
        adu_max: Saturation value (``2**bit_depth - 1``).
    """

    pixels: torch.Tensor
    meta: FrameMeta
    adu_max: int

    def __post_init__(self) -> None:
        if self.pixels.dim() != 2:
            raise ValueError(
                f"RawFrame.pixels must be 2-D (H, W); got shape {tuple(self.pixels.shape)}"
            )

    @property
    def saturation_fraction(self) -> float:
        """Fraction of pixels at the saturation rail (observable, never
        silently clipped away)."""
        return float((self.pixels >= self.adu_max).float().mean().item())


def assert_detector_headroom(
    frame: "RawFrame | torch.Tensor",
    max_fill: float = 0.8,
    *,
    adu_max: int | None = None,
) -> float:
    """Precondition for any comparison against a *linear* reference: the
    detector must not be near its rail.

    A chain whose brightest pixel sits at (or clips against) the ADC ceiling
    is nonlinear there — its flat-topped spot cores cannot correlate with a
    linear rendering — so a test or study that scores a detector chain
    against a linear model must assert headroom *before* the comparison, or
    it measures saturation instead of the effect it meant to.

    Args:
        frame: A :class:`RawFrame`, or a raw-ADU tensor (then ``adu_max`` is
            required).
        max_fill: Peak-ADU fraction of ``adu_max`` allowed; the default 0.8
            leaves shot-noise headroom below the rail.
        adu_max: Saturation ADU value, required only when ``frame`` is a bare
            tensor (a :class:`RawFrame` carries its own).

    Returns:
        The observed fill fraction ``peak_adu / adu_max`` (so a caller can
        also log or bound it).

    Raises:
        AssertionError: if the peak fill is at or above ``max_fill``.
        ValueError: if ``max_fill`` is out of ``(0, 1]`` or a bare tensor is
            passed without ``adu_max``.
    """
    if not (0.0 < max_fill <= 1.0):
        raise ValueError(f"max_fill must be in (0, 1]; got {max_fill}")
    if isinstance(frame, RawFrame):
        pixels, ceiling = frame.pixels, frame.adu_max
    else:
        if adu_max is None:
            raise ValueError("assert_detector_headroom needs adu_max when given a bare tensor")
        pixels, ceiling = frame, int(adu_max)
    peak = float(pixels.max())
    fill = peak / ceiling
    if fill >= max_fill:
        raise AssertionError(
            f"detector saturating: peak {peak:.0f} ADU is {fill:.1%} of adu_max {ceiling} "
            f"(>= the {max_fill:.0%} headroom bar) — a chain this close to its rail is "
            f"nonlinear and cannot be scored against a linear reference; reduce flux/exposure "
            f"before comparing"
        )
    return fill
