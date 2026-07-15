"""Sensor-agnostic detector characterization calibration routines.

Ported from DLWFS but decoupled from CalibrationSet: each routine returns a
plain result dataclass instead of mutating a CalibrationSet. This module is
deliberately import-free of DLWFS/wfsdm (CameraSim is a leaf dependency).

Every routine drives the :class:`camerasim.camera.Camera` via its public
interface (set_flux/set_shutter/set_exposure/grab) and returns a frozen result
object carrying the measured quantities + provenance metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch

__all__ = [
    "DarkResult",
    "GainResult",
    "DarkCurrentResult",
    "LinearityResult",
    "BadPixelResult",
    "measure_dark",
    "measure_gain_photon_transfer",
    "measure_dark_current",
    "measure_linearity",
    "measure_linearity_autorange",
    "build_bad_pixel_mask",
    "choose_exposure",
]


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class DarkResult:
    """Master dark + per-pixel temporal read noise (shutter closed).
    
    Attributes:
        dark_adu: ``(H, W)`` mean dark ADU (bias + dark current).
        read_noise_adu: ``(H, W)`` per-pixel temporal read-noise std (ADU).
        meta: Provenance dict (n_frames, exposure_s).
    """

    dark_adu: torch.Tensor
    read_noise_adu: torch.Tensor
    meta: dict = field(default_factory=dict)


@dataclass
class GainResult:
    """Conversion gain (e⁻/ADU) from the photon-transfer curve.
    
    Attributes:
        gain_e_per_adu: Conversion gain in electrons per ADU.
        meta: Provenance dict (levels, n_pairs, region, mean_adu, var_adu).
    """

    gain_e_per_adu: float
    meta: dict = field(default_factory=dict)


@dataclass
class DarkCurrentResult:
    """Dark-current rate (e⁻/s) from a shutter-closed exposure sweep.
    
    Attributes:
        dark_current_e_s: Dark-current rate in electrons per second.
        meta: Provenance dict (exposures_s, n_frames, mean_adu, slope_adu_s, intercept_adu).
    """

    dark_current_e_s: float
    meta: dict = field(default_factory=dict)


@dataclass
class LinearityResult:
    """Linear response range and full-well ADU from a flux sweep.
    
    Attributes:
        full_well_adu: Full-well ADU (2% deviation point).
        meta: Provenance dict (levels, mean_adu, n_frames, region, fit_slope_adu_per_level,
            fit_intercept_adu, max_in_range_frac_dev, saturation_index, and optionally
            autorange/coarse_levels/coarse_mean_adu/knee_bracket if autorange was used).
    """

    full_well_adu: float
    meta: dict = field(default_factory=dict)


@dataclass
class BadPixelResult:
    """Hot/dead pixel map from dark/flat acquisitions.
    
    Attributes:
        bad_pixel_mask: ``(H, W)`` bool mask (True = bad; flagged, never silently dropped).
        meta: Provenance dict (n_frames, sigma, n_bad_total, n_bad_from_darks, n_bad_from_flat).
    """

    bad_pixel_mask: torch.Tensor
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _line_fit(x: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
    """Least-squares ``y = slope*x + intercept``. Shared by every routine
    below that reduces a scan to a straight-line fit."""
    mx, my = x.mean(), y.mean()
    denom = ((x - mx) ** 2).sum()
    slope = float(((x - mx) * (y - my)).sum() / denom)
    intercept = float(my - slope * mx)
    return slope, intercept


# ---------------------------------------------------------------------------
# Calibration routines
# ---------------------------------------------------------------------------


def measure_dark(camera, *, n_frames: int = 16) -> DarkResult:
    """Master dark + per-pixel temporal read noise (shutter closed).
    
    Acquires ``n_frames`` frames with the shutter closed, computes the
    temporal mean (dark_adu) and std (read_noise_adu), and restores the
    shutter state.
    
    Args:
        camera: A :class:`camerasim.camera.Camera` instance.
        n_frames: Frames to average (>= 2).
    
    Returns:
        A :class:`DarkResult` with ``dark_adu``, ``read_noise_adu``, and ``meta``.
    
    Raises:
        ValueError: if ``n_frames < 2``.
    """
    if n_frames < 2:
        raise ValueError(f"n_frames must be >= 2; got {n_frames}")
    was_open = camera.shutter_open
    camera.set_shutter(False)
    try:
        stack = torch.stack([camera.grab().pixels for _ in range(n_frames)], dim=0)
    finally:
        camera.set_shutter(was_open)
    dark_adu = stack.mean(dim=0)
    read_noise_adu = stack.std(dim=0)
    meta = {"n_frames": n_frames, "exposure_s": camera.exposure_s}
    return DarkResult(dark_adu=dark_adu, read_noise_adu=read_noise_adu, meta=meta)


def measure_gain_photon_transfer(
    camera,
    *,
    levels: Sequence[float],
    n_pairs: int = 8,
    region: int = 32,
) -> GainResult:
    """Conversion gain (e⁻/ADU) from the photon-transfer curve.
    
    At each flat-illumination level (set via ``camera.set_flux``), the signal
    variance is estimated from frame *pairs* (``var(a - b) / 2`` — cancels
    fixed-pattern structure) and the mean from dark-subtracted frames, over a
    central ``region``² window. Shot noise gives ``var_ADU = mean_ADU / g + const``,
    so the fitted slope is ``1/g``.
    
    Requires a prior dark measurement (creates a fresh dark if needed).
    
    Args:
        camera: A :class:`camerasim.camera.Camera` instance.
        levels: Photon-rate levels (photons/pixel/s) to visit (>= 2, spanning a decent range).
        n_pairs: Frame pairs per level.
        region: Side of the central analysis window.
    
    Returns:
        A :class:`GainResult` with ``gain_e_per_adu`` and ``meta``.
    
    Raises:
        ValueError: if ``len(levels) < 2``, or the fit produces a non-positive slope.
    """
    if len(levels) < 2:
        raise ValueError(f"need >= 2 illumination levels; got {len(levels)}")
    
    # Acquire a fresh dark (we need it for subtraction)
    was_open = camera.shutter_open
    camera.set_shutter(False)
    try:
        dark = camera.grab().pixels
    finally:
        camera.set_shutter(was_open)
    
    h, w = camera.shape
    y0, x0 = (h - region) // 2, (w - region) // 2
    sl = (slice(y0, y0 + region), slice(x0, x0 + region))
    dark_crop = dark[sl]

    means, variances = [], []
    camera.set_shutter(True)
    try:
        for level in levels:
            camera.set_flux(float(level))
            level_means, level_vars = [], []
            for _ in range(n_pairs):
                a = camera.grab().pixels[sl]
                b = camera.grab().pixels[sl]
                level_means.append(((a - dark_crop) + (b - dark_crop)).mean() / 2.0)
                level_vars.append((a - b).var() / 2.0)
            means.append(torch.stack(level_means).mean())
            variances.append(torch.stack(level_vars).mean())
    finally:
        camera.set_shutter(was_open)
    
    mean_v = torch.stack(means)
    var_v = torch.stack(variances)

    # least-squares line var = slope * mean + intercept
    mx, my = mean_v.mean(), var_v.mean()
    slope = ((mean_v - mx) * (var_v - my)).sum() / ((mean_v - mx) ** 2).sum()
    if slope <= 0.0:
        raise ValueError(
            f"photon-transfer fit produced non-positive slope ({slope.item():.3g}); "
            "levels likely span too little flux or the source is saturating"
        )
    gain_e_per_adu = float(1.0 / slope)
    meta = {
        "levels": [float(v) for v in levels],
        "n_pairs": n_pairs,
        "region": region,
        "mean_adu": [float(v) for v in mean_v],
        "var_adu": [float(v) for v in var_v],
    }
    return GainResult(gain_e_per_adu=gain_e_per_adu, meta=meta)


def measure_dark_current(
    camera,
    *,
    gain_e_per_adu: float,
    exposures: Sequence[float] = (1e-3, 2e-3, 4e-3, 8e-3),
    n_frames: int = 32,
    gain_measured: bool = True,
) -> DarkCurrentResult:
    """Dark-current rate (e⁻/s) from a shutter-closed exposure sweep.
    
    Mean dark ADU is linear in exposure time (``mean = offset + rate_e_s /
    gain_e_per_adu * exposure``); the fitted slope, converted to
    photoelectrons via the *already-measured* conversion gain, is the
    per-pixel dark-current rate.
    
    Requires a PTC-measured gain first: a still-default ``gain_e_per_adu == 1.0``
    with ``gain_measured=False`` means the ADU-per-second slope has never been
    converted to physical units, so this raises loudly instead.
    
    Restores the camera's exposure and shutter state on return.
    
    Args:
        camera: A :class:`camerasim.camera.Camera` instance.
        gain_e_per_adu: Conversion gain from a prior PTC measurement.
        exposures: Exposure times to sweep (>= 2, seconds).
        n_frames: Frames averaged per exposure.
        gain_measured: Whether the gain was measured (vs still at the 1.0 default).
    
    Returns:
        A :class:`DarkCurrentResult` with ``dark_current_e_s`` and ``meta``.
    
    Raises:
        ValueError: if ``len(exposures) < 2``, or the gain is un-measured.
    """
    if len(exposures) < 2:
        raise ValueError(f"need >= 2 exposures for a slope fit; got {len(exposures)}")
    if not gain_measured and gain_e_per_adu == 1.0:
        raise ValueError(
            "measure_dark_current needs a PTC-measured gain first (gain_e_per_adu "
            "is still the 1.0 default and gain_measured=False); "
            "run measure_gain_photon_transfer before measure_dark_current"
        )
    was_open = camera.shutter_open
    was_exposure = camera.exposure_s
    camera.set_shutter(False)
    means = []
    try:
        for exp in exposures:
            camera.set_exposure(float(exp))
            stack = torch.stack([camera.grab().pixels for _ in range(n_frames)], dim=0)
            means.append(stack.mean())
    finally:
        camera.set_shutter(was_open)
        camera.set_exposure(was_exposure)

    exp_v = torch.as_tensor(exposures, dtype=torch.float32)
    mean_v = torch.stack(means)
    slope_adu_s, intercept_adu = _line_fit(exp_v, mean_v)
    dark_current_e_s = slope_adu_s * gain_e_per_adu
    meta = {
        "exposures_s": [float(e) for e in exposures],
        "n_frames": n_frames,
        "mean_adu": [float(v) for v in mean_v],
        "slope_adu_s": slope_adu_s,
        "intercept_adu": intercept_adu,
    }
    return DarkCurrentResult(dark_current_e_s=dark_current_e_s, meta=meta)


def measure_linearity(
    camera,
    *,
    levels: Sequence[float],
    n_frames: int = 4,
    region: int = 64,
) -> LinearityResult:
    """Linear response range and full-well ADU from a flux sweep.
    
    Like :func:`measure_gain_photon_transfer`'s scan, but tracking the mean
    raw ADU response (no dark subtraction -- full well is a property of the
    raw sensor's saturation rail, not a calibrated quantity) up through
    saturation. The straight line is fit only to the lower half of
    ``levels`` (sorted ascending) -- the genuinely linear regime -- then
    every level's fractional deviation from that line is checked in
    ascending order; ``full_well_adu`` is the (linearly interpolated)
    mean-ADU value at the first level whose deviation exceeds 2%.
    
    Never extrapolates: if no level in ``levels`` deviates by more than 2%,
    the scan didn't reach saturation and this raises rather than guessing.
    
    Args:
        camera: A :class:`camerasim.camera.Camera` instance.
        levels: Photon-rate levels (photons/pixel/s) to visit (>= 4, spanning well past
            the expected saturation point).
        n_frames: Frames averaged per level.
        region: Side of the central analysis window.
    
    Returns:
        A :class:`LinearityResult` with ``full_well_adu`` and ``meta``.
    
    Raises:
        ValueError: if ``len(levels) < 4``, or no level's response deviates
            from the linear fit by more than 2% (scan range too small).
    """
    if len(levels) < 4:
        raise ValueError(f"need >= 4 flux levels to fit + detect saturation; got {len(levels)}")
    h, w = camera.shape
    y0, x0 = (h - region) // 2, (w - region) // 2
    sl = (slice(y0, y0 + region), slice(x0, x0 + region))

    means = []
    was_open = camera.shutter_open
    camera.set_shutter(True)
    try:
        for level in levels:
            camera.set_flux(float(level))
            frame_means = [camera.grab().pixels[sl].mean() for _ in range(n_frames)]
            means.append(torch.stack(frame_means).mean())
    finally:
        camera.set_shutter(was_open)
    
    levels_v = torch.as_tensor(levels, dtype=torch.float32)
    mean_v = torch.stack(means)

    order = torch.argsort(levels_v)
    levels_sorted = levels_v[order]
    mean_sorted = mean_v[order]
    n_lo = max(2, len(levels) // 2)
    slope, intercept = _line_fit(levels_sorted[:n_lo], mean_sorted[:n_lo])

    predicted = slope * levels_sorted + intercept
    frac_dev = ((mean_sorted - predicted).abs() / predicted.abs().clamp_min(1.0))

    sat_idx = None
    max_in_range_dev = 0.0
    for i in range(len(levels_sorted)):
        if float(frac_dev[i]) > 0.02:
            sat_idx = i
            break
        max_in_range_dev = max(max_in_range_dev, float(frac_dev[i]))
    if sat_idx is None or sat_idx == 0:
        raise ValueError(
            "no flux level exceeded the linear fit by >2% (scan range too small to bracket "
            "saturation); widen levels -- never extrapolating"
        )

    y_lo, y_hi = float(frac_dev[sat_idx - 1]), float(frac_dev[sat_idx])
    t = (0.02 - y_lo) / (y_hi - y_lo) if y_hi != y_lo else 0.0
    full_well = float(mean_sorted[sat_idx - 1]) + t * (
        float(mean_sorted[sat_idx]) - float(mean_sorted[sat_idx - 1])
    )
    meta = {
        "levels": [float(v) for v in levels_sorted],
        "mean_adu": [float(v) for v in mean_sorted],
        "n_frames": n_frames,
        "region": region,
        "fit_slope_adu_per_level": slope,
        "fit_intercept_adu": intercept,
        "max_in_range_frac_dev": max_in_range_dev,
        "saturation_index": sat_idx,
    }
    return LinearityResult(full_well_adu=full_well, meta=meta)


def measure_linearity_autorange(
    camera,
    *,
    coarse_levels: Sequence[float],
    n_fine: int = 12,
    n_frames: int = 4,
    region: int = 64,
    saturation_fill: float = 0.98,
) -> LinearityResult:
    """Two-pass linearity + full well: a **coarse wide** sweep locates the
    saturation knee, then a **narrow** sweep bracketing it is handed to
    :func:`measure_linearity` for the actual full-well fit. This removes the
    per-detector level hand-tuning the single-pass routine needs (its levels
    must already straddle saturation with a linear lower half, which differs by
    orders of magnitude across detectors).
    
    Pass 1 (coarse): visit ``coarse_levels`` (sorted ascending; span well below
    to well above the expected knee — decade steps are fine) at half the frame
    budget, recording each level's central mean ADU. The knee is bracketed by
    the last level below the ADC rail (``saturation_fill * adu_max``) and the
    first at/above it.
    
    Pass 2 (fine): build ``n_fine`` levels — a linear-regime anchor cluster
    (a decade below the last-linear coarse level, comfortably linear so the
    fit slope is clean) plus a dense cluster from there up to the first
    saturated coarse level (crossing the rail so the >2% deviation is
    bracketed) — and delegate to :func:`measure_linearity`. ``meta`` gains
    ``coarse_levels``/``coarse_mean_adu``/``knee_bracket``/``autorange`` for
    traceability.
    
    Args:
        camera: A :class:`camerasim.camera.Camera` instance.
        coarse_levels: Wide flux-level sweep (>= 3 levels) to locate the knee.
        n_fine: Levels in the narrow calibration sweep (>= 4).
        n_frames: Frames per level.
        region: Side of the central analysis window.
        saturation_fill: Fraction of ``camera.adu_max`` a coarse level's mean
            must reach to count as saturated.
    
    Returns:
        A :class:`LinearityResult` with ``full_well_adu`` and ``meta`` (including
        autorange provenance).
    
    Raises:
        ValueError: fewer than 3 coarse or 4 fine levels; the coarse sweep
            never reached saturation (extend ``coarse_levels`` upward); or its
            lowest level already saturates (add lower ``coarse_levels``).
    """
    import math
    
    coarse = sorted(float(v) for v in coarse_levels)
    if len(coarse) < 3:
        raise ValueError(f"need >= 3 coarse_levels to bracket the knee; got {len(coarse)}")
    if n_fine < 4:
        raise ValueError(f"need >= 4 fine levels for the linearity fit; got {n_fine}")

    rail = saturation_fill * camera.adu_max
    h, w = camera.shape
    y0, x0 = (h - region) // 2, (w - region) // 2
    sl = (slice(y0, y0 + region), slice(x0, x0 + region))
    n_coarse_frames = max(1, n_frames // 2)

    coarse_means: list[float] = []
    was_open = camera.shutter_open
    camera.set_shutter(True)
    try:
        for level in coarse:
            camera.set_flux(level)
            m = torch.stack([camera.grab().pixels[sl].mean() for _ in range(n_coarse_frames)]).mean()
            coarse_means.append(float(m))
    finally:
        camera.set_shutter(was_open)

    sat_idx = next((i for i, m in enumerate(coarse_means) if m >= rail), None)
    if sat_idx is None:
        raise ValueError(
            f"coarse sweep never reached saturation (max mean {max(coarse_means):.1f} ADU < "
            f"{rail:.1f} = {saturation_fill:.0%} of adu_max {camera.adu_max}); extend "
            "coarse_levels upward"
        )
    if sat_idx == 0:
        raise ValueError(
            f"coarse sweep's lowest level ({coarse[0]:.4g}) already saturates "
            f"(mean {coarse_means[0]:.1f} ADU >= rail {rail:.1f}); add lower coarse_levels"
        )

    l_lin = coarse[sat_idx - 1]  # last level clearly below the rail
    l_sat = coarse[sat_idx]  # first level at/above the rail
    # Fine sweep = clearly-linear geometric anchors (a decade below l_lin, so
    # the fit slope is clean) + a DENSE LINEAR ramp across the knee bracket.
    # Linear (not geometric) knee spacing puts samples right up against a hard
    # ADC clip, where full_well (the first >2% departure from the linear fit)
    # is set by the last sub-clip sample -- geometric spacing under-samples
    # exactly there and reads full_well low.
    n_anchor = max(3, n_fine // 3)
    n_knee = max(3, n_fine - n_anchor)
    anchors = torch.logspace(math.log10(l_lin / 20.0), math.log10(l_lin / 3.0), n_anchor)
    knee = torch.linspace(l_lin * 0.8, l_sat, n_knee)
    fine_levels = sorted({float(v) for v in torch.cat([anchors, knee]).tolist()})

    result = measure_linearity(camera, levels=fine_levels, n_frames=n_frames, region=region)
    result.meta["autorange"] = True
    result.meta["coarse_levels"] = [float(v) for v in coarse]
    result.meta["coarse_mean_adu"] = coarse_means
    result.meta["knee_bracket"] = [l_lin, l_sat]
    return result


def build_bad_pixel_mask(
    camera,
    *,
    n_frames: int = 8,
    sigma: float = 6.0,
    flat_level: float | None = None,
) -> BadPixelResult:
    """Hot/dead pixel map from a fresh dark stack, optionally extended
    with a flat.
    
    Acquires its own ``n_frames``-frame dark stack (shutter closed) and
    flags a pixel *hot* if its temporal mean **or** std is more than
    ``sigma`` robust-sigmas (median + 1.4826*MAD, the normal-consistent MAD
    scale) from the frame's own median.
    
    If ``flat_level`` is given, an additional flat acquisition flags pixels
    *dead*: far **below** the flat frame's median by the same robust-sigma test
    (a dead pixel doesn't respond to light, so it reads low under flat
    illumination even though it may look unremarkable in the dark).
    
    Args:
        camera: A :class:`camerasim.camera.Camera` instance.
        n_frames: Frames per acquisition (dark, and flat if requested).
        sigma: Robust-sigma outlier threshold.
        flat_level: Optional flux level (photons/pixel/s) for the dead-pixel extension.
    
    Returns:
        A :class:`BadPixelResult` with ``bad_pixel_mask`` (True = bad) and ``meta``.
    
    Raises:
        ValueError: if ``n_frames < 2``.
    """
    if n_frames < 2:
        raise ValueError(f"n_frames must be >= 2; got {n_frames}")

    def _robust_outliers(x: torch.Tensor) -> torch.Tensor:
        med = x.median()
        scale = (1.4826 * (x - med).abs().median()).clamp_min(1e-12)
        return ((x - med).abs() / scale) > sigma

    was_open = camera.shutter_open
    camera.set_shutter(False)
    try:
        dark_stack = torch.stack([camera.grab().pixels for _ in range(n_frames)], dim=0)
    finally:
        camera.set_shutter(was_open)
    dark_mean = dark_stack.mean(dim=0)
    dark_std = dark_stack.std(dim=0)
    bad = _robust_outliers(dark_mean) | _robust_outliers(dark_std)
    n_from_darks = int(bad.sum().item())

    n_from_flat = 0
    if flat_level is not None:
        camera.set_flux(float(flat_level))
        camera.set_shutter(True)
        try:
            flat_mean = torch.stack([camera.grab().pixels for _ in range(n_frames)], dim=0).mean(dim=0)
        finally:
            camera.set_shutter(was_open)
        med = flat_mean.median()
        scale = (1.4826 * (flat_mean - med).abs().median()).clamp_min(1e-12)
        dead = (med - flat_mean) / scale > sigma  # far BELOW median -> dead
        n_from_flat = int((dead & ~bad).sum().item())
        bad = bad | dead

    meta = {
        "n_frames": n_frames,
        "sigma": sigma,
        "n_bad_total": int(bad.sum().item()),
        "n_bad_from_darks": n_from_darks,
        "n_bad_from_flat": n_from_flat,
    }
    return BadPixelResult(bad_pixel_mask=bad, meta=meta)


def choose_exposure(
    camera,
    *,
    target_fill: float = 0.5,
    tol: float = 0.05,
    n_frames: int = 4,
    max_iter: int = 12,
    min_exposure_s: float = 1e-6,
    max_exposure_s: float = 10.0,
    full_well_adu: float | None = None,
) -> float:
    """Auto-exposure: set the exposure so the brightest pixel sits at
    ``target_fill`` of full well.
    
    Real benches solve saturation with exposure control, so the twin does
    too. This is the routine that lets bench scripts and tests **stop
    hand-picking photon rates** -- the actual root cause behind saturated-parity
    incidents.
    
    Peak *signal* (above the dark floor) is linear in exposure below the rail,
    so every unsaturated grab predicts the exposure that hits the target and
    the routine converges in a few iterations. A grab that comes back
    saturated (any pixel at the ceiling) carries no usable slope, so the
    routine backs off geometrically (halves the exposure) until it clears the
    rail before resuming the linear step -- it never extrapolates through a
    clip.
    
    Full-well ceiling defaults to the ADC ceiling ``camera.adu_max`` (the twin
    clamps in the ADU domain only). Fill is peak signal over the *usable* range
    ``ceiling - dark_floor``, so the bias offset never counts as fill; the
    per-pixel dark map is acquired fresh (shutter-closed grab), which also
    cancels a hot pixel that would otherwise masquerade as the peak.
    
    Restores the shutter state and **leaves the camera at the chosen exposure**
    (the point of the routine).
    
    Args:
        camera: A :class:`camerasim.camera.Camera` instance.
        target_fill: Peak-signal fraction of usable full well to aim for, in
            ``(0, 1)``. 0.5 (half full well) is the bench default -- headroom
            for shot-noise excursions without wasting dynamic range.
        tol: Convergence tolerance on the achieved fill fraction.
        n_frames: Frames averaged per iteration (denoises the peak estimate;
            saturation is still detected on any single frame).
        max_iter: Maximum grab/adjust iterations.
        min_exposure_s, max_exposure_s: Hard exposure bracket. The routine
            clamps into it and raises if the target cannot be met inside it.
        full_well_adu: Override the full-well ceiling (else ADC max).
    
    Returns:
        The chosen exposure time (seconds), already set on the camera.
    
    Raises:
        ValueError: for a ``target_fill`` outside ``(0, 1)``, a degenerate
            bracket, ``n_frames``/``max_iter`` < 1, or a target unreachable
            inside the exposure bracket.
    """
    if not (0.0 < target_fill < 1.0):
        raise ValueError(f"target_fill must be in (0, 1); got {target_fill}")
    if not (0.0 < min_exposure_s < max_exposure_s):
        raise ValueError(
            f"need 0 < min_exposure_s < max_exposure_s; got {min_exposure_s}, {max_exposure_s}"
        )
    if n_frames < 1:
        raise ValueError(f"n_frames must be >= 1; got {n_frames}")
    if max_iter < 1:
        raise ValueError(f"max_iter must be >= 1; got {max_iter}")

    ceiling = float(full_well_adu) if full_well_adu is not None else float(camera.adu_max)

    # Per-pixel dark floor: grab one shutter-closed frame here
    was_open = camera.shutter_open
    camera.set_shutter(False)
    try:
        dark_map = camera.grab().pixels
    finally:
        camera.set_shutter(was_open)
    dark_floor = float(dark_map.median())
    usable = ceiling - dark_floor
    if usable <= 0.0:
        raise ValueError(
            f"dark floor {dark_floor:.1f} ADU leaves no headroom below full well {ceiling:.1f}"
        )
    target_signal = target_fill * usable

    was_exposure = camera.exposure_s
    exposure = min(max(was_exposure, min_exposure_s), max_exposure_s)
    best = None  # (abs_fill_error, exposure, fill)
    converged = False
    camera.set_shutter(was_open)
    try:
        for _ in range(max_iter):
            camera.set_exposure(exposure)
            frames = [camera.grab() for _ in range(n_frames)]
            peak_raw = max(float(f.pixels.max()) for f in frames)
            saturated = peak_raw >= ceiling
            if saturated:
                if exposure <= min_exposure_s:
                    raise ValueError(
                        f"source saturates even at min_exposure_s={min_exposure_s} s "
                        f"(peak {peak_raw:.0f} >= full well {ceiling:.0f}); reduce the flux"
                    )
                exposure = max(exposure * 0.5, min_exposure_s)
                continue
            mean_px = torch.stack([f.pixels for f in frames], dim=0).mean(dim=0)
            peak_signal = float((mean_px - dark_map).max())
            fill = peak_signal / usable
            err = abs(fill - target_fill)
            if best is None or err < best[0]:
                best = (err, exposure, fill)
            if err <= tol:
                converged = True
                break
            if peak_signal <= 0.0:
                # No measurable signal yet -- jump to the top of the bracket
                exposure = max_exposure_s
                continue
            predicted = exposure * target_signal / peak_signal
            exposure = min(max(predicted, min_exposure_s), max_exposure_s)
    finally:
        camera.set_shutter(was_open)

    if best is None:
        camera.set_exposure(was_exposure)
        raise ValueError(
            "could not find any unsaturated exposure in "
            f"[{min_exposure_s}, {max_exposure_s}] s; check the source/bracket"
        )
    err, exposure, fill = best
    if not converged and err > tol:
        camera.set_exposure(was_exposure)
        raise ValueError(
            f"target fill {target_fill:.2f} unreachable within exposure bracket "
            f"[{min_exposure_s}, {max_exposure_s}] s (best fill {fill:.3f} at "
            f"{exposure:.3e} s) -- the source is too dim or too bright for this range; "
            f"adjust the flux, never extrapolating"
        )

    camera.set_exposure(exposure)
    return float(exposure)
