"""Photometric bridge: convert a relative |E|^2 intensity map to mean electrons.

The wave-optics half of the pipeline (e.g. Phase-reconstructor's coherent
propagator) returns dimensionless ``|E|^2`` — a relative irradiance map.
:func:`electrons_from_intensity` rescales it by the known source photon flux,
applies QE and exposure, and hands the mean-electron map to
:class:`camerasim.detector.DetectorModel`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import _backend


def electrons_from_intensity(
    intensity: Any,
    qe: float,
    exposure_s: float,
    pixel_area: float,
    source_power: float,
) -> Any:
    """Convert relative ``|E|^2`` to per-pixel mean electron rate.

    The conversion treats ``intensity`` as relative irradiance on the
    detector grid and rescales so the total optical power across the grid
    equals ``source_power``. ``pixel_area`` enters the discrete-to-continuous
    integration; it cancels with the normalisation sum, so any consistent
    positive value gives the same answer — it is kept in the signature for
    unit transparency.

    Parameters
    ----------
    intensity : array, ``(..., H, W)``
        Non-negative relative intensity. Accepts numpy or torch.
    qe : float in [0, 1]
        Quantum efficiency, electrons per incident photon.
    exposure_s : float
        Integration time per frame, in seconds.
    pixel_area : float
        Pixel collection area. Any consistent unit (e.g. m^2).
    source_power : float
        Total photon flux across the imaged scene, in photons / second.
        The output is rescaled so summed photons/s equals ``source_power``.

    Returns
    -------
    electrons : same shape and framework as ``intensity``
        Mean per-pixel electron rate, ready for Poisson sampling by
        :meth:`camerasim.detector.DetectorModel.expose`.
    """
    if qe < 0.0 or qe > 1.0:
        raise ValueError(f"qe must be in [0, 1]; got {qe}.")
    if exposure_s <= 0.0:
        raise ValueError(f"exposure_s must be > 0; got {exposure_s}.")
    if pixel_area <= 0.0:
        raise ValueError(f"pixel_area must be > 0; got {pixel_area}.")
    if source_power < 0.0:
        raise ValueError(f"source_power must be >= 0; got {source_power}.")

    intensity_arr, meta = _backend.to_numpy(intensity)
    intensity_arr = intensity_arr.astype(np.float64)
    if np.any(intensity_arr < 0):
        raise ValueError("intensity must be non-negative.")
    total = float(intensity_arr.sum()) * pixel_area
    if total <= 0.0:
        raise ValueError("intensity is identically zero; cannot normalise.")

    per_pixel_power = intensity_arr * pixel_area * (source_power / total)
    electrons = qe * exposure_s * per_pixel_power
    return _backend.from_numpy(electrons, meta)
