"""CameraSim: detector chain + PSF + photometry bridge, shared across projects.

The base package (``import camerasim``) is torch-free and provides:
- :class:`DetectorModel` — full detector chain (numpy/torch transparent)
- :mod:`camerasim.psf` — PSF utilities
- :func:`electrons_from_intensity` — photometry conversion

Torch-requiring submodules (import explicitly):
- :mod:`camerasim.camera` — :class:`Camera`, :class:`CameraConfig` (exposure core)
- :mod:`camerasim.contracts` — :class:`RawFrame`, :class:`FrameMeta`, etc.
- :mod:`camerasim.clock` — :class:`Clock` Protocol, :class:`VirtualClock`, :class:`WallClock`
- :mod:`camerasim.calibration` — Sensor-agnostic detector characterization calibration
  routines (:func:`measure_dark`, :func:`measure_gain_photon_transfer`,
  :func:`measure_dark_current`, :func:`measure_linearity`,
  :func:`measure_linearity_autorange`, :func:`build_bad_pixel_mask`,
  :func:`choose_exposure`)

These submodules require torch and are imported explicitly rather than at
package level (e.g., ``from camerasim.camera import Camera``).
"""

from camerasim import psf
from camerasim.detector import DetectorModel
from camerasim.photometry import electrons_from_intensity

__all__ = ["DetectorModel", "electrons_from_intensity", "psf"]
__version__ = "0.3.0.dev0"
