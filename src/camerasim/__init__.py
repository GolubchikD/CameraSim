"""CameraSim: detector chain + PSF + photometry bridge, shared across projects."""

from camerasim import psf
from camerasim.detector import DetectorModel
from camerasim.photometry import electrons_from_intensity

__all__ = ["DetectorModel", "electrons_from_intensity", "psf"]
__version__ = "0.1.0"
