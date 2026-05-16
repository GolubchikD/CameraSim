# CameraSim

Shared camera-detector simulator extracted from
[Mestechko](https://github.com/GolubchikD/Mestechko) and consumed by
[Phase-reconstructor](https://github.com/GolubchikD/Phase-reconstructor)
and TurbImage. MIT-licensed so MIT consumers can depend on it without
the GPL infection that would come from importing Mestechko directly.

The package has three pieces:

- **`DetectorModel`** — the full imaging chain (PSF, background, dark,
  PRNU, Poisson, DSNU, read noise, hot/dead, full-well clip, gain/bias,
  ADC quantise). Accepts a single frame `(H, W)` or a stack `(N, H, W)`,
  numpy or torch (tensor in -> tensor out on the original device).
- **`psf`** — standalone Gaussian, Airy, and Zernike-aberrated PSF
  kernel generators, all returning numpy arrays summing to 1.
- **`electrons_from_intensity`** — convert the dimensionless `|E|^2`
  output of a wave-optics propagator to a per-pixel mean electron rate
  ready for the detector.

## Install

```bash
pip install git+https://github.com/GolubchikD/CameraSim.git@v0.1.0
```

Optional torch bridge: `pip install "camerasim[torch]"`.

## Quick start

```python
import numpy as np
from camerasim import DetectorModel, psf, electrons_from_intensity

intensity = np.exp(-np.linspace(-2, 2, 64)[:, None] ** 2
                   -np.linspace(-2, 2, 64)[None, :] ** 2)
mean_e = electrons_from_intensity(
    intensity, qe=0.6, exposure_s=1e-3, pixel_area=(5e-6) ** 2,
    source_power=1e9,  # photons / s
)

cam = DetectorModel(
    psf_kernel=psf.gaussian(11, sigma_px=0.9),
    rng_seed=0,
)
adu = cam.expose(mean_e)            # (64, 64) uint16
stack = cam.expose(np.stack([mean_e] * 8))  # (8, 64, 64) uint16
```

## Testing

```bash
pip install -e ".[test]"
pytest
```

## License

MIT. The detector pipeline is ported from
[Mestechko](https://github.com/GolubchikD/Mestechko) (GPL-3.0) by the
same author, who relicenses this extraction as MIT.
