"""Side-by-side demo of the reference camera presets.

Runs the same synthetic scene through three real-world camera models and
saves a comparison PNG showing how the detector chain (noise, saturation,
quantisation) differs between them.

Usage::

    pip install matplotlib
    python examples/demo.py

Writes ``examples/output/comparison.png``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    raise SystemExit(
        "This demo requires matplotlib. Install with `pip install matplotlib`."
    )

# Make `from cameras import ...` work whether run as `python examples/demo.py`
# or `python -m examples.demo` from the repo root.
sys.path.insert(0, str(Path(__file__).parent))
from cameras import (  # noqa: E402  (intentional after sys.path tweak)
    basler_ace_aca1300_200um,
    kaya_iron_cxp_253,
    phantom_miro_c211,
)

from camerasim import electrons_from_intensity  # noqa: E402


OUT_DIR = Path(__file__).parent / "output"


def synthetic_intensity(h: int = 256, w: int = 256) -> np.ndarray:
    """A test scene: bright central Gaussian spot on a horizontal ramp."""
    y, x = np.mgrid[:h, :w].astype(np.float64)
    cy, cx = h / 2.0, w / 2.0
    spot = 12.0 * np.exp(-((y - cy) ** 2 + (x - cx) ** 2) / (2.0 * 14.0 ** 2))
    ramp = 0.05 + 0.5 * x / (w - 1)
    return spot + ramp


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    intensity = synthetic_intensity()

    # Photometric scene definition: enough photons to nudge the brighter
    # cameras near saturation so the differences are visible.
    mean_e = electrons_from_intensity(
        intensity,
        qe=0.6,
        exposure_s=1e-3,
        pixel_area=(5e-6) ** 2,
        source_power=2e10,
    )

    presets = [
        ("KAYA Iron CXP 253", kaya_iron_cxp_253()),
        ("Phantom Miro C211", phantom_miro_c211()),
        ("Basler ace acA1300-200um", basler_ace_aca1300_200um()),
    ]

    fig, axes = plt.subplots(1, len(presets), figsize=(4.5 * len(presets), 4.6))
    for ax, (name, cam) in zip(axes, presets):
        adu = cam.expose(mean_e)
        im = ax.imshow(adu, cmap="gray")
        subtitle = (
            f"{cam.bits}-bit | FWC={cam.full_well_e:.0f} e-\n"
            f"RN={cam.read_noise_e:.2f} e- | "
            f"gain={cam.gain_e_per_adu:.2f} e-/ADU"
        )
        ax.set_title(f"{name}\n{subtitle}", fontsize=9)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, label="ADU")

    fig.suptitle("Same scene through three real cameras", fontsize=12)
    fig.tight_layout()
    out = OUT_DIR / "comparison.png"
    fig.savefig(out, dpi=140)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
