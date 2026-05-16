"""Reference :class:`~camerasim.DetectorModel` presets for real cameras.

Each factory returns a fully-configured :class:`DetectorModel` calibrated
to published vendor specs. Citations are in each docstring. Values that
vendors rarely publish per model (PRNU, DSNU, hot/dead fractions, bias
pedestal) use EMVA 1288 typical ranges; override via ``**kwargs`` when
better data is available.

Each preset takes ``exposure_s`` so dark current scales sensibly with
integration time; ``background_e`` is left at 0 because sky / scene
background is a property of the imaged scene, not the sensor.
"""

from __future__ import annotations

from camerasim import DetectorModel


def kaya_iron_cxp_253(exposure_s: float = 1e-3, **overrides) -> DetectorModel:
    """KAYA Instruments Iron CXP 253 -- industrial CoaXPress camera.

    Sensor: Sony IMX253LLR Pregius global-shutter CMOS, 4096 x 3000,
    3.45 um pixel, 12-bit, up to 63.8 fps at full frame.

    Specs:
      - Full well (saturation capacity): 32,500 e-  (Basler IMX253 white paper)
      - Dynamic range: 73.6 dB  =>  derived read noise ~6.8 e-
      - Peak QE: ~70 % (Sony Pregius typical)

    Sources:
      - https://kaya.vision/product/iron-cxp-253-camera/
      - https://www.sodavision.com/wp-content/uploads/2018/01/basler-sensor-comparison-are-all-imx-sensors-equal.pdf
    """
    full_well_e = 32_500.0
    dark_e_per_s = 150.0  # typical uncooled Pregius at ~25 C
    params = dict(
        psf_sigma_px=0.7,
        background_e=0.0,
        dark_current_e=dark_e_per_s * exposure_s,
        read_noise_e=6.8,
        prnu_sigma=0.010,
        dsnu_sigma_e=1.0,
        full_well_e=full_well_e,
        gain_e_per_adu=full_well_e / (2**12 - 1),  # fit full well into 12 bits
        bits=12,
        bias_adu=25.0,
        hot_pixel_fraction=1e-4,
        dead_pixel_fraction=1e-5,
        rng_seed=0,
    )
    params.update(overrides)
    return DetectorModel(**params)


def phantom_miro_c211(exposure_s: float = 5e-4, **overrides) -> DetectorModel:
    """Phantom Miro C211 -- high-speed industrial CMOS, 1.3 MP @ 1800 fps.

    Sensor: 1280 x 1024, 5.6 um pixel, 12-bit.

    Specs (EMVA 1288 report, Standard Mode, 532 nm color path):
      - Peak QE: 41.7 %
      - Read noise: 9.08 e-
      - Full well (saturation): 6,972 e-
      - Dynamic range: 57.7 dB
      - SNR_max: 38.7 dB

    Trades sensitivity for speed -- short integration, modest full well,
    higher read noise than scientific or cooled cameras.

    Sources:
      - https://www.phantomhighspeed.com/products/cameras/mirocnn/c211
      - https://www.phantomhighspeed.com/-/media/project/ameteksxa/visionresearch/documents/datasheets/web/miroc211_emva1288-report.pdf
    """
    full_well_e = 6_972.0
    dark_e_per_s = 200.0  # uncooled, ambient (vendor does not publish)
    params = dict(
        psf_sigma_px=0.8,
        background_e=0.0,
        dark_current_e=dark_e_per_s * exposure_s,
        read_noise_e=9.08,
        prnu_sigma=0.015,
        dsnu_sigma_e=2.0,
        full_well_e=full_well_e,
        gain_e_per_adu=full_well_e / (2**12 - 1),
        bits=12,
        bias_adu=64.0,
        hot_pixel_fraction=2e-4,
        dead_pixel_fraction=5e-5,
        rng_seed=0,
    )
    params.update(overrides)
    return DetectorModel(**params)


def basler_ace_aca1300_200um(exposure_s: float = 1e-3, **overrides) -> DetectorModel:
    """Basler ace acA1300-200um -- general-purpose USB3 machine-vision camera.

    Sensor: onsemi PYTHON 1300 global-shutter CMOS, 1280 x 1024,
    4.8 um pixel, 8/12-bit selectable, up to 203 fps.

    Specs (Basler EMVA 1288 datasheet, peak QE near 560 nm):
      - Read noise: ~6.83 e-
      - Full well (saturation): ~10,200 e-
      - Dynamic range: ~57 dB
      - Peak QE: ~52 %

    Sources:
      - https://www.baslerweb.com/en/products/cameras/area-scan-cameras/ace/aca1300-200um/
      - https://www.onsemi.com/products/sensors/image-sensors/python1300
    """
    full_well_e = 10_200.0
    dark_e_per_s = 150.0
    params = dict(
        psf_sigma_px=0.6,
        background_e=0.0,
        dark_current_e=dark_e_per_s * exposure_s,
        read_noise_e=6.83,
        prnu_sigma=0.012,
        dsnu_sigma_e=1.5,
        full_well_e=full_well_e,
        gain_e_per_adu=full_well_e / (2**12 - 1),
        bits=12,
        bias_adu=32.0,
        hot_pixel_fraction=1e-4,
        dead_pixel_fraction=1e-5,
        rng_seed=0,
    )
    params.update(overrides)
    return DetectorModel(**params)


PRESETS = {
    "kaya_iron_cxp_253": kaya_iron_cxp_253,
    "phantom_miro_c211": phantom_miro_c211,
    "basler_ace_aca1300_200um": basler_ace_aca1300_200um,
}
