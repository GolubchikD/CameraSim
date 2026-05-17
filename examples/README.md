# Examples

`cameras.py` — preset factories for three real cameras with parameters
sourced from manufacturer datasheets and EMVA 1288 reports:

| Preset | Sensor | Read noise | Full well | Bit depth | Source class |
|---|---|---|---|---|---|
| `kaya_iron_cxp_253()` | Sony IMX253 Pregius | 6.8 e⁻ | 32,500 e⁻ | 12 | Industrial CoaXPress (global shutter) |
| `kaya_iron_2020bsi()` | Gpixel GSENSE2020BSI | 1.6 e⁻ | 55,000 e⁻ | 12 | Scientific BSI sCMOS (rolling shutter, 95% QE) |
| `phantom_miro_c211()` | Custom CMOS | 9.08 e⁻ | 6,972 e⁻ | 12 | High-speed (1800 fps) |
| `basler_ace_aca1300_200um()` | onsemi PYTHON 1300 | 6.83 e⁻ | 10,200 e⁻ | 12 | General machine-vision |

Each takes an optional `exposure_s` and `**overrides` so you can swap
individual fields. Citations are in the docstrings.

## demo.py

Runs the same synthetic scene through each preset and writes
`examples/output/comparison.png`.

```bash
pip install matplotlib
python examples/demo.py
```
