"""Numpy<->torch bridge so the public API accepts and returns either kind transparently.

Internals are pure numpy/scipy. Anywhere the public API meets the user, we
strip torch tensors to numpy with :func:`to_numpy`, do the work, then
round-trip via :func:`from_numpy` so torch callers get a tensor back on the
original device.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:  # torch is an optional dependency.
    import torch as _torch  # type: ignore
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised in torch-less environments
    _torch = None
    _TORCH_AVAILABLE = False


def is_torch_tensor(x: Any) -> bool:
    return _TORCH_AVAILABLE and isinstance(x, _torch.Tensor)


def to_numpy(x: Any) -> tuple[np.ndarray, dict]:
    """Convert input to a contiguous numpy array, returning round-trip metadata.

    The metadata dict carries ``is_torch`` and (when applicable) ``device``,
    which :func:`from_numpy` uses to put the result back on the original
    framework / device.
    """
    if is_torch_tensor(x):
        arr = x.detach().cpu().numpy()
        return arr, {"is_torch": True, "device": x.device}
    return np.asarray(x), {"is_torch": False}


def from_numpy(arr: np.ndarray, meta: dict) -> Any:
    """Round-trip a numpy result back to torch when the input was torch."""
    if not meta.get("is_torch", False):
        return arr
    # torch < 2.3 cannot accept uint16 / uint32 via from_numpy; widen safely.
    if arr.dtype == np.uint16:
        arr = arr.astype(np.int32)
    elif arr.dtype == np.uint32:
        arr = arr.astype(np.int64)
    t = _torch.from_numpy(np.ascontiguousarray(arr))
    device = meta.get("device")
    if device is not None:
        t = t.to(device)
    return t
