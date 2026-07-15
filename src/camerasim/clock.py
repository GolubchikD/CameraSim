"""Clock backends: virtual and wall-clock time.

The virtual clock is what makes latency a *simulated observable*: exposure,
readout, DM settle and inference times all advance it, so telemetry measures
the loop budget in software instead of asserting it on a slide.
"""

from __future__ import annotations

import time
from typing import Protocol, runtime_checkable

__all__ = ["Clock", "VirtualClock", "WallClock"]


@runtime_checkable
class Clock(Protocol):
    """Monotonic bench clock. Virtual in sim, ``time.monotonic`` on metal."""

    def now(self) -> float:
        """Current time in seconds (only differences are meaningful)."""
        ...

    def sleep(self, dt: float) -> None:
        """Block (or, in sim, advance virtual time) for ``dt`` seconds."""
        ...

    def sleep_until(self, t: float) -> None:
        """Sleep until the specified time ``t``. Default implementation uses
        ``now()`` and ``sleep()``."""
        dt = t - self.now()
        if dt > 0.0:
            self.sleep(dt)


class VirtualClock:
    """Deterministic simulated time. ``sleep`` advances instantly."""

    def __init__(self, t0: float = 0.0) -> None:
        self._t = float(t0)

    def now(self) -> float:
        return self._t

    def sleep(self, dt: float) -> None:
        if dt < 0.0:
            raise ValueError(f"cannot sleep a negative duration ({dt})")
        self._t += dt

    def sleep_until(self, t: float) -> None:
        dt = t - self.now()
        if dt > 0.0:
            self.sleep(dt)

    def advance(self, dt: float) -> None:
        """Alias of :meth:`sleep` for device internals (exposure, readout)."""
        self.sleep(dt)


class WallClock:
    """Real time via ``time.monotonic`` / ``time.sleep``."""

    def now(self) -> float:
        return time.monotonic()

    def sleep(self, dt: float) -> None:
        if dt < 0.0:
            raise ValueError(f"cannot sleep a negative duration ({dt})")
        time.sleep(dt)

    def sleep_until(self, t: float) -> None:
        dt = t - self.now()
        if dt > 0.0:
            self.sleep(dt)
