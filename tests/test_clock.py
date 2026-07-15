"""Tests for Clock implementations."""

import pytest

from camerasim.clock import Clock, VirtualClock, WallClock


def test_virtual_clock_advances():
    """VirtualClock.sleep() advances time instantly."""
    clock = VirtualClock(t0=0.0)
    assert clock.now() == 0.0

    clock.sleep(1.5)
    assert clock.now() == 1.5

    clock.sleep(0.5)
    assert clock.now() == 2.0


def test_virtual_clock_advance_alias():
    """VirtualClock.advance() is an alias of sleep()."""
    clock = VirtualClock(t0=0.0)
    clock.advance(3.0)
    assert clock.now() == 3.0


def test_virtual_clock_sleep_until():
    """VirtualClock.sleep_until() advances to the specified time."""
    clock = VirtualClock(t0=0.0)
    clock.sleep_until(5.0)
    assert clock.now() == 5.0

    # sleep_until in the past does nothing
    clock.sleep_until(3.0)
    assert clock.now() == 5.0


def test_virtual_clock_negative_sleep_raises():
    """VirtualClock.sleep() with negative dt raises ValueError."""
    clock = VirtualClock()
    with pytest.raises(ValueError, match="cannot sleep a negative duration"):
        clock.sleep(-1.0)


def test_wall_clock_sleep_until():
    """WallClock.sleep_until() waits until the specified time."""
    clock = WallClock()
    t0 = clock.now()
    target = t0 + 0.01  # 10 ms in the future
    clock.sleep_until(target)
    t1 = clock.now()
    # Should have waited ~10 ms (allow some slop for OS scheduling)
    assert t1 >= target
    assert t1 < target + 0.1  # But not too much longer


def test_wall_clock_negative_sleep_raises():
    """WallClock.sleep() with negative dt raises ValueError."""
    clock = WallClock()
    with pytest.raises(ValueError, match="cannot sleep a negative duration"):
        clock.sleep(-0.1)


def test_clock_protocol_check():
    """VirtualClock and WallClock satisfy the Clock Protocol."""
    virtual = VirtualClock()
    wall = WallClock()

    assert isinstance(virtual, Clock), "VirtualClock should satisfy Clock Protocol"
    assert isinstance(wall, Clock), "WallClock should satisfy Clock Protocol"


def test_clock_protocol_methods():
    """Clock Protocol has the required methods."""
    clock = VirtualClock()

    # Protocol methods exist and are callable
    assert callable(clock.now)
    assert callable(clock.sleep)
    assert callable(clock.sleep_until)

    # Basic smoke test
    t0 = clock.now()
    clock.sleep(1.0)
    t1 = clock.now()
    assert t1 == t0 + 1.0
