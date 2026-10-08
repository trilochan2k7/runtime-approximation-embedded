"""Simulated resource model.

Because this project is a software architectural prototype rather than physical
hardware, the resource state is **simulated**.  It is expressed as a continuous
*pressure* value in ``[0, 1]``:

* ``0.0`` -> resources abundant (ample energy budget, low processor
  utilisation, cool).
* ``1.0`` -> resources scarce (low remaining energy, high utilisation, thermal
  throttling).

A trace generator produces a (deterministic, seeded) non-stationary pressure
sequence over the block stream so that runtime adaptation has a changing
environment to react to.  These are simulated conditions and are never claimed
to be hardware power measurements.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from config import Config, pressure_to_level

__all__ = ["ResourceTrace", "make_resource_trace", "constant_trace"]


@dataclass(frozen=True)
class ResourceTrace:
    """A per-block sequence of simulated resource pressures in ``[0, 1]``."""

    pressures: np.ndarray  # shape (n_blocks,)

    def __len__(self) -> int:
        return int(self.pressures.shape[0])

    def pressure(self, block_index: int) -> float:
        """Pressure for a block, clamping the index into range."""

        idx = int(np.clip(block_index, 0, len(self) - 1))
        return float(self.pressures[idx])

    def level(self, block_index: int) -> str:
        """Coarse LOW/MEDIUM/HIGH label for a block's pressure."""

        return pressure_to_level(self.pressure(block_index))

    def mean_pressure(self) -> float:
        return float(np.mean(self.pressures))


def constant_trace(pressure: float, n_blocks: int) -> ResourceTrace:
    """A flat trace at a fixed pressure (used for the static LOW/MED/HIGH runs)."""

    p = float(np.clip(pressure, 0.0, 1.0))
    return ResourceTrace(pressures=np.full(n_blocks, p, dtype=np.float64))


def make_resource_trace(cfg: Config, n_blocks: int, seed: int) -> ResourceTrace:
    """Generate a resource-pressure trace according to :class:`ResourceConfig`.

    Supported ``trace_kind`` values:

    * ``"constant"``    - flat at ``base_pressure``.
    * ``"oscillating"`` - sinusoid of the configured ``period_blocks`` and
      ``amplitude`` about ``base_pressure`` with a little noise (models periodic
      load / duty-cycled energy harvesting).
    * ``"rising"``      - a ramp from low to high pressure (models a draining
      battery over the run).
    * ``"random"``      - i.i.d. uniform pressure per block.

    All variants are clamped to ``[0, 1]`` and are reproducible for a given seed.
    """

    rc = cfg.resource
    rng = np.random.default_rng(seed)
    t = np.arange(n_blocks, dtype=np.float64)

    if rc.trace_kind == "constant":
        base = np.full(n_blocks, rc.base_pressure, dtype=np.float64)
    elif rc.trace_kind == "rising":
        base = np.linspace(0.1, 0.9, n_blocks)
    elif rc.trace_kind == "random":
        base = rng.uniform(0.0, 1.0, size=n_blocks)
    else:  # oscillating (default)
        period = max(rc.period_blocks, 1)
        base = rc.base_pressure + rc.amplitude * np.sin(2.0 * np.pi * t / period)

    noise = rng.normal(0.0, rc.noise_std, size=n_blocks)
    pressures = np.clip(base + noise, 0.0, 1.0)
    return ResourceTrace(pressures=pressures.astype(np.float64))
