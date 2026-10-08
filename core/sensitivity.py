"""Lightweight, explainable output-sensitivity estimator.

What "sensitivity" means here
-----------------------------
Operand rounding introduces a bounded *relative* perturbation into each
multiply (at most ``2**-b`` for ``b`` retained significand bits).  How strongly
that per-operand perturbation shows up in the *output* depends on the
computation's conditioning - chiefly, how much cancellation occurs in the
accumulation.  A block whose output is a sum with little cancellation passes
rounding error through roughly linearly; a block whose output nearly cancels
amplifies it.

The estimator therefore reports an **error-amplification factor** ``A``:

    A = (observed output relative error at the probe precision)
        / (per-operand relative unit at the probe precision)

``A ~ 1`` means errors propagate linearly, ``A >> 1`` means the block is
ill-conditioned / highly sensitive, and ``A < 1`` means it is robust.  This is a
cheap, finite-difference style conditioning estimate - explicitly a **heuristic**,
not a universally optimal sensitivity metric.

Why it is cheap enough for runtime use
--------------------------------------
The probe quantises only a small representative sub-block (``probe_elements``
output cells) and compares it with the exact sub-block.  Its MAC count is
reported so the controller can charge the probe against the adaptation-overhead
budget (hypothesis H3).

How the controller uses it
--------------------------
Because output relative error is, to first order, linear in the per-operand
unit, the predicted relative error of a mode with ``b`` significand bits is

    predicted_error(b) = A * 2**-b          (and 0 for the exact mode).

This closes the loop: the controller can pick the cheapest mode whose predicted
error stays within the application budget (see :mod:`core.controller`).  The
first-order linearity is accurate for mild approximation and degrades for very
aggressive rounding, which is one reason the controller keeps a safety margin.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from config import Config
from core.metrics import relative_error
from core.workloads import Block, Workload

__all__ = ["SensitivityResult", "estimate_sensitivity", "normalise_amplification"]


@dataclass(frozen=True)
class SensitivityResult:
    """Result of probing one block for output sensitivity.

    Attributes
    ----------
    amplification:
        Raw error-amplification factor ``A`` (``>= 0``, unclipped).  Consumed by
        the controller to predict per-mode error.
    score:
        Amplification mapped to a normalised ``[0, 1]`` sensitivity score for
        display and the decision map (0 = robust, 1 = highly sensitive).
    observed_rel_error:
        L2 relative error measured on the probe sub-block at the probe width.
    probe_bits:
        Significand width used for the probe.
    probe_macs:
        MACs performed by the probe (one exact + one approximate pass over the
        sub-block => ``2 * sub_block_macs``).  Charged to adaptation overhead.
    wall_time_s:
        Measured wall-clock time of the probe (informational; software only).
    """

    amplification: float
    score: float
    observed_rel_error: float
    probe_bits: int
    probe_macs: int
    wall_time_s: float


def normalise_amplification(amp: float, cfg: Config) -> float:
    """Map a raw amplification factor to a ``[0, 1]`` score on a log scale.

    Amplification spans orders of magnitude, so we interpolate between
    ``amp_floor`` and ``amp_ceil`` in ``log2`` space and clip to ``[0, 1]``.
    """

    sc = cfg.sensitivity
    lo, hi = np.log2(sc.amp_floor), np.log2(sc.amp_ceil)
    val = np.log2(max(amp, 1e-12))
    return float(np.clip((val - lo) / (hi - lo), 0.0, 1.0))


def estimate_sensitivity(workload: Workload, block: Block, cfg: Config) -> SensitivityResult:
    """Probe ``block`` and return its sensitivity estimate.

    The probe quantises a small sub-block to ``cfg.sensitivity.probe_bits`` and
    measures the L2 relative output error, then divides by the per-operand unit
    ``2**-probe_bits`` to obtain the amplification factor.
    """

    sc = cfg.sensitivity
    t0 = time.perf_counter()
    probe = workload.probe(block, sc.probe_elements, sc.probe_bits)
    observed_rel = relative_error(probe.approx, probe.exact)
    wall = time.perf_counter() - t0

    unit = 2.0 ** (-sc.probe_bits)
    amplification = observed_rel / unit if unit > 0 else 0.0
    # Two passes over the sub-block (exact + approximate) were performed.
    probe_macs = 2 * probe.n_macs

    return SensitivityResult(
        amplification=float(amplification),
        score=normalise_amplification(amplification, cfg),
        observed_rel_error=float(observed_rel),
        probe_bits=sc.probe_bits,
        probe_macs=int(probe_macs),
        wall_time_s=float(wall),
    )
