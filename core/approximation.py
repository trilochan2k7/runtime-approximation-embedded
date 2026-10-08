"""Approximation engine: reduced-precision (truncated-mantissa) arithmetic.

Technique
---------
All three workloads reduce to **multiply-accumulate (MAC)** operations.  The
approximation applied by ``APPROX-1`` and ``APPROX-2`` is *operand precision
reduction*: before each multiply, the operands are rounded to a reduced number
of significand (mantissa) bits, exactly as a reduced-precision / truncated
hardware multiplier would represent them.  Accumulation is kept in full
precision, mirroring the standard embedded "narrow multiplier, wide
accumulator" MAC datapath (e.g. 8-bit multiply with 32-bit accumulate used by
DSPs and ML accelerators).

Why this saves resources
-------------------------
The dynamic energy and area of a multiplier grow roughly with the square of the
operand bit-width, so narrowing the significand from the exact width to 10 or 6
bits reduces the modelled per-MAC energy substantially (see
:class:`config.CostModelConfig`).  Latency (datapath critical path) shrinks
roughly linearly with width.

How it affects accuracy
-----------------------
Rounding a normalised mantissa (magnitude in ``[0.5, 1)``) to ``b`` bits gives a
*relative* per-operand error of at most ``2**-b``.  How that per-operand error
propagates to the output depends on the computation's conditioning (how much
cancellation occurs in the accumulation); that propagation is exactly what the
sensitivity estimator probes (see :mod:`core.sensitivity`).

Everything here is deterministic and vectorised with NumPy.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

__all__ = ["quantize_significand", "approx_matmul", "approx_dot_blocks"]


def quantize_significand(x: np.ndarray, bits: Optional[int]) -> np.ndarray:
    """Round ``x`` to ``bits`` significand bits (float mantissa truncation).

    Parameters
    ----------
    x:
        Array of real values (any shape).
    bits:
        Number of mantissa bits to retain.  ``None`` returns ``x`` unchanged
        (the exact reference).  ``bits`` must be positive otherwise.

    Returns
    -------
    numpy.ndarray
        A new ``float64`` array whose every element has been rounded to the
        requested precision.

    Notes
    -----
    A finite nonzero value is written as ``x = m * 2**e`` with ``m`` in
    ``[0.5, 1)`` (``numpy.frexp``).  We round ``m`` to the grid of step
    ``2**-(bits+1)`` and reassemble with ``numpy.ldexp``.  Zeros, infinities and
    NaNs pass through unchanged because ``frexp``/``ldexp`` preserve them.
    """

    if bits is None:
        return np.asarray(x, dtype=np.float64)
    if bits <= 0:
        raise ValueError("bits must be a positive integer or None")

    arr = np.asarray(x, dtype=np.float64)
    mant, exp = np.frexp(arr)          # arr == mant * 2**exp, |mant| in [0.5, 1)
    scale = float(2 ** bits)
    mant_q = np.round(mant * scale) / scale
    return np.ldexp(mant_q, exp)


def approx_matmul(a: np.ndarray, b: np.ndarray, bits: Optional[int]) -> np.ndarray:
    """Matrix product ``a @ b`` with operands quantised to ``bits`` significand bits.

    The multiplier operands (both matrices) are reduced to the requested
    precision; the accumulation (the sum over the shared dimension) is performed
    in full ``float64`` precision, modelling a wide accumulator.
    """

    aq = quantize_significand(a, bits)
    bq = quantize_significand(b, bits)
    return aq @ bq


def approx_dot_blocks(
    taps: np.ndarray, windows: np.ndarray, bits: Optional[int]
) -> np.ndarray:
    """Approximate FIR/dot computation.

    Parameters
    ----------
    taps:
        1-D filter coefficients of length ``T``.
    windows:
        2-D array of shape ``(N, T)``; row ``i`` holds the ``T`` input samples
        aligned with output ``i``.
    bits:
        Operand quantisation width (``None`` for exact).

    Returns
    -------
    numpy.ndarray
        Length-``N`` output where ``y[i] = sum_t taps[t] * windows[i, t]`` with
        the multiply operands quantised and the sum accumulated in full
        precision.
    """

    taps_q = quantize_significand(taps, bits)
    win_q = quantize_significand(windows, bits)
    return win_q @ taps_q
