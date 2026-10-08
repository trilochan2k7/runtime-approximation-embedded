"""Error and cost metrics computed from actual experiment outputs.

Nothing in this module fabricates values: every function takes real arrays or
real per-block accounting records and returns derived quantities.  Energy and
latency are *estimated* using the documented cost model (see
:class:`config.CostModelConfig`); they are never presented as hardware
measurements.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

__all__ = [
    "ErrorMetrics",
    "error_metrics",
    "relative_error",
]


@dataclass(frozen=True)
class ErrorMetrics:
    """Container for the standard error metrics of an approximate output."""

    mae: float          # mean absolute error
    mse: float          # mean squared error
    rmse: float         # root mean squared error
    max_abs_error: float
    mean_rel_error: float   # mean |approx - exact| / (|exact| + eps)
    max_rel_error: float
    norm_rel_error: float   # ||approx - exact|| / (||exact|| + eps)  (aggregate)

    def as_dict(self) -> Dict[str, float]:
        return {
            "mae": self.mae,
            "mse": self.mse,
            "rmse": self.rmse,
            "max_abs_error": self.max_abs_error,
            "mean_rel_error": self.mean_rel_error,
            "max_rel_error": self.max_rel_error,
            "norm_rel_error": self.norm_rel_error,
        }


def error_metrics(
    approx: np.ndarray, exact: np.ndarray, eps: float = 1e-12
) -> ErrorMetrics:
    """Compute every error metric of ``approx`` relative to the ``exact`` output.

    ``exact`` is the full-precision reference.  The aggregate ``norm_rel_error``
    is the L2 norm of the error divided by the L2 norm of the reference; it is
    the quantity the controller compares against the application error budget
    because it is robust to individual near-zero reference elements.
    """

    approx = np.asarray(approx, dtype=np.float64).ravel()
    exact = np.asarray(exact, dtype=np.float64).ravel()
    if approx.shape != exact.shape:
        raise ValueError("approx and exact must have the same number of elements")

    diff = approx - exact
    abs_diff = np.abs(diff)

    mae = float(np.mean(abs_diff))
    mse = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))
    max_abs = float(np.max(abs_diff)) if abs_diff.size else 0.0

    denom = np.abs(exact) + eps
    rel = abs_diff / denom
    mean_rel = float(np.mean(rel))
    max_rel = float(np.max(rel)) if rel.size else 0.0

    ref_norm = float(np.linalg.norm(exact))
    norm_rel = float(np.linalg.norm(diff) / (ref_norm + eps))

    return ErrorMetrics(
        mae=mae,
        mse=mse,
        rmse=rmse,
        max_abs_error=max_abs,
        mean_rel_error=mean_rel,
        max_rel_error=max_rel,
        norm_rel_error=norm_rel,
    )


def relative_error(approx: np.ndarray, exact: np.ndarray, eps: float = 1e-12) -> float:
    """Aggregate L2 relative error ``||approx-exact|| / ||exact||``.

    This is the single scalar used as the "achieved error" throughout the
    experiments, matching the quantity the controller predicts and bounds.
    """

    approx = np.asarray(approx, dtype=np.float64).ravel()
    exact = np.asarray(exact, dtype=np.float64).ravel()
    diff = np.linalg.norm(approx - exact)
    ref = np.linalg.norm(exact)
    return float(diff / (ref + eps))
