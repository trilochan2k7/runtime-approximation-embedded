"""Runtime controller: the policy that selects a computation mode per block.

Policies
--------
All policies share one interface (:meth:`Policy.decide`) so the simulator can
treat them uniformly.  Five policies are provided; the first three are the
non-adaptive baselines and the last two are the adaptive controllers compared in
hypotheses H1-H3:

* :class:`ExactOnlyPolicy`         - always EXACT (reference accuracy).
* :class:`FixedPolicy`             - always a chosen fixed approximate mode.
* :class:`ResourceOnlyAdaptive`    - resource- and budget-aware, but *blind to
  per-block sensitivity*: it uses a single workload-average amplification
  calibrated offline.
* :class:`FullAdaptive`            - resource-, sensitivity- and budget-aware:
  it probes each block's sensitivity at runtime.

Decision rule (adaptive policies)
---------------------------------
1. **Budget feasibility (hard constraint).**  For each mode predict the relative
   error ``pred(mode) = A * 2**-bits`` (``0`` for EXACT, where ``A`` is the
   amplification factor).  A mode is *feasible* only if
   ``pred(mode) * safety_margin <= budget``.  EXACT is always feasible, so a
   decision always exists and the budget is never knowingly violated to save
   energy.
2. **Resource-weighted selection among feasible modes.**  Minimise

       score(mode) = energy_ratio(mode) + lambda(pressure) * risk(mode)

   where ``risk = pred(mode) / budget`` is how much of the budget the mode is
   predicted to consume and ``lambda(pressure) = lambda_max*(1-p) +
   lambda_min*p``.  Under abundant resources (``p`` small) risk is penalised, so
   the controller keeps accuracy margin; under scarcity (``p`` large) it spends
   the budget to save energy.  Robust blocks (``risk ~ 0``) are approximated
   aggressively regardless of pressure because doing so is safe and cheap.

The rule is deliberately transparent (no learned model) because the research
goal is to demonstrate and explain the *architecture / policy* behaviour.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from config import Config, Mode, MODE_ORDER, pressure_to_level
from core.sensitivity import estimate_sensitivity
from core.workloads import Block, Workload

__all__ = [
    "ControllerDecision",
    "Policy",
    "ExactOnlyPolicy",
    "FixedPolicy",
    "ResourceOnlyAdaptive",
    "FullAdaptive",
    "predicted_errors",
    "select_mode",
    "calibrate_average_amplification",
    "make_policy",
]


@dataclass
class ControllerDecision:
    """Everything the controller decided for one block, for metrics + logging."""

    mode: Mode
    pressure: float
    predicted_error: Dict[Mode, float]
    feasible: Dict[Mode, bool]
    scores: Dict[Mode, float]
    overhead_units: float               # EXACT-MAC-equivalent energy units
    wall_time_s: float                  # measured decision wall-clock (software)
    reason: str
    sensitivity_amp: Optional[float] = None
    sensitivity_score: Optional[float] = None
    probe_macs: int = 0

    @property
    def level(self) -> str:
        return pressure_to_level(self.pressure)


def predicted_errors(amplification: float, cfg: Config) -> Dict[Mode, float]:
    """Predicted L2 relative error of each mode given an amplification factor.

    ``pred(mode) = amplification * 2**-bits`` for approximate modes and ``0`` for
    EXACT (the reference).  This is the first-order error-propagation estimate
    justified in :mod:`core.sensitivity`.
    """

    return {mode: amplification * cfg.unit_error(mode) for mode in MODE_ORDER}


def select_mode(
    pred: Dict[Mode, float], budget: float, pressure: float, cfg: Config
) -> tuple[Mode, Dict[Mode, bool], Dict[Mode, float]]:
    """Apply the feasibility filter and resource-weighted score.

    Returns the chosen mode, the per-mode feasibility flags, and the per-mode
    scores (``inf`` for infeasible modes).
    """

    cc = cfg.controller
    lam = cc.lambda_max * (1.0 - pressure) + cc.lambda_min * pressure

    feasible: Dict[Mode, bool] = {}
    scores: Dict[Mode, float] = {}
    for mode in MODE_ORDER:
        pe = pred[mode]
        ok = (mode == Mode.EXACT) or (pe * cc.safety_margin <= budget)
        feasible[mode] = ok
        if ok:
            risk = pe / budget if budget > 0 else 0.0
            scores[mode] = cfg.energy_ratio(mode) + lam * risk
        else:
            scores[mode] = float("inf")

    chosen = min(MODE_ORDER, key=lambda m: scores[m])
    return chosen, feasible, scores


def _reason_text(mode: Mode, pressure: float, budget: float,
                 amp: Optional[float], pred: Dict[Mode, float],
                 feasible: Dict[Mode, bool]) -> str:
    lvl = pressure_to_level(pressure)
    parts: List[str] = []
    if amp is not None:
        parts.append(f"sens A={amp:.2f}")
    parts.append(f"pressure={pressure:.2f} ({lvl})")
    parts.append(f"budget={budget * 100:.2f}%")
    feas = ",".join(m.short for m in MODE_ORDER if feasible[m])
    parts.append(f"feasible=[{feas}]")
    parts.append(f"-> {mode.value} (pred {pred[mode] * 100:.3f}%)")
    return "; ".join(parts)


class Policy(ABC):
    """Common interface for every mode-selection policy."""

    name: str
    uses_sensitivity: bool = False

    @abstractmethod
    def decide(self, workload: Workload, block: Block, budget: float,
               pressure: float, cfg: Config) -> ControllerDecision:
        ...


class ExactOnlyPolicy(Policy):
    """Always compute exactly (the accuracy reference, no runtime decision)."""

    name = "exact_only"

    def decide(self, workload, block, budget, pressure, cfg) -> ControllerDecision:
        pred = {m: 0.0 for m in MODE_ORDER}
        return ControllerDecision(
            mode=Mode.EXACT,
            pressure=pressure,
            predicted_error=pred,
            feasible={m: m == Mode.EXACT for m in MODE_ORDER},
            scores={Mode.EXACT: 0.0},
            overhead_units=0.0,
            wall_time_s=0.0,
            reason="fixed exact-only",
        )


class FixedPolicy(Policy):
    """Always use one fixed mode (e.g. fixed APPROX-1 or APPROX-2)."""

    def __init__(self, mode: Mode) -> None:
        self.mode = mode
        self.name = f"fixed_{mode.short.lower()}"

    def decide(self, workload, block, budget, pressure, cfg) -> ControllerDecision:
        return ControllerDecision(
            mode=self.mode,
            pressure=pressure,
            predicted_error={m: 0.0 for m in MODE_ORDER},
            feasible={m: m == self.mode for m in MODE_ORDER},
            scores={self.mode: 0.0},
            overhead_units=0.0,
            wall_time_s=0.0,
            reason=f"fixed {self.mode.value}",
        )


class ResourceOnlyAdaptive(Policy):
    """Resource- and budget-aware, but blind to per-block sensitivity.

    It uses a single workload-average amplification (calibrated offline, once)
    for *every* block, so it cannot tell a robust block from a sensitive one.
    Its only per-block runtime cost is the fixed decision overhead - it runs no
    sensitivity probe.
    """

    name = "resource_only"
    uses_sensitivity = False

    def __init__(self, average_amplification: float) -> None:
        self.avg_amp = float(average_amplification)

    def decide(self, workload, block, budget, pressure, cfg) -> ControllerDecision:
        t0 = time.perf_counter()
        pred = predicted_errors(self.avg_amp, cfg)
        mode, feasible, scores = select_mode(pred, budget, pressure, cfg)
        wall = time.perf_counter() - t0
        reason = "resource-only (avg A={:.2f}); ".format(self.avg_amp) + _reason_text(
            mode, pressure, budget, self.avg_amp, pred, feasible
        )
        return ControllerDecision(
            mode=mode,
            pressure=pressure,
            predicted_error=pred,
            feasible=feasible,
            scores=scores,
            overhead_units=cfg.controller.decision_overhead_units,
            wall_time_s=wall,
            reason=reason,
            sensitivity_amp=self.avg_amp,
            sensitivity_score=None,
            probe_macs=0,
        )


class FullAdaptive(Policy):
    """Resource-, sensitivity- and budget-aware controller (the full policy).

    Each block is probed at runtime for its sensitivity; the resulting
    amplification drives per-mode error prediction.  The probe's MACs plus the
    fixed decision cost are charged to the adaptation overhead.
    """

    name = "full_adaptive"
    uses_sensitivity = True

    def decide(self, workload, block, budget, pressure, cfg) -> ControllerDecision:
        t0 = time.perf_counter()
        sens = estimate_sensitivity(workload, block, cfg)
        pred = predicted_errors(sens.amplification, cfg)
        mode, feasible, scores = select_mode(pred, budget, pressure, cfg)
        wall = time.perf_counter() - t0
        overhead = sens.probe_macs + cfg.controller.decision_overhead_units
        reason = _reason_text(mode, pressure, budget, sens.amplification, pred, feasible)
        return ControllerDecision(
            mode=mode,
            pressure=pressure,
            predicted_error=pred,
            feasible=feasible,
            scores=scores,
            overhead_units=float(overhead),
            wall_time_s=wall,
            reason=reason,
            sensitivity_amp=sens.amplification,
            sensitivity_score=sens.score,
            probe_macs=sens.probe_macs,
        )


def calibrate_average_amplification(workload: Workload, cfg: Config) -> float:
    """Average per-block amplification, used to configure :class:`ResourceOnlyAdaptive`.

    This is a one-time *offline* calibration (not charged per block): it probes
    every block once and averages the amplification, modelling a controller
    tuned to the workload's typical behaviour but unaware of per-block variation.
    """

    amps = [estimate_sensitivity(workload, b, cfg).amplification for b in workload.blocks]
    return float(np.mean(amps)) if amps else 1.0


def make_policy(name: str, cfg: Config, workload: Optional[Workload] = None) -> Policy:
    """Factory for policies by canonical name.

    ``resource_only`` requires ``workload`` so its average amplification can be
    calibrated.
    """

    if name == "exact_only":
        return ExactOnlyPolicy()
    if name == "fixed_a1":
        return FixedPolicy(Mode.APPROX1)
    if name == "fixed_a2":
        return FixedPolicy(Mode.APPROX2)
    if name == "fixed_a3":
        return FixedPolicy(Mode.APPROX3)
    if name == "full_adaptive":
        return FullAdaptive()
    if name == "resource_only":
        if workload is None:
            raise ValueError("resource_only policy requires a workload to calibrate")
        return ResourceOnlyAdaptive(calibrate_average_amplification(workload, cfg))
    raise ValueError(f"unknown policy {name!r}")
