"""Central configuration for the runtime approximation architecture prototype.

Every tunable parameter of the simulator lives here so that experiments and the
Streamlit UI can override a single, well-documented object instead of scattering
magic numbers across the codebase.

Design notes
------------
* The three computation *modes* are described purely by an *effective significand
  (datapath) bit-width*.  ``EXACT`` is the full-precision reference; the two
  approximate modes quantise multiplier operands to fewer bits.  See
  :mod:`core.approximation` for the arithmetic and :mod:`docs/methodology.md`
  for the justification.
* The relative energy / latency cost of a mode is *derived* from its bit-width
  through a documented, configurable model (see :class:`CostModelConfig`).  No
  cost number is hand-picked to flatter the adaptive controller.
* Nothing here measures physical hardware.  Energy and latency are **estimated**
  quantities produced by a transparent model and are always labelled as such.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Dict, List


class Mode(str, Enum):
    """The computation modes offered by the approximation engine."""

    EXACT = "EXACT"
    APPROX1 = "APPROX-1"
    APPROX2 = "APPROX-2"
    APPROX3 = "APPROX-3"

    @property
    def short(self) -> str:
        return {"EXACT": "EXACT", "APPROX-1": "A1", "APPROX-2": "A2", "APPROX-3": "A3"}[self.value]


# Ordered from most to least accurate (and most to least expensive).
MODE_ORDER: List[Mode] = [Mode.EXACT, Mode.APPROX1, Mode.APPROX2, Mode.APPROX3]


@dataclass(frozen=True)
class ModeConfig:
    """Describes one computation mode.

    Attributes
    ----------
    bits:
        Effective significand (mantissa / datapath) bit-width.  For ``EXACT``
        this is a *nominal* width used only for cost normalisation; the exact
        computation itself is performed in full ``float64`` precision and is
        treated as the ground-truth reference.  For the approximate modes this
        is the number of significand bits retained when quantising multiplier
        operands.
    quantise:
        Whether operands are quantised in this mode.  ``False`` only for
        ``EXACT``.
    """

    bits: int
    quantise: bool

    def unit_error(self) -> float:
        """Worst-case per-operand relative quantisation error, ``2**-bits``.

        Mantissa truncation to ``bits`` significand bits rounds the normalised
        mantissa (magnitude in ``[0.5, 1)``) to a grid of step ``2**-(bits+1)``,
        giving a relative error of at most ``2**-bits``.  ``EXACT`` has none.
        """

        return 0.0 if not self.quantise else 2.0 ** (-self.bits)


@dataclass(frozen=True)
class CostModelConfig:
    """Transparent, configurable relative cost model for a multiply-accumulate.

    The energy of a combinational ``n``-bit multiplier scales roughly with the
    number of partial-product bits (``~n**2``), while the accumulating adder
    scales roughly linearly (``~n``).  We therefore model the *estimated* energy
    of one multiply-accumulate (MAC) at significand width ``b`` as::

        energy_mac(b) = w_mul * b**2 + w_add * b

    and the *estimated* latency (critical-path depth of the datapath) as::

        latency_mac(b) = lat_coeff * b + lat_const

    Both quantities are reported **normalised to the EXACT mode**, so EXACT has
    energy = latency = 1.0 by construction and the approximate modes are
    fractions thereof.  The absolute weights below are deliberately simple and
    are exposed so a reviewer can change them and re-run every experiment.

    These are *modelled* costs, not hardware measurements.
    """

    w_mul: float = 1.0
    w_add: float = 1.0
    lat_coeff: float = 1.0
    lat_const: float = 0.0

    def energy_mac(self, bits: int) -> float:
        return self.w_mul * (bits ** 2) + self.w_add * bits

    def latency_mac(self, bits: int) -> float:
        return self.lat_coeff * bits + self.lat_const


@dataclass(frozen=True)
class ControllerConfig:
    """Parameters of the rule-plus-score runtime controller.

    Attributes
    ----------
    safety_margin:
        The controller treats a mode as *budget-feasible* only when its
        predicted relative error multiplied by ``safety_margin`` stays within
        the application error budget.  ``>= 1.0`` makes the controller
        conservative to absorb prediction error.
    lambda_min, lambda_max:
        Weight on *risk* (how much of the error budget a mode is predicted to
        consume) in the selection score.  The effective weight interpolates
        with resource pressure ``p`` in ``[0, 1]`` (1 = scarce)::

            lam(p) = lambda_max * (1 - p) + lambda_min * p

        When resources are abundant (``p`` small) risk is penalised heavily, so
        the controller keeps accuracy margin (prefers exacter modes).  When
        resources are scarce (``p`` large) risk is cheap, so the controller
        spends the available error budget to save energy.
    decision_overhead_units:
        Fixed per-decision controller cost, expressed in EXACT-MAC-equivalent
        energy units (the arithmetic of comparing a handful of candidate modes).
        The sensitivity-probe cost is accounted separately and measured from the
        actual probe work performed.
    """

    safety_margin: float = 1.25
    lambda_min: float = 0.0
    lambda_max: float = 4.0
    decision_overhead_units: float = 8.0


@dataclass(frozen=True)
class SensitivityConfig:
    """Parameters of the lightweight perturbation-based sensitivity estimator.

    Attributes
    ----------
    probe_bits:
        Significand width used for the perturbation probe.  The probe quantises
        a small representative sub-block to this width, measures the resulting
        relative output deviation, and divides by ``2**-probe_bits`` to obtain
        an *error-amplification* estimate ``A`` (a cheap conditioning proxy).
    probe_elements:
        Target number of output elements evaluated by the probe.  Kept small so
        the probe is cheap relative to a full block (this is what makes runtime
        use plausible and is central to hypothesis H3).
    amp_floor, amp_ceil:
        The raw amplification estimate is clipped to this range before being
        mapped (on a log scale) to the normalised ``[0, 1]`` sensitivity score
        used for display and the decision map.  The controller consumes the
        raw (unclipped) amplification for its error prediction.
    """

    probe_bits: int = 8
    probe_elements: int = 32
    amp_floor: float = 0.25
    amp_ceil: float = 64.0


@dataclass(frozen=True)
class ResourceConfig:
    """Parameters of the simulated resource model.

    ``pressure`` is a continuous value in ``[0, 1]`` where 0 means resources are
    abundant and 1 means resources are scarce (e.g. low remaining energy, high
    processor utilisation or thermal throttling).  A trace generator produces a
    non-stationary pressure sequence so that runtime adaptation has something to
    adapt to across blocks.  These are **simulated** conditions, not readings
    from real hardware.
    """

    trace_kind: str = "oscillating"  # one of: oscillating, rising, random, constant
    base_pressure: float = 0.5
    amplitude: float = 0.45
    period_blocks: int = 16
    noise_std: float = 0.05


@dataclass(frozen=True)
class WorkloadConfig:
    """Sizes of the three workloads and how they are split into blocks."""

    # FIR / vector workload
    fir_length: int = 8192
    fir_taps: int = 48
    fir_blocks: int = 32

    # Dense matrix multiplication (A @ B)
    mat_m: int = 192
    mat_k: int = 160
    mat_n: int = 192
    mat_row_block: int = 6  # rows of A processed per block

    # 2-D single-channel convolution
    img_size: int = 160
    conv_kernel: int = 5
    conv_row_block: int = 5  # output rows per block


@dataclass(frozen=True)
class Config:
    """Top-level configuration aggregating every sub-model."""

    seed: int = 20260408
    modes: Dict[Mode, ModeConfig] = field(
        default_factory=lambda: {
            # EXACT: full float64 reference; 16-bit nominal datapath for costing.
            Mode.EXACT: ModeConfig(bits=16, quantise=False),
            # APPROX-1: mild reduced precision (10 significand bits).
            Mode.APPROX1: ModeConfig(bits=10, quantise=True),
            # APPROX-2: moderate reduced precision (6 significand bits).
            Mode.APPROX2: ModeConfig(bits=6, quantise=True),
            # APPROX-3: aggressive reduced precision (4 significand bits).
            Mode.APPROX3: ModeConfig(bits=4, quantise=True),
        }
    )
    cost: CostModelConfig = field(default_factory=CostModelConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    sensitivity: SensitivityConfig = field(default_factory=SensitivityConfig)
    resource: ResourceConfig = field(default_factory=ResourceConfig)
    workload: WorkloadConfig = field(default_factory=WorkloadConfig)

    # ------------------------------------------------------------------ #
    # Derived cost helpers (normalised to EXACT).
    # ------------------------------------------------------------------ #
    def exact_bits(self) -> int:
        return self.modes[Mode.EXACT].bits

    def energy_ratio(self, mode: Mode) -> float:
        """Estimated energy per MAC in this mode, normalised so EXACT == 1.0."""

        base = self.cost.energy_mac(self.exact_bits())
        return self.cost.energy_mac(self.modes[mode].bits) / base

    def latency_ratio(self, mode: Mode) -> float:
        """Estimated latency per MAC in this mode, normalised so EXACT == 1.0."""

        base = self.cost.latency_mac(self.exact_bits())
        return self.cost.latency_mac(self.modes[mode].bits) / base

    def mode_bits(self, mode: Mode) -> int | None:
        """Quantisation width for ``mode`` or ``None`` for the exact reference."""

        mc = self.modes[mode]
        return mc.bits if mc.quantise else None

    def unit_error(self, mode: Mode) -> float:
        """Per-operand relative quantisation unit ``2**-bits`` (0 for EXACT)."""

        return self.modes[mode].unit_error()

    def with_overrides(self, **kwargs) -> "Config":
        """Return a copy with top-level fields replaced (handy for the UI)."""

        return replace(self, **kwargs)


# A ready-to-use default configuration instance.
DEFAULT_CONFIG = Config()


# Standard error budgets (relative) swept in experiments, as fractions.
DEFAULT_ERROR_BUDGETS: List[float] = [0.005, 0.01, 0.02, 0.05, 0.10]

# Named static resource conditions used in experiments and the UI.
RESOURCE_LEVELS: Dict[str, float] = {
    "LOW": 0.15,     # abundant resources -> little pressure
    "MEDIUM": 0.5,
    "HIGH": 0.85,    # scarce resources -> high pressure
}


def pressure_to_level(pressure: float) -> str:
    """Map a continuous pressure value to a coarse LOW/MEDIUM/HIGH label."""

    if pressure < 0.33:
        return "LOW"
    if pressure < 0.66:
        return "MEDIUM"
    return "HIGH"
