"""Top-Level Experiment Framework for Runtime Dynamic Approximation.

This module provides a reproducible, automated benchmarking pipeline that executes
workloads (FIR, Matrix Multiplication, 2-D Convolution) across baseline and adaptive
policies, sweeping error budgets and resource pressure traces.

IMPORTANT NOTE ON METRICS:
--------------------------
All energy, latency, and resource quantities are **SIMULATED AND MODELLED ESTIMATES**
derived from the mathematical models in :mod:`config` (:class:`CostModelConfig` and
:class:`ResourceConfig`). They are NOT physical FPGA or hardware measurements.

Architecture Overview:
----------------------
Workload (Block Stream)
    ↓
Workload Analyzer & Sensitivity Estimator (sub-block perturbation)
    ↓
Resource Monitor (simulated pressure trace)
    ↓
Runtime Controller (Exact / Approx-1 / Approx-2 / Approx-3 selection)
    ↓
Approximation Engine (quantized-mantissa MAC arithmetic)
    ↓
Output & Error Monitor (MAE, MSE, RMSE, Relative L2 Error)
    ↓
Result Logging (CSV & JSON in results/)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from config import (
    DEFAULT_CONFIG,
    DEFAULT_ERROR_BUDGETS,
    MODE_ORDER,
    RESOURCE_LEVELS,
    Config,
    Mode,
    WorkloadConfig,
    pressure_to_level,
)
from core.controller import ControllerDecision, Policy, make_policy
from core.metrics import ErrorMetrics, error_metrics, relative_error
from core.resource_model import ResourceTrace, constant_trace, make_resource_trace
from core.workloads import Block, Workload, make_workload

# Canonical list of supported workloads, policies, and trace types
ALL_WORKLOADS = ["fir", "matmul", "conv"]
ALL_POLICIES = [
    "exact_only",
    "fixed_a1",
    "fixed_a2",
    "fixed_a3",
    "resource_only",
    "full_adaptive",
]
ALL_TRACES = ["oscillating", "rising", "random", "constant_low", "constant_med", "constant_high"]


@dataclass
class BlockRecord:
    """Detailed record of one block computation within an experiment."""

    block_index: int
    character: str
    pressure: float
    pressure_level: str
    sensitivity_amp: Optional[float]
    sensitivity_score: Optional[float]
    probe_macs: int
    selected_mode: str
    mode_bits: Optional[int]
    predicted_error: float
    norm_rel_error: float
    mae: float
    mse: float
    rmse: float
    max_abs_error: float
    budget_satisfied: bool
    compute_energy_macs: float
    overhead_energy_macs: float
    total_energy_macs: float
    baseline_energy_macs: float
    energy_ratio: float
    energy_savings_pct: float
    compute_latency_units: float
    wall_time_s: float
    decision_reason: str


@dataclass
class ExperimentResult:
    """Summary record of one complete workload experiment execution."""

    experiment_id: str
    timestamp: str
    workload: str
    policy: str
    error_budget: float
    trace_kind: str
    bias: str
    seed: int
    n_blocks: int
    total_macs: int
    overall_norm_rel_error: float
    overall_mae: float
    overall_mse: float
    overall_rmse: float
    overall_max_abs_error: float
    overall_budget_satisfied: bool
    block_satisfaction_rate: float
    total_energy_macs: float
    baseline_energy_macs: float
    energy_ratio: float
    energy_savings_pct: float
    total_latency_units: float
    baseline_latency_units: float
    latency_ratio: float
    latency_reduction_pct: float
    mean_pressure: float
    mean_sensitivity_amp: float
    mode_distribution: Dict[str, float]
    total_probe_macs: int
    total_overhead_macs: float
    overhead_pct: float
    blocks: List[BlockRecord] = field(default_factory=list, repr=False)

    def to_dict(self, include_blocks: bool = True) -> Dict[str, Any]:
        """Convert result to a serializable dictionary."""
        d = asdict(self)
        if not include_blocks:
            d.pop("blocks", None)
        return d


def build_trace(trace_kind: str, cfg: Config, n_blocks: int, seed: int) -> ResourceTrace:
    """Construct a ResourceTrace for the given trace kind."""
    if trace_kind == "constant_low":
        return constant_trace(RESOURCE_LEVELS["LOW"], n_blocks)
    if trace_kind == "constant_med":
        return constant_trace(RESOURCE_LEVELS["MEDIUM"], n_blocks)
    if trace_kind == "constant_high":
        return constant_trace(RESOURCE_LEVELS["HIGH"], n_blocks)
    if trace_kind in ("oscillating", "rising", "random", "constant"):
        custom_cfg = cfg.with_overrides(
            resource=cfg.resource.__class__(trace_kind=trace_kind)
        )
        return make_resource_trace(custom_cfg, n_blocks, seed)
    raise ValueError(f"Unknown trace kind: {trace_kind!r}")


def run_single_experiment(
    workload_name: str,
    policy_name: str,
    budget: float,
    trace_kind: str,
    cfg: Optional[Config] = None,
    bias: str = "mixed",
    seed: Optional[int] = None,
) -> ExperimentResult:
    """Execute a single end-to-end experiment and record metrics.

    Parameters
    ----------
    workload_name:
        'fir', 'matmul', or 'conv' (or 'convolution')
    policy_name:
        'exact_only', 'fixed_a1', 'fixed_a2', 'fixed_a3', 'resource_only', 'full_adaptive'
    budget:
        Application error budget as a relative fraction (e.g. 0.05 for 5%)
    trace_kind:
        Resource trace identifier (e.g. 'oscillating', 'rising', 'random', 'constant_med')
    cfg:
        Configuration object (defaults to DEFAULT_CONFIG)
    bias:
        Workload sensitivity bias ('low', 'mixed', 'high')
    seed:
        Random seed for reproducibility

    Returns
    -------
    ExperimentResult
        Comprehensive metrics container.
    """
    if workload_name == "convolution":
        workload_name = "conv"

    cfg = DEFAULT_CONFIG if cfg is None else cfg
    seed = cfg.seed if seed is None else seed

    # 1. Instantiate Workload
    workload: Workload = make_workload(workload_name, cfg, bias=bias, seed=seed)
    n_blocks = len(workload.blocks)

    # 2. Instantiate Policy
    policy: Policy = make_policy(policy_name, cfg, workload=workload)

    # 3. Generate Resource Trace
    trace: ResourceTrace = build_trace(trace_kind, cfg, n_blocks, seed)

    # 4. Block-by-block Execution
    block_records: List[BlockRecord] = []
    approx_outputs: List[np.ndarray] = []
    exact_outputs: List[np.ndarray] = []

    total_compute_energy = 0.0
    total_overhead_energy = 0.0
    total_baseline_energy = 0.0
    total_compute_latency = 0.0
    total_baseline_latency = 0.0
    total_probe_macs = 0
    mode_counts: Dict[str, int] = {m.value: 0 for m in MODE_ORDER}
    sensitivity_amps: List[float] = []

    for block in workload.blocks:
        p = trace.pressure(block.index)

        # A. Runtime Controller Decision
        t0 = time.perf_counter()
        decision: ControllerDecision = policy.decide(workload, block, budget, p, cfg)
        dec_wall = time.perf_counter() - t0

        chosen_mode = decision.mode
        mode_counts[chosen_mode.value] += 1
        bits = cfg.mode_bits(chosen_mode)

        if decision.sensitivity_amp is not None:
            sensitivity_amps.append(decision.sensitivity_amp)

        # B. Approximate and Reference Computation
        approx_block_out = workload.compute(block, bits)
        exact_block_out = workload.compute(block, None)

        approx_outputs.append(approx_block_out)
        exact_outputs.append(exact_block_out)

        # C. Block Error Metrics
        blk_metrics: ErrorMetrics = error_metrics(approx_block_out, exact_block_out)
        budget_met = blk_metrics.norm_rel_error <= budget

        # D. Modelled Energy and Latency Accounting
        blk_macs = block.n_macs
        e_ratio = cfg.energy_ratio(chosen_mode)
        l_ratio = cfg.latency_ratio(chosen_mode)

        compute_energy = blk_macs * e_ratio
        overhead_energy = float(decision.overhead_units)
        tot_blk_energy = compute_energy + overhead_energy
        base_blk_energy = float(blk_macs)  # EXACT has energy ratio == 1.0

        blk_compute_latency = blk_macs * l_ratio

        total_compute_energy += compute_energy
        total_overhead_energy += overhead_energy
        total_baseline_energy += base_blk_energy
        total_compute_latency += blk_compute_latency
        total_baseline_latency += float(blk_macs)
        total_probe_macs += decision.probe_macs

        blk_e_ratio = tot_blk_energy / base_blk_energy if base_blk_energy > 0 else 1.0
        blk_e_savings = (1.0 - blk_e_ratio) * 100.0

        block_records.append(
            BlockRecord(
                block_index=block.index,
                character=block.character,
                pressure=p,
                pressure_level=pressure_to_level(p),
                sensitivity_amp=decision.sensitivity_amp,
                sensitivity_score=decision.sensitivity_score,
                probe_macs=decision.probe_macs,
                selected_mode=chosen_mode.value,
                mode_bits=bits,
                predicted_error=decision.predicted_error.get(chosen_mode, 0.0),
                norm_rel_error=blk_metrics.norm_rel_error,
                mae=blk_metrics.mae,
                mse=blk_metrics.mse,
                rmse=blk_metrics.rmse,
                max_abs_error=blk_metrics.max_abs_error,
                budget_satisfied=budget_met,
                compute_energy_macs=compute_energy,
                overhead_energy_macs=overhead_energy,
                total_energy_macs=tot_blk_energy,
                baseline_energy_macs=base_blk_energy,
                energy_ratio=blk_e_ratio,
                energy_savings_pct=blk_e_savings,
                compute_latency_units=blk_compute_latency,
                wall_time_s=dec_wall,
                decision_reason=decision.reason,
            )
        )

    # 5. Overall Workload Aggregation
    concat_approx = np.concatenate([arr.ravel() for arr in approx_outputs])
    concat_exact = np.concatenate([arr.ravel() for arr in exact_outputs])
    overall_err: ErrorMetrics = error_metrics(concat_approx, concat_exact)

    tot_energy = total_compute_energy + total_overhead_energy
    overall_e_ratio = tot_energy / total_baseline_energy if total_baseline_energy > 0 else 1.0
    overall_e_savings = (1.0 - overall_e_ratio) * 100.0

    overall_l_ratio = total_compute_latency / total_baseline_latency if total_baseline_latency > 0 else 1.0
    overall_l_savings = (1.0 - overall_l_ratio) * 100.0

    satisfied_blocks = sum(1 for b in block_records if b.budget_satisfied)
    satisfaction_rate = satisfied_blocks / n_blocks if n_blocks > 0 else 1.0

    mode_dist = {
        mode_val: (count / n_blocks * 100.0) if n_blocks > 0 else 0.0
        for mode_val, count in mode_counts.items()
    }

    overhead_pct = (total_overhead_energy / tot_energy * 100.0) if tot_energy > 0 else 0.0
    mean_amp = float(np.mean(sensitivity_amps)) if sensitivity_amps else 1.0

    exp_id = f"{workload_name}_{policy_name}_b{int(budget * 1000):03d}_{trace_kind}"
    now_str = datetime.now().isoformat(timespec="seconds")

    return ExperimentResult(
        experiment_id=exp_id,
        timestamp=now_str,
        workload=workload_name,
        policy=policy_name,
        error_budget=budget,
        trace_kind=trace_kind,
        bias=bias,
        seed=seed,
        n_blocks=n_blocks,
        total_macs=int(total_baseline_energy),
        overall_norm_rel_error=overall_err.norm_rel_error,
        overall_mae=overall_err.mae,
        overall_mse=overall_err.mse,
        overall_rmse=overall_err.rmse,
        overall_max_abs_error=overall_err.max_abs_error,
        overall_budget_satisfied=(overall_err.norm_rel_error <= budget),
        block_satisfaction_rate=satisfaction_rate,
        total_energy_macs=tot_energy,
        baseline_energy_macs=total_baseline_energy,
        energy_ratio=overall_e_ratio,
        energy_savings_pct=overall_e_savings,
        total_latency_units=total_compute_latency,
        baseline_latency_units=total_baseline_latency,
        latency_ratio=overall_l_ratio,
        latency_reduction_pct=overall_l_savings,
        mean_pressure=trace.mean_pressure(),
        mean_sensitivity_amp=mean_amp,
        mode_distribution=mode_dist,
        total_probe_macs=total_probe_macs,
        total_overhead_macs=total_overhead_energy,
        overhead_pct=overhead_pct,
        blocks=block_records,
    )


def save_experiment_results(
    results: Sequence[ExperimentResult],
    output_dir: Union[str, Path] = "results",
    save_csv: bool = True,
    save_json: bool = True,
    save_block_details: bool = True,
) -> Dict[str, Path]:
    """Save experiment results to CSV and JSON files in the specified directory."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    generated_files: Dict[str, Path] = {}

    # 1. Summary CSV
    if save_csv:
        summary_csv = out_path / "summary.csv"
        fieldnames = [
            "experiment_id",
            "timestamp",
            "workload",
            "policy",
            "error_budget",
            "trace_kind",
            "bias",
            "seed",
            "n_blocks",
            "total_macs",
            "overall_norm_rel_error",
            "overall_budget_satisfied",
            "block_satisfaction_rate",
            "energy_savings_pct",
            "energy_ratio",
            "latency_reduction_pct",
            "latency_ratio",
            "mean_pressure",
            "mean_sensitivity_amp",
            "overhead_pct",
            "pct_EXACT",
            "pct_APPROX1",
            "pct_APPROX2",
            "pct_APPROX3",
            "overall_mae",
            "overall_mse",
            "overall_rmse",
            "overall_max_abs_error",
            "total_energy_macs",
            "baseline_energy_macs",
        ]

        with open(summary_csv, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                row = {
                    "experiment_id": r.experiment_id,
                    "timestamp": r.timestamp,
                    "workload": r.workload,
                    "policy": r.policy,
                    "error_budget": r.error_budget,
                    "trace_kind": r.trace_kind,
                    "bias": r.bias,
                    "seed": r.seed,
                    "n_blocks": r.n_blocks,
                    "total_macs": r.total_macs,
                    "overall_norm_rel_error": f"{r.overall_norm_rel_error:.6e}",
                    "overall_budget_satisfied": r.overall_budget_satisfied,
                    "block_satisfaction_rate": f"{r.block_satisfaction_rate * 100:.2f}%",
                    "energy_savings_pct": f"{r.energy_savings_pct:.2f}%",
                    "energy_ratio": f"{r.energy_ratio:.4f}",
                    "latency_reduction_pct": f"{r.latency_reduction_pct:.2f}%",
                    "latency_ratio": f"{r.latency_ratio:.4f}",
                    "mean_pressure": f"{r.mean_pressure:.3f}",
                    "mean_sensitivity_amp": f"{r.mean_sensitivity_amp:.3f}",
                    "overhead_pct": f"{r.overhead_pct:.2f}%",
                    "pct_EXACT": f"{r.mode_distribution.get(Mode.EXACT.value, 0.0):.1f}%",
                    "pct_APPROX1": f"{r.mode_distribution.get(Mode.APPROX1.value, 0.0):.1f}%",
                    "pct_APPROX2": f"{r.mode_distribution.get(Mode.APPROX2.value, 0.0):.1f}%",
                    "pct_APPROX3": f"{r.mode_distribution.get(Mode.APPROX3.value, 0.0):.1f}%",
                    "overall_mae": f"{r.overall_mae:.6e}",
                    "overall_mse": f"{r.overall_mse:.6e}",
                    "overall_rmse": f"{r.overall_rmse:.6e}",
                    "overall_max_abs_error": f"{r.overall_max_abs_error:.6e}",
                    "total_energy_macs": f"{r.total_energy_macs:.1f}",
                    "baseline_energy_macs": f"{r.baseline_energy_macs:.1f}",
                }
                writer.writerow(row)
        generated_files["summary_csv"] = summary_csv

    # 2. Block Details CSV (Fine-grained per-block log)
    if save_csv and save_block_details:
        blocks_csv = out_path / "block_details.csv"
        block_fields = [
            "experiment_id",
            "block_index",
            "character",
            "pressure",
            "pressure_level",
            "sensitivity_amp",
            "sensitivity_score",
            "selected_mode",
            "mode_bits",
            "predicted_error",
            "norm_rel_error",
            "budget_satisfied",
            "compute_energy_macs",
            "overhead_energy_macs",
            "total_energy_macs",
            "baseline_energy_macs",
            "energy_savings_pct",
            "compute_latency_units",
            "mae",
            "mse",
            "rmse",
            "probe_macs",
            "decision_reason",
        ]

        with open(blocks_csv, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=block_fields)
            writer.writeheader()
            for r in results:
                for b in r.blocks:
                    row = {
                        "experiment_id": r.experiment_id,
                        "block_index": b.block_index,
                        "character": b.character,
                        "pressure": f"{b.pressure:.3f}",
                        "pressure_level": b.pressure_level,
                        "sensitivity_amp": f"{b.sensitivity_amp:.3f}" if b.sensitivity_amp is not None else "N/A",
                        "sensitivity_score": f"{b.sensitivity_score:.3f}" if b.sensitivity_score is not None else "N/A",
                        "selected_mode": b.selected_mode,
                        "mode_bits": b.mode_bits if b.mode_bits is not None else "EXACT",
                        "predicted_error": f"{b.predicted_error:.6e}",
                        "norm_rel_error": f"{b.norm_rel_error:.6e}",
                        "budget_satisfied": b.budget_satisfied,
                        "compute_energy_macs": f"{b.compute_energy_macs:.1f}",
                        "overhead_energy_macs": f"{b.overhead_energy_macs:.1f}",
                        "total_energy_macs": f"{b.total_energy_macs:.1f}",
                        "baseline_energy_macs": f"{b.baseline_energy_macs:.1f}",
                        "energy_savings_pct": f"{b.energy_savings_pct:.2f}%",
                        "compute_latency_units": f"{b.compute_latency_units:.1f}",
                        "mae": f"{b.mae:.6e}",
                        "mse": f"{b.mse:.6e}",
                        "rmse": f"{b.rmse:.6e}",
                        "probe_macs": b.probe_macs,
                        "decision_reason": b.decision_reason,
                    }
                    writer.writerow(row)
        generated_files["blocks_csv"] = blocks_csv

    # 3. Comprehensive JSON
    if save_json:
        json_file = out_path / "experiments.json"
        data = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "experiment_count": len(results),
                "model_notice": (
                    "Energy, latency, and resource metrics are modelled software estimates "
                    "based on CostModelConfig (MAC arithmetic scaling) and ResourceConfig, "
                    "not physical hardware/FPGA measurements."
                ),
            },
            "experiments": [r.to_dict(include_blocks=True) for r in results],
        }
        with open(json_file, mode="w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        generated_files["json"] = json_file

    return generated_files


def run_smoke_test(output_dir: str = "results") -> bool:
    """Run a fast, lightweight smoke test covering all 3 workloads and 4 policies.

    Uses a smaller WorkloadConfig so the test executes in under 1 second.
    """
    print("\n=======================================================")
    print("RUNNING SMOKE TEST (Fast End-to-End Pipeline Check)")
    print("=======================================================")

    # Lightweight configuration for fast execution
    fast_workload_cfg = WorkloadConfig(
        fir_length=1024,
        fir_taps=16,
        fir_blocks=8,
        mat_m=32,
        mat_k=32,
        mat_n=32,
        mat_row_block=4,
        img_size=32,
        conv_kernel=3,
        conv_row_block=4,
    )
    test_cfg = DEFAULT_CONFIG.with_overrides(workload=fast_workload_cfg)

    test_matrix = [
        ("fir", "exact_only", 0.05, "constant_med"),
        ("fir", "fixed_a2", 0.05, "constant_med"),
        ("matmul", "resource_only", 0.05, "oscillating"),
        ("conv", "full_adaptive", 0.05, "rising"),
    ]

    results: List[ExperimentResult] = []
    t_start = time.perf_counter()

    for w, p, b, tr in test_matrix:
        print(f"  -> Testing {w.upper():<7} | Policy: {p:<15} | Budget: {b*100:.1f}% | Trace: {tr}")
        res = run_single_experiment(w, p, b, tr, cfg=test_cfg, seed=42)
        results.append(res)
        print(
            f"     [Result] Mode Dist: EXACT={res.mode_distribution[Mode.EXACT.value]:.0f}% "
            f"A1={res.mode_distribution[Mode.APPROX1.value]:.0f}% "
            f"A2={res.mode_distribution[Mode.APPROX2.value]:.0f}% "
            f"A3={res.mode_distribution[Mode.APPROX3.value]:.0f}% | "
            f"RelErr: {res.overall_norm_rel_error:.4f} (Budget Met: {res.overall_budget_satisfied}) | "
            f"Energy Saving: {res.energy_savings_pct:.1f}%"
        )

    t_elapsed = time.perf_counter() - t_start
    out_files = save_experiment_results(results, output_dir=output_dir)

    print("\nSmoke test completed successfully in {:.3f} seconds!".format(t_elapsed))
    print(f"Saved summary: {out_files.get('summary_csv')}")
    print(f"Saved details: {out_files.get('blocks_csv')}")
    print(f"Saved JSON:    {out_files.get('json')}")
    print("=======================================================\n")
    return True


def run_experiment_suite(
    workloads: Sequence[str],
    policies: Sequence[str],
    budgets: Sequence[float],
    traces: Sequence[str],
    bias: str = "mixed",
    seed: Optional[int] = None,
    output_dir: str = "results",
    cfg: Optional[Config] = None,
    verbose: bool = True,
) -> List[ExperimentResult]:
    """Run a full factorial grid of experiments and save the results."""
    total_runs = len(workloads) * len(policies) * len(budgets) * len(traces)
    if verbose:
        print(f"\nLaunching Experiment Suite: {total_runs} total runs planned.")
        print(f"  Workloads: {list(workloads)}")
        print(f"  Policies:  {list(policies)}")
        print(f"  Budgets:   {[f'{b*100:.1f}%' for b in budgets]}")
        print(f"  Traces:    {list(traces)}")
        print(f"  Output:    {output_dir}/\n")

    results: List[ExperimentResult] = []
    run_idx = 0
    t0 = time.perf_counter()

    for w in workloads:
        for p in policies:
            for b in budgets:
                for tr in traces:
                    run_idx += 1
                    res = run_single_experiment(w, p, b, tr, cfg=cfg, bias=bias, seed=seed)
                    results.append(res)
                    if verbose:
                        print(
                            f"[{run_idx:>3}/{total_runs}] {w.upper():<6} | {p:<14} | "
                            f"Budg:{b*100:>4.1f}% | Tr:{tr:<12} => "
                            f"RelErr:{res.overall_norm_rel_error:>8.4e} | "
                            f"ESave:{res.energy_savings_pct:>6.1f}% | "
                            f"LRed:{res.latency_reduction_pct:>6.1f}% | "
                            f"Met:{'YES' if res.overall_budget_satisfied else 'NO '}"
                        )

    t_total = time.perf_counter() - t0
    save_experiment_results(results, output_dir=output_dir)

    if verbose:
        print(f"\nExperiment suite completed in {t_total:.2f} seconds.")
        print(f"Results recorded in '{output_dir}/'.\n")

    return results


def main() -> None:
    """CLI interface for experiment runner."""
    parser = argparse.ArgumentParser(
        description="Top-Level Experiment Framework for Runtime Approximation Prototype.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a fast 1-second smoke test verifying all workloads and policies.",
    )
    parser.add_argument(
        "--workload",
        "-w",
        nargs="+",
        default=["all"],
        help=f"Workloads to evaluate: {ALL_WORKLOADS} or 'all' (or 'convolution').",
    )
    parser.add_argument(
        "--policy",
        "-p",
        nargs="+",
        default=["all"],
        help=f"Policies to evaluate: {ALL_POLICIES} or 'all'.",
    )
    parser.add_argument(
        "--budget",
        "-b",
        type=float,
        nargs="+",
        default=None,
        help="Error budgets as fractions (e.g. 0.01 0.05 0.10). Defaults to standard config budgets.",
    )
    parser.add_argument(
        "--trace",
        "-t",
        nargs="+",
        default=["oscillating"],
        help=f"Resource traces: {ALL_TRACES} or 'all'.",
    )
    parser.add_argument(
        "--bias",
        choices=["low", "mixed", "high"],
        default="mixed",
        help="Workload sensitivity bias distribution.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_CONFIG.seed,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="results",
        help="Directory to store CSV and JSON results.",
    )

    args = parser.parse_args()

    if args.smoke_test:
        run_smoke_test(output_dir=args.output_dir)
        return

    # Parse Workloads
    if "all" in args.workload:
        selected_workloads = ALL_WORKLOADS
    else:
        selected_workloads = [
            "conv" if w.lower() == "convolution" else w.lower() for w in args.workload
        ]
        for w in selected_workloads:
            if w not in ALL_WORKLOADS:
                print(f"Error: unknown workload {w!r}. Choose from {ALL_WORKLOADS}")
                sys.exit(1)

    # Parse Policies
    if "all" in args.policy:
        selected_policies = ALL_POLICIES
    else:
        selected_policies = [p.lower() for p in args.policy]
        for p in selected_policies:
            if p not in ALL_POLICIES:
                print(f"Error: unknown policy {p!r}. Choose from {ALL_POLICIES}")
                sys.exit(1)

    # Parse Budgets
    selected_budgets = DEFAULT_ERROR_BUDGETS if args.budget is None else args.budget

    # Parse Traces
    if "all" in args.trace:
        selected_traces = ALL_TRACES
    else:
        selected_traces = args.trace
        for tr in selected_traces:
            if tr not in ALL_TRACES:
                print(f"Error: unknown trace {tr!r}. Choose from {ALL_TRACES}")
                sys.exit(1)

    run_experiment_suite(
        workloads=selected_workloads,
        policies=selected_policies,
        budgets=selected_budgets,
        traces=selected_traces,
        bias=args.bias,
        seed=args.seed,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
