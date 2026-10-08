"""Comprehensive Hypothesis Testing Runner (H1, H2, H3).

This script orchestrates the systematic evaluation of the three core research hypotheses:
- H1 (Adaptive Precision): Dynamic precision reduces energy/latency within error budget.
- H2 (Sensitivity Awareness): Full adaptive controller outperforms resource-only adaptation.
- H3 (Overhead & Break-Even): Net energy benefits after accounting for probe and controller overhead.
- Multi-seed Statistical Repeatability (5 seeds: mean, std, min, max).

All energy and latency metrics are MODELLED analytical estimates derived from
CostModelConfig and ResourceConfig, NOT physical FPGA/hardware measurements.
"""

from __future__ import annotations

import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from config import DEFAULT_CONFIG, Config, Mode
from core.metrics import error_metrics, relative_error
from run_experiments import (
    ALL_POLICIES,
    ALL_TRACES,
    ALL_WORKLOADS,
    BlockRecord,
    ExperimentResult,
    run_single_experiment,
)

OUTPUT_DIR = Path("results/hypothesis")
SEEDS = [20260408, 42, 12345, 999, 777]
BUDGETS = [0.005, 0.01, 0.02, 0.05, 0.10]
WORKLOADS = ["fir", "matmul", "conv"]
TRACES = ["constant_low", "constant_med", "constant_high", "oscillating", "rising", "random"]


def run_h1_experiments(cfg: Config) -> Tuple[List[ExperimentResult], List[Dict[str, Any]]]:
    """Execute H1 parameter sweep across workloads, budgets, traces, and policies."""
    print("\n" + "=" * 78)
    print("  EXPERIMENT 1: HYPOTHESIS H1 (ADAPTIVE PRECISION vs BASELINES)")
    print("=" * 78)

    results: List[ExperimentResult] = []
    rows: List[Dict[str, Any]] = []

    total_runs = len(WORKLOADS) * len(BUDGETS) * len(TRACES) * len(ALL_POLICIES)
    print(f"  Running H1 Grid: {total_runs} total experiments...")

    count = 0
    t0 = time.perf_counter()
    for w in WORKLOADS:
        for b in BUDGETS:
            for tr in TRACES:
                for p in ALL_POLICIES:
                    count += 1
                    res = run_single_experiment(w, p, b, tr, cfg=cfg, bias="mixed", seed=SEEDS[0])
                    results.append(res)

                    row = {
                        "workload": w,
                        "policy": p,
                        "budget": b,
                        "budget_pct": f"{b*100:.1f}%",
                        "trace": tr,
                        "rel_error": res.overall_norm_rel_error,
                        "budget_satisfied": res.overall_budget_satisfied,
                        "block_satisfaction_rate": res.block_satisfaction_rate,
                        "energy_savings_pct": res.energy_savings_pct,
                        "energy_ratio": res.energy_ratio,
                        "latency_reduction_pct": res.latency_reduction_pct,
                        "latency_ratio": res.latency_ratio,
                        "pct_EXACT": res.mode_distribution.get(Mode.EXACT.value, 0.0),
                        "pct_APPROX1": res.mode_distribution.get(Mode.APPROX1.value, 0.0),
                        "pct_APPROX2": res.mode_distribution.get(Mode.APPROX2.value, 0.0),
                        "pct_APPROX3": res.mode_distribution.get(Mode.APPROX3.value, 0.0),
                        "total_energy_macs": res.total_energy_macs,
                        "baseline_energy_macs": res.baseline_energy_macs,
                        "overhead_pct": res.overhead_pct,
                        "mae": res.overall_mae,
                        "rmse": res.overall_rmse,
                    }
                    rows.append(row)

    t_el = time.perf_counter() - t0
    print(f"  H1 execution finished in {t_el:.2f}s ({len(results)} runs).")
    return results, rows


def run_h2_experiments(cfg: Config) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Execute H2 direct head-to-head comparison: full_adaptive vs resource_only."""
    print("\n" + "=" * 78)
    print("  EXPERIMENT 2: HYPOTHESIS H2 (SENSITIVITY-AWARE vs RESOURCE-ONLY)")
    print("=" * 78)

    h2_comparisons: List[Dict[str, Any]] = []
    block_comparisons: List[Dict[str, Any]] = []

    # Test across workloads, budgets, and traces for mixed and high sensitivity biases
    biases = ["mixed", "high"]
    for w in WORKLOADS:
        for bias in biases:
            for b in BUDGETS:
                for tr in ["oscillating", "rising", "constant_med"]:
                    res_full = run_single_experiment(w, "full_adaptive", b, tr, cfg=cfg, bias=bias, seed=SEEDS[0])
                    res_res = run_single_experiment(w, "resource_only", b, tr, cfg=cfg, bias=bias, seed=SEEDS[0])

                    comp = {
                        "workload": w,
                        "bias": bias,
                        "budget": b,
                        "budget_pct": f"{b*100:.1f}%",
                        "trace": tr,
                        "full_budget_satisfied": res_full.overall_budget_satisfied,
                        "res_budget_satisfied": res_res.overall_budget_satisfied,
                        "full_block_sat_rate": res_full.block_satisfaction_rate,
                        "res_block_sat_rate": res_res.block_satisfaction_rate,
                        "full_energy_savings_pct": res_full.energy_savings_pct,
                        "res_energy_savings_pct": res_res.energy_savings_pct,
                        "delta_energy_savings_pct": res_full.energy_savings_pct - res_res.energy_savings_pct,
                        "full_rel_error": res_full.overall_norm_rel_error,
                        "res_rel_error": res_res.overall_norm_rel_error,
                        "full_overhead_pct": res_full.overhead_pct,
                        "res_overhead_pct": res_res.overhead_pct,
                        "full_pct_exact": res_full.mode_distribution.get(Mode.EXACT.value, 0.0),
                        "res_pct_exact": res_res.mode_distribution.get(Mode.EXACT.value, 0.0),
                    }
                    h2_comparisons.append(comp)

                    # Also compare per-block discrimination between robust and sensitive blocks
                    for blk_f, blk_r in zip(res_full.blocks, res_res.blocks):
                        block_comparisons.append({
                            "workload": w,
                            "bias": bias,
                            "budget": b,
                            "trace": tr,
                            "block_index": blk_f.block_index,
                            "character": blk_f.character,
                            "sensitivity_amp": blk_f.sensitivity_amp,
                            "full_mode": blk_f.selected_mode,
                            "res_mode": blk_r.selected_mode,
                            "full_error": blk_f.norm_rel_error,
                            "res_error": blk_r.norm_rel_error,
                            "full_sat": blk_f.budget_satisfied,
                            "res_sat": blk_r.budget_satisfied,
                        })

    print(f"  H2 comparison generated {len(h2_comparisons)} pairwise runs.")
    return h2_comparisons, block_comparisons


def run_h3_experiments(cfg: Config) -> List[Dict[str, Any]]:
    """Execute H3 controller and probe overhead analysis and break-even study."""
    print("\n" + "=" * 78)
    print("  EXPERIMENT 3: HYPOTHESIS H3 (CONTROLLER & PROBE OVERHEAD ANALYSIS)")
    print("=" * 78)

    h3_rows: List[Dict[str, Any]] = []

    for w in WORKLOADS:
        for b in [0.01, 0.02, 0.05, 0.10]:
            for tr in ["oscillating", "rising", "constant_med"]:
                res = run_single_experiment(w, "full_adaptive", b, tr, cfg=cfg, bias="mixed", seed=SEEDS[0])

                # Gross savings: compute energy saved without overhead
                base_energy = res.baseline_energy_macs
                total_compute_energy = sum(
                    b_rec.compute_energy_macs for b_rec in res.blocks
                )
                gross_energy_saved = base_energy - total_compute_energy
                gross_savings_pct = (gross_energy_saved / base_energy * 100.0) if base_energy > 0 else 0.0

                overhead_energy = res.total_overhead_macs
                net_energy_saved = gross_energy_saved - overhead_energy
                net_savings_pct = (net_energy_saved / base_energy * 100.0) if base_energy > 0 else 0.0

                overhead_to_savings_ratio = (overhead_energy / gross_energy_saved) if gross_energy_saved > 0 else float("inf")
                is_net_beneficial = net_energy_saved > 0

                h3_rows.append({
                    "workload": w,
                    "budget": b,
                    "budget_pct": f"{b*100:.1f}%",
                    "trace": tr,
                    "total_workload_macs": base_energy,
                    "gross_savings_macs": gross_energy_saved,
                    "gross_savings_pct": gross_savings_pct,
                    "overhead_macs": overhead_energy,
                    "overhead_pct_of_total": res.overhead_pct,
                    "net_savings_macs": net_energy_saved,
                    "net_savings_pct": net_savings_pct,
                    "overhead_to_savings_ratio": overhead_to_savings_ratio,
                    "net_beneficial": is_net_beneficial,
                    "probe_macs": res.total_probe_macs,
                    "norm_rel_error": res.overall_norm_rel_error,
                    "budget_satisfied": res.overall_budget_satisfied,
                })

    print(f"  H3 overhead analysis completed across {len(h3_rows)} configurations.")
    return h3_rows


def run_multiseed_repeatability(cfg: Config) -> List[Dict[str, Any]]:
    """Execute Multi-Seed Repeatability Check across 5 seeds for statistical stability."""
    print("\n" + "=" * 78)
    print("  EXPERIMENT 4: MULTI-SEED STATISTICAL REPEATABILITY CHECK (5 SEEDS)")
    print("=" * 78)

    stats_rows: List[Dict[str, Any]] = []

    for w in WORKLOADS:
        for p in ["exact_only", "fixed_a1", "fixed_a2", "fixed_a3", "resource_only", "full_adaptive"]:
            for b in [0.01, 0.05]:
                tr = "oscillating"

                e_savings_list: List[float] = []
                rel_err_list: List[float] = []
                sat_list: List[bool] = []

                for s in SEEDS:
                    res = run_single_experiment(w, p, b, tr, cfg=cfg, bias="mixed", seed=s)
                    e_savings_list.append(res.energy_savings_pct)
                    rel_err_list.append(res.overall_norm_rel_error)
                    sat_list.append(res.overall_budget_satisfied)

                stats_rows.append({
                    "workload": w,
                    "policy": p,
                    "budget": b,
                    "budget_pct": f"{b*100:.1f}%",
                    "trace": tr,
                    "n_seeds": len(SEEDS),
                    "energy_savings_mean": float(np.mean(e_savings_list)),
                    "energy_savings_std": float(np.std(e_savings_list)),
                    "energy_savings_min": float(np.min(e_savings_list)),
                    "energy_savings_max": float(np.max(e_savings_list)),
                    "rel_error_mean": float(np.mean(rel_err_list)),
                    "rel_error_std": float(np.std(rel_err_list)),
                    "rel_error_min": float(np.min(rel_err_list)),
                    "rel_error_max": float(np.max(rel_err_list)),
                    "satisfaction_rate_pct": float(np.mean([100.0 if s else 0.0 for s in sat_list])),
                })

    print(f"  Multi-seed stability evaluated for {len(stats_rows)} benchmark groups.")
    return stats_rows


def build_overall_summary_table(h1_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Aggregate high-level baseline comparison table across all policies."""
    summary_rows: List[Dict[str, Any]] = []

    for p in ALL_POLICIES:
        p_rows = [r for r in h1_rows if r["policy"] == p]
        if not p_rows:
            continue

        e_savings = [r["energy_savings_pct"] for r in p_rows]
        l_reductions = [r["latency_reduction_pct"] for r in p_rows]
        rel_errors = [r["rel_error"] for r in p_rows]
        satisfactions = [1.0 if r["budget_satisfied"] else 0.0 for r in p_rows]
        overheads = [r["overhead_pct"] for r in p_rows]

        summary_rows.append({
            "policy": p,
            "total_runs": len(p_rows),
            "mean_energy_savings_pct": float(np.mean(e_savings)),
            "std_energy_savings_pct": float(np.std(e_savings)),
            "mean_latency_reduction_pct": float(np.mean(l_reductions)),
            "mean_norm_rel_error": float(np.mean(rel_errors)),
            "budget_compliance_rate_pct": float(np.mean(satisfactions) * 100.0),
            "mean_overhead_pct": float(np.mean(overheads)),
        })

    return summary_rows


def save_csv(data: List[Dict[str, Any]], filepath: Path) -> None:
    """Helper to save list of dicts to CSV."""
    if not data:
        return
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(data[0].keys()))
        writer.writeheader()
        writer.writerows(data)
    print(f"  Saved: {filepath} ({len(data)} rows)")


def main() -> None:
    t_global = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "#" * 78)
    print("  LAUNCHING COMPREHENSIVE H1/H2/H3 HYPOTHESIS BENCHMARK SUITE")
    print("#" * 78)

    cfg = DEFAULT_CONFIG

    # 1. Run H1
    h1_results, h1_rows = run_h1_experiments(cfg)
    save_csv(h1_rows, OUTPUT_DIR / "H1_results.csv")

    # 2. Run H2
    h2_comparisons, h2_blocks = run_h2_experiments(cfg)
    save_csv(h2_comparisons, OUTPUT_DIR / "H2_results.csv")
    save_csv(h2_blocks, OUTPUT_DIR / "H2_block_discrimination.csv")

    # 3. Run H3
    h3_rows = run_h3_experiments(cfg)
    save_csv(h3_rows, OUTPUT_DIR / "H3_results.csv")

    # 4. Multi-Seed Statistical Repeatability
    stats_rows = run_multiseed_repeatability(cfg)
    save_csv(stats_rows, OUTPUT_DIR / "statistical_multi_seed.csv")

    # 5. Overall Summary Table
    overall_summary = build_overall_summary_table(h1_rows)
    save_csv(overall_summary, OUTPUT_DIR / "overall_summary.csv")

    # 6. Save consolidated JSON metadata
    json_data = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "total_h1_runs": len(h1_rows),
            "total_h2_comparisons": len(h2_comparisons),
            "total_h3_evaluations": len(h3_rows),
            "seeds_tested": SEEDS,
            "cost_model_notice": (
                "Energy, latency, and resource metrics are modelled software estimates "
                "based on CostModelConfig and ResourceConfig, not physical FPGA measurements."
            ),
        },
        "summary": overall_summary,
    }
    with open(OUTPUT_DIR / "hypothesis_data.json", "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)
    print(f"  Saved: {OUTPUT_DIR / 'hypothesis_data.json'}")

    t_total = time.perf_counter() - t_global
    print("\n" + "=" * 78)
    print(f"  ALL HYPOTHESIS EXPERIMENTS COMPLETED IN {t_total:.2f} SECONDS")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
