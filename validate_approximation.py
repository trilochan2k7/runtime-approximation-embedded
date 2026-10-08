"""Validation Suite for the Approximation Engine, Error Metrics, and Cost Models.

This script scientifically validates:
1. Exact Reference Integrity (EXACT mode produces zero approximation error).
2. Progressive Error Scaling (EXACT -> APPROX-1 -> APPROX-2 -> APPROX-3).
3. Monotonic Cost/Energy/Latency Reductions across approximation levels.
4. Workload Condition Sensitivity (low vs mixed vs high sensitivity regimes).
5. Mathematical Correctness of Error Metrics (MAE, MSE, RMSE, Relative Error).
6. Internal Consistency of the analytical cost model.

NOTE ON METRICS:
Energy, latency, and resource metrics are modelled software estimates based on
CostModelConfig (MAC arithmetic scaling) and are NOT physical FPGA measurements.
"""

from __future__ import annotations

import sys
import time
from typing import Dict, List, Tuple

import numpy as np

from config import DEFAULT_CONFIG, MODE_ORDER, Config, Mode
from core.approximation import approx_dot_blocks, approx_matmul, quantize_significand
from core.metrics import ErrorMetrics, error_metrics, relative_error
from core.sensitivity import estimate_sensitivity
from core.workloads import Workload, make_workload


def print_section(title: str) -> None:
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)


def validate_metric_formulas() -> bool:
    """Test 1: Validate mathematical correctness of error metric functions."""
    print_section("TEST 1: MATHEMATICAL CORRECTNESS OF ERROR METRICS")

    # Known synthetic test vectors
    exact = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    approx = np.array([1.1, 1.8, 3.3, 3.6], dtype=np.float64)
    diff = approx - exact  # [+0.1, -0.2, +0.3, -0.4]

    # Hand-calculated reference values:
    expected_mae = (0.1 + 0.2 + 0.3 + 0.4) / 4.0  # 0.25
    expected_mse = (0.01 + 0.04 + 0.09 + 0.16) / 4.0  # 0.075
    expected_rmse = np.sqrt(0.075)  # ~0.27386127875258304
    expected_max_abs = 0.4
    expected_norm_rel = np.linalg.norm(diff) / np.linalg.norm(exact)  # sqrt(0.30) / sqrt(30) = sqrt(0.01) = 0.10

    metrics = error_metrics(approx, exact)

    checks = [
        ("MAE", np.isclose(metrics.mae, expected_mae), metrics.mae, expected_mae),
        ("MSE", np.isclose(metrics.mse, expected_mse), metrics.mse, expected_mse),
        ("RMSE", np.isclose(metrics.rmse, expected_rmse), metrics.rmse, expected_rmse),
        ("Max Absolute Error", np.isclose(metrics.max_abs_error, expected_max_abs), metrics.max_abs_error, expected_max_abs),
        ("Norm L2 Relative Error", np.isclose(metrics.norm_rel_error, expected_norm_rel), metrics.norm_rel_error, expected_norm_rel),
    ]

    all_passed = True
    for name, passed, actual, expected in checks:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {name:<24}: actual={actual:.6f}, expected={expected:.6f}")
        if not passed:
            all_passed = False

    # Also test identical array gives 0.0
    zero_metrics = error_metrics(exact, exact)
    zero_ok = (
        zero_metrics.mae == 0.0
        and zero_metrics.mse == 0.0
        and zero_metrics.rmse == 0.0
        and zero_metrics.max_abs_error == 0.0
        and zero_metrics.norm_rel_error == 0.0
    )
    print(f"  {'[PASS]' if zero_ok else '[FAIL]'} Zero Error on Identical Arrays: norm_rel={zero_metrics.norm_rel_error:.1e}")
    if not zero_ok:
        all_passed = False

    return all_passed


def validate_cost_model_consistency(cfg: Config) -> bool:
    """Test 2: Validate cost model monotonic scaling for energy and latency."""
    print_section("TEST 2: ANALYTICAL COST & LATENCY MODEL CONSISTENCY")

    print(f"  Cost weights: w_mul={cfg.cost.w_mul}, w_add={cfg.cost.w_add}, lat_coeff={cfg.cost.lat_coeff}")
    print(f"  Nominal EXACT bits: {cfg.exact_bits()}")
    print("-" * 78)
    print(f"  {'Mode':<12} {'Significand Bits':<18} {'Energy Ratio':<15} {'Latency Ratio':<15} {'Modelled E-Savings':<18}")
    print("-" * 78)

    prev_energy = float("inf")
    prev_latency = float("inf")
    all_monotonic = True

    for mode in MODE_ORDER:
        mc = cfg.modes[mode]
        e_ratio = cfg.energy_ratio(mode)
        l_ratio = cfg.latency_ratio(mode)
        savings = (1.0 - e_ratio) * 100.0

        print(f"  {mode.value:<12} {str(mc.bits) + (' (exact)' if not mc.quantise else ''):<18} {e_ratio:<15.4f} {l_ratio:<15.4f} {savings:>6.1f}%")

        if mode == Mode.EXACT:
            if not np.isclose(e_ratio, 1.0) or not np.isclose(l_ratio, 1.0):
                print(f"    [FAIL] EXACT must have energy ratio == 1.0 and latency ratio == 1.0")
                all_monotonic = False
        else:
            if e_ratio >= prev_energy or l_ratio >= prev_latency:
                print(f"    [FAIL] Mode {mode.value} did not strictly reduce energy/latency compared to previous mode!")
                all_monotonic = False

        prev_energy = e_ratio
        prev_latency = l_ratio

    status = "[PASS]" if all_monotonic else "[FAIL]"
    print(f"\n  {status} Cost model strictly monotonic (Energy & Latency decrease with reduced bit-width)")
    return all_monotonic


def validate_workload_approximation(
    workload_name: str, cfg: Config, bias: str = "mixed", seed: int = 42
) -> Tuple[bool, Dict[str, ErrorMetrics]]:
    """Test 3: Validate approximation behavior on a specific workload."""
    workload: Workload = make_workload(workload_name, cfg, bias=bias, seed=seed)

    # 1. Compute full exact reference
    exact_blocks = [workload.compute(b, None) for b in workload.blocks]
    exact_full = np.concatenate([arr.ravel() for arr in exact_blocks])

    results: Dict[str, ErrorMetrics] = {}
    print(f"\n  Workload: {workload_name.upper()} (Bias: {bias}, Blocks: {len(workload.blocks)})")
    print("  " + "-" * 74)
    print(f"  {'Mode':<10} {'Bits':<6} {'Norm RelErr':<14} {'MAE':<14} {'RMSE':<14} {'MaxAbsErr':<14}")
    print("  " + "-" * 74)

    for mode in MODE_ORDER:
        bits = cfg.mode_bits(mode)
        approx_blocks = [workload.compute(b, bits) for b in workload.blocks]
        approx_full = np.concatenate([arr.ravel() for arr in approx_blocks])

        metrics = error_metrics(approx_full, exact_full)
        results[mode.value] = metrics
        bit_str = str(bits) if bits is not None else "EXACT"
        print(
            f"  {mode.value:<10} {bit_str:<6} "
            f"{metrics.norm_rel_error:<14.6e} "
            f"{metrics.mae:<14.6e} "
            f"{metrics.rmse:<14.6e} "
            f"{metrics.max_abs_error:<14.6e}"
        )

    # Check assertions:
    # A. EXACT must have zero error
    exact_err = results[Mode.EXACT.value]
    exact_is_zero = np.isclose(exact_err.norm_rel_error, 0.0, atol=1e-15) and np.isclose(exact_err.mae, 0.0, atol=1e-15)

    # B. Progressive error scaling: EXACT < APPROX-1 < APPROX-2 < APPROX-3
    err_a1 = results[Mode.APPROX1.value].norm_rel_error
    err_a2 = results[Mode.APPROX2.value].norm_rel_error
    err_a3 = results[Mode.APPROX3.value].norm_rel_error

    progressive = (0.0 == exact_err.norm_rel_error) and (err_a1 < err_a2) and (err_a2 < err_a3)

    ok = exact_is_zero and progressive
    status = "[PASS]" if ok else "[FAIL]"
    print(f"  {status} Exact is zero reference ({exact_is_zero}) & Progressive Error ({progressive}): 0 < A1({err_a1:.2e}) < A2({err_a2:.2e}) < A3({err_a3:.2e})")

    return ok, results


def validate_sensitivity_regimes(cfg: Config) -> bool:
    """Test 4: Validate that high sensitivity bias exhibits higher error amplification than low sensitivity."""
    print_section("TEST 4: SENSITIVITY BIAS REGIME BEHAVIOR")

    all_passed = True
    for w_name in ["fir", "matmul", "conv"]:
        w_low = make_workload(w_name, cfg, bias="low", seed=42)
        w_high = make_workload(w_name, cfg, bias="high", seed=42)

        # Average sensitivity amplification factor
        amps_low = [estimate_sensitivity(w_low, b, cfg).amplification for b in w_low.blocks]
        amps_high = [estimate_sensitivity(w_high, b, cfg).amplification for b in w_high.blocks]

        mean_a_low = float(np.mean(amps_low))
        mean_a_high = float(np.mean(amps_high))

        # Full-run relative error at APPROX-2
        ex_low = np.concatenate([w_low.compute(b, None).ravel() for b in w_low.blocks])
        ap_low = np.concatenate([w_low.compute(b, cfg.mode_bits(Mode.APPROX2)).ravel() for b in w_low.blocks])
        err_low = relative_error(ap_low, ex_low)

        ex_high = np.concatenate([w_high.compute(b, None).ravel() for b in w_high.blocks])
        ap_high = np.concatenate([w_high.compute(b, cfg.mode_bits(Mode.APPROX2)).ravel() for b in w_high.blocks])
        err_high = relative_error(ap_high, ex_high)

        passed = (mean_a_high > mean_a_low) and (err_high > err_low)
        if not passed:
            all_passed = False

        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {w_name.upper():<7}: LowBias (Amp={mean_a_low:.2f}, Err={err_low:.4e}) < HighBias (Amp={mean_a_high:.2f}, Err={err_high:.4e})")

    return all_passed


def main() -> None:
    t_start = time.perf_counter()
    cfg = DEFAULT_CONFIG

    print("\n" + "#" * 78)
    print("  RUNTIME APPROXIMATION ENGINE VALIDATION REPORT")
    print("#" * 78)

    t1_ok = validate_metric_formulas()
    t2_ok = validate_cost_model_consistency(cfg)

    print_section("TEST 3: WORKLOAD-SPECIFIC APPROXIMATION PROGRESSION")
    t3_fir_ok, _ = validate_workload_approximation("fir", cfg, bias="mixed")
    t3_mat_ok, _ = validate_workload_approximation("matmul", cfg, bias="mixed")
    t3_cnv_ok, _ = validate_workload_approximation("conv", cfg, bias="mixed")
    t3_all_ok = t3_fir_ok and t3_mat_ok and t3_cnv_ok

    t4_ok = validate_sensitivity_regimes(cfg)

    t_total = time.perf_counter() - t_start

    print_section("VALIDATION SUMMARY")
    print(f"  1. Error Metrics Correctness:       {'PASSED' if t1_ok else 'FAILED'}")
    print(f"  2. Cost Model Consistency:          {'PASSED' if t2_ok else 'FAILED'}")
    print(f"  3. Workload Progression (FIR):      {'PASSED' if t3_fir_ok else 'FAILED'}")
    print(f"  4. Workload Progression (MatMul):   {'PASSED' if t3_mat_ok else 'FAILED'}")
    print(f"  5. Workload Progression (Conv):     {'PASSED' if t3_cnv_ok else 'FAILED'}")
    print(f"  6. Sensitivity Regime Scaling:      {'PASSED' if t4_ok else 'FAILED'}")
    print(f"\n  Total Execution Time: {t_total:.3f} seconds")

    all_passed = t1_ok and t2_ok and t3_all_ok and t4_ok
    if all_passed:
        print("\n>>> ALL VALIDATION CHECKS PASSED PERFECTLY <<<\n")
        sys.exit(0)
    else:
        print("\n>>> ONE OR MORE VALIDATION CHECKS FAILED <<<\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
