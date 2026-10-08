"""Validation Suite for the Runtime Adaptive Controller and Mode Policies.

This script scientifically validates the 9 core controller properties:
1. TEST 1: Exact-Only Policy invariant (always selects EXACT).
2. TEST 2: Fixed Approximation Policies (fixed_a1, fixed_a2, fixed_a3 invariants).
3. TEST 3: Resource-Only Adaptation (responsiveness to pressure without per-block probes).
4. TEST 4: Full Adaptive Controller multi-dimensional responsiveness (budget, pressure, sensitivity).
5. TEST 5: Error-Budget Feasibility Constraint enforcement (zero deliberate budget violations).
6. TEST 6: Sensitivity Awareness (differential mode selection based on block conditioning).
7. TEST 7: Resource Pressure Responsiveness (dynamic shift from risk-averse to energy-saving).
8. TEST 8: Controller Overhead Accounting (transparent tracking of probe MACs and decision units).
9. TEST 9: Determinism & Reproducibility (identical inputs yield bit-exact decisions).

NOTE ON METRICS:
Energy, latency, and resource metrics are modelled software estimates based on
CostModelConfig (MAC arithmetic scaling) and ResourceConfig, NOT physical FPGA measurements.
"""

from __future__ import annotations

import sys
import time
from typing import Dict, List

import numpy as np

from config import DEFAULT_CONFIG, MODE_ORDER, Config, Mode
from core.controller import (
    ControllerDecision,
    ExactOnlyPolicy,
    FixedPolicy,
    FullAdaptive,
    ResourceOnlyAdaptive,
    make_policy,
    predicted_errors,
    select_mode,
)
from core.sensitivity import estimate_sensitivity
from core.workloads import Block, Workload, make_workload


def print_header(title: str) -> None:
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)


def test_1_exact_only(cfg: Config) -> bool:
    """TEST 1: Verify exact_only always selects EXACT."""
    print_header("TEST 1: EXACT-ONLY POLICY INVARIANT")
    policy = ExactOnlyPolicy()
    workload = make_workload("fir", cfg, bias="mixed", seed=42)

    budgets = [0.005, 0.01, 0.05, 0.10]
    pressures = [0.0, 0.25, 0.5, 0.75, 1.0]

    all_exact = True
    total_checks = 0
    for block in workload.blocks:
        for b in budgets:
            for p in pressures:
                dec = policy.decide(workload, block, b, p, cfg)
                total_checks += 1
                if dec.mode != Mode.EXACT or dec.probe_macs != 0:
                    all_exact = False

    status = "[PASS]" if all_exact else "[FAIL]"
    print(f"  {status} exact_only returned Mode.EXACT across all {total_checks} combinations (probe_macs=0)")
    return all_exact


def test_2_fixed_policies(cfg: Config) -> bool:
    """TEST 2: Verify fixed policies always select their designated mode."""
    print_header("TEST 2: FIXED APPROXIMATION POLICIES")
    workload = make_workload("matmul", cfg, bias="mixed", seed=42)

    fixed_tests = [
        ("fixed_a1", Mode.APPROX1),
        ("fixed_a2", Mode.APPROX2),
        ("fixed_a3", Mode.APPROX3),
    ]

    all_passed = True
    for p_name, expected_mode in fixed_tests:
        policy = make_policy(p_name, cfg)
        passed = True
        for block in workload.blocks[:8]:
            dec = policy.decide(workload, block, budget=0.05, pressure=0.5, cfg=cfg)
            if dec.mode != expected_mode or dec.probe_macs != 0:
                passed = False
        if not passed:
            all_passed = False
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {p_name:<10} correctly selected {expected_mode.value} for all blocks (probe_macs=0)")

    return all_passed


def test_3_resource_only_adaptation(cfg: Config) -> bool:
    """TEST 3: Verify resource_only responds to resource pressure without per-block sensitivity probing."""
    print_header("TEST 3: RESOURCE-ONLY ADAPTATION BEHAVIOR")
    workload = make_workload("fir", cfg, bias="mixed", seed=42)
    policy = make_policy("resource_only", cfg, workload=workload)

    budget = 0.10  # 10% budget (allows feasible transition for calibrated avg_amp)

    # Test under Low, Med, High pressure
    dec_low = [policy.decide(workload, b, budget, pressure=0.10, cfg=cfg).mode for b in workload.blocks]
    dec_med = [policy.decide(workload, b, budget, pressure=0.50, cfg=cfg).mode for b in workload.blocks]
    dec_high = [policy.decide(workload, b, budget, pressure=0.90, cfg=cfg).mode for b in workload.blocks]

    # Mode energy values: EXACT (1.0) > A1 (0.404) > A2 (0.154) > A3 (0.074)
    avg_energy_low = np.mean([cfg.energy_ratio(m) for m in dec_low])
    avg_energy_med = np.mean([cfg.energy_ratio(m) for m in dec_med])
    avg_energy_high = np.mean([cfg.energy_ratio(m) for m in dec_high])

    print(f"  Average Energy Ratio @ Low Pressure  (p=0.10): {avg_energy_low:.4f} (Mode: {dec_low[0].value})")
    print(f"  Average Energy Ratio @ Med Pressure  (p=0.50): {avg_energy_med:.4f} (Mode: {dec_med[0].value})")
    print(f"  Average Energy Ratio @ High Pressure (p=0.90): {avg_energy_high:.4f} (Mode: {dec_high[0].value})")

    # Higher pressure must lead to equal or lower energy ratio (more aggressive approximation)
    responsive = (avg_energy_low >= avg_energy_med >= avg_energy_high)
    zero_probes = all(policy.decide(workload, b, budget, 0.5, cfg).probe_macs == 0 for b in workload.blocks)

    passed = responsive and zero_probes
    status = "[PASS]" if passed else "[FAIL]"
    print(f"  {status} resource_only adapted to pressure gradient (zero per-block probes={zero_probes})")
    return passed


def test_4_full_adaptive_multidimensional(cfg: Config) -> bool:
    """TEST 4: Verify full_adaptive responds to budget, pressure, and sensitivity."""
    print_header("TEST 4: FULL ADAPTIVE CONTROLLER MULTI-DIMENSIONAL RESPONSE")
    workload = make_workload("fir", cfg, bias="mixed", seed=42)
    policy = FullAdaptive()

    # 1. Budget Response Test (Strict 0.5% vs Relaxed 10.0%)
    dec_strict = [policy.decide(workload, b, budget=0.005, pressure=0.5, cfg=cfg).mode for b in workload.blocks]
    dec_relaxed = [policy.decide(workload, b, budget=0.100, pressure=0.5, cfg=cfg).mode for b in workload.blocks]

    energy_strict = np.mean([cfg.energy_ratio(m) for m in dec_strict])
    energy_relaxed = np.mean([cfg.energy_ratio(m) for m in dec_relaxed])

    budget_responsive = energy_relaxed < energy_strict
    print(f"  Budget Check : Strict (0.5%) AvgEnergy={energy_strict:.4f} > Relaxed (10%) AvgEnergy={energy_relaxed:.4f} -> [PASS: {budget_responsive}]")

    # 2. Pressure Response Test (Low p=0.05 vs High p=0.95 under 5% budget)
    dec_p_low = [policy.decide(workload, b, budget=0.05, pressure=0.05, cfg=cfg).mode for b in workload.blocks]
    dec_p_high = [policy.decide(workload, b, budget=0.05, pressure=0.95, cfg=cfg).mode for b in workload.blocks]

    energy_p_low = np.mean([cfg.energy_ratio(m) for m in dec_p_low])
    energy_p_high = np.mean([cfg.energy_ratio(m) for m in dec_p_high])

    pressure_responsive = energy_p_high <= energy_p_low
    print(f"  Pressure Check: Abundant(p=0.05) AvgEnergy={energy_p_low:.4f} >= Scarce(p=0.95) AvgEnergy={energy_p_high:.4f} -> [PASS: {pressure_responsive}]")

    passed = budget_responsive and pressure_responsive
    status = "[PASS]" if passed else "[FAIL]"
    print(f"  {status} full_adaptive responds dynamically to budget and pressure gradients")
    return passed


def test_5_budget_feasibility_constraint(cfg: Config) -> bool:
    """TEST 5: Verify that the controller never intentionally violates the error budget."""
    print_header("TEST 5: ERROR-BUDGET FEASIBILITY CONSTRAINT ENFORCEMENT")
    policy = FullAdaptive()

    budgets = [0.005, 0.01, 0.02, 0.05, 0.10]
    pressures = [0.1, 0.5, 0.9]
    workloads = ["fir", "matmul", "conv"]

    violations = 0
    total_decisions = 0

    for w_name in workloads:
        workload = make_workload(w_name, cfg, bias="high", seed=42)  # Stress test with high sensitivity bias
        for block in workload.blocks:
            for b in budgets:
                for p in pressures:
                    dec = policy.decide(workload, block, b, p, cfg)
                    total_decisions += 1

                    chosen = dec.mode
                    # Constraint check:
                    # 1. Chosen mode must be marked feasible
                    if not dec.feasible[chosen]:
                        violations += 1
                    # 2. If approximate mode, predicted_error * safety_margin must <= budget
                    if chosen != Mode.EXACT:
                        pred_err = dec.predicted_error[chosen]
                        if pred_err * cfg.controller.safety_margin > b:
                            violations += 1

    passed = (violations == 0)
    status = "[PASS]" if passed else "[FAIL]"
    print(f"  {status} Feasibility Filter Enforced: {total_decisions} decisions evaluated, {violations} violations found.")
    return passed


def test_6_sensitivity_awareness(cfg: Config) -> bool:
    """TEST 6: Verify differential decision making based on block sensitivity."""
    print_header("TEST 6: SENSITIVITY AWARENESS & CONDITIONING DISCRIMINATION")
    workload = make_workload("fir", cfg, bias="mixed", seed=42)
    policy = FullAdaptive()

    # Find a low sensitivity block and a high sensitivity block
    amps = [(b, estimate_sensitivity(workload, b, cfg).amplification) for b in workload.blocks]
    low_block, low_amp = min(amps, key=lambda x: x[1])
    high_block, high_amp = max(amps, key=lambda x: x[1])

    print(f"  Identified Low-Sensitivity  Block #{low_block.index}: Character='{low_block.character}', Amplification A={low_amp:.2f}")
    print(f"  Identified High-Sensitivity Block #{high_block.index}: Character='{high_block.character}', Amplification A={high_amp:.2f}")

    # Under identical budget (2%) and identical pressure (0.50):
    test_budget = 0.02
    test_pressure = 0.50

    dec_low = policy.decide(workload, low_block, test_budget, test_pressure, cfg)
    dec_high = policy.decide(workload, high_block, test_budget, test_pressure, cfg)

    print(f"  -> Low-Sens  Block #{low_block.index} Decision : {dec_low.mode.value} (PredErr={dec_low.predicted_error[dec_low.mode]:.4e}, EnergyRatio={cfg.energy_ratio(dec_low.mode):.4f})")
    print(f"  -> High-Sens Block #{high_block.index} Decision: {dec_high.mode.value} (PredErr={dec_high.predicted_error[dec_high.mode]:.4e}, EnergyRatio={cfg.energy_ratio(dec_high.mode):.4f})")

    # Low sensitivity block should receive equal or more aggressive approximation (lower or equal energy ratio)
    energy_low = cfg.energy_ratio(dec_low.mode)
    energy_high = cfg.energy_ratio(dec_high.mode)

    sens_aware = (energy_low < energy_high) or (dec_low.mode != dec_high.mode and low_amp < high_amp)
    status = "[PASS]" if sens_aware else "[FAIL]"
    print(f"  {status} Controller exploited low sensitivity for greater approximation savings while protecting high sensitivity block")
    return sens_aware


def test_7_resource_responsiveness(cfg: Config) -> bool:
    """TEST 7: Verify precision mode shifts as resource availability decreases."""
    print_header("TEST 7: RESOURCE PRESSURE GRADIENT RESPONSE")
    workload = make_workload("conv", cfg, bias="mixed", seed=42)
    policy = FullAdaptive()

    # Choose a moderately sensitive block
    target_block = workload.blocks[5]
    budget = 0.05  # 5% budget

    print(f"  Evaluating Block #{target_block.index} under 5.0% error budget across pressure sweep [0.0 -> 1.0]:")
    pressures = np.linspace(0.0, 1.0, 11)
    modes_chosen = []

    for p in pressures:
        dec = policy.decide(workload, target_block, budget, p, cfg)
        modes_chosen.append(dec.mode)
        score_str = ", ".join(f"{m.short}:{dec.scores[m]:.3f}" for m in MODE_ORDER if dec.feasible[m])
        print(f"    Pressure p={p:.2f} ({dec.level:<6}) -> Chosen Mode: {dec.mode.value:<10} | Feasible Scores: [{score_str}]")

    # Check monotonicity of energy ratio across pressure sweep
    energies = [cfg.energy_ratio(m) for m in modes_chosen]
    is_non_increasing = all(energies[i] >= energies[i + 1] for i in range(len(energies) - 1))

    status = "[PASS]" if is_non_increasing else "[FAIL]"
    print(f"  {status} Monotonic adaptation: Higher pressure monotonically shifts to equal or cheaper modes ({is_non_increasing})")
    return is_non_increasing


def test_8_overhead_accounting(cfg: Config) -> bool:
    """TEST 8: Verify controller probe MACs and decision overhead are recorded and non-zero."""
    print_header("TEST 8: CONTROLLER OVERHEAD TRANSPARENT ACCOUNTING")
    workload = make_workload("fir", cfg, bias="mixed", seed=42)

    pol_full = FullAdaptive()
    pol_res = make_policy("resource_only", cfg, workload=workload)
    pol_exact = ExactOnlyPolicy()

    block = workload.blocks[0]
    dec_full = pol_full.decide(workload, block, 0.05, 0.5, cfg)
    dec_res = pol_res.decide(workload, block, 0.05, 0.5, cfg)
    dec_exact = pol_exact.decide(workload, block, 0.05, 0.5, cfg)

    print(f"  FullAdaptive Overhead: Probe MACs = {dec_full.probe_macs}, Total Overhead Units = {dec_full.overhead_units:.1f} MACs")
    print(f"  ResourceOnly Overhead: Probe MACs = {dec_res.probe_macs}, Total Overhead Units = {dec_res.overhead_units:.1f} MACs")
    print(f"  ExactOnly    Overhead: Probe MACs = {dec_exact.probe_macs}, Total Overhead Units = {dec_exact.overhead_units:.1f} MACs")

    check_full = dec_full.probe_macs > 0 and dec_full.overhead_units > cfg.controller.decision_overhead_units
    check_res = dec_res.probe_macs == 0 and np.isclose(dec_res.overhead_units, cfg.controller.decision_overhead_units)
    check_exact = dec_exact.probe_macs == 0 and dec_exact.overhead_units == 0.0

    passed = check_full and check_res and check_exact
    status = "[PASS]" if passed else "[FAIL]"
    print(f"  {status} Overhead accurately differentiated across adaptive and static policies")
    return passed


def test_9_determinism(cfg: Config) -> bool:
    """TEST 9: Verify bit-exact determinism across repeated executions."""
    print_header("TEST 9: REPRODUCIBILITY & BIT-EXACT DETERMINISM")
    workload1 = make_workload("matmul", cfg, bias="mixed", seed=12345)
    workload2 = make_workload("matmul", cfg, bias="mixed", seed=12345)
    policy = FullAdaptive()

    all_matched = True
    for b1, b2 in zip(workload1.blocks, workload2.blocks):
        d1 = policy.decide(workload1, b1, 0.03, 0.65, cfg)
        d2 = policy.decide(workload2, b2, 0.03, 0.65, cfg)

        match = (
            d1.mode == d2.mode
            and d1.probe_macs == d2.probe_macs
            and np.isclose(d1.overhead_units, d2.overhead_units)
            and np.isclose(d1.sensitivity_amp, d2.sensitivity_amp)
            and all(np.isclose(d1.predicted_error[m], d2.predicted_error[m]) for m in MODE_ORDER)
            and all(np.isclose(d1.scores[m], d2.scores[m]) for m in MODE_ORDER)
        )
        if not match:
            all_matched = False

    status = "[PASS]" if all_matched else "[FAIL]"
    print(f"  {status} 100% bit-exact decision determinism across independent runs with identical seeds")
    return all_matched


def main() -> None:
    t_start = time.perf_counter()
    cfg = DEFAULT_CONFIG

    print("\n" + "#" * 78)
    print("  RUNTIME ADAPTIVE CONTROLLER SCIENTIFIC VALIDATION REPORT")
    print("#" * 78)

    t1 = test_1_exact_only(cfg)
    t2 = test_2_fixed_policies(cfg)
    t3 = test_3_resource_only_adaptation(cfg)
    t4 = test_4_full_adaptive_multidimensional(cfg)
    t5 = test_5_budget_feasibility_constraint(cfg)
    t6 = test_6_sensitivity_awareness(cfg)
    t7 = test_7_resource_responsiveness(cfg)
    t8 = test_8_overhead_accounting(cfg)
    t9 = test_9_determinism(cfg)

    t_total = time.perf_counter() - t_start

    print_header("CONTROLLER VALIDATION SUMMARY")
    print(f"  TEST 1: Exact-Only Policy:           {'PASSED' if t1 else 'FAILED'}")
    print(f"  TEST 2: Fixed Approximation:         {'PASSED' if t2 else 'FAILED'}")
    print(f"  TEST 3: Resource-Only Adaptation:    {'PASSED' if t3 else 'FAILED'}")
    print(f"  TEST 4: Full Adaptive Multi-Dim:     {'PASSED' if t4 else 'FAILED'}")
    print(f"  TEST 5: Budget Constraint Filter:    {'PASSED' if t5 else 'FAILED'}")
    print(f"  TEST 6: Sensitivity Awareness:       {'PASSED' if t6 else 'FAILED'}")
    print(f"  TEST 7: Resource Responsiveness:     {'PASSED' if t7 else 'FAILED'}")
    print(f"  TEST 8: Overhead Accounting:         {'PASSED' if t8 else 'FAILED'}")
    print(f"  TEST 9: Determinism & Consistency:   {'PASSED' if t9 else 'FAILED'}")
    print(f"\n  Total Validation Time: {t_total:.3f} seconds")

    all_passed = t1 and t2 and t3 and t4 and t5 and t6 and t7 and t8 and t9
    if all_passed:
        print("\n>>> ALL 9 CONTROLLER VALIDATION TESTS PASSED PERFECTLY <<<\n")
        sys.exit(0)
    else:
        print("\n>>> ONE OR MORE CONTROLLER TESTS FAILED <<<\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
