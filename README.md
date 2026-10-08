# Resource- and Sensitivity-Aware Runtime Dynamic Approximation

A software research simulation prototype for evaluating dynamic runtime arithmetic precision adaptation in resource-constrained embedded computing.

> **IMPORTANT NOTICE ON METRICS & HARDWARE:**  
> This repository is a **software-only research simulation prototype**. Energy, latency, and resource metrics are **modelled analytical estimates** derived from standard hardware MAC cost formulations (`CostModelConfig`) and dynamic resource pressure traces (`ResourceConfig`). They are **not physical FPGA or silicon hardware measurements**.

---

## 1. Architecture Overview

The system models a streaming, block-based adaptive precision datapath:

```
Workload Stream (FIR / MatMul / Convolution)
    ↓
Workload Analyzer & Sensitivity Estimator (sub-block perturbation probe)
    ↓
Resource Monitor (simulated resource pressure trace p ∈ [0, 1])
    ↓
Runtime Adaptation Controller (evaluates budget feasibility & minimizes Energy + Risk)
    ↓
Approximation Engine (Exact / Approx-1 / Approx-2 / Approx-3 mantissa quantization)
    ↓
Output & Error Monitor (MAE, MSE, RMSE, Normalized L2 Relative Error)
    ↓
Feedback & Result Logging (CSV & JSON in results/)
```

---

## 2. Computation Modes & Bit-Widths

| Mode | Significand (Mantissa) Bits | Quantization | Error Unit ($2^{-b}$) | Modelled Relative Energy | Modelled Relative Latency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **EXACT** | 16 (nominal reference) | None (`float64`) | $0.0$ | $1.000$ (Reference) | $1.000$ (Reference) |
| **APPROX-1 (A1)** | 10 bits | Truncated Mantissa | $9.77 \times 10^{-4}$ | $\approx 0.404$ (~59.6% savings) | $0.625$ (37.5% reduction) |
| **APPROX-2 (A2)** | 6 bits | Truncated Mantissa | $1.56 \times 10^{-2}$ | $\approx 0.154$ (~84.6% savings) | $0.375$ (62.5% reduction) |
| **APPROX-3 (A3)** | 4 bits | Truncated Mantissa | $6.25 \times 10^{-2}$ | $\approx 0.074$ (~92.6% savings) | $0.250$ (75.0% reduction) |

*Cost Model:* For significand width $b$, modelled per-MAC energy is $E(b) = w_{\text{mul}} b^2 + w_{\text{add}} b$ and latency is $L(b) = \text{lat\_coeff} \cdot b$.

---

## 3. Evaluated Policies & Baselines

1. **`exact_only` (Baseline 1):**  
   Always computes in full `float64` precision. Guarantees 0% error, but consumes maximum energy and latency.
2. **`fixed_a1` / `fixed_a2` / `fixed_a3` (Baseline 2):**  
   Always uses a static approximate precision mode regardless of data sensitivity or resource pressure. Can violate error budgets on ill-conditioned data.
3. **`resource_only` (Baseline 3):**  
   Resource- and budget-aware adaptive policy, but blind to per-block sensitivity (uses a static offline-calibrated average amplification $\bar{A}$).
4. **`full_adaptive` (Proposed Architecture):**  
   Probes each data block at runtime using a lightweight perturbation probe, estimates the block's true error-amplification factor $A$, predicts per-mode error, filters for budget feasibility, and selects the optimal mode to balance resource pressure and accuracy.

---

## 4. Installation & Requirements

* **Python:** 3.10+
* **Dependencies:** `numpy`, `pandas`, `streamlit`

```bash
pip install -r requirements.txt
```

---

## 5. Running the Interactive Dashboard

Launch the Streamlit research dashboard to explore all experimental data, hypothesis validations, trade-offs, and live controller simulations:

```bash
streamlit run dashboard.py
```

### Dashboard Tabs Overview:
* **Tab 1: Overview & Architecture** — Executive summary, headline metrics, and closed-loop dataflow architecture.
* **Tab 2: Precision Trade-Off** — Bit-width vs numerical error vs modelled energy/latency scaling.
* **Tab 3: Baseline Policy Comparison** — Comparative analysis across Exact-Only, Fixed Modes, Resource-Only, and Full-Adaptive.
* **Tab 4: H1: Adaptive Precision** — Interactive filtering across workloads, budgets, and traces for Hypothesis H1.
* **Tab 5: H2: Sensitivity Awareness** — Pairwise comparison between Full-Adaptive and Resource-Only with block discrimination analysis.
* **Tab 6: H3: Overhead & Break-Even** — Gross savings vs probe/controller overhead vs net savings and granularity break-even study.
* **Tab 7: Error Budget Compliance** — Compliance rates across 5 budget tiers ($0.5\%$ to $10.0\%$).
* **Tab 8: Interactive Runtime Demo** — Live invocation of `core.controller.FullAdaptive` with customizable budgets, workloads, and pressure.
* **Tab 9: Research Conclusions** — Formal hypothesis verdicts (H1 SUPPORTED, H2 SUPPORTED, H3 SUPPORTED) and scientific limitations.

---

## 6. Running Validation Suites

Execute the automated test suites to scientifically verify all components:

```bash
# 1. Approximation Engine Validation (< 0.3s)
python validate_approximation.py

# 2. Runtime Adaptive Controller Validation (< 0.4s)
python validate_controller.py

# 3. Metrics & Net-Benefit Audit (< 0.2s)
python audit_metrics_and_generate_final.py
```

---

## 7. Running Full Research Experiments

```bash
# 1. Fast Pipeline Smoke Test (< 0.1s)
python run_experiments.py --smoke-test

# 2. Standard Workload Sweeps
python run_experiments.py --workload all --policy all --budget 0.01 0.05 --trace oscillating

# 3. Comprehensive H1/H2/H3 Benchmark Suite (540 experiments, 5 seeds, < 17s)
python run_hypothesis_experiments.py
```

---

## 8. Data Storage & Artifacts

* **`results/final/`:** Master summary datasets used by publications and the dashboard:
  * `baseline_comparison.csv` — Master policy metrics table.
  * `workload_comparison.csv` — Workload-specific breakdowns (FIR, MatMul, Conv).
  * `precision_tradeoff.csv` — Accuracy vs cost trade-offs per precision mode.
  * `budget_compliance.csv` — Error-budget compliance rates across tiers.
  * `overhead_analysis.csv` — Gross vs Net savings and probe overhead breakdown.
  * `statistical_summary.csv` — 5-seed repeatability data.
* **`results/hypothesis/`:** Full-grid raw experiment runs (`H1_results.csv`, `H2_results.csv`, `H3_results.csv`, `hypothesis_data.json`).
* **`results/`:** Top-level run outputs (`summary.csv`, `block_details.csv`, `experiments.json`).