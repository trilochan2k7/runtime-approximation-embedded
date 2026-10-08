# Scientific Metrics Audit and Baseline Evaluation Report

**Document Version:** 1.0  
**Audit Date:** October 8, 2026  
**Auditor:** Antigravity Engineering & Research  
**Target System:** Software Simulation Prototype of Runtime Dynamic Approximation  
**Disclaimer:** All energy, latency, and resource metrics are **modelled software estimates** based on analytical hardware complexity formulations (`CostModelConfig`) and dynamic resource traces (`ResourceConfig`), **not physical FPGA or silicon hardware measurements**.

---

## 1. Metric Definitions & Mathematical Validation

Every error, energy, latency, and overhead metric in the codebase was audited for mathematical correctness, dimensional consistency, normalization integrity, and numerical stability (absence of NaN/Inf or division-by-zero).

### A. Numerical Error Metrics (`core/metrics.py`)
Given approximate block output $\mathbf{y}_{\text{approx}} \in \mathbb{R}^N$ and exact reference output $\mathbf{y}_{\text{exact}} \in \mathbb{R}^N$ with error vector $\mathbf{e} = \mathbf{y}_{\text{approx}} - \mathbf{y}_{\text{exact}}$:

1. **Mean Absolute Error (MAE):**
   $$\text{MAE} = \frac{1}{N} \sum_{i=1}^N |e_i|$$
   *Status:* Verified mathematically correct.
2. **Mean Squared Error (MSE):**
   $$\text{MSE} = \frac{1}{N} \sum_{i=1}^N e_i^2$$
   *Status:* Verified mathematically correct.
3. **Root Mean Squared Error (RMSE):**
   $$\text{RMSE} = \sqrt{\text{MSE}}$$
   *Status:* Verified mathematically correct.
4. **Maximum Absolute Error:**
   $$\text{MaxAbsErr} = \max_{1 \le i \le N} |e_i|$$
   *Status:* Verified mathematically correct.
5. **Normalized $L_2$ Relative Error (Primary Error Metric):**
   $$\text{RelError}_{L_2} = \frac{\|\mathbf{y}_{\text{approx}} - \mathbf{y}_{\text{exact}}\|_2}{\|\mathbf{y}_{\text{exact}}\|_2 + \epsilon} \quad (\epsilon = 10^{-12})$$
   *Status:* Verified mathematically correct. Robust against near-zero individual components.

---

## 2. Energy Model Audit (`config.py:CostModelConfig`)

### Analytical Formulation
The energy consumption of an $n$-bit multiply-accumulate (MAC) datapath is modelled after combinational multiplier and adder gate scaling:
$$E_{\text{MAC}}(b) = w_{\text{mul}} \cdot b^2 + w_{\text{add}} \cdot b$$
where $w_{\text{mul}} = 1.0$, $w_{\text{add}} = 1.0$, and $b$ is the significand bit-width.

Energy is normalized relative to the **EXACT mode reference** ($b_{\text{exact}} = 16$ bits):
$$E_{\text{exact}} = 1.0 \cdot (16)^2 + 1.0 \cdot (16) = 256 + 16 = 272 \text{ units}$$

$$\text{Energy Ratio}(b) = \frac{E_{\text{MAC}}(b)}{E_{\text{exact}}} = \frac{b^2 + b}{272}$$

$$\text{Modelled Energy Savings \%} = \left(1 - \text{Energy Ratio}(b)\right) \times 100\%$$

### Recalculation and Validation of Baseline Modes:

| Mode | Bits ($b$) | Quantized? | Raw Energy $E(b)$ | Energy Ratio | Modelled Energy Savings | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **EXACT** | 16 | No | $272.0$ | $1.0000$ | **$0.00\%$** (Reference) | Verified |
| **APPROX-1** | 10 | Yes | $10^2 + 10 = 110.0$ | $\frac{110}{272} \approx 0.404412$ | **$59.56\%$** | Verified |
| **APPROX-2** | 6 | Yes | $6^2 + 6 = 42.0$ | $\frac{42}{272} \approx 0.154412$ | **$84.56\%$** | Verified |
| **APPROX-3** | 4 | Yes | $4^2 + 4 = 20.0$ | $\frac{20}{272} \approx 0.073529$ | **$92.65\%$** | Verified |

---

## 3. Latency Model Audit (`config.py:CostModelConfig`)

### Analytical Formulation
Critical-path delay of the reduced-width datapath scales linearly with operand bit-width:
$$L_{\text{MAC}}(b) = \text{lat\_coeff} \cdot b + \text{lat\_const} \quad (\text{lat\_coeff} = 1.0, \text{lat\_const} = 0.0)$$

Normalized relative to EXACT ($b = 16$):
$$\text{Latency Ratio}(b) = \frac{b}{16}$$

$$\text{Modelled Latency Reduction \%} = \left(1 - \frac{b}{16}\right) \times 100\%$$

### Validation:
* **EXACT (16 bits):** Ratio $= 1.0000 \implies \mathbf{0.00\%}$ reduction.
* **APPROX-1 (10 bits):** Ratio $= 0.6250 \implies \mathbf{37.50\%}$ reduction.
* **APPROX-2 (6 bits):** Ratio $= 0.3750 \implies \mathbf{62.50\%}$ reduction.
* **APPROX-3 (4 bits):** Ratio $= 0.2500 \implies \mathbf{75.00\%}$ reduction.

---

## 4. Comprehensive Overhead Audit

### Components of Adaptation Overhead
1. **Raw Controller Decision Overhead:**  
   Fixed cost of evaluating candidate mode scores $\text{Score}(m) = \text{EnergyRatio}(m) + \lambda(p) \cdot \text{Risk}(m)$:
   $$\text{Decision Overhead} = 8.0 \text{ EXACT-MAC equivalent units per decision}$$
2. **Sensitivity Probe Overhead:**  
   Sub-block evaluation on $k$ elements ($k \le 32$) requiring one exact pass and one reduced-precision pass ($2 \times k \times \text{inner\_dim}$ MACs):
   $$\text{Probe MACs} = 2 \cdot k \cdot \text{inner\_dim}$$
   Total Overhead per block: $\text{Overhead}_{\text{block}} = \text{Probe MACs} + 8.0$.

### Resolution of the Overall 22.16% Overhead Figure:
The reported overall mean overhead of **$22.16\%$** in `overall_summary.csv` is calculated as:
$$\text{Overhead \% of Consumed Energy} = \frac{\text{Total Overhead Energy}}{\text{Total Consumed Energy}} \times 100\%$$
Because consumed energy is in the denominator, small workloads with aggressive approximation have higher overhead percentages. 

#### Breakdown by Workload:
* **FIR Filtering ($12,288$ MACs/block):** Probe takes $3,072$ MACs ($25\%$ of block compute), representing **$39.22\%$** of consumed energy.
* **Image Convolution ($80,000$ MACs/block):** Probe takes $1,210$ MACs ($1.5\%$ of block compute), representing **$18.23\%$** of consumed energy.
* **Matrix Multiplication ($184,320$ MACs/block):** Probe takes $8,000$ MACs ($4.3\%$ of block compute), representing **$10.91\%$** of consumed energy.

---

## 5. Net Energy Benefit Formulation

Net energy savings strictly subtract all probing and decision overhead from gross compute energy savings:
$$\text{Gross Savings (MACs)} = E_{\text{baseline}} - E_{\text{compute}}$$
$$\text{Net Savings (MACs)} = \text{Gross Savings} - \text{Overhead} = E_{\text{baseline}} - (E_{\text{compute}} + \text{Overhead})$$
$$\text{Net Savings \%} = \frac{\text{Net Savings}}{E_{\text{baseline}}} \times 100\%$$

*Verification:* $\text{Net Savings} \le \text{Gross Savings}$ holds unconditionally across all 540 evaluated configurations.

---

## 6. Master Baseline Comparison Table

| Policy | Gross Energy Savings (%) | Net Energy Savings (%) | Std Net Savings (%) | Latency Reduction (%) | Mean $L_2$ Rel. Error | Budget Compliance | Overhead (% of consumed) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`exact_only`** | $0.00\%$ | $0.00\%$ | $0.00\%$ | $0.00\%$ | $0.0000$ | **$100.00\%$** | $0.00\%$ |
| **`fixed_a1`** | $59.56\%$ | $59.56\%$ | $0.00\%$ | $37.50\%$ | $1.90 \times 10^{-4}$ | **$100.00\%$** | $0.00\%$ |
| **`fixed_a2`** | $84.56\%$ | $84.56\%$ | $0.00\%$ | $62.50\%$ | $2.97 \times 10^{-3}$ | **$93.33\%$** | $0.00\%$ |
| **`fixed_a3`** | $92.65\%$ | $92.65\%$ | $0.00\%$ | $75.00\%$ | $1.60 \times 10^{-2}$ | **$60.00\%$** | $0.00\%$ |
| **`resource_only`** | $37.99\%$ | $37.96\%$ | $29.59\%$ | $24.64\%$ | $6.36 \times 10^{-4}$ | **$100.00\%$** | $0.08\%$ |
| **`full_adaptive`** | **$61.08\%$** | **$49.13\%$** | $16.87\%$ | **$43.05\%$** | $3.89 \times 10^{-3}$ | **$100.00\%$** | $22.16\%$ |

---

## 7. Precision Mode Trade-Off Table

| Mode | Bit-Width | Quantization Error Bound ($2^{-b}$) | Modelled Energy Ratio | Energy Savings | Modelled Latency Ratio | Latency Reduction |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **EXACT** | 16 | $0.000000$ (Exact) | $1.0000$ | $0.00\%$ | $1.0000$ | $0.00\%$ |
| **APPROX-1** | 10 | $9.765625 \times 10^{-4}$ | $0.4044$ | $59.56\%$ | $0.6250$ | $37.50\%$ |
| **APPROX-2** | 6 | $1.562500 \times 10^{-2}$ | $0.1544$ | $84.56\%$ | $0.3750$ | $62.50\%$ |
| **APPROX-3** | 4 | $6.250000 \times 10^{-2}$ | $0.0735$ | $92.65\%$ | $0.2500$ | $75.00\%$ |

---

## 8. Budget Compliance Breakdown by Error Budget Tier

| Policy | Evaluated | Satisfied | Violated | Overall Compliance | $0.5\%$ Budget | $1.0\%$ Budget | $2.0\%$ Budget | $5.0\%$ Budget | $10.0\%$ Budget |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`exact_only`** | 90 | 90 | 0 | **$100.00\%$** | $100\%$ ($18/18$) | $100\%$ ($18/18$) | $100\%$ ($18/18$) | $100\%$ ($18/18$) | $100\%$ ($18/18$) |
| **`fixed_a1`** | 90 | 90 | 0 | **$100.00\%$** | $100\%$ ($18/18$) | $100\%$ ($18/18$) | $100\%$ ($18/18$) | $100\%$ ($18/18$) | $100\%$ ($18/18$) |
| **`fixed_a2`** | 90 | 84 | 6 | **$93.33\%$** | **$66.7\%$** ($12/18$) | $100\%$ ($18/18$) | $100\%$ ($18/18$) | $100\%$ ($18/18$) | $100\%$ ($18/18$) |
| **`fixed_a3`** | 90 | 54 | 36 | **$60.00\%$** | **$0.0\%$** ($0/18$) | **$0.0\%$** ($0/18$) | $100\%$ ($18/18$) | $100\%$ ($18/18$) | $100\%$ ($18/18$) |
| **`resource_only`** | 90 | 90 | 0 | **$100.00\%$** | $100\%$ ($18/18$) | $100\%$ ($18/18$) | $100\%$ ($18/18$) | $100\%$ ($18/18$) | $100\%$ ($18/18$) |
| **`full_adaptive`** | 90 | 90 | 0 | **$100.00\%$** | **$100\%$** ($18/18$) | **$100\%$** ($18/18$) | **$100\%$** ($18/18$) | **$100\%$** ($18/18$) | **$100\%$** ($18/18$) |

---

## 9. Statistical Repeatability (5 Seeds)

Evaluated across 5 seeds (`20260408`, `42`, `12345`, `999`, `777`):
* Standard deviation across all 36 workload/policy groups was $\pm 0.00\%$, confirming **100% deterministic reproducibility** of the software simulator.

---

## 10. Summary Data Files Generated (`results/final/`)
* [`results/final/baseline_comparison.csv`](file:///c:/Users/trilo/runtime-approximation-embedded/results/final/baseline_comparison.csv)
* [`results/final/workload_comparison.csv`](file:///c:/Users/trilo/runtime-approximation-embedded/results/final/workload_comparison.csv)
* [`results/final/precision_tradeoff.csv`](file:///c:/Users/trilo/runtime-approximation-embedded/results/final/precision_tradeoff.csv)
* [`results/final/budget_compliance.csv`](file:///c:/Users/trilo/runtime-approximation-embedded/results/final/budget_compliance.csv)
* [`results/final/overhead_analysis.csv`](file:///c:/Users/trilo/runtime-approximation-embedded/results/final/overhead_analysis.csv)
* [`results/final/statistical_summary.csv`](file:///c:/Users/trilo/runtime-approximation-embedded/results/final/statistical_summary.csv)
