# PROJECT AUDIT: runtime-approximation-embedded

**Audit Date:** October 8, 2026  
**Auditor:** Antigravity Engineering Takeover Agent  
**Repository State:** Static Inspection & Non-destructive Validation Complete  

---

## 1. Existing Project Purpose

The existing codebase implements a **Python-based architectural simulation and benchmarking framework for Runtime Dynamic Approximation in Embedded/DSP Computing**. 

Its goal is to evaluate adaptive arithmetic precision control across streaming signal-processing and linear algebra workloads (FIR filtering, Matrix Multiplication, 2D Image Convolution). It dynamically adjusts multiplier operand precision (floating-point significand/mantissa bit-width truncation) across individual data blocks to minimize simulated energy and latency costs while strictly bounding output relative error within user-specified budgets under fluctuating resource pressure.

---

## 2. Existing Architecture

The existing Python simulator is structured around a block-based closed-loop adaptive control architecture:

```
+-----------------------------------------------------------------------------------+
|                               SIMULATION ARCHITECTURE                             |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  [ Workload Stream ] -----> [ Block Sub-probe ]                                   |
|   (FIR / MatMul / Conv)           |                                               |
|           |                       v                                               |
|           |             [ Sensitivity Estimator ] -> Error Amplification (A)      |
|           |                       |                                               |
|           |                       v                                               |
|           |             [ Adaptive Controller ] <--- [ Simulated Resource Trace ] |
|           |             - Predicts error per mode    (Pressure p in [0, 1])       |
|           |             - Checks budget feasibility                               |
|           |             - Minimizes Energy + Risk                                 |
|           |                       |                                               |
|           |                       v Chosen Mode (EXACT / APPROX-1 / APPROX-2)     |
|           +---------------------->+                                               |
|                                   v                                               |
|                       [ Approximation Engine ]                                    |
|                     (Quantized Significand MACs)                                  |
|                                   |                                               |
|                                   v                                               |
|                         [ Output & Error Metrics ]                                |
|                        (MAE, MSE, RMSE, NormRelErr)                               |
+-----------------------------------------------------------------------------------+
```

---

## 3. Existing Files and Their Responsibilities

### 1. `config.py` (Repository Root)
* **Location:** `c:\Users\trilo\runtime-approximation-embedded\config.py` (293 lines)
* **Responsibility:** Central configuration schema containing dataclasses for modes (`ModeConfig`), cost models (`CostModelConfig`), controller policies (`ControllerConfig`), sensitivity probes (`SensitivityConfig`), simulated resources (`ResourceConfig`), and workloads (`WorkloadConfig`).
* **Connections:** Imported by `core.resource_model`, `core.sensitivity`, `core.workloads`, `core.controller`, and top-level execution scripts.

### 2. `core/approximation.py`
* **Location:** `c:\Users\trilo\runtime-approximation-embedded\core\approximation.py` (118 lines)
* **Responsibility:** Core numerical approximation engine implementing float mantissa bit-truncation via `frexp`/`ldexp`, approximate matrix multiplication (`approx_matmul`), and approximate sliding window FIR filtering (`approx_dot_blocks`).
* **Connections:** Called directly by `core.workloads` and `core.sensitivity`.

### 3. `core/metrics.py`
* **Location:** `c:\Users\trilo\runtime-approximation-embedded\core\metrics.py` (103 lines)
* **Responsibility:** Numerical accuracy assessment container `ErrorMetrics` calculating MAE, MSE, RMSE, maximum absolute error, mean relative error, max relative error, and aggregate L2 normalized relative error (`relative_error`).
* **Connections:** Used by `core.sensitivity` for probe error calculation and end-to-end output validation.

### 4. `core/resource_model.py`
* **Location:** `c:\Users\trilo\runtime-approximation-embedded\core\resource_model.py` (94 lines)
* **Responsibility:** Simulates dynamic embedded operational conditions (battery drain, thermal throttling, periodic harvesting) generating continuous pressure traces $p \in [0, 1]$ (constant, oscillating, rising, random).
* **Connections:** Consumed by `core.controller` decision policies.

### 5. `core/sensitivity.py`
* **Location:** `c:\Users\trilo\runtime-approximation-embedded\core\sensitivity.py` (128 lines)
* **Responsibility:** Implements a finite-difference perturbation probe that evaluates a small subset of block elements at reduced precision, measures output relative deviation, and computes the error-amplification factor $A$.
* **Connections:** Interfaces between `core.workloads` and `core.controller`.

### 6. `core/workloads.py`
* **Location:** `c:\Users\trilo\runtime-approximation-embedded\core\workloads.py` (375 lines)
* **Responsibility:** Generates test workloads with non-stationary sensitivity profiles (`FIRWorkload`, `MatMulWorkload`, `ConvWorkload`). Splits data into discrete `Block` units with attached payload arrays and MAC counts.
* **Connections:** Consumed by `core.sensitivity`, `core.approximation`, and `core.controller`.

### 7. `core/controller.py`
* **Location:** `c:\Users\trilo\runtime-approximation-embedded\core\controller.py` (298 lines)
* **Responsibility:** Multi-policy decision engine implementing `ExactOnlyPolicy`, `FixedPolicy`, `ResourceOnlyAdaptive`, and `FullAdaptive`. Evaluates error feasibility and optimizes energy-risk score.
* **Connections:** Orchestrates `core.sensitivity`, `core.workloads`, and `config.py`.

### 8. `README.md`
* **Location:** `c:\Users\trilo\runtime-approximation-embedded\README.md` (1 line)
* **Responsibility:** Placeholder containing only `# runtime-approximation-embedded`.

---

## 4. Existing Implemented Functionality

1. **Mantissa Quantization Arithmetic:** Software emulation of truncated hardware multipliers (`numpy.frexp` mantissa scaling, rounding, and `numpy.ldexp` reassembly).
2. **Analytical MAC Cost Modeling:** Quadratic energy scaling ($E \propto b^2$) and linear critical path latency ($L \propto b$) normalized against a 16-bit reference.
3. **Synthetic Non-Stationary Signal/Matrix/Image Generators:** Generates DC-offset signals (low sensitivity), near-Nyquist oscillations (high sensitivity), and mixed frequency bands.
4. **Sub-block Sensitivity Probing:** Measures perturbation amplification on $k$ sub-elements to predict whole-block error propagation.
5. **Runtime Policy Selector:** Mathematical optimization balancing resource pressure ($p$), error risk ($\text{predicted\_error} / \text{budget}$), and energy ratio.
6. **Error Verification Suite:** Mathematical comparison of approximated arrays against full `float64` ground truth.

---

## 5. Existing Algorithms

* **Floating-Point Mantissa Truncation:**
  $$x = m \cdot 2^e, \quad m_q = \frac{\text{round}(m \cdot 2^b)}{2^b}, \quad x_q = m_q \cdot 2^e$$
* **Error Amplification ($A$):**
  $$A = \frac{\|\mathbf{y}_{\text{approx\_probe}} - \mathbf{y}_{\text{exact\_probe}}\|_2}{\|\mathbf{y}_{\text{exact\_probe}}\|_2 \cdot 2^{-b_{\text{probe}}}}$$
* **Predicted Block Error:**
  $$\text{pred}(mode) = A \cdot 2^{-b_{\text{mode}}}$$
* **Budget Feasibility:**
  $$\text{Feasible}(mode) \iff \text{pred}(mode) \cdot \gamma_{\text{safety}} \le \text{Budget}$$
* **Controller Selection Scoring:**
  $$\text{Score}(mode) = \text{EnergyRatio}(mode) + \lambda(p) \cdot \frac{\text{pred}(mode)}{\text{Budget}}$$
  where $\lambda(p) = \lambda_{\max}(1 - p) + \lambda_{\min} p$.

---

## 6. Existing Inputs and Outputs

* **Inputs:**
  * Workload definitions (FIR filter length/taps, Matrix dimensions $M \times K \times N$, Image dimensions $H \times W \times K$).
  * Sensitivity bias flags (`low`, `mixed`, `high`).
  * Configuration objects specifying error budget (e.g. 1%, 5%, 10%) and resource pressure trace type.
* **Outputs:**
  * `ControllerDecision` objects containing selected `Mode`, predicted error, feasibility map, scores, and adaptation overhead MACs.
  * Approximated output arrays (`np.ndarray`).
  * `ErrorMetrics` objects detailing exact vs approximate numerical differences.

---

## 7. Existing Dependencies

* **Language:** Python 3.10+
* **Libraries:**
  * `numpy` (>= 1.22.0)
  * Standard library: `dataclasses`, `enum`, `typing`, `abc`, `time`, `math`
* *Note: Comments in `config.py` mention a potential Streamlit UI and markdown documentation (`docs/methodology.md`), but neither was created before the previous agent reached limits.*

---

## 8. What Appears Complete

* **Core Algorithmic Foundation for Approximation Simulation:**
  * Complete arithmetic engine (`core/approximation.py`).
  * Complete error evaluation metrics (`core/metrics.py`).
  * Complete resource trace generator (`core/resource_model.py`).
  * Complete sub-block sensitivity estimation routine (`core/sensitivity.py`).
  * Complete synthetic workload data structures for FIR, MatMul, and Conv (`core/workloads.py`).
  * Complete rule-plus-score adaptive controller with 5 distinct policies (`core/controller.py`).
  * Complete centralized configuration schema (`config.py`).

---

## 9. What Appears Incomplete in the Existing Python Framework

1. **Simulation Runner / Experiment Pipeline:** No top-level benchmark script (`run_experiments.py` or `evaluate.py`) exists to execute sweeps across budgets, traces, and workloads.
2. **UI / Visualization:** No Streamlit dashboard (despite references in `config.py` docstrings).
3. **Documentation:** `README.md` is empty, and referenced `docs/methodology.md` does not exist.
4. **Unit Tests:** No formal `test_*.py` test suite was authored (only verified via manual Python invocation).

---

## 10. Potential Bugs, Gaps & Nuances in Existing Code

1. **Config Import Path Inconsistency:** `config.py` is at the repository root, but `core/sensitivity.py`, `core/resource_model.py`, `core/workloads.py`, and `core/controller.py` do `from config import ...`. This works when running from the repo root with `PYTHONPATH=.`, but if treated as a standalone package `core`, package imports would need clean packaging.
2. **Zero-Division Handling in Probing:** In `core/sensitivity.py:116`, if `norm(exact) == 0`, `relative_error` returns 0.0, which results in $A = 0.0$. This is mathematically safe due to `eps=1e-12`, but marks all-zero signals as maximally robust.
3. **Overhead Cost Accounting Unit Mixing:** In `core/controller.py:249`, `overhead = sens.probe_macs + cfg.controller.decision_overhead_units`. `probe_macs` is an absolute MAC integer count, while `decision_overhead_units` is in EXACT-MAC equivalent units.

---

## 11. Alignment with Intended Project Objective

### Target Project:
**"Arduino Uno Based Multi-Parameter Motor Monitoring for Fault Severity Estimation and Fault Cause Identification"**

### Target Features:
* **Hardware:** Arduino Uno (ATmega328P: 16 MHz, 2 KB SRAM, 32 KB Flash, 10-bit ADC).
* **Sensors:** Vibration (piezo / accelerometer), Temperature (LM35 / DS18B20 / NTC), Current (ACS712 / shunt), RPM (Hall effect / optical sensor).
* **Signal Pipeline:** Acquisition $\to$ Preprocessing $\to$ Operating Condition Analysis $\to$ Abnormality/Severity Estimation $\to$ Fault Cause Classification $\to$ Peripherals (LCD / Buzzer / LEDs) $\to$ Serial Logging.

### Comparison Matrix:

| Aspect | Existing Codebase | Intended Target System |
| :--- | :--- | :--- |
| **Execution Environment** | Host PC / Python 3 / NumPy | Arduino Uno (ATmega328P C/C++ Embedded Firmware) + Optional PC Host Tool |
| **Primary Domain** | Dynamic Floating-Point Mantissa Truncation | Real-Time Motor Sensor Acquisition, DSP & Fault Diagnosis |
| **Sensor Processing** | None (Synthetic math matrices/signals) | Analog/Digital sampling of Vibration, Temp, Current, RPM |
| **Fault Diagnosis** | None | Feature extraction (RMS, peak, Crest factor), Severity scoring, Fault classification (Bearing, Unbalance, Misalignment, Overload) |
| **Actuation & UI** | Planned Streamlit UI (unimplemented) | I2C LCD (16x2 / 20x4), Alert Buzzer, Status LEDs, Serial CSV Stream |

---

## 12. What Can Be Reused vs What Needs to Be Added

### What Can Be Reused / Adapted:
1. **Mathematical Concepts:** 
   * Preprocessing & windowing algorithms (sliding window logic in `core/workloads.py` can be translated to C/C++ ring buffers).
   * Threshold-based and adaptive decision scoring logic (can be adapted for resource-constrained AVR execution).
2. **Dual-Layer Architecture Strategy:**
   * Keep the Python framework as the **PC-Side Data Logger, Calibration Engine, Algorithm Verification & Visualization Host**.
   * Add the **Embedded C/C++ Firmware (`firmware/`)** targeting the Arduino Uno.
3. **Metrics Framework:** `core/metrics.py` is directly reusable for evaluating classification accuracy, estimation errors, and sensor noise characteristics on the host.

### What Needs to Be Built:
1. **Embedded Firmware (`firmware/`):**
   * Arduino C/C++ sketch (`.ino` / `.cpp` / `.h`).
   * Sensor driver modules:
     * Vibration ADC sampling & time-domain feature extraction (RMS, Peak-to-Peak).
     * Temperature conversion & calibration.
     * ACS712 Current sampling (True RMS calculation).
     * RPM tachometer pulse interrupt counting.
   * On-chip signal preprocessing and ring-buffer management in 2 KB SRAM.
   * Rule-based / threshold-based Fault Severity Estimator (Normal, Warning, Critical).
   * Fault Cause Classifier (Bearing fault, Mechanical Unbalance, Angular/Parallel Misalignment, Electrical Overload/Overheating).
   * Peripheral drivers (LiquidCrystal_I2C, Buzzer PWM/tone, LED status indicators).
   * Serial Telemetry Protocol (Structured JSON / CSV frames).
2. **PC-Side Interface & Telemetry (`tools/` or `host/`):**
   * Real-time serial ingestion bridge.
   * Dataset logger and synthetic sensor simulator (for testing when physical Arduino hardware is offline).
   * Real-time dashboard / plotting tool for multi-parameter motor state.

---

## 13. Recommended Next Development Phase

1. **Phase 1: Architecture Blueprint & Target System Specification**
   * Define the complete embedded memory map (SRAM allocation for ATmega328P).
   * Formulate exact pin assignments, sensor transfer functions, and fault classification decision matrices.
2. **Phase 2: Arduino Uno Embedded Firmware Implementation**
   * Author modular C/C++ embedded drivers and analytical engine.
3. **Phase 3: Host Telemetry & Simulation Tools**
   * Build the serial monitoring and synthetic sensor stream bridge in Python.
4. **Phase 4: Verification, Benchmarking & End-to-End Validation**
   * Validate motor fault identification against synthetic and real test profiles.

---
*Audit completed without modifying existing implementation files.*
