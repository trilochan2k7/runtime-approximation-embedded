"""Interactive Research Dashboard for Runtime Dynamic Approximation Prototype.

This Streamlit application visualizes the experimental results, hypothesis validations
(H1, H2, H3), baseline comparisons, and an interactive runtime controller demonstration.

DISCLAIMER:
All energy, latency, and resource metrics are MODELLED / SIMULATED analytical estimates
derived from CostModelConfig and ResourceConfig, NOT physical FPGA or silicon measurements.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

from config import DEFAULT_CONFIG, MODE_ORDER, Config, Mode
from core.controller import FullAdaptive, select_mode
from core.sensitivity import estimate_sensitivity
from core.workloads import Workload, make_workload

# Paths to verified experimental data
RESULTS_DIR = Path("results/final")
HYPOTHESIS_DIR = Path("results/hypothesis")

# Set Page Config
st.set_page_config(
    page_title="Runtime Dynamic Approximation | Research Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for Clean Engineering Theme
st.markdown(
    """
    <style>
    .main-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #64748B;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        padding: 12px;
        text-align: center;
    }
    .metric-val {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0F172A;
    }
    .metric-lbl {
        font-size: 0.8rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .model-notice {
        background-color: #FEF3C7;
        border-left: 4px solid #F59E0B;
        padding: 8px 12px;
        font-size: 0.85rem;
        color: #92400E;
        margin-bottom: 15px;
    }
    .status-pill {
        display: inline-block;
        padding: 2px 8px;
        font-size: 0.75rem;
        font-weight: 600;
        border-radius: 4px;
        background-color: #DCFCE7;
        color: #166534;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data(filepath: Path) -> Optional[pd.DataFrame]:
    """Helper to safely load CSV datasets."""
    if not filepath.exists():
        st.error(f"Missing required dataset: `{filepath}`. Please run experiments first.")
        return None
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        st.error(f"Error loading `{filepath}`: {e}")
        return None


# Sidebar Navigation
st.sidebar.title("⚡ Navigation")
selected_tab = st.sidebar.radio(
    "Select View",
    [
        "1. Overview & Architecture",
        "2. Precision Trade-Off",
        "3. Baseline Policy Comparison",
        "4. H1: Adaptive Precision",
        "5. H2: Sensitivity Awareness",
        "6. H3: Overhead & Break-Even",
        "7. Error Budget Compliance",
        "8. Interactive Runtime Demo",
        "9. Research Conclusions",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Project Scope:**  
    *Software Simulation Prototype*  
    **Models:** `CostModelConfig`, `ResourceConfig`  
    **Workloads:** FIR, MatMul, 2D Conv
    """
)

# Global Notice
st.markdown(
    """
    <div class="model-notice">
        <b>NOTICE:</b> Energy, latency, and resource numbers are <b>MODELLED / SIMULATED SOFTWARE ESTIMATES</b>
        based on analytical datapath scaling (<code>CostModelConfig</code>) and dynamic pressure traces, NOT physical FPGA measurements.
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# TAB 1: OVERVIEW & ARCHITECTURE
# -----------------------------------------------------------------------------
if selected_tab == "1. Overview & Architecture":
    st.markdown('<div class="main-header">Resource- and Sensitivity-Aware Runtime Approximation</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">A dynamic arithmetic adaptation architecture for resource-constrained embedded computing</div>', unsafe_allow_html=True)

    st.markdown(
        """
        ### Abstract
        The proposed architecture dynamically selects exact or reduced-precision arithmetic (**EXACT**, **APPROX-1**, **APPROX-2**, **APPROX-3**)
        per computational block by monitoring real-time **Resource Pressure** ($p \in [0, 1]$), probing **Output Sensitivity** ($A$),
        and strictly enforcing an **Application Error Budget**.
        """
    )

    # Headline Metrics
    df_baseline = load_data(RESULTS_DIR / "baseline_comparison.csv")
    if df_baseline is not None:
        full_row = df_baseline[df_baseline["policy"] == "full_adaptive"].iloc[0]
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.metric("Full Adaptive Net Energy Savings", full_row["mean_net_energy_savings_pct"], help="Modelled net energy savings after subtracting probe and decision overhead.")
        with c2:
            st.metric("Latency Reduction", full_row["mean_latency_reduction_pct"], help="Modelled critical-path latency reduction.")
        with c3:
            st.metric("Budget Compliance", full_row["budget_compliance_rate_pct"], help="Percentage of evaluated runs satisfying the target error budget.")
        with c4:
            st.metric("Mean Relative Error", f"{float(full_row['mean_norm_rel_error']):.4f}", help="Average achieved normalized L2 relative error.")
        with c5:
            st.metric("Mean Overhead", full_row["mean_overhead_pct_of_consumed_energy"], help="Probing and controller overhead as % of consumed energy.")

    st.markdown("---")
    st.markdown("### Closed-Loop Architecture Flow")
    st.code(
        """
        Workload Stream (FIR Filter / Matrix Multiplication / 2-D Convolution)
             │
             ▼
        Workload Analyzer & Sensitivity Estimator (Lightweight 32-element sub-probe)
             │  --> Output Error Amplification Factor (A)
             ▼
        Resource Monitor (Continuous Environmental Pressure Trace p ∈ [0, 1])
             │
             ▼
        Runtime Adaptation Controller (Filters Feasible Modes & Minimizes Energy + Risk)
             │
             ├──────────────────────┬──────────────────────┬──────────────────────┐
             ▼                      ▼                      ▼                      ▼
        [ EXACT (16-bit) ]   [ APPROX-1 (10-bit) ]  [ APPROX-2 (6-bit) ]   [ APPROX-3 (4-bit) ]
             │                      │                      │                      │
             └──────────────────────┴──────────┬───────────┴──────────────────────┘
                                               ▼
                                 [ Approximation Engine ]
                               (Quantized Mantissa MACs)
                                               │
                                               ▼
                                  [ Output & Error Monitor ]
                             (MAE, RMSE, Normalized L2 Error)
                                               │
                                               ▼
                                 [ Feedback & Result Logging ]
        """,
        language="text",
    )

# -----------------------------------------------------------------------------
# TAB 2: PRECISION TRADE-OFF
# -----------------------------------------------------------------------------
elif selected_tab == "2. Precision Trade-Off":
    st.markdown('<div class="main-header">Precision vs Cost vs Error Trade-Off</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Analytical scaling of mantissa truncation, modelled MAC energy, and critical-path latency</div>', unsafe_allow_html=True)

    df_prec = load_data(RESULTS_DIR / "precision_tradeoff.csv")
    if df_prec is not None:
        st.dataframe(df_prec, use_container_width=True, hide_index=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Modelled Energy vs Latency Savings")
            chart_data = df_prec.copy()
            chart_data["Energy Savings (%)"] = chart_data["modelled_energy_savings_pct"].str.rstrip("%").astype(float)
            chart_data["Latency Reduction (%)"] = chart_data["modelled_latency_reduction_pct"].str.rstrip("%").astype(float)
            st.bar_chart(chart_data.set_index("mode")[["Energy Savings (%)", "Latency Reduction (%)"]])

        with c2:
            st.markdown("#### Per-Operand Relative Quantization Error Bound ($2^{-b}$)")
            err_data = pd.DataFrame({
                "Mode": ["EXACT (16b)", "APPROX-1 (10b)", "APPROX-2 (6b)", "APPROX-3 (4b)"],
                "Error Bound": [0.0, 2**-10, 2**-6, 2**-4],
            })
            st.line_chart(err_data.set_index("Mode"))

# -----------------------------------------------------------------------------
# TAB 3: BASELINE POLICY COMPARISON
# -----------------------------------------------------------------------------
elif selected_tab == "3. Baseline Policy Comparison":
    st.markdown('<div class="main-header">Policy Baseline Benchmark Comparison</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Cross-comparison of Exact-Only, Fixed Approximation, Resource-Only, and Full Adaptive policies</div>', unsafe_allow_html=True)

    df_base = load_data(RESULTS_DIR / "baseline_comparison.csv")
    if df_base is not None:
        st.dataframe(df_base, use_container_width=True, hide_index=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Gross vs Net Energy Savings (%)")
            b_chart = df_base.copy()
            b_chart["Gross Savings (%)"] = b_chart["mean_gross_energy_savings_pct"].str.rstrip("%").astype(float)
            b_chart["Net Savings (%)"] = b_chart["mean_net_energy_savings_pct"].str.rstrip("%").astype(float)
            st.bar_chart(b_chart.set_index("policy")[["Gross Savings (%)", "Net Savings (%)"]])

        with c2:
            st.markdown("#### Budget Compliance Rate (%)")
            comp_chart = df_base.copy()
            comp_chart["Compliance (%)"] = comp_chart["budget_compliance_rate_pct"].str.rstrip("%").astype(float)
            st.bar_chart(comp_chart.set_index("policy")[["Compliance (%)"]])

    st.markdown("---")
    st.markdown("### Workload-Specific Policy Breakdown")
    df_workload = load_data(RESULTS_DIR / "workload_comparison.csv")
    if df_workload is not None:
        selected_w = st.selectbox("Filter Workload", ["FIR", "MATMUL", "CONV"])
        filtered_w = df_workload[df_workload["workload"] == selected_w]
        st.dataframe(filtered_w, use_container_width=True, hide_index=True)

# -----------------------------------------------------------------------------
# TAB 4: H1: ADAPTIVE PRECISION
# -----------------------------------------------------------------------------
elif selected_tab == "4. H1: Adaptive Precision":
    st.markdown('<div class="main-header">Hypothesis H1: Adaptive Precision Validation</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Evaluating whether dynamic adaptation achieves energy/latency reductions while staying within error budget</div>', unsafe_allow_html=True)

    st.info("🎯 **H1 VERDICT: SUPPORTED** — Full adaptive controller maintains 100% budget compliance while saving 49.1% net energy, unlike aggressive fixed modes which violate strict budgets.")

    df_h1 = load_data(HYPOTHESIS_DIR / "H1_results.csv")
    if df_h1 is not None:
        c1, c2, c3 = st.columns(3)
        with c1:
            w_filter = st.selectbox("Workload", ["all"] + list(df_h1["workload"].unique()))
        with c2:
            b_filter = st.selectbox("Budget", ["all"] + list(df_h1["budget_pct"].unique()))
        with c3:
            tr_filter = st.selectbox("Resource Trace", ["all"] + list(df_h1["trace"].unique()))

        filtered_h1 = df_h1.copy()
        if w_filter != "all":
            filtered_h1 = filtered_h1[filtered_h1["workload"] == w_filter]
        if b_filter != "all":
            filtered_h1 = filtered_h1[filtered_h1["budget_pct"] == b_filter]
        if tr_filter != "all":
            filtered_h1 = filtered_h1[filtered_h1["trace"] == tr_filter]

        st.markdown(f"**Showing {len(filtered_h1)} experimental configurations:**")
        st.dataframe(
            filtered_h1[["workload", "policy", "budget_pct", "trace", "rel_error", "budget_satisfied", "energy_savings_pct", "latency_reduction_pct", "overhead_pct"]],
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("#### Energy Savings vs Accuracy Distribution by Policy")
        chart_df = filtered_h1.groupby("policy")[["energy_savings_pct", "rel_error"]].mean().reset_index()
        st.bar_chart(chart_df.set_index("policy")[["energy_savings_pct"]])

# -----------------------------------------------------------------------------
# TAB 5: H2: SENSITIVITY AWARENESS
# -----------------------------------------------------------------------------
elif selected_tab == "5. H2: Sensitivity Awareness":
    st.markdown('<div class="main-header">Hypothesis H2: Sensitivity Awareness vs Resource-Only</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Demonstrating fine-grained block discrimination based on output sensitivity conditioning</div>', unsafe_allow_html=True)

    st.info("🎯 **H2 VERDICT: SUPPORTED** — Full adaptive controller discriminates robust blocks from sensitive ones, unlocking 19%–49% energy savings under strict budgets where resource-only is paralyzed in 100% EXACT mode.")

    df_h2 = load_data(HYPOTHESIS_DIR / "H2_results.csv")
    df_disc = load_data(HYPOTHESIS_DIR / "H2_block_discrimination.csv")

    if df_h2 is not None:
        st.markdown("#### Pairwise Comparison: `full_adaptive` vs `resource_only`")
        st.dataframe(
            df_h2[["workload", "bias", "budget_pct", "trace", "full_energy_savings_pct", "res_energy_savings_pct", "delta_energy_savings_pct", "full_rel_error", "res_rel_error", "full_budget_satisfied", "res_budget_satisfied"]],
            use_container_width=True,
            hide_index=True,
        )

    if df_disc is not None:
        st.markdown("---")
        st.markdown("#### Block-Level Sensitivity Discrimination Sample")
        st.dataframe(df_disc.head(20), use_container_width=True, hide_index=True)

# -----------------------------------------------------------------------------
# TAB 6: H3: OVERHEAD & BREAK-EVEN
# -----------------------------------------------------------------------------
elif selected_tab == "6. H3: Overhead & Break-Even":
    st.markdown('<div class="main-header">Hypothesis H3: Overhead Analysis & Break-Even Study</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Evaluating net benefit = gross compute savings − adaptation overhead</div>', unsafe_allow_html=True)

    st.info("🎯 **H3 VERDICT: SUPPORTED** — Adaptation produces substantial net positive energy savings (>48%) for moderate and large block sizes (MatMul, Conv). Probe overhead becomes significant only on very small block streams (FIR).")

    df_ovh = load_data(RESULTS_DIR / "overhead_analysis.csv")
    df_h3 = load_data(HYPOTHESIS_DIR / "H3_results.csv")

    if df_ovh is not None:
        st.markdown("#### Overhead Summary by Workload Block Granularity")
        st.dataframe(df_ovh, use_container_width=True, hide_index=True)

    if df_h3 is not None:
        st.markdown("---")
        st.markdown("#### Gross Savings vs Net Savings across Tested Configurations")
        h3_chart = df_h3.groupby("workload")[["gross_savings_pct", "net_savings_pct", "overhead_pct_of_total"]].mean().reset_index()
        st.bar_chart(h3_chart.set_index("workload")[["gross_savings_pct", "net_savings_pct"]])

# -----------------------------------------------------------------------------
# TAB 7: ERROR BUDGET COMPLIANCE
# -----------------------------------------------------------------------------
elif selected_tab == "7. Error Budget Compliance":
    st.markdown('<div class="main-header">Error Budget Compliance Heatmap & Tiers</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Strict feasibility enforcement across error budget levels (0.5% – 10.0%)</div>', unsafe_allow_html=True)

    df_comp = load_data(RESULTS_DIR / "budget_compliance.csv")
    if df_comp is not None:
        st.dataframe(df_comp, use_container_width=True, hide_index=True)

        st.markdown("#### Compliance Summary by Policy")
        st.markdown(
            """
            * **`full_adaptive`:** **100% compliance** across all tiers (0.5%, 1%, 2%, 5%, 10%).
            * **`resource_only`:** **100% compliance** across all tiers.
            * **`fixed_a1`:** **100% compliance** across all tiers.
            * **`fixed_a2`:** **93.3% compliance** (fails 33.3% of runs at 0.5% strict budget).
            * **`fixed_a3`:** **60.0% compliance** (fails 100% of runs at 0.5% and 1.0% budgets).
            """
        )

# -----------------------------------------------------------------------------
# TAB 8: INTERACTIVE RUNTIME DEMO
# -----------------------------------------------------------------------------
elif selected_tab == "8. Interactive Runtime Demo":
    st.markdown('<div class="main-header">Interactive Runtime Controller Demonstration</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Live invocation of <code>core.controller.FullAdaptive</code> on actual workload data blocks</div>', unsafe_allow_html=True)

    cfg = DEFAULT_CONFIG
    c1, c2, c3 = st.columns(3)
    with c1:
        demo_workload_name = st.selectbox("Select Workload", ["fir", "matmul", "conv"])
    with c2:
        demo_budget = st.select_slider("Error Budget", options=[0.005, 0.01, 0.02, 0.05, 0.10], value=0.05, format_func=lambda x: f"{x*100:.1f}%")
    with c3:
        demo_pressure = st.slider("Simulated Resource Pressure (p)", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

    workload = make_workload(demo_workload_name, cfg, bias="mixed", seed=42)
    block_idx = st.slider("Select Data Block Index", min_value=0, max_value=len(workload.blocks) - 1, value=0)
    target_block = workload.blocks[block_idx]

    # Live Controller Execution
    policy = FullAdaptive()
    sens = estimate_sensitivity(workload, target_block, cfg)
    dec = policy.decide(workload, target_block, demo_budget, demo_pressure, cfg)

    st.markdown("---")
    st.markdown(f"### Controller Decision for Block #{target_block.index}")

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Selected Mode", dec.mode.value)
    with m2:
        st.metric("Sensitivity Amplification (A)", f"{sens.amplification:.2f}")
    with m3:
        st.metric("Predicted Error", f"{dec.predicted_error[dec.mode]*100:.4f}%")
    with m4:
        st.metric("Modelled Energy Ratio", f"{cfg.energy_ratio(dec.mode):.4f}")

    st.code(f"Decision Reason: {dec.reason}\nProbe Overhead: {dec.probe_macs} MACs | Fixed Decision Cost: 8.0 units", language="text")

# -----------------------------------------------------------------------------
# TAB 9: RESEARCH CONCLUSIONS
# -----------------------------------------------------------------------------
elif selected_tab == "9. Research Conclusions":
    st.markdown('<div class="main-header">Scientific Findings & Architectural Conclusions</div>', unsafe_allow_html=True)

    st.markdown(
        """
        ### Summary of Hypotheses
        1. **H1 (Adaptive Precision):** **SUPPORTED**  
           *Dynamic multi-precision adaptation allows embedded systems to exploit data non-stationarity, achieving 49.1% average energy savings and 43.1% latency reduction while strictly maintaining application error bounds.*
        2. **H2 (Sensitivity Awareness):** **SUPPORTED**  
           *Per-block sensitivity probing is essential under tight error budgets ($b \le 1\%$), where resource-only controllers freeze into 100% exact computation.*
        3. **H3 (Controller Overhead):** **SUPPORTED**  
           *Sub-block perturbation probing is highly net-beneficial whenever block size exceeds $\approx 10,000$ MACs, consuming $<15\%$ of total compute energy.*

        ### Modelled vs Measured Boundary
        * All energy, area, and latency numbers are **analytical software estimates** and are not claimed as physical FPGA silicon measurements.
        """
    )
