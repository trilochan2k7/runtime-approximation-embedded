"""Interactive Application Workspace for Runtime Dynamic Approximation Architecture.

A complete software-based interactive engineering application demonstrating
Resource- and Sensitivity-Aware Dynamic Arithmetic Precision Selection
for Resource-Constrained Embedded Systems.

DISCLAIMER:
All energy, latency, and resource metrics are MODELLED / SIMULATED analytical estimates
derived from CostModelConfig and ResourceConfig, NOT physical FPGA/silicon measurements.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from config import DEFAULT_CONFIG, MODE_ORDER, Config, Mode, pressure_to_level
from core.app_workloads import (
    AppExecutionResult,
    ImageWorkloadAdapter,
    MatrixWorkloadAdapter,
    SignalWorkloadAdapter,
    array_to_display_image,
)
from core.controller import FullAdaptive, predicted_errors, select_mode
from core.sensitivity import estimate_sensitivity
from core.workloads import Workload, make_workload

# Paths to verified experimental data
RESULTS_DIR = Path("results/final")
HYPOTHESIS_DIR = Path("results/hypothesis")

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Resource-Aware Runtime Approximation | Embedded Systems",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Professional Engineering Theme CSS
st.markdown(
    """
    <style>
    /* Global Typography & Palette */
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    code, pre, .mono-font {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Clean Card Layouts */
    .app-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    }
    
    .decision-hero-card {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 20px 24px;
        color: #F8FAFC;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.15);
    }

    .metric-box {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        padding: 12px 14px;
        text-align: center;
    }
    .metric-box-val {
        font-size: 1.45rem;
        font-weight: 700;
        color: #0F172A;
        font-family: 'JetBrains Mono', monospace;
    }
    .metric-box-lbl {
        font-size: 0.72rem;
        color: #64748B;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.05em;
        margin-top: 2px;
    }

    /* Badges */
    .mode-badge-exact {
        background-color: #DBEAFE;
        color: #1E40AF;
        padding: 4px 10px;
        border-radius: 4px;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.9rem;
        display: inline-block;
    }
    .mode-badge-a1 {
        background-color: #CFFAFE;
        color: #0E7490;
        padding: 4px 10px;
        border-radius: 4px;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.9rem;
        display: inline-block;
    }
    .mode-badge-a2 {
        background-color: #DCFCE7;
        color: #15803D;
        padding: 4px 10px;
        border-radius: 4px;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.9rem;
        display: inline-block;
    }
    .mode-badge-a3 {
        background-color: #FEF3C7;
        color: #B45309;
        padding: 4px 10px;
        border-radius: 4px;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.9rem;
        display: inline-block;
    }
    
    .status-pass {
        background-color: #DCFCE7;
        color: #166534;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.78rem;
        font-weight: 700;
    }
    .status-fail {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.78rem;
        font-weight: 700;
    }

    .disclaimer-banner {
        background-color: #FFFBEB;
        border: 1px solid #FDE68A;
        border-left: 4px solid #F59E0B;
        padding: 8px 14px;
        border-radius: 4px;
        font-size: 0.82rem;
        color: #92400E;
        margin-bottom: 16px;
    }

    .step-badge {
        background-color: #E2E8F0;
        color: #334155;
        padding: 2px 6px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.75rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------
@st.cache_data
def load_data(filepath: Path) -> Optional[pd.DataFrame]:
    """Helper to safely load CSV datasets."""
    if not filepath.exists():
        return None
    try:
        return pd.read_csv(filepath)
    except Exception:
        return None


def get_mode_badge(mode: Mode) -> str:
    """Return HTML badge for precision mode."""
    if mode == Mode.EXACT:
        return f'<span class="mode-badge-exact">EXACT (16-bit)</span>'
    elif mode == Mode.APPROX1:
        return f'<span class="mode-badge-a1">APPROX-1 (10-bit)</span>'
    elif mode == Mode.APPROX2:
        return f'<span class="mode-badge-a2">APPROX-2 (6-bit)</span>'
    else:
        return f'<span class="mode-badge-a3">APPROX-3 (4-bit)</span>'


# Initialize session state for history
if "last_run" not in st.session_state:
    st.session_state["last_run"] = None
if "run_history" not in st.session_state:
    st.session_state["run_history"] = []


# -----------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# -----------------------------------------------------------------------------
st.sidebar.markdown("### ⚡ Runtime Approximation")
selected_nav = st.sidebar.radio(
    "Application Navigation",
    [
        "1. RUN APPLICATION",
        "2. RESULTS",
        "3. LIVE CONTROLLER",
        "4. RESEARCH / EVALUATION",
        "5. ABOUT SYSTEM",
    ],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Architecture Overview:**  
    * **Engine:** Truncated Mantissa MACs
    * **Modes:** EXACT (16b), A1 (10b), A2 (6b), A3 (4b)
    * **Controller:** Budget & Risk Weighted
    * **Cost Model:** $E(b) = b^2 + b$, $L(b) \propto b$
    
    *All metrics are modelled software estimates.*
    """
)


# Global Top Disclaimer
st.markdown(
    """
    <div class="disclaimer-banner">
        <b>NOTICE:</b> Energy, latency, and resource numbers are <b>MODELLED / SIMULATED SOFTWARE ESTIMATES</b>
        based on analytical datapath scaling (<code>CostModelConfig</code>) and dynamic pressure traces, NOT physical FPGA measurements.
    </div>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# SECTION 1: RUN APPLICATION (Main Engineering Workspace)
# =============================================================================
if selected_nav == "1. RUN APPLICATION":
    st.markdown("## RESOURCE-AWARE RUNTIME APPROXIMATION")
    st.markdown(
        "##### *Software Implementation & Evaluation — Runtime precision is selected dynamically using resource pressure, output sensitivity and an application error budget.*"
    )

    # Workload Type Selector
    st.markdown("---")
    st.markdown("#### 1. Select Computational Workload & Input")

    workload_type = st.radio(
        "Choose Workload Domain",
        ["🖼️ Image Processing", "📻 Signal Processing (FIR)", "🔢 Matrix Processing (MatMul)"],
        horizontal=True,
    )

    cfg = DEFAULT_CONFIG

    # -------------------------------------------------------------------------
    # WORKLOAD A: IMAGE PROCESSING
    # -------------------------------------------------------------------------
    if workload_type == "🖼️ Image Processing":
        col_in1, col_in2 = st.columns([1, 1])

        with col_in1:
            img_source = st.radio(
                "Image Source",
                ["Choose Preset Test Pattern", "Upload Custom Image (PNG/JPG)"],
                horizontal=True,
            )

            if img_source == "Choose Preset Test Pattern":
                pattern_name = st.selectbox(
                    "Select Synthetic Pattern",
                    [
                        "Sensor Grayscale Pattern (Mixed Gradients & Texture)",
                        "Checkerboard & Edges (High Contrast)",
                        "Gradient Waves (Smooth Low Sensitivity)",
                        "High-Frequency Oscillation (High Sensitivity)",
                    ],
                )
                img_size = st.slider("Image Size (N × N)", min_value=96, max_value=256, value=160, step=32)
                raw_pattern = pattern_name.split(" (")[0]
                img_array = ImageWorkloadAdapter.generate_synthetic_image(raw_pattern, size=img_size, seed=42)
            else:
                uploaded_file = st.file_uploader("Upload Image File", type=["png", "jpg", "jpeg"])
                if uploaded_file is not None:
                    pil_img = Image.open(uploaded_file)
                    img_array = ImageWorkloadAdapter.load_user_image(pil_img, max_dim=256)
                else:
                    st.info("No file uploaded yet. Using default sensor test pattern.")
                    img_array = ImageWorkloadAdapter.generate_synthetic_image("Sensor Grayscale Pattern", size=160, seed=42)

            op_name = st.selectbox(
                "Filter Operation (2-D Kernel)",
                [
                    "3x3 Gaussian Blur",
                    "3x3 Box Blur",
                    "3x3 Sharpen",
                    "3x3 Sobel Horizontal (Edges)",
                    "3x3 Laplacian (High-Pass)",
                ],
            )

        with col_in2:
            st.markdown("##### Input Image Preview")
            preview_img = array_to_display_image(img_array, is_diff=False)
            st.image(preview_img, caption=f"Input: {img_array.shape[0]} × {img_array.shape[1]} ({img_array.size:,} pixels)", width=220)

        # Runtime Constraints Panel
        st.markdown("---")
        st.markdown("#### 2. Configure Runtime Constraints")
        c_c1, c_c2, c_c3 = st.columns(3)
        with c_c1:
            budget_choice = st.select_slider(
                "Error Budget (Max Output Error)",
                options=[0.005, 0.01, 0.02, 0.05, 0.10],
                value=0.02,
                format_func=lambda x: f"{x*100:.1f}%",
                help="Maximum allowable L2 relative error for the application.",
            )
        with c_c2:
            pressure_val = st.slider(
                "Simulated Resource Pressure (p)",
                min_value=0.0,
                max_value=1.0,
                value=0.60,
                step=0.05,
                help="0.0 = Abundant resources, 1.0 = Critical resource scarcity.",
            )
            lvl = pressure_to_level(pressure_val)
            st.caption(f"Pressure Regime: **{lvl}** (Score weight $\lambda(p)={cfg.controller.lambda_max*(1-pressure_val) + cfg.controller.lambda_min*pressure_val:.2f}$)")
        with c_c3:
            st.markdown("##### Output Sensitivity")
            st.caption("Automatically estimated at runtime by probing image patches (no manual guessing needed).")

        st.markdown("---")
        if st.button("⚡ RUN ADAPTIVE COMPUTATION", type="primary", use_container_width=True):
            with st.spinner("Analyzing input, estimating sensitivity, and selecting optimal precision..."):
                exec_res = ImageWorkloadAdapter.execute(
                    img_array, op_name, budget_choice, pressure_val, cfg
                )
                st.session_state["last_run"] = exec_res
                st.session_state["run_history"].append(exec_res)

    # -------------------------------------------------------------------------
    # WORKLOAD B: SIGNAL PROCESSING (FIR)
    # -------------------------------------------------------------------------
    elif workload_type == "📻 Signal Processing (FIR)":
        col_in1, col_in2 = st.columns([1, 1])

        with col_in1:
            sig_type = st.selectbox(
                "Signal Type",
                [
                    "Chirp & Harmonics",
                    "Noisy Multi-Tone (Audio-like)",
                    "Non-Stationary (Dynamic Sensitivity)",
                    "Clean Carrier Wave",
                ],
            )
            sig_len = st.select_slider("Signal Length (Samples)", options=[256, 512, 1024, 2048], value=1024)
            fir_taps = st.selectbox("FIR Filter Taps (Hann Window LPF)", [16, 32, 48, 64], index=1)
            noise_level = st.slider("Additive Noise Level ($\sigma$)", min_value=0.0, max_value=0.20, value=0.04, step=0.01)

            signal_data = SignalWorkloadAdapter.generate_signal(sig_type, length=sig_len, noise_std=noise_level, seed=42)

        with col_in2:
            st.markdown("##### Input Waveform Preview")
            st.line_chart(signal_data[: min(200, len(signal_data))], height=180)
            st.caption(f"Showing first {min(200, len(signal_data))} samples of {len(signal_data):,} total samples.")

        # Runtime Constraints Panel
        st.markdown("---")
        st.markdown("#### 2. Configure Runtime Constraints")
        c_c1, c_c2, c_c3 = st.columns(3)
        with c_c1:
            budget_choice = st.select_slider(
                "Error Budget (Max Output Error)",
                options=[0.005, 0.01, 0.02, 0.05, 0.10],
                value=0.02,
                format_func=lambda x: f"{x*100:.1f}%",
            )
        with c_c2:
            pressure_val = st.slider(
                "Simulated Resource Pressure (p)",
                min_value=0.0,
                max_value=1.0,
                value=0.60,
                step=0.05,
            )
            lvl = pressure_to_level(pressure_val)
            st.caption(f"Pressure Regime: **{lvl}**")
        with c_c3:
            st.markdown("##### Output Sensitivity")
            st.caption("Probed via sliding sub-window of filter taps.")

        st.markdown("---")
        if st.button("⚡ RUN ADAPTIVE COMPUTATION", type="primary", use_container_width=True):
            with st.spinner("Analyzing signal conditioning and evaluating candidate modes..."):
                exec_res = SignalWorkloadAdapter.execute(
                    signal_data, taps_count=fir_taps, budget=budget_choice, pressure=pressure_val, cfg=cfg
                )
                st.session_state["last_run"] = exec_res
                st.session_state["run_history"].append(exec_res)

    # -------------------------------------------------------------------------
    # WORKLOAD C: MATRIX PROCESSING (MATMUL)
    # -------------------------------------------------------------------------
    else:
        col_in1, col_in2 = st.columns([1, 1])

        with col_in1:
            mat_preset = st.selectbox(
                "Matrix Conditioning Preset",
                [
                    "Smooth & Positive (Low Sensitivity)",
                    "Mixed Dynamic (Realistic DSP)",
                    "Near-Zero Mean (High Sensitivity)",
                    "Uniform Random",
                ],
            )
            dim_size = st.select_slider("Matrix Dimensions ($M = K = N$)", options=[32, 64, 96, 128], value=64)
            A_mat, B_mat = MatrixWorkloadAdapter.generate_matrices(
                mat_preset, M=dim_size, K=dim_size, N=dim_size, seed=42
            )

        with col_in2:
            st.markdown("##### Matrix A Heatmap Preview")
            img_mat = array_to_display_image(A_mat, is_diff=False)
            st.image(img_mat, caption=f"Matrix A: {dim_size} × {dim_size} ({dim_size*dim_size} elements)", width=180)

        # Runtime Constraints Panel
        st.markdown("---")
        st.markdown("#### 2. Configure Runtime Constraints")
        c_c1, c_c2, c_c3 = st.columns(3)
        with c_c1:
            budget_choice = st.select_slider(
                "Error Budget (Max Output Error)",
                options=[0.005, 0.01, 0.02, 0.05, 0.10],
                value=0.02,
                format_func=lambda x: f"{x*100:.1f}%",
            )
        with c_c2:
            pressure_val = st.slider(
                "Simulated Resource Pressure (p)",
                min_value=0.0,
                max_value=1.0,
                value=0.60,
                step=0.05,
            )
            lvl = pressure_to_level(pressure_val)
            st.caption(f"Pressure Regime: **{lvl}**")
        with c_c3:
            st.markdown("##### Output Sensitivity")
            st.caption("Probed via representative submatrix product.")

        st.markdown("---")
        if st.button("⚡ RUN ADAPTIVE COMPUTATION", type="primary", use_container_width=True):
            with st.spinner("Analyzing matrix conditioning and computing adaptive dot products..."):
                exec_res = MatrixWorkloadAdapter.execute(
                    A_mat, B_mat, budget=budget_choice, pressure=pressure_val, cfg=cfg
                )
                st.session_state["last_run"] = exec_res
                st.session_state["run_history"].append(exec_res)

    # -------------------------------------------------------------------------
    # DISPLAY EXECUTION RESULTS (If available)
    # -------------------------------------------------------------------------
    if st.session_state["last_run"] is not None:
        res: AppExecutionResult = st.session_state["last_run"]

        st.markdown("---")
        st.markdown("### 3. Execution Sequence & Controller Decision")

        # Step Sequence Checklist
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.markdown(f"✅ **Input Analyzed**<br><span class='step-badge'>{res.input_summary.get('Total MACs', 'MACs')}</span>", unsafe_allow_html=True)
        with s2:
            st.markdown(f"✅ **Sensitivity Probed**<br><span class='step-badge'>Amp A = {res.sensitivity_amp:.2f}</span>", unsafe_allow_html=True)
        with s3:
            st.markdown(f"✅ **Controller Selection**<br><span class='step-badge'>{res.decision.mode.value}</span>", unsafe_allow_html=True)
        with s4:
            compliance_html = '<span class="status-pass">PASS</span>' if res.compliant else '<span class="status-fail">FAIL</span>'
            st.markdown(f"✅ **Error Verified**<br>{compliance_html}", unsafe_allow_html=True)

        # Controller Decision Hero Card
        badge_html = get_mode_badge(res.decision.mode)
        st.markdown(
            f"""
            <div class="decision-hero-card">
                <div style="font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.08em; color: #94A3B8; margin-bottom: 6px;">
                    Runtime Adaptation Controller Decision
                </div>
                <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; margin-bottom: 12px;">
                    <div style="font-size: 1.6rem; font-weight: 800; color: #FFFFFF;">
                        Selected Mode: {badge_html}
                    </div>
                    <div style="font-size: 0.9rem; color: #CBD5E1; font-family: 'JetBrains Mono', monospace;">
                        Predicted Error: <b>{res.decision.predicted_error[res.decision.mode]*100:.3f}%</b> | Budget: <b>{res.budget*100:.1f}%</b>
                    </div>
                </div>
                <div style="font-size: 0.92rem; color: #E2E8F0; line-height: 1.5; border-top: 1px solid #334155; padding-top: 10px;">
                    <b>Controller Rationale:</b> {res.decision.reason}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Visual Computational Output Centerpiece
        st.markdown("### 4. Computational Output Comparison")

        if res.workload_type == "image":
            v1, v2, v3, v4 = st.columns(4)
            with v1:
                st.markdown("**1. Original Input**")
                st.image(array_to_display_image(img_array, False), use_container_width=True)
            with v2:
                st.markdown("**2. Exact Reference (Float64)**")
                st.image(array_to_display_image(res.exact_output, False), use_container_width=True)
            with v3:
                st.markdown(f"**3. Adaptive ({res.decision.mode.short})**")
                st.image(array_to_display_image(res.adaptive_output, False), use_container_width=True)
            with v4:
                st.markdown("**4. Error Difference (Heatmap)**")
                st.image(array_to_display_image(res.diff_output, True), use_container_width=True)

        elif res.workload_type == "signal":
            v1, v2 = st.columns(2)
            with v1:
                st.markdown("**Exact Reference vs Adaptive Filter Output**")
                plot_df = pd.DataFrame({
                    "Exact Output": res.exact_output[: min(300, len(res.exact_output))],
                    f"Adaptive ({res.decision.mode.short})": res.adaptive_output[: min(300, len(res.adaptive_output))],
                })
                st.line_chart(plot_df, height=260)
            with v2:
                st.markdown("**Error Residual Waveform ($|y_{adapt} - y_{exact}|$)**")
                st.line_chart(res.diff_output[: min(300, len(res.diff_output))], height=260)

        else:  # Matrix
            v1, v2, v3 = st.columns(3)
            with v1:
                st.markdown("**Exact Result Matrix ($A \\times B$)**")
                st.image(array_to_display_image(res.exact_output, False), use_container_width=True)
            with v2:
                st.markdown(f"**Adaptive Result Matrix ({res.decision.mode.short})**")
                st.image(array_to_display_image(res.adaptive_output, False), use_container_width=True)
            with v3:
                st.markdown("**Absolute Error Difference Matrix**")
                st.image(array_to_display_image(res.diff_output, True), use_container_width=True)

        # Performance Metrics Row
        st.markdown("### 5. Derived Performance & Cost Metrics")
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-box-val">{res.metrics.norm_rel_error*100:.3f}%</div>
                    <div class="metric-box-lbl">Normalized L2 Error</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-box-val" style="color: #16A34A;">{res.energy_savings_pct:.1f}%</div>
                    <div class="metric-box-lbl">Modelled Energy Savings</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m3:
            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-box-val" style="color: #0284C7;">{res.latency_reduction_pct:.1f}%</div>
                    <div class="metric-box-lbl">Modelled Latency Reduction</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m4:
            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-box-val">{res.overhead_pct:.2f}%</div>
                    <div class="metric-box-lbl">Adaptation Overhead</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m5:
            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-box-val">{res.execution_time_ms:.1f} ms</div>
                    <div class="metric-box-lbl">Software Execution Time</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# =============================================================================
# SECTION 2: RESULTS (Execution Summary & History)
# =============================================================================
elif selected_nav == "2. RESULTS":
    st.markdown("## Execution Summary & Comparison")
    st.markdown("##### Detailed comparative analysis between EXACT reference and Runtime Adaptive execution.")

    if st.session_state["last_run"] is None:
        st.info("No interactive run executed in this session yet. Please navigate to **'1. RUN APPLICATION'** and click 'Run Adaptive Computation'.")
    else:
        res: AppExecutionResult = st.session_state["last_run"]

        st.markdown("### Latest Execution Breakdown")
        
        comp_data = {
            "Dimension": [
                "Arithmetic Mode",
                "Operand Bit-Width",
                "Normalized L2 Relative Error",
                "Mean Absolute Error (MAE)",
                "Root Mean Squared Error (RMSE)",
                "Application Error Budget",
                "Budget Compliance",
                "Modelled Energy Ratio",
                "Modelled Latency Ratio",
                "Adaptation Overhead (MACs)",
                "Net Energy Savings vs Exact",
                "Latency Reduction vs Exact",
            ],
            "EXACT Reference": [
                "EXACT",
                "16-bit nominal (float64 ref)",
                "0.0000%",
                "0.0000",
                "0.0000",
                f"{res.budget*100:.1f}%",
                "PASS (0.0% error)",
                "1.0000 (272 cost units/MAC)",
                "1.0000 (16 cycles/MAC)",
                "0 MACs (No controller)",
                "0.0%",
                "0.0%",
            ],
            "Adaptive Execution": [
                res.decision.mode.value,
                f"{cfg.modes[res.decision.mode].bits}-bit significand",
                f"{res.metrics.norm_rel_error*100:.4f}%",
                f"{res.metrics.mae:.6f}",
                f"{res.metrics.rmse:.6f}",
                f"{res.budget*100:.1f}%",
                "PASS" if res.compliant else "VIOLATION",
                f"{cfg.energy_ratio(res.decision.mode):.4f}",
                f"{cfg.latency_ratio(res.decision.mode):.4f}",
                f"{res.overhead_macs:,.0f} MACs ({res.overhead_pct:.2f}%)",
                f"{res.energy_savings_pct:.1f}%",
                f"{res.latency_reduction_pct:.1f}%",
            ],
        }
        df_comp_table = pd.DataFrame(comp_data)
        st.dataframe(df_comp_table, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("### Session History Log")
        if st.session_state["run_history"]:
            hist_rows = []
            for i, r in enumerate(st.session_state["run_history"]):
                hist_rows.append({
                    "Run #": i + 1,
                    "Workload": r.workload_type.upper(),
                    "Operation": r.operation_name,
                    "Budget": f"{r.budget*100:.1f}%",
                    "Pressure": f"{r.pressure:.2f}",
                    "Sensitivity A": f"{r.sensitivity_amp:.2f}",
                    "Selected Mode": r.decision.mode.value,
                    "Rel Error": f"{r.metrics.norm_rel_error*100:.3f}%",
                    "Net Energy Savings": f"{r.energy_savings_pct:.1f}%",
                    "Latency Reduction": f"{r.latency_reduction_pct:.1f}%",
                    "Compliance": "PASS" if r.compliant else "FAIL",
                })
            st.dataframe(pd.DataFrame(hist_rows), use_container_width=True, hide_index=True)


# =============================================================================
# SECTION 3: LIVE CONTROLLER & "WHAT-IF?"
# =============================================================================
elif selected_nav == "3. LIVE CONTROLLER":
    st.markdown("## Live Controller Decision Explorer")
    st.markdown("##### Real-time dynamic evaluation of precision selection across resource pressure, error budget, and sensitivity.")

    st.markdown("### 1. Interactive 'What If?' Parameters")
    c1, c2, c3 = st.columns(3)
    with c1:
        what_pressure = st.slider("Resource Pressure (p)", min_value=0.0, max_value=1.0, value=0.65, step=0.05)
    with c2:
        what_budget = st.select_slider("Error Budget", options=[0.005, 0.01, 0.02, 0.05, 0.10], value=0.02, format_func=lambda x: f"{x*100:.1f}%")
    with c3:
        what_amp = st.slider("Sensitivity Error Amplification Factor (A)", min_value=0.1, max_value=20.0, value=2.5, step=0.2)

    cfg = DEFAULT_CONFIG
    pred = predicted_errors(what_amp, cfg)
    chosen_mode, feasible, scores = select_mode(pred, what_budget, what_pressure, cfg)

    # Candidate Modes Evaluation Table
    st.markdown("### 2. Candidate Precision Modes Evaluation")
    eval_rows = []
    for m in MODE_ORDER:
        b = cfg.modes[m].bits
        pe = pred[m]
        feas = feasible[m]
        e_rat = cfg.energy_ratio(m)
        risk = pe / what_budget if what_budget > 0 else 0.0
        sc = scores[m]
        is_sel = (m == chosen_mode)
        
        eval_rows.append({
            "Mode": m.value,
            "Bit-Width": f"{b}-bit",
            "Predicted Error": f"{pe*100:.3f}%",
            "Safety Check ($1.25 \\times \\text{pred} \\le \\text{budget}$)": f"{pe * 1.25 * 100:.3f}% $\\le$ {what_budget*100:.1f}%",
            "Feasibility": "✓ FEASIBLE" if feas else "✗ INFEASIBLE",
            "Modelled Energy Ratio": f"{e_rat:.4f}",
            "Risk ($P / \\epsilon$)": f"{risk:.2f}" if feas else "N/A",
            "Controller Score": f"{sc:.4f}" if feas else "$\infty$",
            "Controller Decision": "👉 SELECTED" if is_sel else ("Candidate" if feas else "Rejected"),
        })

    st.dataframe(pd.DataFrame(eval_rows), use_container_width=True, hide_index=True)

    # Explanation Callout
    lvl = pressure_to_level(what_pressure)
    st.markdown(
        f"""
        <div class="decision-hero-card">
            <div style="font-size: 1.4rem; font-weight: 700; margin-bottom: 6px;">
                Decision: {get_mode_badge(chosen_mode)}
            </div>
            <div style="font-size: 0.95rem; color: #CBD5E1;">
                Under <b>{lvl} resource pressure (p={what_pressure:.2f})</b> and <b>sensitivity A={what_amp:.2f}</b>, 
                the controller picked <b>{chosen_mode.value}</b> as the lowest-cost feasible mode satisfying the <b>{what_budget*100:.1f}%</b> error budget.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2D Operational Map
    st.markdown("### 3. Full 2D Decision Map Across Operating Region")
    st.caption("Visualizing controller decision across Resource Pressure (x-axis) vs Error Budget (y-axis) for current sensitivity A=" + f"{what_amp:.2f}:")

    pressures = np.linspace(0.0, 1.0, 11)
    budgets = [0.005, 0.01, 0.02, 0.05, 0.10]
    map_matrix = []
    for b in budgets:
        row = []
        for p in pressures:
            m_choice, _, _ = select_mode(pred, b, p, cfg)
            row.append(m_choice.short)
        map_matrix.append(row)

    df_map = pd.DataFrame(map_matrix, index=[f"{b*100:.1f}% Budget" for b in budgets], columns=[f"p={p:.1f}" for p in pressures])
    st.dataframe(df_map, use_container_width=True)


# =============================================================================
# SECTION 4: RESEARCH / EVALUATION (Preserved Empirical Research)
# =============================================================================
elif selected_nav == "4. RESEARCH / EVALUATION":
    st.markdown("## Research Findings & Experimental Validation")
    st.markdown("##### Empirical validation data generated from 540-run statistical experiment suite.")

    eval_tab = st.selectbox(
        "Select Research Evaluation View",
        [
            "1. Baseline Policy Comparison",
            "2. Precision vs Cost Trade-Off",
            "3. Workload Characterization",
            "4. Hypothesis H1: Adaptive Precision",
            "5. Hypothesis H2: Sensitivity Awareness",
            "6. Hypothesis H3: Overhead & Break-Even",
            "7. Error Budget Compliance & Multi-Seed Summary",
        ],
    )

    # 1. Baseline Comparison
    if eval_tab == "1. Baseline Policy Comparison":
        st.markdown("### Baseline Policy Comparison")
        df_base = load_data(RESULTS_DIR / "baseline_comparison.csv")
        if df_base is not None:
            st.dataframe(df_base, use_container_width=True, hide_index=True)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### Net Energy Savings (%)")
                b_chart = df_base.copy()
                b_chart["Net Savings (%)"] = b_chart["mean_net_energy_savings_pct"].str.rstrip("%").astype(float)
                st.bar_chart(b_chart.set_index("policy")[["Net Savings (%)"]])
            with c2:
                st.markdown("#### Budget Compliance Rate (%)")
                comp_chart = df_base.copy()
                comp_chart["Compliance (%)"] = comp_chart["budget_compliance_rate_pct"].str.rstrip("%").astype(float)
                st.bar_chart(comp_chart.set_index("policy")[["Compliance (%)"]])

    # 2. Precision Trade-off
    elif eval_tab == "2. Precision vs Cost Trade-Off":
        st.markdown("### Precision vs Cost Trade-Off")
        df_prec = load_data(RESULTS_DIR / "precision_tradeoff.csv")
        if df_prec is not None:
            st.dataframe(df_prec, use_container_width=True, hide_index=True)
            c1, c2 = st.columns(2)
            with c1:
                chart_data = df_prec.copy()
                chart_data["Energy Savings (%)"] = chart_data["modelled_energy_savings_pct"].str.rstrip("%").astype(float)
                chart_data["Latency Reduction (%)"] = chart_data["modelled_latency_reduction_pct"].str.rstrip("%").astype(float)
                st.bar_chart(chart_data.set_index("mode")[["Energy Savings (%)", "Latency Reduction (%)"]])
            with c2:
                err_data = pd.DataFrame({
                    "Mode": ["EXACT (16b)", "APPROX-1 (10b)", "APPROX-2 (6b)", "APPROX-3 (4b)"],
                    "Worst-Case Per-Operand Unit Error": [0.0, 2**-10, 2**-6, 2**-4],
                })
                st.line_chart(err_data.set_index("Mode"))

    # 3. Workload Comparison
    elif eval_tab == "3. Workload Characterization":
        st.markdown("### Workload Characterization across Sensitivity Regimes")
        df_workload = load_data(RESULTS_DIR / "workload_comparison.csv")
        if df_workload is not None:
            w_choice = st.selectbox("Filter Workload", ["FIR", "MATMUL", "CONV"])
            st.dataframe(df_workload[df_workload["workload"] == w_choice], use_container_width=True, hide_index=True)

    # 4. H1
    elif eval_tab == "4. Hypothesis H1: Adaptive Precision":
        st.markdown("### Hypothesis H1: Dynamic Multi-Precision Adaptation")
        st.info("🎯 **H1 VERDICT: SUPPORTED** — Adaptive controller achieves 49.1% net energy savings and 43.1% latency reduction while maintaining 100% budget compliance.")
        df_h1 = load_data(HYPOTHESIS_DIR / "H1_results.csv")
        if df_h1 is not None:
            st.dataframe(df_h1.head(25), use_container_width=True, hide_index=True)

    # 5. H2
    elif eval_tab == "5. Hypothesis H2: Sensitivity Awareness":
        st.markdown("### Hypothesis H2: Sensitivity-Awareness vs Resource-Only")
        st.info("🎯 **H2 VERDICT: SUPPORTED** — Per-block sensitivity probing unlocks 19%–49% energy savings under strict budgets where resource-only is frozen in 100% EXACT mode.")
        df_h2 = load_data(HYPOTHESIS_DIR / "H2_results.csv")
        if df_h2 is not None:
            st.dataframe(df_h2.head(25), use_container_width=True, hide_index=True)

    # 6. H3
    elif eval_tab == "6. Hypothesis H3: Overhead & Break-Even":
        st.markdown("### Hypothesis H3: Controller Overhead & Break-Even")
        st.info("🎯 **H3 VERDICT: SUPPORTED** — Net benefit is strongly positive for blocks $\ge 10,000$ MACs, consuming $<15\%$ of compute energy.")
        df_ovh = load_data(RESULTS_DIR / "overhead_analysis.csv")
        if df_ovh is not None:
            st.dataframe(df_ovh, use_container_width=True, hide_index=True)

    # 7. Compliance & Multi-seed
    else:
        st.markdown("### Budget Compliance & Statistical Multi-Seed Stability")
        df_comp = load_data(RESULTS_DIR / "budget_compliance.csv")
        df_stat = load_data(RESULTS_DIR / "statistical_summary.csv")
        if df_comp is not None:
            st.markdown("##### Budget Compliance Matrix")
            st.dataframe(df_comp, use_container_width=True, hide_index=True)
        if df_stat is not None:
            st.markdown("---")
            st.markdown("##### Statistical Summary Across 5 Random Seeds")
            st.dataframe(df_stat, use_container_width=True, hide_index=True)


# =============================================================================
# SECTION 5: ABOUT SYSTEM
# =============================================================================
else:
    st.markdown("## Architecture & System Reference")
    st.markdown("##### Technical specification of the runtime dynamic approximation prototype.")

    st.markdown("### 1. Closed-Loop Architectural Flow")
    st.code(
        """
        WORKLOAD INPUT (Image / Signal / Matrix)
             │
             ▼
        WORKLOAD ANALYZER & SENSITIVITY ESTIMATOR (Lightweight 32-MAC sub-probe)
             │  --> Output Error Amplification Factor (A)
             ▼
        RESOURCE MONITOR (Dynamic Environmental Pressure p ∈ [0, 1])
             │
             ▼
        RUNTIME CONTROLLER (Filters Feasible Modes & Minimizes Energy + Risk)
             │
             ├──────────────────────┬──────────────────────┬──────────────────────┐
             ▼                      ▼                      ▼                      ▼
        [ EXACT (16-bit) ]   [ APPROX-1 (10-bit) ]  [ APPROX-2 (6-bit) ]   [ APPROX-3 (4-bit) ]
             │                      │                      │                      │
             └──────────────────────┴──────────┬───────────┴──────────────────────┘
                                               ▼
                                 [ APPROXIMATION ENGINE ]
                               (Quantized Mantissa MACs)
                                               │
                                               ▼
                                  [ OUTPUT & ERROR MONITOR ]
                             (MAE, RMSE, Normalized L2 Error)
                                               │
                                               ▼
                                 [ FEEDBACK & LOGGING ]
        """,
        language="text",
    )

    st.markdown("---")
    st.markdown("### 2. Analytical Cost Model Formulations")
    st.markdown(
        """
        * **Per-MAC Energy Scaling:**  
          $E(b) = w_{\\text{mul}} b^2 + w_{\\text{add}} b$  
          Normalised to 16-bit EXACT ($E_{\\text{exact}} = 16^2 + 16 = 272$ units):  
          * **EXACT (16b):** Energy Ratio = $1.0000$
          * **APPROX-1 (10b):** Energy Ratio = $110 / 272 \\approx 0.4044$
          * **APPROX-2 (6b):** Energy Ratio = $42 / 272 \\approx 0.1544$
          * **APPROX-3 (4b):** Energy Ratio = $20 / 272 \\approx 0.0735$

        * **Per-MAC Latency Scaling:**  
          $L(b) = \\text{lat\\_coeff} \\cdot b$  
          * EXACT: $1.000$, A1: $0.625$, A2: $0.375$, A3: $0.250$

        * **Controller Selection Score:**  
          $\\text{Score}(m) = E(m) + \\lambda(p) \\cdot \\frac{\\text{pred}(m)}{\\epsilon_{\\text{budget}}}$  
          where $\\lambda(p) = \\lambda_{\\max}(1-p) + \\lambda_{\\min} p$.
        """
    )

    st.markdown("---")
    st.markdown("### 3. Research Questions & Hypotheses")
    st.markdown(
        """
        * **Research Question:** Can a lightweight resource-, sensitivity-, and budget-aware runtime controller achieve a better energy–accuracy–latency trade-off than static policies?
        * **H1 (Adaptive Precision):** Dynamic multi-precision adaptation reduces energy and latency while staying strictly within the error budget.
        * **H2 (Sensitivity Awareness):** Conditioning-aware sensitivity probing provides superior energy savings over blind resource-only adaptation under tight error budgets.
        * **H3 (Controller Overhead):** Probing overhead is amortised across realistic block sizes, resulting in positive net energy savings.
        """
    )

    st.markdown("---")
    st.markdown("### 4. Important Scope & Limitations")
    st.markdown(
        """
        1. **Software Prototype:** This repository is an algorithmic simulation and research evaluation prototype implemented in Python/NumPy.
        2. **Analytical Estimates:** All reported energy, area, and latency numbers are derived from the documented polynomial cost models.
        3. **No Silicon Claims:** No physical FPGA hardware, ASIC tapeouts, or physical power measurements are claimed.
        """
    )
