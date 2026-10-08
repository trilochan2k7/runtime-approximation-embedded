"""Interactive Application Workload Adapters for Real-Time Runtime Approximation.

This module provides dedicated, high-performance adapters for interactive execution
of Image Processing, Signal Processing (FIR), and Matrix Multiplication workloads
in the Streamlit application workspace.

All numerical computations use the project's real approximation engine
(:mod:`core.approximation`), sensitivity estimator (:mod:`core.sensitivity`),
controller (:mod:`core.controller`), and metrics (:mod:`core.metrics`).

DISCLAIMER:
All energy and latency values are MODELLED / SIMULATED analytical estimates based
on CostModelConfig, NOT physical FPGA measurements.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from config import DEFAULT_CONFIG, MODE_ORDER, Config, Mode, pressure_to_level
from core.approximation import approx_dot_blocks, approx_matmul, quantize_significand
from core.controller import ControllerDecision, predicted_errors, select_mode
from core.metrics import ErrorMetrics, error_metrics


# --------------------------------------------------------------------------- #
# Helper Utilities
# --------------------------------------------------------------------------- #
def array_to_display_image(arr: np.ndarray, is_diff: bool = False) -> Image.Image:
    """Convert a 2-D numpy array to a PIL display image with appropriate scaling."""
    arr = np.asarray(arr, dtype=np.float64)
    if is_diff:
        # Scale difference with percentile normalization so small errors are visible
        mx = float(np.percentile(arr, 99.5)) if arr.size > 0 else 1.0
        norm = np.clip(arr / max(mx, 1e-9), 0.0, 1.0)
        # Hot/magma false-color map for difference heatmap
        r = (255.0 * np.sqrt(norm)).astype(np.uint8)
        g = (255.0 * (norm ** 2)).astype(np.uint8)
        b = (120.0 * (1.0 - norm)).astype(np.uint8)
        rgb = np.stack([r, g, b], axis=-1)
        return Image.fromarray(rgb, mode="RGB")
    else:
        # Grayscale normalization
        vmin, vmax = float(np.min(arr)), float(np.max(arr))
        denom = max(vmax - vmin, 1e-9)
        norm = np.clip((arr - vmin) / denom, 0.0, 1.0)
        u8 = (norm * 255.0).astype(np.uint8)
        return Image.fromarray(u8, mode="L")


# --------------------------------------------------------------------------- #
# Data Containers
# --------------------------------------------------------------------------- #
@dataclass
class AppExecutionResult:
    """Complete record of an interactive workload execution."""

    workload_type: str                  # "image", "signal", "matrix"
    operation_name: str                 # e.g., "3x3 Gaussian Blur", "FIR Low-Pass"
    input_summary: Dict[str, str]       # Input dimensions, type, etc.
    exact_output: np.ndarray            # Exact full-precision reference output
    adaptive_output: np.ndarray         # Adaptive reduced-precision output
    diff_output: np.ndarray             # Absolute difference |adaptive - exact|
    decision: ControllerDecision        # Controller decision record
    metrics: ErrorMetrics               # Error metrics (MAE, RMSE, Norm L2, etc.)
    budget: float                       # Configured error budget (e.g., 0.02)
    pressure: float                     # Configured resource pressure (0.0 to 1.0)
    sensitivity_amp: float              # Estimated error amplification factor A
    n_macs: int                         # Total MAC operations in workload
    energy_exact: float                 # Modelled energy for exact computation
    energy_adaptive: float              # Modelled energy for adaptive computation
    energy_savings_pct: float           # Energy savings percentage vs exact
    latency_exact: float                # Modelled latency for exact computation
    latency_adaptive: float             # Modelled latency for adaptive computation
    latency_reduction_pct: float        # Latency reduction percentage vs exact
    overhead_macs: float                # Controller + probe overhead in MAC units
    overhead_pct: float                 # Overhead as % of workload MACs
    execution_time_ms: float            # Real wall-clock execution time in ms
    compliant: bool                     # True if norm_rel_error <= budget


# --------------------------------------------------------------------------- #
# Image Processing Workload Adapter
# --------------------------------------------------------------------------- #
class ImageWorkloadAdapter:
    """Adapter for interactive 2-D image filtering using the approximation engine."""

    KERNELS: Dict[str, np.ndarray] = {
        "3x3 Gaussian Blur": np.array(
            [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]], dtype=np.float64
        ) / 16.0,
        "3x3 Box Blur": np.ones((3, 3), dtype=np.float64) / 9.0,
        "3x3 Sharpen": np.array(
            [[0.0, -1.0, 0.0], [-1.0, 5.0, -1.0], [0.0, -1.0, 0.0]], dtype=np.float64
        ),
        "3x3 Sobel Horizontal (Edges)": np.array(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=np.float64
        ),
        "3x3 Laplacian (High-Pass)": np.array(
            [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float64
        ),
    }

    @staticmethod
    def generate_synthetic_image(pattern: str, size: int = 256, seed: int = 42) -> np.ndarray:
        """Generate a synthetic test image with known frequency characteristics."""
        rng = np.random.default_rng(seed)
        x = np.linspace(0, 4 * np.pi, size)
        y = np.linspace(0, 4 * np.pi, size)
        xx, yy = np.meshgrid(x, y)

        if pattern == "Sensor Grayscale Pattern":
            # Realistic sensor image: smooth gradients + high-frequency texture patches
            base = 128.0 + 60.0 * np.sin(0.5 * xx) * np.cos(0.5 * yy)
            texture = 25.0 * np.sin(4.0 * xx + 3.0 * yy)
            noise = rng.normal(0.0, 5.0, (size, size))
            img = base + texture + noise
        elif pattern == "Checkerboard & Edges":
            # High-contrast geometric edges
            sq_size = max(8, size // 8)
            board = ((xx // (4 * np.pi / 8)) + (yy // (4 * np.pi / 8))) % 2
            img = 40.0 + 175.0 * board + rng.normal(0.0, 3.0, (size, size))
        elif pattern == "Gradient Waves (Smooth)":
            # Low-frequency smooth gradient (low sensitivity)
            img = 128.0 + 100.0 * np.sin(0.3 * xx) + 20.0 * np.cos(0.2 * yy)
        else:  # High-frequency oscillation (high sensitivity)
            img = 128.0 + 100.0 * np.sin(3.0 * xx) * np.sin(3.0 * yy)

        return np.clip(img, 0.0, 255.0).astype(np.float64)

    @staticmethod
    def load_user_image(pil_img: Image.Image, max_dim: int = 384) -> np.ndarray:
        """Convert a user-uploaded PIL Image to a single-channel float64 array."""
        # Convert to grayscale
        gray = pil_img.convert("L")
        # Resize if image is too large for responsive interactive execution
        w, h = gray.size
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            gray = gray.resize((new_w, new_h), Image.Resampling.BILINEAR)
        return np.array(gray, dtype=np.float64)

    @classmethod
    def _conv2d_fast(cls, img: np.ndarray, kernel: np.ndarray, bits: Optional[int]) -> np.ndarray:
        """2-D valid convolution with operands quantized to `bits` significand bits."""
        kh, kw = kernel.shape
        H, W = img.shape
        out_h = H - kh + 1
        out_w = W - kw + 1
        if out_h <= 0 or out_w <= 0:
            return np.zeros((max(0, out_h), max(0, out_w)), dtype=np.float64)

        img_q = quantize_significand(img, bits)
        ker_q = quantize_significand(kernel, bits)
        out = np.zeros((out_h, out_w), dtype=np.float64)

        for i in range(kh):
            for j in range(kw):
                out += ker_q[i, j] * img_q[i : i + out_h, j : j + out_w]

        return out

    @classmethod
    def estimate_image_sensitivity(
        cls, img: np.ndarray, kernel: np.ndarray, cfg: Config
    ) -> Tuple[float, float, int]:
        """Probe representative image patches to estimate error amplification A."""
        kh, kw = kernel.shape
        probe_bits = cfg.sensitivity.probe_bits
        probe_elements = cfg.sensitivity.probe_elements

        # Extract representative sub-patch (center)
        side = max(4, int(np.sqrt(probe_elements)))
        cr, cc = img.shape[0] // 2, img.shape[1] // 2
        r0 = max(0, cr - side // 2)
        r1 = min(img.shape[0], r0 + side + kh - 1)
        c0 = max(0, cc - side // 2)
        c1 = min(img.shape[1], c0 + side + kw - 1)

        patch = img[r0:r1, c0:c1]
        exact_probe = cls._conv2d_fast(patch, kernel, None)
        approx_probe = cls._conv2d_fast(patch, kernel, probe_bits)

        diff_norm = float(np.linalg.norm(approx_probe - exact_probe))
        ref_norm = float(np.linalg.norm(exact_probe)) + 1e-12
        observed_rel_err = diff_norm / ref_norm

        unit_err = float(2.0 ** (-probe_bits))
        amp = observed_rel_err / max(unit_err, 1e-15)
        # MAC count for 1 exact + 1 approx pass over probe patch
        probe_macs = 2 * exact_probe.size * kh * kw
        score = float(np.clip(np.log2(max(amp, 1e-6)) / 6.0, 0.0, 1.0))

        return amp, score, probe_macs

    @classmethod
    def execute(
        cls,
        img: np.ndarray,
        op_name: str,
        budget: float,
        pressure: float,
        cfg: Optional[Config] = None,
    ) -> AppExecutionResult:
        """Run the full adaptive execution pipeline for image processing."""
        cfg = cfg or DEFAULT_CONFIG
        t0 = time.perf_counter()

        kernel = cls.KERNELS.get(op_name, cls.KERNELS["3x3 Gaussian Blur"])
        kh, kw = kernel.shape
        H, W = img.shape
        out_h, out_w = H - kh + 1, W - kw + 1
        n_macs = out_h * out_w * kh * kw

        # 1. Sensitivity Estimation
        t_probe0 = time.perf_counter()
        amp, score, probe_macs = cls.estimate_image_sensitivity(img, kernel, cfg)
        probe_time = time.perf_counter() - t_probe0

        # 2. Runtime Controller Decision
        pred = predicted_errors(amp, cfg)
        chosen_mode, feasible, scores = select_mode(pred, budget, pressure, cfg)

        overhead_macs = float(probe_macs + cfg.controller.decision_overhead_units)
        reason = (
            f"Resource pressure={pressure:.2f} ({pressure_to_level(pressure)}), "
            f"Sensitivity A={amp:.2f}, Budget={budget*100:.2f}%. "
            f"Selected {chosen_mode.value} (pred error={pred[chosen_mode]*100:.3f}%)."
        )

        decision = ControllerDecision(
            mode=chosen_mode,
            pressure=pressure,
            predicted_error=pred,
            feasible=feasible,
            scores=scores,
            overhead_units=overhead_macs,
            wall_time_s=probe_time,
            reason=reason,
            sensitivity_amp=amp,
            sensitivity_score=score,
            probe_macs=probe_macs,
        )

        # 3. Exact vs Adaptive Computation
        exact_out = cls._conv2d_fast(img, kernel, None)
        bits = cfg.mode_bits(chosen_mode)  # None for EXACT
        adaptive_out = cls._conv2d_fast(img, kernel, bits)
        diff_out = np.abs(adaptive_out - exact_out)

        # 4. Error Metrics
        metrics = error_metrics(adaptive_out, exact_out)
        wall_time_ms = (time.perf_counter() - t0) * 1000.0

        # 5. Energy & Latency Models
        energy_ratio = cfg.energy_ratio(chosen_mode)
        latency_ratio = cfg.latency_ratio(chosen_mode)

        energy_exact = float(n_macs * 1.0)
        energy_adaptive = float(n_macs * energy_ratio + overhead_macs)
        energy_savings_pct = max(0.0, (1.0 - (energy_adaptive / energy_exact)) * 100.0)

        latency_exact = float(n_macs * 1.0)
        latency_adaptive = float(n_macs * latency_ratio)
        latency_reduction_pct = max(0.0, (1.0 - (latency_adaptive / latency_exact)) * 100.0)

        overhead_pct = (overhead_macs / max(1, n_macs)) * 100.0
        compliant = bool(metrics.norm_rel_error <= (budget * cfg.controller.safety_margin + 1e-9))

        return AppExecutionResult(
            workload_type="image",
            operation_name=op_name,
            input_summary={
                "Image Size": f"{H} × {W}",
                "Total Pixels": f"{H * W:,}",
                "Output Size": f"{out_h} × {out_w}",
                "Kernel Size": f"{kh} × {kw}",
                "Total MACs": f"{n_macs:,}",
            },
            exact_output=exact_out,
            adaptive_output=adaptive_out,
            diff_output=diff_out,
            decision=decision,
            metrics=metrics,
            budget=budget,
            pressure=pressure,
            sensitivity_amp=amp,
            n_macs=n_macs,
            energy_exact=energy_exact,
            energy_adaptive=energy_adaptive,
            energy_savings_pct=energy_savings_pct,
            latency_exact=latency_exact,
            latency_adaptive=latency_adaptive,
            latency_reduction_pct=latency_reduction_pct,
            overhead_macs=overhead_macs,
            overhead_pct=overhead_pct,
            execution_time_ms=wall_time_ms,
            compliant=compliant,
        )


# --------------------------------------------------------------------------- #
# Signal Processing Workload Adapter (FIR)
# --------------------------------------------------------------------------- #
class SignalWorkloadAdapter:
    """Adapter for interactive 1-D FIR filtering using the approximation engine."""

    @staticmethod
    def generate_signal(
        signal_type: str, length: int = 1024, noise_std: float = 0.05, seed: int = 42
    ) -> np.ndarray:
        """Generate a 1-D synthetic signal with realistic spectral components."""
        rng = np.random.default_rng(seed)
        t = np.linspace(0.0, 1.0, length, endpoint=False)

        if signal_type == "Chirp & Harmonics":
            # Linear frequency sweep + fundamental tone
            f0, f1 = 5.0, 60.0
            chirp = np.sin(2.0 * np.pi * (f0 + 0.5 * (f1 - f0) * t) * t)
            harm = 0.5 * np.sin(2.0 * np.pi * 15.0 * t)
            sig = 1.0 + chirp + harm
        elif signal_type == "Noisy Multi-Tone (Audio-like)":
            # Multi-frequency audio band simulation
            tones = [
                0.8 * np.sin(2.0 * np.pi * 12.0 * t),
                0.5 * np.sin(2.0 * np.pi * 35.0 * t + 0.5),
                0.3 * np.sin(2.0 * np.pi * 80.0 * t + 1.2),
            ]
            sig = 1.5 + sum(tones)
        elif signal_type == "Non-Stationary (Dynamic Sensitivity)":
            # Split signal into robust DC segment, mixed ripple, and high-frequency near-Nyquist
            n3 = length // 3
            s1 = 4.0 + 0.3 * np.sin(2.0 * np.pi * 4.0 * t[:n3])
            s2 = 1.0 + 0.8 * np.sin(2.0 * np.pi * 20.0 * t[n3 : 2 * n3])
            s3 = np.sin(np.pi * 0.9 * np.arange(length - 2 * n3))
            sig = np.concatenate([s1, s2, s3])
        else:  # Clean Carrier
            sig = 2.0 + np.sin(2.0 * np.pi * 10.0 * t)

        if noise_std > 0:
            sig = sig + rng.normal(0.0, noise_std, size=length)

        return sig.astype(np.float64)

    @staticmethod
    def design_fir_filter(taps: int = 32) -> np.ndarray:
        """Create a Hann-windowed low-pass FIR filter."""
        w = np.hanning(taps + 2)[1:-1]
        if not np.any(w):
            w = np.ones(taps)
        return (w / w.sum()).astype(np.float64)

    @classmethod
    def execute(
        cls,
        signal: np.ndarray,
        taps_count: int = 32,
        budget: float = 0.02,
        pressure: float = 0.5,
        cfg: Optional[Config] = None,
    ) -> AppExecutionResult:
        """Run the full adaptive execution pipeline for FIR filtering."""
        cfg = cfg or DEFAULT_CONFIG
        t0 = time.perf_counter()

        taps = cls.design_fir_filter(taps_count)
        T = taps_count
        N = len(signal)
        out_len = N - T + 1

        if out_len <= 0:
            raise ValueError(f"Signal length ({N}) must be greater than filter taps ({T})")

        # Build sliding windows: shape (out_len, T)
        from numpy.lib.stride_tricks import sliding_window_view
        windows = sliding_window_view(signal, T)  # (out_len, T)
        n_macs = out_len * T

        # 1. Sensitivity Estimation (probe sub-segment)
        probe_bits = cfg.sensitivity.probe_bits
        probe_k = min(cfg.sensitivity.probe_elements, out_len)
        start_idx = max(0, (out_len - probe_k) // 2)
        sub_windows = windows[start_idx : start_idx + probe_k]

        exact_probe = approx_dot_blocks(taps, sub_windows, None)
        approx_probe = approx_dot_blocks(taps, sub_windows, probe_bits)

        diff_norm = float(np.linalg.norm(approx_probe - exact_probe))
        ref_norm = float(np.linalg.norm(exact_probe)) + 1e-12
        obs_rel_err = diff_norm / ref_norm
        amp = obs_rel_err / max(float(2.0 ** (-probe_bits)), 1e-15)
        probe_macs = 2 * probe_k * T
        score = float(np.clip(np.log2(max(amp, 1e-6)) / 6.0, 0.0, 1.0))

        # 2. Runtime Controller Decision
        pred = predicted_errors(amp, cfg)
        chosen_mode, feasible, scores = select_mode(pred, budget, pressure, cfg)

        overhead_macs = float(probe_macs + cfg.controller.decision_overhead_units)
        reason = (
            f"Resource pressure={pressure:.2f} ({pressure_to_level(pressure)}), "
            f"Sensitivity A={amp:.2f}, Budget={budget*100:.2f}%. "
            f"Selected {chosen_mode.value} (pred error={pred[chosen_mode]*100:.3f}%)."
        )

        decision = ControllerDecision(
            mode=chosen_mode,
            pressure=pressure,
            predicted_error=pred,
            feasible=feasible,
            scores=scores,
            overhead_units=overhead_macs,
            wall_time_s=0.0,
            reason=reason,
            sensitivity_amp=amp,
            sensitivity_score=score,
            probe_macs=probe_macs,
        )

        # 3. Exact vs Adaptive Computation
        exact_out = approx_dot_blocks(taps, windows, None)
        bits = cfg.mode_bits(chosen_mode)
        adaptive_out = approx_dot_blocks(taps, windows, bits)
        diff_out = np.abs(adaptive_out - exact_out)

        # 4. Error Metrics
        metrics = error_metrics(adaptive_out, exact_out)
        wall_time_ms = (time.perf_counter() - t0) * 1000.0

        # 5. Energy & Latency Models
        energy_ratio = cfg.energy_ratio(chosen_mode)
        latency_ratio = cfg.latency_ratio(chosen_mode)

        energy_exact = float(n_macs * 1.0)
        energy_adaptive = float(n_macs * energy_ratio + overhead_macs)
        energy_savings_pct = max(0.0, (1.0 - (energy_adaptive / energy_exact)) * 100.0)

        latency_exact = float(n_macs * 1.0)
        latency_adaptive = float(n_macs * latency_ratio)
        latency_reduction_pct = max(0.0, (1.0 - (latency_adaptive / latency_exact)) * 100.0)

        overhead_pct = (overhead_macs / max(1, n_macs)) * 100.0
        compliant = bool(metrics.norm_rel_error <= (budget * cfg.controller.safety_margin + 1e-9))

        return AppExecutionResult(
            workload_type="signal",
            operation_name=f"1-D FIR Filter ({T} taps)",
            input_summary={
                "Signal Length": f"{N:,} samples",
                "Filter Taps": f"{T} taps (Hann LPF)",
                "Output Length": f"{out_len:,} samples",
                "Total MACs": f"{n_macs:,}",
            },
            exact_output=exact_out,
            adaptive_output=adaptive_out,
            diff_output=diff_out,
            decision=decision,
            metrics=metrics,
            budget=budget,
            pressure=pressure,
            sensitivity_amp=amp,
            n_macs=n_macs,
            energy_exact=energy_exact,
            energy_adaptive=energy_adaptive,
            energy_savings_pct=energy_savings_pct,
            latency_exact=latency_exact,
            latency_adaptive=latency_adaptive,
            latency_reduction_pct=latency_reduction_pct,
            overhead_macs=overhead_macs,
            overhead_pct=overhead_pct,
            execution_time_ms=wall_time_ms,
            compliant=compliant,
        )


# --------------------------------------------------------------------------- #
# Matrix Processing Workload Adapter (MatMul)
# --------------------------------------------------------------------------- #
class MatrixWorkloadAdapter:
    """Adapter for interactive matrix multiplication using the approximation engine."""

    @staticmethod
    def generate_matrices(
        preset: str, M: int = 64, K: int = 64, N: int = 64, seed: int = 42
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate matrices A (M x K) and B (K x N) according to specified sensitivity presets."""
        rng = np.random.default_rng(seed)

        if preset == "Smooth & Positive (Low Sensitivity)":
            # Well-conditioned positive values: minimal cancellation
            A = 1.0 + rng.uniform(0.1, 1.0, (M, K))
            B = 1.0 + rng.uniform(0.1, 1.0, (K, N))
        elif preset == "Near-Zero Mean (High Sensitivity)":
            # Zero-mean oscillatory rows: high cancellation risk
            t = np.linspace(0, 10 * np.pi, K)
            A = np.stack([np.sin(t + rng.uniform(0, np.pi)) - 0.05 for _ in range(M)])
            B = np.stack([np.cos(t + rng.uniform(0, np.pi)) for _ in range(N)], axis=1)
        elif preset == "Mixed Dynamic (Realistic DSP)":
            # Mixed features
            A = rng.normal(1.0, 0.8, (M, K))
            B = rng.uniform(0.5, 2.0, (K, N))
        else:  # Uniform Random
            A = rng.uniform(-1.0, 1.0, (M, K))
            B = rng.uniform(-1.0, 1.0, (K, N))

        return A.astype(np.float64), B.astype(np.float64)

    @classmethod
    def execute(
        cls,
        A: np.ndarray,
        B: np.ndarray,
        budget: float = 0.02,
        pressure: float = 0.5,
        cfg: Optional[Config] = None,
    ) -> AppExecutionResult:
        """Run the full adaptive execution pipeline for matrix multiplication."""
        cfg = cfg or DEFAULT_CONFIG
        t0 = time.perf_counter()

        M, K = A.shape
        K2, N = B.shape
        if K != K2:
            raise ValueError(f"Matrix dimension mismatch: A is {M}x{K}, B is {K2}x{N}")

        n_macs = M * N * K

        # 1. Sensitivity Estimation (probe small submatrix)
        probe_bits = cfg.sensitivity.probe_bits
        side = max(2, int(np.sqrt(cfg.sensitivity.probe_elements)))
        kr = min(side, M)
        kc = min(side, N)

        sub_A = A[:kr]
        sub_B = B[:, :kc]
        exact_probe = approx_matmul(sub_A, sub_B, None)
        approx_probe = approx_matmul(sub_A, sub_B, probe_bits)

        diff_norm = float(np.linalg.norm(approx_probe - exact_probe))
        ref_norm = float(np.linalg.norm(exact_probe)) + 1e-12
        obs_rel_err = diff_norm / ref_norm
        amp = obs_rel_err / max(float(2.0 ** (-probe_bits)), 1e-15)
        probe_macs = 2 * kr * kc * K
        score = float(np.clip(np.log2(max(amp, 1e-6)) / 6.0, 0.0, 1.0))

        # 2. Runtime Controller Decision
        pred = predicted_errors(amp, cfg)
        chosen_mode, feasible, scores = select_mode(pred, budget, pressure, cfg)

        overhead_macs = float(probe_macs + cfg.controller.decision_overhead_units)
        reason = (
            f"Resource pressure={pressure:.2f} ({pressure_to_level(pressure)}), "
            f"Sensitivity A={amp:.2f}, Budget={budget*100:.2f}%. "
            f"Selected {chosen_mode.value} (pred error={pred[chosen_mode]*100:.3f}%)."
        )

        decision = ControllerDecision(
            mode=chosen_mode,
            pressure=pressure,
            predicted_error=pred,
            feasible=feasible,
            scores=scores,
            overhead_units=overhead_macs,
            wall_time_s=0.0,
            reason=reason,
            sensitivity_amp=amp,
            sensitivity_score=score,
            probe_macs=probe_macs,
        )

        # 3. Exact vs Adaptive Computation
        exact_out = approx_matmul(A, B, None)
        bits = cfg.mode_bits(chosen_mode)
        adaptive_out = approx_matmul(A, B, bits)
        diff_out = np.abs(adaptive_out - exact_out)

        # 4. Error Metrics
        metrics = error_metrics(adaptive_out, exact_out)
        wall_time_ms = (time.perf_counter() - t0) * 1000.0

        # 5. Energy & Latency Models
        energy_ratio = cfg.energy_ratio(chosen_mode)
        latency_ratio = cfg.latency_ratio(chosen_mode)

        energy_exact = float(n_macs * 1.0)
        energy_adaptive = float(n_macs * energy_ratio + overhead_macs)
        energy_savings_pct = max(0.0, (1.0 - (energy_adaptive / energy_exact)) * 100.0)

        latency_exact = float(n_macs * 1.0)
        latency_adaptive = float(n_macs * latency_ratio)
        latency_reduction_pct = max(0.0, (1.0 - (latency_adaptive / latency_exact)) * 100.0)

        overhead_pct = (overhead_macs / max(1, n_macs)) * 100.0
        compliant = bool(metrics.norm_rel_error <= (budget * cfg.controller.safety_margin + 1e-9))

        return AppExecutionResult(
            workload_type="matrix",
            operation_name=f"Dense MatMul ({M}×{K} × {K}×{N})",
            input_summary={
                "Matrix A": f"{M} × {K}",
                "Matrix B": f"{K} × {N}",
                "Result Matrix": f"{M} × {N}",
                "Total MACs": f"{n_macs:,}",
            },
            exact_output=exact_out,
            adaptive_output=adaptive_out,
            diff_output=diff_out,
            decision=decision,
            metrics=metrics,
            budget=budget,
            pressure=pressure,
            sensitivity_amp=amp,
            n_macs=n_macs,
            energy_exact=energy_exact,
            energy_adaptive=energy_adaptive,
            energy_savings_pct=energy_savings_pct,
            latency_exact=latency_exact,
            latency_adaptive=latency_adaptive,
            latency_reduction_pct=latency_reduction_pct,
            overhead_macs=overhead_macs,
            overhead_pct=overhead_pct,
            execution_time_ms=wall_time_ms,
            compliant=compliant,
        )
