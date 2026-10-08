"""Workloads that execute through the approximation engine.

Three workloads are provided, all of which reduce to multiply-accumulate (MAC)
operations and therefore exercise the reduced-precision engine uniformly:

* :class:`FIRWorkload`      - 1-D FIR filtering (sliding dot products).
* :class:`MatMulWorkload`   - dense matrix multiplication ``A @ B``.
* :class:`ConvWorkload`     - 2-D single-channel convolution (image blur).

Each workload is split into **blocks** (signal frames, row-blocks of ``A``, or
bands of output rows).  Blocks are deliberately *non-stationary*: some carry
smooth / single-sign content that is numerically robust to operand rounding
(low output sensitivity), while others carry zero-mean oscillatory content whose
filtered output nearly cancels, strongly amplifying rounding error (high output
sensitivity).  Real signals and images are non-stationary in exactly this way,
and this heterogeneity is what gives a per-block, sensitivity-aware controller
something to exploit (hypothesis H2).

All generation is seeded, so a given configuration always produces identical
blocks (reproducibility).  A workload is agnostic to which mode a block will be
computed in; the mode (operand bit-width) is supplied at ``compute`` time.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from config import Config
from core.approximation import (
    approx_dot_blocks,
    approx_matmul,
    quantize_significand,
)

WORKLOAD_NAMES: List[str] = ["fir", "matmul", "conv"]

# Overall sensitivity bias of a workload instance: controls the fraction of
# high-sensitivity blocks, giving the "multiple sensitivity conditions" axis.
SENSITIVITY_BIASES: List[str] = ["low", "mixed", "high"]


@dataclass
class Block:
    """One unit of work scheduled by the controller.

    Attributes
    ----------
    index:
        Position of the block in the stream.
    payload:
        Workload-specific arrays needed to compute the block.
    n_macs:
        Number of multiply-accumulate operations in the block (a real count,
        used by the cost model - not an estimate).
    inner_dim:
        Number of terms accumulated per output element.
    character:
        Ground-truth sensitivity class assigned at generation time
        (``"low"``/``"med"``/``"high"``).  Used only for analysis and to
        validate the estimator - the controller never reads it.
    """

    index: int
    payload: Dict[str, np.ndarray]
    n_macs: int
    inner_dim: int
    character: str = "med"


@dataclass
class ProbeResult:
    """Output of a sensitivity probe on a small sub-block of a block."""

    exact: np.ndarray     # full-precision reference output of the sub-block
    approx: np.ndarray    # output with operands quantised to the probe width
    n_macs: int           # MACs performed for ONE pass over the sub-block


def _character_sequence(n_blocks: int, bias: str, rng: np.random.Generator) -> List[str]:
    """Assign a sensitivity character to each block according to ``bias``."""

    if bias == "low":
        weights = {"low": 0.70, "med": 0.25, "high": 0.05}
    elif bias == "high":
        weights = {"low": 0.10, "med": 0.30, "high": 0.60}
    else:  # mixed
        weights = {"low": 0.34, "med": 0.33, "high": 0.33}
    choices = list(weights.keys())
    probs = np.array([weights[c] for c in choices], dtype=np.float64)
    probs /= probs.sum()
    return list(rng.choice(choices, size=n_blocks, p=probs))


def _segment_signal(length: int, character: str, rng: np.random.Generator) -> np.ndarray:
    """Generate a 1-D signal segment with the requested robustness character.

    * ``low``  - large positive DC offset plus a slow ripple.  A low-pass filter
      preserves it almost exactly, so operand rounding barely perturbs the
      output (low sensitivity).
    * ``high`` - zero-mean, near-Nyquist oscillation.  A smoothing / averaging
      filter nearly cancels it, so a small operand perturbation produces a large
      *relative* output change (high sensitivity).
    * ``med``  - a blend of the two.
    """

    t = np.arange(length, dtype=np.float64)
    phase = rng.uniform(0.0, 2.0 * np.pi)
    if character == "low":
        offset = 4.0 + rng.uniform(0.0, 1.0)
        ripple = 0.3 * np.sin(2.0 * np.pi * 0.01 * t + phase)
        sig = offset + ripple
    elif character == "high":
        freq = np.pi * (0.5 - 0.02 * rng.uniform())  # near Nyquist
        hf = np.sin(freq * t + phase)
        sig = hf - np.mean(hf)  # enforce (near) zero mean
    else:  # med
        offset = 1.2
        slow = 0.8 * np.sin(2.0 * np.pi * 0.02 * t + phase)
        fast = 0.9 * np.sin(np.pi * 0.45 * t + phase)
        sig = offset + slow + fast
    return sig.astype(np.float64)


class Workload(ABC):
    """Abstract base class for all workloads."""

    name: str

    def __init__(self, config: Config, bias: str, seed: int) -> None:
        self.config = config
        self.bias = bias
        self.seed = seed
        self._blocks: List[Block] = self._build_blocks()

    @property
    def blocks(self) -> List[Block]:
        return self._blocks

    @abstractmethod
    def _build_blocks(self) -> List[Block]:
        ...

    @abstractmethod
    def compute(self, block: Block, bits: Optional[int]) -> np.ndarray:
        """Compute a block's output with operands quantised to ``bits`` bits.

        ``bits is None`` performs the full-precision exact reference.
        """

    @abstractmethod
    def probe(self, block: Block, elements: int, probe_bits: int) -> ProbeResult:
        """Compute a small representative sub-block exactly and at ``probe_bits``.

        Used by the sensitivity estimator.  ``elements`` is the target number of
        output cells to evaluate; the actual count may be smaller for tiny
        blocks.
        """


# --------------------------------------------------------------------------- #
# FIR / vector workload
# --------------------------------------------------------------------------- #
class FIRWorkload(Workload):
    name = "fir"

    @staticmethod
    def _make_filter(taps: int) -> np.ndarray:
        # Normalised low-pass (Hann-windowed moving average): all-positive taps
        # summing to 1, so robustness is governed by the signal content.
        w = np.hanning(taps + 2)[1:-1]
        if not np.any(w):
            w = np.ones(taps)
        return (w / w.sum()).astype(np.float64)

    def _build_blocks(self) -> List[Block]:
        cfg = self.config.workload
        rng = np.random.default_rng(self.seed)
        self.taps = self._make_filter(cfg.fir_taps)
        T = cfg.fir_taps
        n_blocks = cfg.fir_blocks
        block_len = cfg.fir_length // n_blocks
        chars = _character_sequence(n_blocks, self.bias, rng)

        blocks: List[Block] = []
        for i in range(n_blocks):
            seg = _segment_signal(block_len, chars[i], rng)
            # Causal sliding windows: pad with the segment's leading sample so
            # every output accumulates T taps.  windows[j] = seg[j-T+1 .. j].
            padded = np.concatenate([np.full(T - 1, seg[0]), seg])
            windows = sliding_window_view(padded, T)  # (block_len, T)
            blocks.append(
                Block(
                    index=i,
                    payload={"windows": np.ascontiguousarray(windows)},
                    n_macs=block_len * T,
                    inner_dim=T,
                    character=chars[i],
                )
            )
        return blocks

    def compute(self, block: Block, bits: Optional[int]) -> np.ndarray:
        return approx_dot_blocks(self.taps, block.payload["windows"], bits)

    def probe(self, block: Block, elements: int, probe_bits: int) -> ProbeResult:
        windows = block.payload["windows"]
        n = windows.shape[0]
        k = int(min(elements, n))
        # Sample a centred slice so the probe is not biased by the padded
        # leading samples at the start of the block.
        start = max(0, (n - k) // 2)
        sub = windows[start : start + k]
        exact = approx_dot_blocks(self.taps, sub, None)
        approx = approx_dot_blocks(self.taps, sub, probe_bits)
        return ProbeResult(exact=exact, approx=approx, n_macs=k * windows.shape[1])


# --------------------------------------------------------------------------- #
# Dense matrix multiplication
# --------------------------------------------------------------------------- #
class MatMulWorkload(Workload):
    name = "matmul"

    def _build_blocks(self) -> List[Block]:
        cfg = self.config.workload
        rng = np.random.default_rng(self.seed)
        K, N = cfg.mat_k, cfg.mat_n
        # Shared right operand B: smooth, mostly single-sign columns, so that
        # cancellation (and hence sensitivity) is governed by the left rows.
        tcol = np.arange(K, dtype=np.float64)
        self.B = np.stack(
            [1.0 + 0.5 * np.sin(2.0 * np.pi * (c + 1) * 0.004 * tcol + 0.1 * c)
             for c in range(N)],
            axis=1,
        ).astype(np.float64)

        rb = cfg.mat_row_block
        n_blocks = cfg.mat_m // rb
        chars = _character_sequence(n_blocks, self.bias, rng)
        blocks: List[Block] = []
        for i in range(n_blocks):
            rows = np.stack(
                [_segment_signal(K, chars[i], rng) for _ in range(rb)], axis=0
            )
            blocks.append(
                Block(
                    index=i,
                    payload={"A": np.ascontiguousarray(rows)},
                    n_macs=rb * N * K,
                    inner_dim=K,
                    character=chars[i],
                )
            )
        return blocks

    def compute(self, block: Block, bits: Optional[int]) -> np.ndarray:
        return approx_matmul(block.payload["A"], self.B, bits)

    def probe(self, block: Block, elements: int, probe_bits: int) -> ProbeResult:
        A = block.payload["A"]
        side = max(1, int(np.sqrt(elements)))
        kr = int(min(side, A.shape[0]))
        kc = int(min(side, self.B.shape[1]))
        sub_a = A[:kr]
        sub_b = self.B[:, :kc]
        exact = approx_matmul(sub_a, sub_b, None)
        approx = approx_matmul(sub_a, sub_b, probe_bits)
        return ProbeResult(exact=exact, approx=approx, n_macs=kr * kc * A.shape[1])


# --------------------------------------------------------------------------- #
# 2-D single-channel convolution
# --------------------------------------------------------------------------- #
class ConvWorkload(Workload):
    name = "conv"

    @staticmethod
    def _make_kernel(k: int) -> np.ndarray:
        # Separable Gaussian-like smoothing kernel, normalised to sum 1.
        w = np.hanning(k + 2)[1:-1]
        if not np.any(w):
            w = np.ones(k)
        w = w / w.sum()
        ker = np.outer(w, w)
        return (ker / ker.sum()).astype(np.float64)

    def _build_image(self, H: int, W: int, chars: List[str], band: int,
                     rng: np.random.Generator) -> np.ndarray:
        img = np.zeros((H, W), dtype=np.float64)
        for i, ch in enumerate(chars):
            r0 = i * band
            r1 = min(H, r0 + band + (self.conv_k - 1))  # include halo rows
            for r in range(r0, r1):
                img[r] = _segment_signal(W, ch, rng)
        # Any trailing rows beyond the last band inherit a smooth fill.
        return img

    def _build_blocks(self) -> List[Block]:
        cfg = self.config.workload
        rng = np.random.default_rng(self.seed)
        self.conv_k = cfg.conv_kernel
        self.kernel = self._make_kernel(self.conv_k)
        H = W = cfg.img_size
        kh = kw = self.conv_k
        band = cfg.conv_row_block
        out_rows_total = H - kh + 1
        n_blocks = out_rows_total // band
        chars = _character_sequence(n_blocks, self.bias, rng)
        self.image = self._build_image(H, W, chars, band, rng)
        self.out_w = W - kw + 1

        blocks: List[Block] = []
        for i in range(n_blocks):
            r0 = i * band
            img_slice = self.image[r0 : r0 + band + kh - 1]
            blocks.append(
                Block(
                    index=i,
                    payload={"img": np.ascontiguousarray(img_slice)},
                    n_macs=band * self.out_w * kh * kw,
                    inner_dim=kh * kw,
                    character=chars[i],
                )
            )
        return blocks

    def _conv_valid(self, img: np.ndarray, bits: Optional[int]) -> np.ndarray:
        kh, kw = self.kernel.shape
        out_rows = img.shape[0] - kh + 1
        out_cols = img.shape[1] - kw + 1
        if out_rows <= 0 or out_cols <= 0:
            return np.zeros((max(out_rows, 0), max(out_cols, 0)), dtype=np.float64)
        img_q = quantize_significand(img, bits)
        ker_q = quantize_significand(self.kernel, bits)
        acc = np.zeros((out_rows, out_cols), dtype=np.float64)
        for di in range(kh):
            for dj in range(kw):
                acc += ker_q[di, dj] * img_q[di : di + out_rows, dj : dj + out_cols]
        return acc

    def compute(self, block: Block, bits: Optional[int]) -> np.ndarray:
        return self._conv_valid(block.payload["img"], bits)

    def probe(self, block: Block, elements: int, probe_bits: int) -> ProbeResult:
        img = block.payload["img"]
        kh, kw = self.kernel.shape
        side = max(1, int(np.sqrt(elements)))
        rows = int(min(img.shape[0], side + kh - 1))
        cols = int(min(img.shape[1], side + kw - 1))
        patch = img[:rows, :cols]
        exact = self._conv_valid(patch, None)
        approx = self._conv_valid(patch, probe_bits)
        out_r = max(0, rows - kh + 1)
        out_c = max(0, cols - kw + 1)
        return ProbeResult(exact=exact, approx=approx, n_macs=out_r * out_c * kh * kw)


def make_workload(name: str, config: Config, bias: str = "mixed",
                  seed: Optional[int] = None) -> Workload:
    """Factory returning a built workload instance by name."""

    seed = config.seed if seed is None else seed
    if name == "fir":
        return FIRWorkload(config, bias, seed)
    if name == "matmul":
        return MatMulWorkload(config, bias, seed)
    if name == "conv":
        return ConvWorkload(config, bias, seed)
    raise ValueError(f"unknown workload {name!r}; expected one of {WORKLOAD_NAMES}")
