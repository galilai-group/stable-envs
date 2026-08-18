"""Capture a batched ``step`` into a CUDA graph once, and record why it could not be if it could not.

A batched GPU simulator for a small env is not compute-bound, it is LAUNCH-bound: CartPole's step is
about forty elementwise kernels over a ``(B, 4)`` tensor, which costs the same wall-clock at B=256 and
at B=16384 because the arithmetic is invisible next to the launches. Capturing the whole step into a
graph replaces those launches with one, and the step finally starts responding to how much work it is
doing — measured per env in the module docstrings of ``classic_control`` and ``minigrid``.

The contract a caller must keep, which is the whole reason this is a shared helper rather than three
copies of the same twenty lines:

  * ``advance()`` reads and writes ONLY pre-allocated tensors at fixed addresses. Anything that REBINDS
    a state attribute instead of writing through ``copy_`` leaves the replayed graph pointing at dead
    memory, and the symptom is an env that silently ignores ``set_state`` rather than an error.
  * ``advance()`` contains no host sync — no ``.item()``, no boolean indexing, no ``if`` on a tensor.
  * the buffers listed in ``state`` are restored afterwards, because the warm-up iterations before
    capture genuinely run and would otherwise advance the env three steps as a side effect of building
    it. Capture itself records without executing, so only the warm-up has to be undone.

Failure is returned, never raised and never hidden: a machine without graph support, or a step that
turns out not to be capturable, should fall back to the eager path and say so. Both paths run the same
``advance``, which is what makes "graph and eager are bit-identical" a property of the design rather
than a coincidence to be tested for — it is tested for anyway.
"""
from __future__ import annotations

import warnings
from typing import Callable, Sequence

import torch


def capture(advance: Callable[[], None], state: Sequence[torch.Tensor], warmup: int = 3):
    """``(graph, "")`` on success, ``(None, reason)`` on failure. ``state`` is restored either way."""
    saved = [b.clone() for b in state]
    try:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(int(warmup)):          # warm up autotune and the allocator before capture
                advance()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        # A warning during capture is treated as a FAILED capture, because the one that matters —
        # "The CUDA Graph is empty", which torch raises as a UserWarning — leaves a graph whose
        # ``replay()`` does nothing at all. An env that silently stops stepping is the worst possible
        # failure here: every downstream number keeps being produced and every one of them is wrong.
        # (Measured: an ``advance`` that built a constant tensor from a python tuple captured empty.)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            with torch.cuda.graph(g):
                advance()
        return g, ""
    except Exception as e:                        # noqa: BLE001 - fall back to eager, do not hide it
        return None, f"{type(e).__name__}: {e}"
    finally:
        for b, s0 in zip(state, saved):
            b.copy_(s0)
