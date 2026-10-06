from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence

import torch


def capture(advance: Callable[[], None], state: Sequence[torch.Tensor], warmup: int = 3):
    """Capture a fixed-buffer transition and return (graph, "") or (None, reason).

    ``advance`` must mutate preallocated tensors in place without host reads or
    host-to-device copies. Warmup executes real transitions; all mutable buffers
    must appear in ``state`` so they can be restored in the finally block.
    Capture failures allow the caller to fall back to eager execution.
    """
    saved = [b.clone() for b in state]
    try:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(int(warmup)):  # warm up autotune and the allocator before capture
                advance()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        # Treat capture warnings as failures: an empty graph would silently perform no transitions.
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            with torch.cuda.graph(g):
                advance()
        return g, ""
    except Exception as e:  # noqa: BLE001 - fall back to eager, do not hide it
        return None, f"{type(e).__name__}: {e}"
    finally:
        for b, s0 in zip(state, saved):
            b.copy_(s0)
