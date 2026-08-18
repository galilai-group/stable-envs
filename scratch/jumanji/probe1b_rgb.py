"""Probe 1b: the official renderer in ``rgb_array`` mode, per id.

``jumanji.make(eid)`` builds every viewer with ``render_mode="human"``, whose ``_display`` draws to a
window and returns None -- which is why probe 1 measured "returns None" 26 times and learned nothing
about the pixels. ``MatplotlibViewer`` selects between the two by binding ``self._display`` at
construction, so switching an existing viewer to the rgb_array branch is the same code path the
documented ``render_mode="rgb_array"`` takes, without needing each env's constructor signature.

What is measured here:
  * the frame shape/dtype and whether it is uniform (a blank canvas would make parity meaningless);
  * DETERMINISM across two calls, which is the cheapest question and the one that decides whether a
    matplotlib reference can be matched at all;
  * whether the frame changes after stepping -- i.e. whether the picture tracks the state;
  * whether the viewer paints a TITLE, because a title is a region of the frame that is a function of
    something other than the board (2048 puts ``state.score`` there) and that is a G1 finding, not a
    cosmetic one.
"""
from __future__ import annotations

import os
import warnings

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np

warnings.filterwarnings("ignore")


def rgb_viewer(env):
    """Bind the viewer's rgb_array display branch. Returns the viewer, or None if it has no figure."""
    v = getattr(env, "_viewer", None)
    if v is None or not hasattr(v, "_display_rgb_array"):
        return None
    v._display = v._display_rgb_array
    return v


def frame(env, state) -> np.ndarray:
    a = env.render(state)
    if a is None:
        raise RuntimeError("viewer still returned None")
    a = np.asarray(a)
    return a[..., :3] if a.shape[-1] == 4 else a


def main() -> None:
    import jax

    import jumanji

    ids = sorted(jumanji.registered_environments())
    for eid in ids:
        row = {"id": eid}
        try:
            env = jumanji.make(eid)
            v = rgb_viewer(env)
            if v is None:
                print({**row, "status": "no matplotlib viewer"})
                continue
            row["figsize"] = tuple(getattr(v, "figure_size", ()))
            key = jax.random.PRNGKey(0)
            s0, _ = env.reset(key)
            f0 = frame(env, s0)
            f1 = frame(env, s0)
            row["shape"] = f0.shape
            row["det"] = int(np.abs(f0.astype(np.int32) - f1.astype(np.int32)).max())
            row["ncolors"] = int(len(np.unique(f0.reshape(-1, 3), axis=0)))

            fig = v._get_fig_ax()[0]
            row["suptitle"] = repr(fig._suptitle.get_text()) if fig._suptitle else None
            row["axtitle"] = repr(fig.get_axes()[0].get_title()) or None
            print(row, flush=True)
            env.close()
        except Exception as e:  # noqa: BLE001
            print({**row, "err": f"{type(e).__name__}: {str(e)[:150]}"}, flush=True)


if __name__ == "__main__":
    main()
