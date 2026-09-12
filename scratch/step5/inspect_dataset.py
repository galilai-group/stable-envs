"""What is actually in the Reacher corpus, and does its renderer draw the target?

Two questions the harness cannot proceed without.

**Columns.** ``_extract_init_goal`` hands the planner whatever the dataset stores, and the swap
re-renders the ``pixels`` column from ``qpos``. If ``qpos`` is not a column, a stored frame cannot
be regenerated from state and the two arms would be comparing a regenerated frame against a stored
one.

**The target.** ``stable_worldmodel``'s reacher hides its target by setting the material's alpha to
zero, and its variation default is ``render_target = 0``. Whether the CORPUS was recorded that way
is a separate fact, recoverable only from the stored pixels: if the frames show a target, the model
was trained on frames with one, and the eval must produce one too.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="dmc/reacher_random.h5")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--frames", type=int, default=8)
    args = ap.parse_args()

    import stable_worldmodel as swm

    ds = swm.data.load_dataset(args.dataset, keys_to_cache=["action"])
    cols = list(ds.column_names)
    print("columns:", cols)

    col = "episode_idx" if "episode_idx" in cols else "ep_idx"
    ep = ds.get_col_data(col)
    step = ds.get_col_data("step_idx")
    eps = np.unique(ep)
    lens = np.array([step[ep == e].max() + 1 for e in eps[:200]])
    info = {
        "dataset": args.dataset,
        "columns": cols,
        "rows": int(len(ep)),
        "episodes": int(len(eps)),
        "episode_len_first200": {"min": int(lens.min()), "max": int(lens.max()),
                                 "median": float(np.median(lens))},
    }
    # NOT `pixels`. get_dim reads the column to find its width, and this column is 92 GiB of
    # frames -- asking for its dimension OOM-killed a 96 GB allocation. The frame shape is read
    # from a handful of rows below instead.
    for c in cols:
        if c.startswith("pixels"):
            continue
        try:
            info[f"dim_{c}"] = int(ds.get_dim(c))
        except Exception:
            pass

    rows = ds.get_row_data(np.arange(args.frames))
    px = rows["pixels"]
    px = px.numpy() if hasattr(px, "numpy") else np.asarray(px)
    if px.ndim == 4 and px.shape[1] in (1, 3):
        px = px.transpose(0, 2, 3, 1)
    info["pixels_shape"] = list(px.shape)
    info["pixels_dtype"] = str(px.dtype)
    # A dm_control reacher target is a saturated red disc on a blue-grey floor; the arm is tan.
    # Counting strongly-red pixels is a cheap, explicit test for "is the target drawn", and the
    # contact sheet beside it is the one that settles an argument.
    f = px.astype(np.int32)
    reddish = ((f[..., 0] > 110) & (f[..., 0] - f[..., 1] > 40) & (f[..., 0] - f[..., 2] > 40))
    info["reddish_px_per_frame"] = [int(x) for x in reddish.reshape(len(f), -1).sum(1)]
    for c in ("qpos", "qvel", "action", "observation", "proprio", "state"):
        if c in cols:
            v = rows[c]
            v = v.numpy() if hasattr(v, "numpy") else np.asarray(v)
            info[f"sample_{c}"] = np.asarray(v[0]).reshape(-1).tolist()[:12]

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "dataset_inspect.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))
    try:
        import imageio.v2 as imageio
        imageio.imwrite(args.out / "dataset_frames.png",
                        np.concatenate(list(px[:min(6, len(px))].astype(np.uint8)), axis=1))
        print("wrote", args.out / "dataset_frames.png")
    except Exception as e:
        print("(no frame sheet:", e, ")")


if __name__ == "__main__":
    main()
