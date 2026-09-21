"""Do our regenerated frames match the frames the model was actually trained on?

The parity probe compared our renderer against the LIVE reference. That is the right check for the
live observation surface and the wrong one for the other two: the planner's first observation and
its goal image come from the STORED corpus, and the corpus is not the live renderer's current
output. Every sampled corpus frame carries ~285 saturated-red pixels -- a target disc -- while the
live env hides its target behind a zero alpha. The data was recorded before that default changed.

So this probe asks the question the other one could not:

  * ours, target where the episode put it   vs the stored frame
  * ours, target off camera                 vs the stored frame
  * the live env, as configured today       vs the stored frame
  * the live env with render_target forced on vs the stored frame

The third row measures how far the OFFICIAL arm's own live frames have drifted from its stored
ones -- a mismatch inside the baseline that has nothing to do with us. The fourth says whether
turning the knob back on reproduces the corpus, which is what would identify how it was recorded.

    python docs/research/artifacts/step5_downstream/reacher/corpus_probe.py --n 32 --out tests/fixtures/regression/join/step5/reacher
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())

PROBE_SEED = 8712
OFFSCREEN = (10.0, 10.0, 0.01)


def compare(ours: np.ndarray, ref: np.ndarray, C) -> dict:
    a = C.pixel_agreement(ours, ref)
    return {
        "exact_frac": float(a["exact_frac"]),
        "interior_exact_frac": (None if a["interior_exact_frac"] is None else float(a["interior_exact_frac"])),
        "interior_px": int(a["interior_px"]),
        "mean_abs": float(a["mean_abs"]),
        "max_abs": int(a["max_abs"]),
        "within8_frac": float(a["within8_frac"]),
        "ssim": float(C.ssim(ours, ref)),
    }


def reddish(frames: np.ndarray) -> float:
    f = frames.astype(np.int32)
    m = (f[..., 0] > 110) & (f[..., 0] - f[..., 1] > 40) & (f[..., 0] - f[..., 2] > 40)
    return float(m.reshape(len(f), -1).sum(1).mean())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--dataset", default="dmc/reacher_random.h5")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--task", default="qpos_match")
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--out", type=Path, default=ROOT / "tests" / "fixtures" / "regression" / "join" / "step5" / "reacher"
    )
    args = ap.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    import sys

    import mujoco
    import stable_worldmodel as swm
    import torch  # noqa: F401
    from stable_worldmodel.envs.dmcontrol.reacher import ReacherDMControlWrapper

    import stable_envs as se
    from stable_envs.testing import conformance as C

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from state_bridge import slot_names

    ds = swm.data.load_dataset(args.dataset, keys_to_cache=["action"])
    rng = np.random.default_rng(PROBE_SEED)
    rows = np.sort(rng.choice(ds.get_col_data("step_idx").shape[0], size=args.n, replace=False))
    data = ds.get_row_data(rows)

    def arr(k):
        v = data[k]
        return v.numpy() if hasattr(v, "numpy") else np.asarray(v)

    stored = arr("pixels")
    if stored.ndim == 4 and stored.shape[1] in (1, 3):
        stored = stored.transpose(0, 2, 3, 1)
    stored = stored.astype(np.uint8)
    qpos, qvel, tpos = arr("qpos"), arr("qvel"), arr("target_pos")
    print(f"{len(stored)} stored frames {stored.shape[1:]}, {reddish(stored):.0f} red px/frame")

    ours = se.make("DMCReacherHard", num_envs=1, size=args.size, device=args.device)
    slots = slot_names(ours)

    def render_ours(placement: str) -> np.ndarray:
        st = np.zeros((len(stored), ours.state_dim), np.float32)
        st[:, : ours.nq] = qpos[:, : ours.nq]
        for j, (_a, name, comp) in enumerate(slots):
            if placement == "offscreen":
                st[:, ours.nq + j] = OFFSCREEN[comp] if name == "target" else 0.0
            else:
                st[:, ours.nq + j] = tpos[:, comp]
        return ours.render(st).cpu().numpy()

    def render_official(render_target: int) -> np.ndarray:
        env = ReacherDMControlWrapper(task=args.task, seed=PROBE_SEED)
        opts = {"variation_values": {"rendering.render_target": render_target}}
        try:
            env.reset(seed=PROBE_SEED, options=opts)
        except Exception as e:
            print(f"  (could not force render_target={render_target}: {type(e).__name__} {e})")
            env.reset(seed=PROBE_SEED)
        p = env.env.physics
        m = p.model.ptr if hasattr(p.model, "ptr") else p.model
        tid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "target")
        out = []
        for i in range(len(stored)):
            m.geom_pos[tid, :2] = tpos[i]
            with p.reset_context():
                p.data.qpos[:] = qpos[i]
                p.data.qvel[:] = qvel[i]
            out.append(env.render(args.size, args.size).copy())
        return np.stack(out)

    res = {
        "dataset": args.dataset,
        "n": int(args.n),
        "size": args.size,
        "stored_red_px_per_frame": reddish(stored),
        "comparisons": {},
    }
    panels = {}
    for label, frames in (
        ("ours_as_placed", render_ours("as_placed")),
        ("ours_offscreen", render_ours("offscreen")),
        ("official_default", render_official(0)),
        ("official_target_on", render_official(1)),
    ):
        res["comparisons"][label] = compare(frames, stored, C)
        res["comparisons"][label]["red_px_per_frame"] = reddish(frames)
        panels[label] = frames
        r = res["comparisons"][label]
        print(
            f"{label:20s} vs stored: exact {r['exact_frac']:.4f}  mean|d| {r['mean_abs']:7.3f}  "
            f"ssim {r['ssim']:.4f}  red {r['red_px_per_frame']:6.0f}"
        )

    best = min(res["comparisons"], key=lambda k: res["comparisons"][k]["mean_abs"])
    res["closest_to_stored"] = best
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "corpus_probe.json").write_text(json.dumps(res, indent=2))
    try:
        import imageio.v2 as imageio

        k = min(3, len(stored))
        rows_img = [
            np.concatenate(
                [stored[i], panels["ours_as_placed"][i], panels["ours_offscreen"][i], panels["official_default"][i]],
                axis=1,
            )
            for i in range(k)
        ]
        imageio.imwrite(args.out / "corpus_probe.png", np.concatenate(rows_img, axis=0))
        print(f"wrote {args.out / 'corpus_probe.png'} (stored | ours as_placed | ours offscreen | official)")
    except Exception as e:
        print(f"(no sheet: {e})")
    print(f"closest to the stored frames: {best}")
    print(f"wrote {args.out / 'corpus_probe.json'}")


if __name__ == "__main__":
    main()
