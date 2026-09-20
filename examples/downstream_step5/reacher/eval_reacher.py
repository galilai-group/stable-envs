"""Run ``stable_worldmodel``'s own Reacher evaluation with one of three renderers.

This is a wrapper, not a fork. ``scripts/plan/eval_wm.py`` keeps its config, its dataset, its
episode sampling, its ``StandardScaler``, its transform, its CEM solver and its success criterion;
this script composes that config with hydra, installs the renderer swap, calls the script's own
entry point, and intercepts ``World.evaluate`` on the way out to keep the PER-EPISODE outcomes that
the paired statistics need and that the script itself only prints in aggregate.

Everything that decides WHICH episodes run is inside ``eval_wm.run`` and is a pure function of
``cfg.seed`` and the dataset, so the three arms select the same episodes at the same start steps by
construction rather than by a check afterwards -- and the check is done anyway, in
``paired_gate.py``, because "by construction" is what everybody says before the arms diverge.

    python examples/downstream_step5/reacher/eval_reacher.py --arm official --seed 42 --n 50 \
        --policy reacher/prejepa --out tests/fixtures/regression/join/step5/reacher/runs
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

ARMS = ("official", "stable_envs", "blank", "official_regen")


def git_commit(path: Path) -> str:
    try:
        return (
            subprocess.run(
                ["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=30
            ).stdout.strip()
            or "unknown"
        )
    except Exception:
        return "unknown"


def versions() -> dict:
    import importlib.metadata as md

    out = {}
    for p in (
        "torch",
        "torchvision",
        "mujoco",
        "dm-control",
        "stable-worldmodel",
        "stable-pretraining",
        "numpy",
        "scikit-learn",
        "hydra-core",
    ):
        try:
            out[p] = md.version(p)
        except Exception:
            out[p] = "absent"
    return out


def checkpoint_identity(policy: str, cache: Path | None) -> dict:
    """Hash the weights actually loaded, so 'the same checkpoint' is checkable and not asserted."""
    import hashlib

    if policy == "random":
        return {"policy": "random"}
    root = cache or Path(os.environ.get("STABLEWM_HOME", Path.home() / ".stable_worldmodel"))
    d = root / "checkpoints" / policy
    if not d.is_dir():
        # A HuggingFace id caches under models--<user>--<repo>, not under the id's own path.
        hf = root / "checkpoints" / f"models--{policy.replace('/', '--')}"
        if hf.is_dir():
            d = hf
    files = sorted(d.glob("*.pt")) if d.is_dir() else ([d] if d.suffix == ".pt" else [])
    out = {"policy": policy, "dir": str(d), "files": {}}
    for f in files:
        h = hashlib.sha256()
        with f.open("rb") as fh:
            for chunk in iter(lambda: fh.read(2**24), b""):
                h.update(chunk)
        out["files"][f.name] = {"bytes": f.stat().st_size, "sha256": h.hexdigest()[:16]}
    cfg = d / "config.json"
    if cfg.is_file():
        out["config"] = json.loads(cfg.read_text())
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=ARMS, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=50, help="episodes in this run (one env each)")
    ap.add_argument("--policy", default="reacher/prejepa", help="checkpoint name under the swm cache")
    ap.add_argument(
        "--swm",
        type=Path,
        default=Path(os.environ.get("SWM_REPO", "")),
        help="stable-worldmodel checkout (its scripts/plan/config is the config tree)",
    )
    ap.add_argument("--size", type=int, default=224)
    # Per surface. The live env hides its target and the stored corpus draws it, so one policy for
    # both would make our frames disagree with whichever surface it was not chosen for.
    # ``hidden`` is Step 6's REPRESENTATION -- the env's own ``visibility.target`` style leaf, which
    # is what stable_worldmodel's ``render_target = 0`` actually does -- and ``offscreen`` is the
    # Step-5 mechanism, which moved the geometry out of the camera frustum instead. The DEFAULT stays
    # ``offscreen`` so ``tests/fixtures/regression/join/step5/reacher/`` reproduces exactly; the two are measured
    # BIT-IDENTICAL on this scene (probe_mode_equivalence.py: max |d| 0 over 16 frozen states at both
    # 84 and 224 px), so the choice cannot move a downstream number either way.
    ap.add_argument(
        "--target-mode",
        default="offscreen",
        choices=("offscreen", "hidden", "as_placed"),
        help="target placement for the LIVE observation",
    )
    # MEASURED, not assumed. corpus_probe.py compared the stored frames against both placements:
    # drawing the target puts max |d| at 217 and omitting it at 115, so the corpus does NOT contain
    # one -- the ~285 "red" pixels per frame that first suggested otherwise were the tan arm. Both
    # surfaces therefore hide it, which is also what the live environment does.
    ap.add_argument(
        "--dataset-target-mode",
        default="offscreen",
        choices=("offscreen", "hidden", "as_placed"),
        help="target placement when regenerating the stored start and goal frames",
    )
    # Three surfaces feed the model an image. Swapping them separately is what turned the PushT
    # null result into a diagnosis, and it is the first thing to reach for if this arm moves.
    ap.add_argument(
        "--surfaces",
        default="all",
        choices=("all", "live", "dataset"),
        help="which pixel sources to swap: the live observation, the stored corpus frames (start and goal), or both",
    )
    ap.add_argument("--device", default="cuda")
    # PyTorch leaves TF32 matmul OFF by default, so every 768-token projection in the planner runs
    # in full fp32 and uses none of the 3090's tensor cores. This is a backend flag, not a change
    # to the model or the search -- but it changes the arithmetic, so it is recorded per run and
    # validated on the official arm before any renderer comparison uses it.
    ap.add_argument(
        "--tf32", action="store_true", help="allow TF32 matmul/cudnn on Ampere (recorded in the run provenance)"
    )
    ap.add_argument(
        "--out", type=Path, default=ROOT / "tests" / "fixtures" / "regression" / "join" / "step5" / "reacher" / "runs"
    )
    ap.add_argument("--extra", nargs="*", default=[], help="extra hydra overrides, verbatim")
    args = ap.parse_args()

    if not args.swm or not (args.swm / "scripts" / "plan" / "eval_wm.py").is_file():
        raise SystemExit(f"--swm must point at a stable-worldmodel checkout; got {args.swm!r}")

    os.environ.setdefault("MUJOCO_GL", "egl")
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    import numpy as np
    import stable_worldmodel as swm
    import torch
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("[precision] TF32 matmul enabled", flush=True)

    # The renderer goes in BEFORE the world is built: the live swap replaces a method on
    # DMControlWrapper, and swm.World constructs its envs at __init__.
    renderer = None
    if args.arm != "official":
        import swap_renderer

        renderer = swap_renderer.install(
            mode={"blank": "blank", "official_regen": "official_regen"}.get(args.arm, "stable_envs"),
            size=args.size,
            device=args.device,
            target_mode=args.target_mode,
            dataset_target_mode=args.dataset_target_mode,
            swap_live=args.surfaces in ("all", "live"),
            swap_dataset=args.surfaces in ("all", "dataset"),
        )

    captured: dict = {}
    _orig_evaluate = swm.World.evaluate

    def evaluate(self, *a, **kw):
        # video=None: the mp4 writer is not part of the measurement and its imageio path is broken
        # in this environment. Nothing else about the call is touched.
        kw["video"] = None
        t0 = time.time()
        m = _orig_evaluate(self, *a, **kw)
        captured.update(
            {
                "metrics": m,
                "seconds": time.time() - t0,
                "episodes_idx": list(map(int, kw.get("episodes_idx") or [])),
                "start_steps": list(map(int, kw.get("start_steps") or [])),
                "goal_offset": kw.get("goal_offset"),
                "eval_budget": kw.get("eval_budget"),
            }
        )
        return m

    swm.World.evaluate = evaluate

    cfg_dir = str((args.swm / "scripts" / "plan" / "config").resolve())
    overrides = [f"seed={args.seed}", f"eval.num_eval={args.n}", f"policy={args.policy}", *args.extra]
    with initialize_config_dir(version_base=None, config_dir=cfg_dir):
        cfg = compose(config_name="reacher", overrides=overrides)

    sys.path.insert(0, str((args.swm / "scripts" / "plan").resolve()))
    import eval_wm

    print(f"[{args.arm}] seed={args.seed} n={args.n} policy={args.policy}", flush=True)
    eval_wm.run(cfg)

    if "metrics" not in captured:
        raise SystemExit("World.evaluate was never called; nothing was measured")

    m = captured["metrics"]
    succ = np.asarray(m["episode_successes"]).astype(int).tolist()
    rec = {
        "env": "swm/ReacherDMControl-v0",
        "task": cfg.world.task,
        "model_family": "prejepa" if "prejepa" in args.policy else args.policy,
        "arm": args.arm,
        "renderer_target_mode": args.target_mode if args.arm == "stable_envs" else None,
        "renderer_dataset_target_mode": (args.dataset_target_mode if args.arm == "stable_envs" else None),
        "renderer_surfaces": args.surfaces if args.arm != "official" else None,
        "tf32": bool(args.tf32),
        "seed": args.seed,
        "num_eval": args.n,
        "success_rate": float(m["success_rate"]),
        "episode_successes": succ,
        "episodes_idx": captured["episodes_idx"],
        "start_steps": captured["start_steps"],
        "goal_offset": captured["goal_offset"],
        "eval_budget": captured["eval_budget"],
        "eval_seconds": captured["seconds"],
        "render_calls": None if renderer is None else {"live": renderer.n_live, "dataset": renderer.n_dataset},
        "config": OmegaConf.to_container(cfg, resolve=True),
        "provenance": {
            "stable_envs_commit": git_commit(ROOT),
            "stable_worldmodel_commit": git_commit(args.swm),
            "checkpoint": checkpoint_identity(args.policy, None),
            "versions": versions(),
            "host": socket.gethostname(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "slurm_job": os.environ.get("SLURM_JOB_ID"),
            "mujoco_gl": os.environ.get("MUJOCO_GL"),
        },
    }
    args.out.mkdir(parents=True, exist_ok=True)
    tag = "" if args.surfaces == "all" else f"_{args.surfaces}"
    p = args.out / f"seed{args.seed}_n{args.n}_{args.arm}{tag}.json"
    p.write_text(json.dumps(rec, indent=2))
    print(
        f"[{args.arm}] {int(sum(succ))}/{len(succ)} = {rec['success_rate']:.1f}%  ({rec['eval_seconds']:.0f}s) -> {p}"
    )


if __name__ == "__main__":
    main()
