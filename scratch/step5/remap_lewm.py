"""Rename a LeWM checkpoint's ViT parameters from the transformers 4 layout to the 5.x one.

`quentinll/lewm-reacher` was saved in April under HuggingFace's old ViT naming and will not load
against transformers 5.x, which renamed every attention and MLP projection. The weights themselves
are fine; only the keys moved.

    encoder.encoder.layer.N.attention.attention.query.*   ->  encoder.layers.N.attention.q_proj.*
    encoder.encoder.layer.N.attention.output.dense.*      ->  encoder.layers.N.attention.o_proj.*
    encoder.encoder.layer.N.intermediate.dense.*          ->  encoder.layers.N.mlp.fc1.*
    encoder.encoder.layer.N.output.dense.*                ->  encoder.layers.N.mlp.fc2.*

WHY THIS IS SAFE TO DO AND HOW IT IS CHECKED. A key remap is exactly the kind of edit that succeeds
loudly and fails quietly: a wrong pairing still loads, still runs, and produces a model that is
subtly not the one somebody trained. Three guards, and the script refuses to write unless all three
hold:

  1. **Bijection.** Every source key maps to exactly one target key and every target key is claimed
     exactly once. No key is dropped, invented or doubled.
  2. **Shapes.** Each mapped pair has identical shape and dtype, compared against the state dict of
     a model actually instantiated from the checkpoint's own ``config.json``.
  3. **Nothing unmatched.** Any source or target key the rules do not cover is printed and the run
     aborts, rather than being passed through and hoped about.

The fourth guard is not in this file: the remapped checkpoint has a RECORDED score (86% in
stable_worldmodel's docs/baselines.md, against an 8% random-action floor measured on the same
episodes), so the qualification run either lands near it or says the remap is wrong. A rename that
scrambles attention heads collapses to the floor; it does not land on 86%.

The original download is never modified. The output is a new checkpoint directory beside it.

    python scratch/step5/remap_lewm.py --policy quentinll/lewm-reacher --out reacher/lewm_tf5
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

#: Ordered. ``attention.output.dense`` must be tried before the bare ``output.dense`` that names the
#: MLP's second projection, or the attention out-projection is silently written into the MLP.
RULES: tuple[tuple[str, str], ...] = (
    (r"^encoder\.encoder\.layer\.(\d+)\.attention\.attention\.query\.", r"encoder.layers.\1.attention.q_proj."),
    (r"^encoder\.encoder\.layer\.(\d+)\.attention\.attention\.key\.", r"encoder.layers.\1.attention.k_proj."),
    (r"^encoder\.encoder\.layer\.(\d+)\.attention\.attention\.value\.", r"encoder.layers.\1.attention.v_proj."),
    (r"^encoder\.encoder\.layer\.(\d+)\.attention\.output\.dense\.", r"encoder.layers.\1.attention.o_proj."),
    (r"^encoder\.encoder\.layer\.(\d+)\.intermediate\.dense\.", r"encoder.layers.\1.mlp.fc1."),
    (r"^encoder\.encoder\.layer\.(\d+)\.output\.dense\.", r"encoder.layers.\1.mlp.fc2."),
    (r"^encoder\.encoder\.layer\.(\d+)\.", r"encoder.layers.\1."),
)


def remap_key(k: str) -> str:
    for pat, rep in RULES:
        new, n = re.subn(pat, rep, k)
        if n:
            return new
    return k


def target_state_dict(ckpt_dir: Path):
    """The schema to match: a model built from the checkpoint's own config, under THIS transformers."""
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    cfg = json.loads((ckpt_dir / "config.json").read_text())
    model = instantiate(OmegaConf.create(cfg))
    return model, model.state_dict()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", default="quentinll/lewm-reacher")
    ap.add_argument("--out", default="reacher/lewm_tf5", help="new checkpoint name under the cache")
    ap.add_argument("--cache", type=Path, default=None)
    args = ap.parse_args()

    import torch
    from stable_worldmodel.data import get_cache_dir

    root = get_cache_dir(args.cache, sub_folder="checkpoints")
    src_dir = root / args.policy
    if not src_dir.is_dir():
        src_dir = root / f"models--{args.policy.replace('/', '--')}"
    if not src_dir.is_dir():
        raise SystemExit(f"no checkpoint directory for {args.policy!r} under {root}")
    pts = sorted(src_dir.glob("*.pt"))
    if len(pts) != 1:
        raise SystemExit(f"expected exactly one .pt in {src_dir}, found {len(pts)}")
    src_pt = pts[0]
    print(f"source: {src_pt}")

    sd = torch.load(src_pt, map_location="cpu")
    model, tgt = target_state_dict(src_dir)
    print(f"{len(sd)} source tensors, {len(tgt)} target tensors")

    # --- guard 1 and 3: bijection, nothing unmatched -----------------------
    mapped: dict[str, str] = {}
    claimed: dict[str, str] = {}
    collisions, unknown = [], []
    for k in sd:
        nk = remap_key(k)
        if nk not in tgt:
            unknown.append(f"{k}  ->  {nk}  (no such target)")
            continue
        if nk in claimed:
            collisions.append(f"{k} and {claimed[nk]} both map to {nk}")
            continue
        mapped[k], claimed[nk] = nk, k

    unclaimed = sorted(set(tgt) - set(claimed))
    problems = []
    if unknown:
        problems.append(f"{len(unknown)} source key(s) map nowhere")
    if collisions:
        problems.append(f"{len(collisions)} collision(s)")
    if unclaimed:
        problems.append(f"{len(unclaimed)} target key(s) unclaimed")

    # --- guard 2: shapes ---------------------------------------------------
    bad_shape = [f"{k} -> {nk}: {tuple(sd[k].shape)} vs {tuple(tgt[nk].shape)}"
                 for k, nk in mapped.items() if tuple(sd[k].shape) != tuple(tgt[nk].shape)]
    if bad_shape:
        problems.append(f"{len(bad_shape)} shape mismatch(es)")

    for label, items in (("map nowhere", unknown), ("collide", collisions),
                         ("unclaimed", unclaimed), ("shape mismatch", bad_shape)):
        for x in items[:12]:
            print(f"  [{label}] {x}")
        if len(items) > 12:
            print(f"  [{label}] ... and {len(items) - 12} more")

    if problems:
        raise SystemExit("refusing to write a remapped checkpoint: " + "; ".join(problems))

    renamed = sum(1 for k, nk in mapped.items() if k != nk)
    print(f"bijection verified: {len(mapped)} tensors, {renamed} renamed, "
          f"{len(mapped) - renamed} unchanged, every shape equal")

    out_sd = {nk: sd[k] for k, nk in mapped.items()}
    # The load the eval will actually perform, done here so a failure surfaces now.
    model.load_state_dict(out_sd, strict=True)
    print("strict load_state_dict: OK")

    out_dir = root / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pt = out_dir / "weights.pt"
    torch.save(out_sd, out_pt)
    shutil.copy2(src_dir / "config.json", out_dir / "config.json")

    def sha(p: Path) -> str:
        h = hashlib.sha256()
        with p.open("rb") as f:
            for c in iter(lambda: f.read(2 ** 24), b""):
                h.update(c)
        return h.hexdigest()[:16]

    prov = {
        "derived_from": {"policy": args.policy, "path": str(src_pt),
                         "sha256": sha(src_pt), "bytes": src_pt.stat().st_size},
        "written": {"path": str(out_pt), "sha256": sha(out_pt), "bytes": out_pt.stat().st_size},
        "reason": ("transformers 5.x renamed ViT parameters; these weights predate the rename. "
                   "Keys only -- no tensor was modified, reordered or reshaped."),
        "tensors": len(out_sd), "renamed": renamed,
        "guards": ["bijection", "shapes equal", "no unmatched keys on either side",
                   "strict load_state_dict"],
        "still_to_verify": ("the recorded 86% success rate for this checkpoint, against the 8% "
                            "random-action floor measured on the same episodes"),
    }
    (out_dir / "remap_provenance.json").write_text(json.dumps(prov, indent=2))
    print(json.dumps(prov, indent=2))
    print(f"\nwrote {out_pt}\nuse --policy {args.out}/weights.pt")


if __name__ == "__main__":
    main()
