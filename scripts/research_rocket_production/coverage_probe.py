"""Diagnostic float-coverage reference; not a production renderer or timing claim."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import stable_envs as se
from stable_envs.envs import rocket_render


def coverage_reference(rast, raw_pos, faces, *, exhaustive=False, y_offset=0):
    b, h, w, _ = rast.shape
    ids = rast[..., 3].long()
    padded = torch.nn.functional.pad(ids, (1, 1, 1, 1))
    candidates = torch.stack([padded[:, dy : dy + h, dx : dx + w] for dy in range(3) for dx in range(3)], -1)
    edge = (candidates != ids[..., None]).any(-1)
    if exhaustive:
        candidates = torch.arange(1, len(faces) + 1, device=rast.device).expand(b, h, w, -1)
    face = faces[(candidates - 1).clamp_min(0)].long()
    batch = torch.arange(b, device=rast.device)[:, None, None, None, None]
    v = raw_pos[batch, face]
    # TinyRenderer computes the float32 viewport multiply before division.
    screen = ((v[..., :2] * (w / 2)) + (v[..., 3:4] * (w / 2))) / v[..., 3:4]
    xy = screen.double()
    x0, y0 = xy[..., 0, 0], xy[..., 0, 1]
    x1, y1 = xy[..., 1, 0], xy[..., 1, 1]
    x2, y2 = xy[..., 2, 0], xy[..., 2, 1]
    py, px = torch.meshgrid(
        torch.arange(y_offset, y_offset + h, device=rast.device, dtype=torch.float64),
        torch.arange(w, device=rast.device, dtype=torch.float64),
        indexing="ij",
    )
    px, py = px[None, ..., None], py[None, ..., None]
    det = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    a = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) / det
    c = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) / det
    d = 1 - a - c
    valid = (candidates > 0) & (det.abs() > 1e-12) & (a >= 0) & (c >= 0) & (d >= 0) & (v[..., 3] > 0).all(-1)
    bary = torch.stack([a, c, d], -1)
    perspective = bary / v[..., 3].double()
    perspective = perspective / perspective.sum(-1, keepdim=True)
    depth = (perspective * v[..., 2].double()).sum(-1)
    ndc = (bary * (v[..., 2] / v[..., 3]).double()).sum(-1)
    valid = valid & (ndc >= -1) & (ndc <= 1)
    best = torch.where(valid, depth, torch.inf).argmin(-1)
    selected = torch.gather(candidates, -1, best[..., None])[..., 0]
    hit = torch.gather(valid, -1, best[..., None])[..., 0]
    pick = lambda x: torch.gather(x, -1, best[..., None])[..., 0]
    result = torch.stack([pick(perspective[..., 0]), pick(perspective[..., 1]), pick(ndc), selected], -1).float()
    result = torch.where(hit[..., None], result, torch.zeros_like(result))
    # Only boundary pixels can differ in coverage; keep interiors byte-identical.
    return result.contiguous() if exhaustive else torch.where(edge[..., None], result, rast).contiguous()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    import nvdiffrast.torch as dr

    raw = np.load(args.input, allow_pickle=False)
    arrays = {"source": raw["source"]}
    report = dict(job=os.environ.get("SLURM_JOB_ID"), trials=[])
    with se.make(
        "PyFlytRocketLanding",
        num_envs=2,
        device="cuda",
        size=96,
        simulator_config=dict(render_mode="rgb_array", render_resolution=(96, 96)),
    ) as env:
        renderer, _ = rocket_render._gpu_context(env, 96, 1)
        clip = renderer.scene.clip
        rasterize = dr.rasterize
        positions = []

        def capture(*a, **kw):
            result = clip(*a, **kw)
            positions[:] = [result[0]]
            return result

        def corrected(*a, **kw):
            rast, db = rasterize(*a, **kw)
            from triangle_raster import rasterize as exact_rasterize

            return exact_rasterize(positions[0], renderer.scene.face, rast.shape[1]), db

        for label in ("baseline", "triangle_coverage"):
            frames = []
            if label == "triangle_coverage":
                renderer.scene.clip = capture
                dr.rasterize = corrected
            try:
                for t in range(8):
                    prefix = f"state/{t}/"
                    state = TensorDict(
                        {
                            tuple(k[len(prefix) :].split("/")): torch.from_numpy(raw[k])
                            for k in raw.files
                            if k.startswith(prefix)
                        },
                        [2],
                    )
                    frames.append(env.render(state).cpu().numpy())
            finally:
                renderer.scene.clip = clip
                dr.rasterize = rasterize
            data = np.stack(frames, axis=1)
            arrays[label] = data
            report["trials"].append(
                dict(
                    kind=label,
                    mse=np.mean(((data.astype(float) - raw["source"]) / 255) ** 2, axis=(-3, -2, -1)).tolist(),
                )
            )
    np.savez_compressed(args.output / "witness.npz", **arrays)
    report.update(
        status="DIAGNOSTIC_COMPLETE",
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
        gpu=torch.cuda.get_device_name(0),
    )
    (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
