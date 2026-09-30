"""Research-only two-layer alpha and moving-shadow composition ablation.

Does not change production defaults. Tests whether bounded extra MJWarp passes
can recover the missing effects while retaining batched GPU execution.
"""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from .collect import candidate
from .common import catalog, load_state, rgb, state_hash, timed, write
from .config_fixes import mse


def main():
    p = argparse.ArgumentParser()
    p.add_argument("campaign", type=Path)
    p.add_argument("name", choices=["OGBCube", "OGBScene"])
    p.add_argument("output", type=Path)
    p.add_argument("--ss", type=int, default=2)
    p.add_argument("--shadow-transmission", type=float)
    p.add_argument("--all-frames", action="store_true")
    p.add_argument("--throughput", action="store_true")
    p.add_argument("--compensate-shadow", action="store_true")
    p.add_argument("--corpus", type=Path)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    if a.shadow_transmission is not None:
        import importlib

        shader = importlib.import_module("mujoco_warp._src.render")
        assert shader.NO_LIGHT_AMBIENT_FALLBACK == 0.3
        # Process-local research ablation: opaque occluders block direct light.
        # Upstream currently reuses its no-lights ambient fallback for this.
        shader.NO_LIGHT_AMBIENT_FALLBACK = a.shadow_transmission
    import mujoco

    from stable_envs.envs.ogbench import manipulation as frozen
    from stable_envs.envs.ogbench.manipulation_renderer import _ConfiguredRenderer, configured_bundle

    row = next(r for r in catalog().values() if r["name"] == a.name)
    env = candidate(row, "cuda", 17)
    baseline = state_hash(env.get_state())
    corpus = a.campaign / "run-6751001_3/results" / a.name
    state_root = a.corpus or corpus / "cpu"
    image_root = a.corpus or corpus / "source_gpu"
    meta = json.loads((state_root / "result.json").read_text())
    result = dict(status="RUNNING", name=a.name, ss=a.ss, shadow_transmission=a.shadow_transmission, frames=[])
    static_cache = {}
    for f in meta["frames"]:
        if not a.all_frames and f["index"] not in (0, 1, 5, 36, 72, 108, 110, 114, 116, 120, 122, 128, 131):
            continue
        state = load_state(state_root / f["state_file"], "cuda", env.get_state())
        assert state_hash(state) == f["state_hash"]
        rs = env.to_render_state(state)
        size = int(rs["renderer", "size"][0, 0])
        res = size * a.ss
        token, b = configured_bundle(a.name[3:], rs[0])
        alpha = frozen.MW.effective_alpha(b.model)
        transparent = np.flatnonzero((alpha > 0) & (alpha < 1))
        for suffix, hidden in [("opaque", transparent), ("static", np.union1d(transparent, b.movers))]:
            key = token + "/fast-probe/" + suffix
            if key not in frozen._BUNDLE_CACHE:
                alternate = copy.copy(b)
                alternate.unwrapped = copy.copy(b.unwrapped)
                alternate.unwrapped._model = copy.deepcopy(b.model)
                alternate.unwrapped._data = mujoco.MjData(alternate.model)
                alternate.model.geom_group[hidden] = frozen.MW.HIDDEN_GROUP
                alternate.pristine = {k: v.copy() for k, v in b.pristine.items()}
                alternate.pristine["geom_group"] = alternate.model.geom_group.copy()
                frozen._BUNDLE_CACHE[key] = alternate

        def draw(which):
            key = token if which == "original" else token + "/fast-probe/" + which
            bundle = frozen._BUNDLE_CACHE[key]
            renderer = _ConfiguredRenderer(key, bundle, "cuda", res, 1)
            try:
                composite = renderer.draw(rs).clone()
                nw = 1 if len(rs) == 1 else 1 << (len(rs) - 1).bit_length()
                context = frozen._context(key, res, nw)
                return composite, context.bgra[: len(rs)][..., [2, 1, 0]].clone(), context.seg[: len(rs)].clone()
            finally:
                renderer.close()

        original, first_rgb, first_seg = draw("original")
        opaque, shadowed, opaque_seg = draw("opaque")
        static_key = (token, res)
        if static_key not in static_cache:
            static_cache[static_key] = draw("static")[1]
        reference = static_cache[static_key]
        ratio = (shadowed.float() / reference.float().clamp_min(1)).clamp(0, 1)
        if a.compensate_shadow:
            ratio = ((ratio - 0.3) / 0.7).clamp(0, 1)
        lut = torch.zeros(b.model.ngeom + 1, dtype=torch.bool, device="cuda")
        lut[torch.as_tensor(np.asarray(b.movers) + 1, device="cuda")] = True
        mover = lut[(opaque_seg + 1).long()]
        with_shadows = torch.where(mover[..., None], opaque.float(), opaque.float() * ratio)
        opacity = np.ones(b.model.ngeom + 1, np.float32)
        opacity[0] = 1
        for g in transparent:
            # Static transparency already exists in the source-rendered plate.
            if g in b.movers:
                opacity[g + 1] = alpha[g]
        opacity = torch.as_tensor(opacity, device="cuda")[(first_seg + 1).long()][..., None]
        composed = torch.where(opacity < 1, opacity * first_rgb.float() + (1 - opacity) * with_shadows, with_shadows)

        def down(image):
            return rgb(image.float().reshape(1, size, a.ss, size, a.ss, 3).mean((2, 4)).round().clamp(0, 255).byte())

        source = np.load(image_root / f["oracle_file"], allow_pickle=False)
        images = dict(
            source=source,
            original=down(original),
            alpha_only=down(
                torch.where(opacity < 1, opacity * first_rgb.float() + (1 - opacity) * opaque.float(), opaque.float())
            ),
            shadows_only=down(with_shadows),
            combined=down(composed),
        )
        np.savez_compressed(a.output / f"frame-{f['index']:04d}.npz", **images)
        assert state_hash(state) == f["state_hash"] and state_hash(env.get_state()) == baseline
        result["frames"].append(
            dict(
                index=f["index"],
                case=f["case"],
                seed=f["seed"],
                step=f["step"],
                state_hash=f["state_hash"],
                mse={k: mse(v, source) for k, v in images.items() if k != "source"},
            )
        )
        write(a.output / "result.json", result)
        print(a.name, f["index"], result["frames"][-1]["mse"], flush=True)
    if a.throughput:
        # Rendering cost lower bound: research composition lacks the final
        # public boundary's full validation and configuration grouping.
        single = rs.clone()
        result["throughput_scope"] = (
            "Last measured pose repeated in batch; research fast path excludes public validation/grouping, so is a cost lower bound. Quality frames remain unchanged."
        )
        result["throughput"] = []
        alpha_lut = torch.ones(b.model.ngeom + 1, device="cuda")
        for g in transparent:
            if g in b.movers:
                alpha_lut[g + 1] = float(alpha[g])

        def fast_draw(batch_state):
            nonlocal rs
            outputs = []
            for start in range(0, len(batch_state), 64):
                rs = batch_state[start : start + 64]
                _, first, first_seg = draw("original")
                opaque, shadowed, opaque_seg = draw("opaque")
                ratio = (shadowed.float() / reference.float().clamp_min(1)).clamp(0, 1)
                if a.compensate_shadow:
                    ratio = ((ratio - 0.3) / 0.7).clamp(0, 1)
                mover = lut[(opaque_seg + 1).long()]
                shadows = torch.where(mover[..., None], opaque.float(), opaque.float() * ratio)
                opacity = alpha_lut[(first_seg + 1).long()][..., None]
                image = torch.where(opacity < 1, opacity * first.float() + (1 - opacity) * shadows, shadows)
                outputs.append(
                    image.reshape(len(rs), size, a.ss, size, a.ss, 3).mean((2, 4)).round().clamp(0, 255).byte()
                )
            return torch.cat(outputs)

        for batch in (1, 8, 64, 256):
            batched = torch.cat([single] * batch)
            for backend in ("mjwarp", "fast_research"):
                timing, _ = timed(
                    lambda: (
                        fast_draw(batched) if backend == "fast_research" else env.render(batched, backend="mjwarp")
                    ),
                    gpu=True,
                    samples=10,
                    warmups=3,
                )
                result["throughput"].append(
                    dict(
                        batch=batch,
                        backend=backend,
                        timing=timing,
                        mean_ms_per_frame=timing["statistics_ms"]["mean"] / batch,
                    )
                )
                write(a.output / "result.json", result)
    result["status"] = "COMPLETE"
    write(a.output / "result.json", result)


if __name__ == "__main__":
    main()
