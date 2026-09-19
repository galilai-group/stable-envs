"""Do our frames and the official ones look the same to a pretrained vision encoder?

Pixel parity is a claim about bytes. Table~\\ref{tab:lewm} answers the strongest version of the
consequence question for ONE env, by running a pretrained world model's own evaluation on our pixels.
This is the cheap, wide version of the same question: freeze a DINOv2 encoder, embed both renders of
the SAME state, and measure the distance in feature space across every family whose reference renderer
is installed.

WHY THE CONTROLS ARE THE POINT. "Cosine distance 0.001" means nothing on its own -- a metric that
cannot move produces a small number for a broken renderer too. Every env is therefore measured against
three distances, on identical states and one encoder:

    ours vs official        the claim. Should be ~0.
    official_i vs official_j  the SCALE. Two different states of the same env, through the SAME
                            renderer: this is what "a real difference" costs in this feature space,
                            and the claim is only meaningful as a fraction of it.
    blank vs official       the broken-renderer floor, the analogue of the blank-frame arm that
                            costs the LeWM checkpoint 92 points.

Reported as a RATIO of the first to the second, because the absolute number is a property of the
encoder and the ratio is a property of the renderer.

    python examples/dino_agreement.py --out examples/results/dino
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import stable_envs as se
from stable_envs.testing import conformance as C

#: Seeded and constant. The comparison is only reproducible if both arms drew the same states, and a
#: bare ``sample_states(n)`` consumes the env's own rng, which depends on how many draws came before.
STATE_SEED = 20260813

#: DINOv2 ViT-B/14 through timm, from the local HF cache. ViT-B rather than S because the question is
#: whether a STRONG encoder can separate the two renders; a weaker one failing to is weaker evidence.
DEFAULT_MODEL = "vit_base_patch14_dinov2.lvd142m"


def _encoder(name: str, device: str):
    import timm

    m = timm.create_model(name, pretrained=True, num_classes=0).eval().to(device)
    cfg = timm.data.resolve_model_data_config(m)
    mean = torch.tensor(cfg["mean"], device=device).view(1, 3, 1, 1)
    std = torch.tensor(cfg["std"], device=device).view(1, 3, 1, 1)
    size = cfg["input_size"][-1]
    return m, mean, std, size


@torch.no_grad()
def _embed(m, mean, std, size, frames: np.ndarray, device: str, chunk: int = 32) -> torch.Tensor:
    """``(N,H,W,3)`` uint8 -> L2-normalised ``(N,D)`` features.

    Resized to the encoder's own input size with bilinear+antialias. Both arms go through the IDENTICAL
    transform, so any resampling the resize introduces is common-mode and cannot flatter either one.
    """
    out = []
    for i in range(0, len(frames), chunk):
        x = torch.as_tensor(frames[i:i + chunk], device=device).permute(0, 3, 1, 2).float() / 255.0
        if x.shape[-1] != size or x.shape[-2] != size:
            x = torch.nn.functional.interpolate(x, size=(size, size), mode="bilinear",
                                                align_corners=False, antialias=True)
        f = m((x - mean) / std)
        out.append(torch.nn.functional.normalize(f, dim=-1))
    return torch.cat(out)


def _cos_dist(a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
    return (1.0 - (a * b).sum(-1)).clamp_min(0).cpu().numpy()


def _jpeg_roundtrip(frames: np.ndarray, quality: int = 95) -> np.ndarray:
    """The frames a stored corpus would have held: JPEG q95 encode + decode of the OFFICIAL render.

    THE CONTROL THAT DECIDES THE QUESTION. "Our frames differ from the reference by cosine 0.013" is
    only bad if the alternative is 0. It is not: the alternative is a stored image dataset, and storing
    is lossy. So the honest comparison is not ours-vs-perfect but ours-vs-what-you-would-have-had, and
    q95 is the generous end of what people actually store (the LeWM PushT corpus this repo measures is
    worse -- its stored frames match the live renderer on 91.9%% of interior pixels while ours match on
    100%%, and they carry 1,582 distinct colours against the renderer's 146).
    """
    import io

    from PIL import Image

    out = np.empty_like(frames)
    for i, f in enumerate(frames):
        buf = io.BytesIO()
        Image.fromarray(f).save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        out[i] = np.asarray(Image.open(buf).convert("RGB"))
    return out


def measure(name: str, n: int, model, device: str) -> dict:
    m, mean, std, size = model
    cls = se.registry.REGISTRY[name]
    env = se.make(name, num_envs=n, device=device)
    ref = C.reference_env(cls, env.size)                    # cached: never bind an EGL context locally

    states = env.sample_states(n, rng=np.random.default_rng(STATE_SEED))
    official = cls.reference_render(ref, states)
    ours = env.render(states).cpu().numpy()
    if ours.shape != official.shape:
        return {"env": name, "error": f"shape {ours.shape} vs official {official.shape}"}

    px = C.pixel_agreement(ours, official)
    blank = np.zeros_like(official)
    stored = _jpeg_roundtrip(official)

    f_ours, f_off, f_blank, f_jpg = (_embed(m, mean, std, size, a, device)
                                     for a in (ours, official, blank, stored))
    # The scale control: official_i against official_j, i != j, one fixed derangement (roll by 1) so
    # the pairing is deterministic and every frame is used exactly once on each side.
    roll = torch.roll(f_off, 1, dims=0)

    d_claim = _cos_dist(f_ours, f_off)
    d_scale = _cos_dist(f_off, roll)
    d_blank = _cos_dist(f_blank, f_off)
    d_jpeg = _cos_dist(f_jpg, f_off)
    o16, f16, j16 = ours.astype(np.int32), official.astype(np.int32), stored.astype(np.int32)
    mse_claim = float(((o16 - f16) ** 2).mean())
    mse_scale = float(((f16 - np.roll(f16, 1, axis=0)) ** 2).mean())
    mse_jpeg = float(((j16 - f16) ** 2).mean())

    return {
        "env": name, "n": n, "size": int(env.size), "tier": env.fidelity_tier,
        "reference_id": cls.reference_id,
        "interior_exact_frac": px.get("interior_exact_frac"),
        "all_exact_frac": px.get("exact_frac"), "max_abs": px.get("max_abs"),
        "pixel_mse_ours_vs_official": mse_claim,
        "pixel_mse_state_vs_state": mse_scale,
        "dino_cos_ours_vs_official": float(d_claim.mean()),
        "dino_cos_ours_vs_official_max": float(d_claim.max()),
        "dino_cos_state_vs_state": float(d_scale.mean()),
        "dino_cos_blank_vs_official": float(d_blank.mean()),
        "dino_cos_jpeg95_vs_official": float(d_jpeg.mean()),
        "pixel_mse_jpeg95_vs_official": mse_jpeg,
        # THE headline comparison: <1 means regenerating from the state lands CLOSER to the official
        # renderer than storing its output as JPEG and reading it back.
        "ratio_ours_over_jpeg95": float(d_claim.mean() / max(d_jpeg.mean(), 1e-12)),
        # The headline: the claim as a FRACTION of what a genuinely different state costs. Small means
        # the encoder separates two states far more easily than it separates two renders of one state.
        "ratio_claim_over_scale": float(d_claim.mean() / max(d_scale.mean(), 1e-12)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--envs", default="")
    ap.add_argument("--out", default="examples/results/dino")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _encoder(a.model, device)
    wanted = [n for n in sorted(se.registry.REGISTRY)
              if (not a.envs or n in a.envs.split(","))
              and getattr(se.registry.REGISTRY[n], "reference_id", None)
              and hasattr(se.registry.REGISTRY[n], "reference_env")]
    print(f"{len(wanted)} env(s) declare a reference renderer; encoder {a.model} on {device}")

    outdir = Path(a.out); outdir.mkdir(parents=True, exist_ok=True)
    res = {"model": a.model, "n": a.n, "state_seed": STATE_SEED,
           "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu", "envs": {}}
    for name in wanted:
        t0 = time.perf_counter()
        try:
            r = measure(name, a.n, model, device)
        except ImportError as e:
            r = {"env": name, "skipped": f"reference not installed: {e}"}
        except Exception as e:                                              # noqa: BLE001
            r = {"env": name, "error": f"{type(e).__name__}: {e}"}
        r["seconds"] = round(time.perf_counter() - t0, 2)
        res["envs"][name] = r
        if "dino_cos_ours_vs_official" in r:
            print(f"[{name}] ours {r['dino_cos_ours_vs_official']:.6f} | jpeg95 "
                  f"{r['dino_cos_jpeg95_vs_official']:.6f} | ours/jpeg "
                  f"{r['ratio_ours_over_jpeg95']:.3f}x | state-scale "
                  f"{r['dino_cos_state_vs_state']:.4f} | blank "
                  f"{r['dino_cos_blank_vs_official']:.4f}", flush=True)
        else:
            print(f"[{name}] {r.get('skipped') or r.get('error')}", flush=True)
        (outdir / "dino_agreement.json").write_text(json.dumps(res, indent=2))

    ok = [r for r in res["envs"].values() if "dino_cos_ours_vs_official" in r]
    if ok:
        claim = np.array([r["dino_cos_ours_vs_official"] for r in ok])
        scale = np.array([r["dino_cos_state_vs_state"] for r in ok])
        blank = np.array([r["dino_cos_blank_vs_official"] for r in ok])
        mse = np.array([r["pixel_mse_ours_vs_official"] for r in ok])
        res["summary"] = {
            "n_envs": len(ok),
            "cos_ours_vs_official_mean": float(claim.mean()),
            "cos_ours_vs_official_median": float(np.median(claim)),
            "cos_ours_vs_official_max": float(claim.max()),
            "cos_state_vs_state_median": float(np.median(scale)),
            "cos_blank_vs_official_median": float(np.median(blank)),
            "ratio_median": float(np.median(claim / np.maximum(scale, 1e-12))),
            "pixel_mse_median": float(np.median(mse)),
            "n_exact_zero_mse": int((mse == 0).sum()),
            "cos_jpeg95_vs_official_median": float(np.median(
                [r["dino_cos_jpeg95_vs_official"] for r in ok])),
            "ratio_ours_over_jpeg95_median": float(np.median(
                [r["ratio_ours_over_jpeg95"] for r in ok])),
            "n_closer_than_jpeg95": int(sum(r["ratio_ours_over_jpeg95"] < 1.0 for r in ok)),
        }
        print("\n" + json.dumps(res["summary"], indent=2))
    (outdir / "dino_agreement.json").write_text(json.dumps(res, indent=2))
    print(f"\nwrote {outdir / 'dino_agreement.json'}")


if __name__ == "__main__":
    main()
