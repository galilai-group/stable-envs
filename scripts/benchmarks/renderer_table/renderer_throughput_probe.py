"""Same-GPU warmed latency/throughput comparison, separate from quality corpus."""

import argparse
import json
from pathlib import Path

import torch

from .collect import candidate
from .common import catalog, load_state, profile, state_hash, timed, write


def main():
    p = argparse.ArgumentParser()
    p.add_argument("campaign", type=Path)
    p.add_argument("name", choices=["OGBCube", "OGBScene"])
    p.add_argument("output", type=Path)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    row = next(r for r in catalog().values() if r["name"] == a.name)
    env = candidate(row, "cuda", 17)
    corpus = a.campaign / "run-6751001_3/results" / a.name / "cpu"
    meta = json.loads((corpus / "result.json").read_text())
    inputs = [
        env.to_render_state(load_state(corpus / f["state_file"], "cuda", env.get_state())) for f in meta["frames"][:6]
    ]
    result = dict(
        status="RUNNING",
        name=a.name,
        profile=profile(),
        measurements=[],
        scope="Six real same-configuration poses repeated across batch sizes; throughput test, not additional independent quality frames",
    )
    for batch in (1, 8, 64, 256):
        rs = torch.cat([inputs[i % len(inputs)] for i in range(batch)], dim=0)
        before = state_hash(rs)
        for backend in ("mjwarp", "opengl"):
            timing, image = timed(lambda: env.render(rs, backend=backend), gpu=True, samples=10, warmups=3)
            assert image.shape[0] == batch and state_hash(rs) == before
            stats = timing["statistics_ms"]
            result["measurements"].append(
                dict(
                    batch=batch,
                    backend=backend,
                    timing=timing,
                    mean_ms_per_frame=stats["mean"] / batch,
                    frames_per_second=1000 * batch / stats["mean"],
                )
            )
            write(a.output / "result.json", result)
            print(a.name, backend, batch, stats["mean"], flush=True)
    result["status"] = "COMPLETE"
    write(a.output / "result.json", result)


if __name__ == "__main__":
    main()
