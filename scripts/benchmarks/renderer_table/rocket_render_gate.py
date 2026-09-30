"""Rocket renderer batch invariance and same-GPU performance gate."""
import argparse
from pathlib import Path

import torch

from .collect import candidate
from .common import catalog, profile, state_hash, timed, write


def main():
    p = argparse.ArgumentParser()
    p.add_argument("output", type=Path)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    env = candidate(catalog()["swm/PFRocketLanding-v0"], "cuda", 17)
    try:
        full = env.get_state()
        before = state_hash(full)
        state = env.to_render_state(full)
        state = state.expand(3).clone()
        state["renderer", "rgba"][1, 0, :3] = torch.tensor([.1, .7, .3], device="cuda")
        state["renderer", "source_pose"][2, 1, 0] += .2
        saved = state_hash(state)
        images = env.render(state, chunk_size=2)
        for i in range(3):
            assert torch.equal(images[i], env.render(state[i:i+1])[0])
        expanded = torch.cat([state[i % 3:i % 3 + 1] for i in range(33)])
        actual = env.render(expanded, chunk_size=7)
        assert torch.equal(actual, images[torch.arange(33, device="cuda") % 3])
        out = torch.empty_like(images)
        assert env.render(state, out=out).data_ptr() == out.data_ptr() and torch.equal(out, images)
        bad = state.clone()
        bad["renderer", "camera_pose"][2, 0] = float("nan")
        try:
            env.render(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid supplied state was accepted")
        assert state_hash(state) == saved and state_hash(env.get_state()) == before
        result = dict(status="RUNNING", profile=profile(), batch_invariance="EXACT", measurements=[])
        keys = [("renderer", k) for k in ("pose", "style", "size", "supersample")]
        for batch in (1, 8, 64, 256):
            rows = env.to_render_state(full).expand(batch).clone()
            for backend in ("legacy", "corrected"):
                supplied = rows.select(*keys) if backend == "legacy" else rows
                timing, _ = timed(lambda: env.render(supplied), gpu=True, samples=10, warmups=3)
                result["measurements"].append(dict(batch=batch, backend=backend, timing=timing,
                    mean_ms_per_frame=timing["statistics_ms"]["mean"] / batch))
                write(a.output / "result.json", result)
        result["status"] = "PASS"
        write(a.output / "result.json", result)
    finally:
        env.close()


if __name__ == "__main__":
    main()
