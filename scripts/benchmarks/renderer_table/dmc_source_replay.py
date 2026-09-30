"""Pinned DMC renderer at exact saved State using the accepted oracle adapter."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from OpenGL import GL

from scripts.benchmarks.renderer_table.bridges import Bridge
from scripts.benchmarks.renderer_table.common import catalog, load_state, profile, rgb, state_hash, timed, write


def main():
    p = argparse.ArgumentParser()
    p.add_argument("registration")
    p.add_argument("corpus", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    row = catalog()[a.registration]
    assert row["name"] == "DMCQuadrupedWalk"
    corpus = json.loads((a.corpus / "result.json").read_text())
    result = dict(
        registration=a.registration,
        role="source_gpu",
        backend="upstream-opengl",
        profile=profile(),
        job_id=os.environ.get("SLURM_JOB_ID"),
        status="RUNNING",
        frames=[],
        corpus_sha256=hashlib.sha256((a.corpus / "result.json").read_bytes()).hexdigest(),
        method="Accepted oracle set_state; exact full-State recapture and render nonmutation",
    )
    output = dict(corpus, status="RUNNING", frames=[])
    write(a.output / "campaign.json", result)
    active = None
    bridges = []
    for f in corpus["frames"]:
        if f["case"] != active:
            recipe = corpus["recipes"][f["case"]]
            bridge = Bridge(row, f["seed"], recipe["reset_options"])
            bridges.append(bridge)
            bridge.reset(f["seed"])
            active = f["case"]
        state = load_state(a.corpus / f["state_file"], "cpu", bridge.ref.get_state())
        bridge.ref.set_state(state)
        assert state_hash(bridge.ref.get_state()) == f["state_hash"]
        timing, image = timed(bridge.render, gpu=True, samples=1, warmups=1)
        assert state_hash(bridge.ref.get_state()) == f["state_hash"]
        if not result.get("gl_vendor"):
            # dm_control uses a render thread; get its owning context explicitly.
            physics = bridge.source.env.physics

            def vendor():
                return GL.glGetString(GL.GL_VENDOR).decode(), GL.glGetString(GL.GL_RENDERER).decode()

            with physics.contexts.gl.make_current() as ctx:
                result["gl_vendor"], result["gl_renderer"] = ctx.call(vendor)
            assert "nvidia" in result["gl_vendor"].lower()
        image = rgb(image).copy()
        name = f"frame-{f['index']:04d}.npy"
        np.save(a.output / name, image, allow_pickle=False)
        output["frames"].append(
            dict(
                f,
                oracle_file=name,
                oracle_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                original_timing=timing,
                state_witness="EXACT_FULL_STATE_RECAPTURE",
            )
        )
        write(a.output / "result.json", output)
        print(a.registration, f["index"], flush=True)
    result["status"] = output["status"] = "COMPLETE"
    result["collected_frames"] = len(output["frames"])
    write(a.output / "result.json", output)
    write(a.output / "campaign.json", result)


if __name__ == "__main__":
    main()
