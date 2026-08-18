"""Reproduce ``test_state_is_sufficient`` verbatim and find what makes it fail.

Constraints the answer has to satisfy, from what is already known:
  * it fails on pointmaze too, and pointmaze's state is (x, y) with no quaternion at all, so a
    non-unit free-joint quat cannot be the whole story;
  * one reference env driven through two histories fails just as two envs did, so it is not context
    count;
  * ``set_state`` demonstrably calls ``mj_forward`` and leaves ``geom_xpos`` equal to a fresh
    ``mj_kinematics``, so it is not a stale pose;
  * hand-chosen states through a bare ``_build_reference`` gave 0/255.

The remaining difference between the passing check and the failing test is everything that happens
AROUND the render: which states, and what the process did beforehand. So each is varied one at a time:

  A  bare env, hand-chosen states                    (known good, control)
  B  bare env, sample_states                         (does the sampler matter?)
  C  after a mujoco.Renderer has been created AND CLOSED on another model
  D  after the full stable-envs construction path (which bakes, and closes a Renderer per bake)
  E  the same states rendered twice in one pass      (is it even per-state?)

Case C is pointed: ``_bake`` still calls ``Renderer.close()``, and releasing an EGL context is exactly
what was already shown to corrupt other live renderers. If C fails and A/B pass, that is the answer.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

OUT = "/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/ogbench/out"
sys.path.insert(0, "/mnt/vast/home/randall.balestriero/code/stable-envs")


def test_body(cls, ref, states):
    """Exactly what tests/gpu/test_conformance.py::test_state_is_sufficient does."""
    fa = cls.reference_render(ref, states)
    fb = cls.reference_render(ref, states[::-1])[::-1]
    d = np.abs(fa.astype(np.int16) - fb.astype(np.int16)).max(-1)
    per = [int(d[i].max()) for i in range(len(states))]
    out = dict(max=int(d.max()), per_frame_max=per,
               n_frames_differ=int(sum(1 for x in per if x > 0)))
    # off-by-one signature: does reverse frame i match forward frame i-1 or i+1?
    if out["max"] > 0:
        shifts = {}
        for s in (-1, 1):
            k = [i for i in range(len(states)) if 0 <= i + s < len(states)]
            shifts[s] = int(max(np.abs(fb[i].astype(np.int16)
                                       - fa[i + s].astype(np.int16)).max() for i in k))
        out["reverse_vs_forward_shifted"] = shifts
    return out


def hand_states(u, n=16):
    m = u.model
    rng = np.random.default_rng(0)
    q = np.tile(np.asarray(m.qpos0, np.float64), (n, 1))
    q[:, 0] = rng.uniform(0, 12, n)
    q[:, 1] = rng.uniform(0, 12, n)
    return q.astype(np.float32)


def case(name, cls, states_fn, pre=None):
    from stable_envs.envs.ogbench import _build_reference
    if pre is not None:
        pre()
    ref = _build_reference(cls.LOCO, cls.MAZE, 224)
    st = states_fn(ref.unwrapped)
    r = test_body(cls, ref, st)
    r["case"] = name
    q = np.asarray(st, np.float64)
    if q.shape[1] > 7:
        r["quat_norms"] = dict(min=float(np.linalg.norm(q[:, 3:7], axis=1).min()),
                               max=float(np.linalg.norm(q[:, 3:7], axis=1).max()))
    # does the reference land exactly on the requested qpos?
    u = ref.unwrapped
    err = 0.0
    for s in q:
        u.set_state(s, np.zeros(u.model.nv))
        err = max(err, float(np.abs(u.data.qpos - s).max()))
    r["qpos_roundtrip_max_err"] = err
    return r


def open_and_close_a_renderer(loco, maze):
    """What ``_bake`` does: build a Renderer on a scene model and release its EGL context."""
    import mujoco
    from stable_envs.envs.ogbench import _build_reference
    e = _build_reference(loco, maze, 224)
    r = mujoco.Renderer(e.unwrapped.model, width=224, height=224)
    r.update_scene(e.unwrapped.data, camera=e.unwrapped.custom_camera)
    r.render()
    r.close()


def main():
    os.makedirs(OUT, exist_ok=True)
    log = open(os.path.join(OUT, "sufficiency_repro.log"), "w")
    import stable_envs as se
    for nm in ("PointmazeMediumNavigate", "AntmazeMediumNavigate"):
        cls = se.REGISTRY[nm]
        results = []
        try:
            results.append(case("A_bare_handstates", cls, lambda u: hand_states(u)))
        except Exception:
            import traceback
            log.write(f"FAIL A {nm}\n{traceback.format_exc()}\n")

        # B and D need an env instance to use its sampler
        try:
            e = se.make(nm, batch=16, device="cuda")
            results.append(case("D_after_stable_envs_construction", cls,
                                lambda u, e=e: e.sample_states(16)))
        except Exception:
            import traceback
            log.write(f"FAIL D {nm}\n{traceback.format_exc()}\n")
        try:
            results.append(case("C_after_renderer_close", cls, lambda u: hand_states(u),
                                pre=lambda: open_and_close_a_renderer(cls.LOCO, cls.MAZE)))
        except Exception:
            import traceback
            log.write(f"FAIL C {nm}\n{traceback.format_exc()}\n")
        for r in results:
            log.write(json.dumps(dict(env=nm, **r)) + "\n")
            log.flush()
    log.close()


if __name__ == "__main__":
    main()
