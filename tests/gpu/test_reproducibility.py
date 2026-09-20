"""Determinism, reproducibility and controllability — the release gate for "delete your frames".

``test_conformance.py`` asks whether our pixels are the right pixels: parity against the official
renderer, no dead state dimensions, the kernel agreeing with the torch reference. That is necessary and
it is not the promise. The promise is that a user can delete 46 GB of images, keep a few megabytes of
states, and get **those exact frames back** — next month, on another machine, from another process, out
of a loader that hands the rows over in a different order and a different batch size, after a
save/load, at whatever resolution they ask for. A renderer can pass every check in the other file and
break every one of those.

So each test here names one way the promise can fail while parity stays perfect:

  fresh process        a frame that depends on anything the process accumulated — an autotuner's
                       choice, a cached plate, an RNG someone advanced — is reproducible only until
                       you close the interpreter. This is the check a dataset consumer actually leans
                       on, and it is the only one that cannot be faked from inside one process.
  seeded API           ``sample_states(rng=)`` / ``sample_style(rng=)`` / ``sample(seed=)`` must give
                       the same numbers in a new process. A seed that does not reproduce is worse than
                       no seed: the caller believes they hold a handle they do not hold.
  batch size           1, 3, 7, 8, 64 and 1024 must agree, INCLUDING the odd ones. A kernel whose grid
                       is ``cdiv(H*W, BLOCK)`` is the classic case of correct at powers of two and
                       wrong at 7, and a loader's last batch is always the odd one. It is also where a
                       batched matmul lives: cuBLAS picks its kernel from the batch COUNT, so a 3x3
                       frame change written as ``einsum`` sums its three terms in a different order at
                       B=1 and at B=8 — which is exactly how the OGBench ray-caster used to flip a
                       silhouette pixel between hit and miss.
  chunking             ``render_states(chunk=c)`` is the dataset entry point; the chunk boundary is
                       where a stale ``out`` view or a mis-sliced style row lives.
  resolution           rendering at 2S and box-downsampling is NOT required to equal rendering at S,
                       and asserting that it is would be asserting a falsehood. What is asserted is a
                       MEASURED per-env bound, recorded below with the run that produced it.
  input form           numpy / torch / CPU / CUDA / float32 / float64 / non-contiguous. Every one is a
                       real call site and ``frames[:, t]`` has already crashed this repo once.
  purity               ``render`` must not mutate the env, and ``set_state(get_state())`` twice must be
                       the same as once. Otherwise every other check here is order-dependent.
  ordering             ``render(x[perm]) == render(x)[perm]``. Catches a batch-strided read, and a
                       grouped renderer scattering its results back through a stale index.
  serialization        numpy, torch and the HuggingFace corpus must all regenerate bit-identical
                       frames. A corpus that round-trips its numbers but not its pixels is not a
                       corpus.
  CUDA graph           where a step is captured into a graph, the graph and the eager path must agree,
                       and ``set_state`` must still be visible to the graph — which it only is because
                       the state buffer is written through rather than rebound.

COST. The suite is a gate, so it has to finish. Two rules are followed throughout and each is stated
where it applies: batches are small (8) except in the one test whose subject IS the batch size, and no
test asks for more than TWO distinct static-scene style values, because the plate-baking families
re-render a MuJoCo plate per distinct appearance and an n-wide ramp turns a one-second check into
minutes.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.gpu

if not torch.cuda.is_available():
    pytest.skip("stable-envs GPU tests require a CUDA device", allow_module_level=True)

from _shared import family_params, make_or_skip, resolve_family, sampled_states

import stable_envs as se
from stable_envs._data import StateDataset, TrajectoryDataset, collate_batched
from stable_envs.testing import conformance as C
from stable_envs.testing import render_worker as W


def _recorded_name(name: str) -> str:
    """Measured-table spelling: frozen downsample floors are keyed by historical class names."""
    return se.registry._frozen_from_canonical(name)

ENV_NAMES = sorted(se.registry.REGISTRY)

#: States per env. Small on purpose: every property here except batch-size invariance is a statement
#: about indexing and purity, not about coverage, and it is falsified at 8 rows as surely as at 8192.
N = 8

#: Batch sizes the loader-facing check sweeps. Neither 7 nor 8 is decoration — 7 is the odd tail every
#: loader ends on, and 8 is the size at which the OGBench ray-caster's batched matmul first disagreed
#: with 1. See the module docstring for both.
BATCH_SIZES = (1, 3, 7, 8, 64)

#: Peak bytes of frames one test is allowed to materialise at once. The large-batch check picks the
#: biggest size from ``BIG_BATCHES`` that fits: MiniHack draws 336x1264 (1.27 MB a frame) and the
#: intersection envs 600x600, so a fixed 1024 is 1.3 GB for one and 50 MB for another.
BIG_FRAME_BUDGET = 1 << 30
BIG_BATCHES = (1024, 512, 256, 128, 64)

#: Defects this file has carried, and what closed them. There are NONE outstanding — no
#: ``xfail(strict=True)`` entry is left in this suite — and the measurements are kept because they are
#: the only record of what the wrong behaviour looked like, which is what makes a regression
#: recognisable rather than merely detectable.
#:
#: Each of them was held as a strict xfail while it was open, and that is what deleted it: the
#: assertion stayed exactly as strong as it should be, the suite said out loud that the property was
#: broken, and the day the code was fixed the XPASS failed the run and said "remove this entry". A
#: tolerance, a skip or a softened comparison would each have hidden one.
#:
#: HIGHWAY_SIZE_LEAK. ``HighwayEnvBase``
#: derived ``self.H``/``self.W``/the camera from ``render(size=)`` and STORED them, so
#:     env.render(st).shape -> (4,150,600,3); env.render(st, size=300); env.render(st).shape
#: came back (4,300,1200,3) with ``env.size`` still 150 — and it cascaded into four unrelated dataset
#: round-trip failures. The screen is now a frozen ``_Screen`` value derived inside the call, so there
#: is nowhere left to store it; the four highway envs assert the property below like everyone else.
#:
#: RAYCAST_CAPSULE_BATCH_DEPENDENCE.
#: ``raycast3d.intersect`` returned different ``t`` for BITWISE IDENTICAL per-ray inputs when the
#: batch dimension changed, on the capsule and box branches. Instrumented at B=1 against B=8 on
#: AntmazeMediumNavigate:
#:     d bitwise_equal=True   sz/pos/R bitwise_equal=True   t bitwise_equal=False   max|dt| = 1e30
#: 1e30 is the miss sentinel — a ray that HIT at one batch size MISSED at another and a whole
#: silhouette pixel flipped between agent colour and plate (8-44 pixels a frame, up to 59/255, on all
#: 17 legged cells; pointmaze's single-sphere agent was bit-identical throughout). The cause was the
#: ``einsum`` that rotates the ray into the geom frame: it lowers to ``bmm``, cuBLAS selects its kernel
#: from the batch COUNT, and the three products of a 3x3-by-3x1 were accumulated in a different order
#: at B=1 and at B=8. One ulp in ``dl`` became 2.4e-4 in ``disc = b*b - a*c`` and ``disc > 0`` is a
#: hit/miss decision. It is written out as elementwise multiplies and adds now, so nothing in the
#: intersection reduces across threads or changes shape with the batch, and the tiled path is back on.
#:
#: Two more lived in a separate ``test_known_defects.py``, which is gone for the same reason: a
#: per-environment style collapsing to environment 0's on a sub-batch render, and ``render(out=)``
#: accepting a strided destination and scattering the frames across it. Their measurements are in the
#: docstrings of the tests that now assert them (sections 4 and 4b), attached to the assertion rather
#: than to a list, which is where a reproduction is worth keeping.


def _continuous_leaf(env):
    """The first continuous variation leaf, as ``(path, offset, width, midpoint)``, or None.

    Continuous because a midpoint is then an unambiguous value that is in the space and is not the
    default, which is what both style/variation-agreement tests need; a Discrete leaf has no midpoint
    and "flip it" is a different probe. Shared by the two tests so they cannot end up asking their
    question about different knobs and reporting the same verdict.
    """
    for path, off, w in env.variation_space.flat_layout():
        node = env.variation_space
        for part in path.split("."):
            node = node[part]
        if hasattr(node, "low"):
            lo = np.broadcast_to(node.low, node.shape).astype(np.float64).reshape(-1)
            hi = np.broadcast_to(node.high, node.shape).astype(np.float64).reshape(-1)
            return path, off, w, (0.5 * (lo + hi)).astype(np.float32)
    return None


# ------------------------------------------------------------------ fixtures


@pytest.fixture(scope="module", params=ENV_NAMES)
def env(request):
    return se.make(request.param, num_envs=N, device="cuda")


# The "put the env back at its default appearance before every test" fixture now lives in
# tests/gpu/conftest.py, and moving it was not cosmetic. Declared here it took ``env`` as an argument,
# and an AUTOUSE fixture that requests a parameterised fixture parameterises every test in the module —
# including the ones that build their own env and never touch the shared one. Measured:
# ``test_highway_render_size_is_an_argument_not_a_mode`` is parameterised over 4 highway envs and was
# collected 4 x 49 = 196 times, running 192 duplicates of a test whose whole point is that it uses a
# FRESH env (36 items and 24.5s of the smoke subset alone). The conftest version asks
# ``request.fixturenames`` and only then calls ``getfixturevalue``, which resets exactly the same envs
# without dragging every test in the directory into the registry sweep.


@pytest.fixture(scope="module")
def inputs(env):
    """The (states, style) pair, drawn exactly as the subprocess draws it.

    Imported from ``render_worker`` rather than written again here: two copies of "how the inputs are
    drawn" is how a cross-process check quietly becomes a comparison of two different experiments that
    both pass.
    """
    states, style = W.sample_inputs(env, N)
    return states, (style if env.style_dim else None)


@pytest.fixture(scope="session")
def fresh_process(request, tmp_path_factory):
    """Run the registry through ``stable_envs.testing.render_worker`` in a NEW interpreter, once.

    One subprocess for 44 envs rather than one per env. A fresh interpreter costs a CUDA context, a
    Triton cache lookup and every env's construction — for the MuJoCo mazes, a plate bake — so per-env
    it would be twenty minutes of process startup to answer a question that one process answers for
    all of them.

    The env list comes from what pytest actually collected, so ``-k PushT`` costs one env in the
    subprocess too. Without that, narrowing the run to debug one env would still pay for all 44 and
    the obvious reaction would be to stop running the cross-process check at all.
    """
    selected = sorted(
        {
            item.callspec.params["env"]
            for item in request.session.items
            if hasattr(item, "callspec") and "env" in item.callspec.params
        }
    )
    d = tmp_path_factory.mktemp("xproc")
    npz, js = d / "inputs.npz", d / "digests.json"
    root = str(Path(se.__file__).resolve().parents[1])
    env_vars = dict(os.environ)
    env_vars["PYTHONPATH"] = os.pathsep.join([root, env_vars.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    cmd = [
        sys.executable,
        "-m",
        "stable_envs.testing.render_worker",
        "--out-npz",
        str(npz),
        "--out-json",
        str(js),
        "--n",
        str(N),
        "--envs",
        ",".join(selected),
    ]
    # STREAMED, not captured, and each line becomes an event. This fixture builds and renders EVERY
    # selected env in one subprocess, which is the right trade (a fresh interpreter costs a CUDA
    # context and, for the MuJoCo mazes, a plate bake) -- but it means the parent sits in one call for
    # as long as that takes. The supervisor kills a segment after 900s with no test starting or
    # finishing, so at a large registry this fixture out-lived the watchdog, the segment was killed,
    # the resumed segment restarted the worker from scratch, and the run LIVELOCKED: 12 segments and
    # 4h56m of wall clock with zero tests completed, on a suite that finished in 45 minutes at 64
    # envs. Emitting the worker's own per-env progress is what makes a long setup distinguishable
    # from a wedged one, which is the distinction the watchdog exists to make.
    try:
        from stable_envs.testing.safety_plugin import _emit
    except Exception:  # noqa: BLE001

        def _emit(**_rec):  # pragma: no cover
            return

    _emit(kind="fixture_start", fixture="fresh_process", n_envs=len(selected))
    out: list[str] = []
    proc = subprocess.Popen(cmd, env=env_vars, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    deadline = time.time() + 3600
    for line in proc.stdout:
        out.append(line)
        if line.startswith("render_worker: done "):
            _emit(kind="fixture_progress", fixture="fresh_process", env=line.split("done ", 1)[1].strip(), i=len(out))
        if time.time() > deadline:
            proc.kill()
            raise RuntimeError(f"the cross-process render worker exceeded 3600s after {len(out)} envs")
    rc = proc.wait()
    _emit(kind="fixture_end", fixture="fresh_process", rc=rc, lines=len(out))
    if rc != 0 or not js.exists():
        raise RuntimeError(
            f"the cross-process render worker did not produce its digests (exit {rc}).\n"
            f"command: {' '.join(cmd)}\n--- output ---\n{''.join(out)[-6000:]}"
        )
    with open(js) as f:
        report = json.load(f)
    return report, np.load(npz)


def _record(fresh_process, name: str) -> dict:
    report, _ = fresh_process
    rec = report["envs"].get(name)
    assert rec is not None, (
        f"{name} is registered in this process but the fresh process did not report it — the two "
        f"interpreters do not agree on the registry, so nothing below is comparing like with like"
    )
    if "error" in rec:
        pytest.fail(
            f"{name} could not be built or rendered in a fresh process, while it builds here. "
            f"That asymmetry is the bug, not a missing dependency:\n{rec['error']}"
        )
    return rec


def _digest(x) -> str:
    return C.frames_digest(x)


# ------------------------------------------------------------------ 1. cross-process


def test_seeded_sampling_reproduces_in_a_fresh_process(env, fresh_process):
    """``sample_states(rng=)`` and ``sample_style(rng=)`` give the same bytes in a new interpreter.

    Stated separately from the frame check below, and asserted first, because the two failures have
    different owners. If the states differ, the renderer is not implicated at all and a frame
    comparison built on them would be comparing two different experiments; if the states agree and the
    frames do not, the renderer is carrying process state.
    """
    rec = _record(fresh_process, type(env).__name__)
    states, style = W.sample_inputs(env, N)
    assert _digest(states) == rec["states_sha"], (
        f"{type(env).__name__}.sample_states(rng=default_rng({W.STATE_SEED})) gave different numbers "
        f"in this process than in a fresh one — the seeded sampler reads something other than its "
        f"generator (a global RNG, the env's own advanced state, a hash seed), so a caller who "
        f"recorded the seed cannot regenerate the corpus"
    )
    assert _digest(style) == rec["style_sha"], (
        f"{type(env).__name__}.sample_style(rng=default_rng({W.STYLE_SEED})) is not reproducible "
        f"across processes; the appearance half of the state cannot be regenerated from its seed"
    )


def test_frames_are_bit_identical_in_a_fresh_process(env, fresh_process):
    """The gate. Same states, same style, new interpreter, identical bytes.

    Unconditional on the seeded sampler: the states come over the wire in the subprocess's own npz, so
    this holds even if the seeded API were broken. What it rules out is everything a process
    accumulates — a cached plate keyed on something mutable, an autotuner picking a different kernel
    on the second run, a texture atlas built from a dict whose order changed, a float that was
    initialised from an address. Every one of those is invisible to a same-process determinism check
    and fatal to a stored corpus.
    """
    name = type(env).__name__
    rec = _record(fresh_process, name)
    _, arrays = fresh_process
    states = arrays[f"{name}::states"]
    style = arrays[f"{name}::style"]
    frames = env.render(states)
    assert list(frames.shape) == rec["shape"], (
        f"{name}: frame shape {list(frames.shape)} here, {rec['shape']} in a fresh process"
    )
    assert _digest(frames) == rec["frames_sha"], (
        f"{name}: rendering the SAME states in a fresh process gave different pixels. The frame is "
        f"not a function of the state alone — it depends on something this interpreter accumulated, "
        f"which means a stored corpus does not regenerate"
    )
    if env.style_dim:
        assert _digest(env.render(states, style=style)) == rec["frames_styled_sha"], (
            f"{name}: frames agree under the default appearance but not under a stored style, so the "
            f"style half of the state does not survive a process boundary"
        )


def test_rollout_seed_reproduces_in_a_fresh_process(env, fresh_process):
    """The private seeded rollout is the one helper that draws randomness at every step.

    That makes it where a seed leaks: a policy that seeds on ``set_env`` but not on ``reset``, a CUDA
    generator bound to whatever device was current, a global numpy call inside a step. The states and
    the actions are checked separately because a reproducible action sequence driving a
    non-reproducible simulator and the reverse are different bugs.
    """
    from stable_envs.testing._rollout import sample

    name = type(env).__name__
    rec = _record(fresh_process, name)
    if "rollout_states_sha" not in rec:
        pytest.skip(f"{name}: no batched GPU simulator, so there is no rollout to seed")
    tr = sample(env, steps=W.ROLLOUT_STEPS, batch=N, seed=W.ROLLOUT_SEED)
    assert _digest(tr["states"].detach().cpu().numpy()) == rec["rollout_states_sha"], (
        f"{name} sample(seed={W.ROLLOUT_SEED}) produced a different trajectory in a fresh process — "
        f"the rollout reads randomness the seed does not reach"
    )
    if tr.get("actions") is not None and "rollout_actions_sha" in rec:
        assert _digest(tr["actions"].detach().cpu().numpy()) == rec["rollout_actions_sha"], (
            f"{name}: the seeded policy chose different actions in a fresh process"
        )


# ------------------------------------------------------------------ 2. batch size and chunking


def _big_batch(env) -> int:
    """The largest batch from ``BIG_BATCHES`` whose frames fit ``BIG_FRAME_BUDGET``."""
    probe = env.render(env.sample_states(1, rng=np.random.default_rng(0)))
    per = int(np.prod(probe.shape[1:]))
    for b in BIG_BATCHES:
        if b * per <= BIG_FRAME_BUDGET:
            return b
    return BIG_BATCHES[-1]


def test_frame_i_is_independent_of_batch_size(env):
    """Row ``i`` renders the same alone, in 3, in 7, in 8, in 64, and in ~1024.

    The one test that deliberately uses a large batch, because the batch size IS the subject. Odd
    sizes are included on purpose: a kernel launched over ``cdiv(H*W, BLOCK)`` blocks with a masked
    tail, or an output offset computed in int32, is correct at every power of two and wrong at 7 or at
    the size that first overflows — this repo has already shipped one 64-bit offset fix for exactly
    that, found at 224px and batch 16383.

    It is also the check that found ``RAYCAST_CAPSULE_BATCH_DEPENDENCE`` (see the comment above), and
    3 and 8 are in the sweep because 8 is where that one was localised; the existing 1-versus-16
    batch-invariance check never saw it, because 16 happens to agree with 1 on the states it draws.
    """
    big = _big_batch(env)
    states = env.sample_states(big, rng=np.random.default_rng(11))
    r = C.check_batch_size_invariance(env, states, sizes=BATCH_SIZES + (big,))
    assert r["max_abs_diff"] == 0, (
        f"{type(env).__name__}: the same state renders differently depending on how many other "
        f"environments share the launch (max {r['max_abs_diff']}/255, {r['worst_case']}) — a loader's "
        f"last partial batch would then produce different pixels from every batch before it: {r}"
    )


def test_chunked_render_states_is_chunk_independent(env, inputs):
    """``render_states(chunk=c)`` must give the same corpus for every ``c``.

    This is the function ``StateDataset`` and ``TrajectoryDataset.all_frames`` are built on, and it is
    the only path where the renderer writes into a slice of a caller-owned buffer. A chunk that is
    rendered into a stale view, a style tensor sliced with the corpus index instead of the chunk
    index, or a non-blocking host-to-device copy read before it lands all show up here and nowhere
    else. Chunk 1 and 7 are included because they exercise the boundary on every row.
    """
    states, style = inputs
    r = C.check_chunked_render_matches(env, states, style=style, chunks=(N, 1, 7, 4096))
    assert r["max_abs_diff"] == 0, (
        f"{type(env).__name__}: render_states gives different pixels at different chunk sizes "
        f"({r['worst_case']}, max {r['max_abs_diff']}/255) — the chunk is supposed to bound memory, "
        f"not to change the answer: {r}"
    )


# ------------------------------------------------------------------ 3. resolution

#: Measured, per env: the fraction of pixels on which rendering at 2S and box-downsampling agrees
#: EXACTLY with rendering at S, and the mean absolute difference. These are floors taken from a
#: measured run and are NOT a specification — the two pictures are not required to be equal and it
#: would be wrong to assert that they are. Rendering at S asks which colour covers a pixel's centre;
#: rendering at 2S and averaging asks for the mean colour over its area. They differ exactly where a
#: shape edge crosses a pixel.
#:
#: Recorded by ``stable_envs.testing.conformance.check_resolution_consistency`` over 8 sampled states
#: per env; see MEASURED_ON for provenance. A regression that MOVES a shape drops ``exact_frac``
#: immediately, which is what the floor is for; anti-aliasing does not move it at all.
MEASURED_ON = (
    "H200, driver 590.48.01, torch 2.11+cu130, triton 3.6, mujoco 3.9; 8 states per env "
    "from sample_states(rng=default_rng(render_worker.STATE_SEED)), default appearance"
)
#: The floor is the measured value less this. A different GPU may land an edge pixel on the other side
#: of a rounding, and half a percentage point is comfortably more than that and far less than any
#: geometry change: the numbers below span 0.708 to 1.000, and moving a shape by one pixel costs whole
#: percent, not tenths.
DOWNSAMPLE_MARGIN = 0.005
#: What the three renderer families actually cost, and why the numbers differ so much:
#:   1.000       gather from a bitmap atlas whose tile size scales exactly (Craftax, MiniHack) — the
#:               2S frame IS the S frame at twice the size, so the box filter inverts it exactly.
#:   0.007-0.092 Procgen: a gather from a bitmap atlas whose sampling RATE scales instead, so the 2S
#:               frame resamples the same source art more finely. See the note on its entries.
#:   0.97-1.00   MiniGrid: same idea, but the atlas is re-rendered by MiniGrid at the requested tile
#:               size with its own 3x supersample, so the two are close rather than identical.
#:   0.897       PushT: an analytic rasterizer point-sampling shape edges. Every disagreement is an
#:               edge pixel, mean 2.27/255.
#:   0.708-0.726 the OGBench mazes: the plate is re-baked by MuJoCo at 2S with its own multisampling,
#:               so the static scene is genuinely re-rendered rather than re-sampled.
#:   0.282-0.730 DM Control: the same re-bake, over a scene that has almost no flat pixel to survive
#:               it. Its floor is a checkerboard under a spotlight gradient and its sky is a speckled
#:               gradient, so nearly every pixel moves by a level or two when MuJoCo re-renders the
#:               plate at 2S -- mean 1.44/255 on cartpole against the mazes' 0.6-1.1. This is the
#:               loosest family, and the two entries bracket it: point_mass (0.730) has no sky and a
#:               floor the camera looks straight down on, cartpole (0.282) is 62% speckled skybox.
#: TWO Step-7 cells are DELIBERATELY ABSENT from this table and neither absence is an oversight:
#:
#:   ``CraftaxFull``               the class declares ``resolution_consistency_applicability`` with
#:                                 "all bands: not_applicable" and the oracle behind it -- craftax's
#:                                 own ``load_texture`` resamples every PNG with ``Image.NEAREST`` at
#:                                 each size, so the OFFICIAL renderer's 2S atlas is not a 2x
#:                                 replication of its S one either (official world rows 0.4817 in
#:                                 full daylight, 0.0991 at night, max abs 191). This test skips it on
#:                                 that declaration. A floor recorded here would be a number for a
#:                                 relation the reference does not satisfy in any band. Measured for
#:                                 the record, not asserted: 0.794537 at S=10 (closure2 gate_a).
#:   ``DMCManipulatorBringBall``   carries ``PLATE_PATHS`` and is not a ``PLATE_REPRESENTATIVE``, so
#:                                 this test skips it exactly as it skips every other PARITY-tier
#:                                 plate cell in the DM Control family -- ``DMCCartpoleSwingup`` and
#:                                 ``DMCPointMassEasy`` stand for the two camera geometries. Measured
#:                                 for the record, not asserted: 0.850842 (cpu and cuda, closure2).
#:
#: Both numbers live with their provenance in
#: ``tests/fixtures/regression/join/step8/certification/downsample_provenance.json``.
DOWNSAMPLE_EXACT_FRAC: dict[str, float] = {
    "AntmazeGiantNavigate": 0.708287,  # mean 1.1419/255
    "AntmazeGiantStitch": 0.708287,  # mean 1.1419/255
    "AntmazeLargeExplore": 0.722699,  # mean 0.9589/255
    "AntmazeLargeNavigate": 0.722699,  # mean 0.9589/255
    "AntmazeLargeStitch": 0.722699,  # mean 0.9589/255
    "AntmazeMediumExplore": 0.722771,  # mean 0.6205/255
    "AntmazeMediumNavigate": 0.722771,  # mean 0.6205/255
    "AntmazeMediumStitch": 0.722771,  # mean 0.6205/255
    "AntmazeTeleportExplore": 0.717594,  # mean 1.1558/255
    "AntmazeTeleportNavigate": 0.717594,  # mean 1.1558/255
    "AntmazeTeleportStitch": 0.717594,  # mean 1.1558/255
    "BabyAIGoToLocal": 0.987213,  # mean 0.1315/255
    #: Crafter is the outlier of the tile families and the reason is arithmetic, not geometry.
    #: ``Env.render`` CENTRES the 9-tile view inside the frame — ``border = (size - (size // 9) * 9)
    #: // 2`` — so 64 px puts the view at offset 0 and 128 px puts it at offset 1. The 2S picture is
    #: therefore the S picture shifted by half an output pixel, and no box filter inverts a half-pixel
    #: shift; on top of that the atlas is resampled from 16x16 source art at each tile size, so 14 px
    #: tiles are not 7 px tiles doubled. Both are properties of the reference. The floor is low and it
    #: is still a live regression detector: it moves the moment a tile lands in a different place.
    "Crafter": 0.382263,  # mean 12.5190/255
    #: STILL 1.000000, unchanged, and now asserted only where it can be. Step 7 measured the night
    #: band: the noise field is drawn at the FRAME's resolution, so 2S draws a different field and
    #: not a finer sampling of the same one -- 0.282 at light 0.45 and 0.035 at light 0.1 against
    #: 1.000 at light >= 0.5. The official craftax renderer is not resolution-consistent there
    #: either (world rows 0.1634 for both, the same number), so the class declares
    #: ``resolution_consistency_applicability`` and this test judges the applicable band. On the
    #: protocol's own 8-state draw that is 6 states and they measure exactly 1.000000.
    #: Oracle: tests/fixtures/regression/join/step7/craftax-classic-full/oracle/resolution_oracle_S16.json.
    #: Ruling: Sami 2026-09-15 (tests/fixtures/regression/join/step7/dashboard.yaml, decisions).
    "CraftaxClassic": 1.000000,  # mean 0.0000/255 on light >= 0.5; see applicability
    "DMCCartpoleSwingup": 0.282065,  # mean 1.4428/255  see DM Control note below
    "DMCPointMassEasy": 0.730424,  # mean 0.2418/255
    #: The five VISUAL-tier DM Control cells draw the whole scene through MJWarp with no plate, so
    #: PLATE_REPRESENTATIVES does not cover them and each one is measured in its own right. They sit
    #: together at 0.53-0.63 for one reason: 2S resolves the speckled skybox and the checker floor at a
    #: finer rate and the box filter does not invert either, so the number is a property of those two
    #: textures rather than of the geometry. It is still a live detector — the bodies and the camera
    #: are what a regression would move, and moving them collapses it.
    "DMCCheetahRun": 0.533850,  # mean 3.0869/255
    #: Step 7 added a sixth cell to that VISUAL-tier group and it belongs here for the same reason:
    #: DMCQuadrupedWalk draws the whole scene through MJWarp with no plate, so PLATE_REPRESENTATIVES
    #: does not cover it. It sits ABOVE the other five because its camera is much closer to the body
    #: and sees proportionally less speckled skybox. MEASURED under this table's own protocol
    #: (8 states at render_worker.STATE_SEED, default appearance, states sha256 52d82312...): the
    #: minimum over the four committed Step-7 runs -- cpu 0.8118772 (closure2) and 0.8118797
    #: (closure), cuda 0.8120117 and 0.8120217, at mujoco 3.11.0 and 3.12.0 -- so the floor holds on
    #: every device and reference pair measured, and the whole spread is 2.5e-05 wide inside a
    #: 0.005 margin. Artifacts: tests/fixtures/regression/join/step7/closure2/gate_a/{cpu,cuda}_vis311.json.
    "DMCQuadrupedWalk": 0.811877,  # mean 0.6329/255
    "DMCHopperHop": 0.550220,  # mean 2.9202/255
    "DMCHumanoidStand": 0.603650,  # mean 2.4394/255
    "DMCSwimmerSwimmer6": 0.609800,  # mean 0.8424/255
    "DMCWalkerWalk": 0.628950,  # mean 2.5596/255
    #: Griddly is the low end of the TILE families and the mechanism is the reference's, not ours.
    #: `size` here means `Observers.Sprite2D.TileSize`, and griddly RESAMPLES its 24 px sprite PNGs to
    #: whatever tile size is asked for -- so the 2S atlas is a finer resampling of the same source art
    #: rather than the S atlas doubled, and box-averaging four resampled texels does not return the one
    #: the S atlas picked. The wall art is high-frequency pixel art by design, which is why the
    #: wall-dense games sit lowest (Sokoban 0.22, Labyrinth 0.35) and the sparsest highest
    #: (Butterflies 0.70). Mean absolute difference stays at 1.1-2.9/255 throughout -- the two pictures
    #: agree, it is exact-pixel agreement that the resample costs. Still a live detector: the number is
    #: fixed for a fixed sampling grid and moves the moment a tile lands somewhere else.
    "GriddlyBait": 0.432180,  # mean 1.7029/255
    "GriddlyButterflies": 0.704126,  # mean 1.2416/255
    "GriddlyClusters": 0.392938,  # mean 1.4342/255
    "GriddlyCookMePasta": 0.298415,  # mean 1.0666/255
    "GriddlyLabyrinth": 0.354689,  # mean 2.1506/255
    "GriddlySokoban": 0.222717,  # mean 2.9219/255
    "GriddlySokoban2": 0.248369,  # mean 2.8664/255
    "Highway": 0.922090,  # mean 7.6935/255  size re-frames; recorded, not asserted
    #: HumanoidBench draws the whole scene through MJWarp with no plate (its camera is TRACKCOM,
    #: so no static plate exists), and it carries no PLATE_PATHS -- so every cell is measured in
    #: its own right. They sit together at 0.95-0.97 because the two things 2S resamples
    #: differently are both small here: the floor's texture is REPLACED by its dominant colour
    #: (docs/research/humanoidbench.md Section 3.4), so the only rate-dependent content left is
    #: the H1's mesh silhouettes and its shadow edge. Spoon is the low end because it is the one
    #: cell here with textured primitives on the table, which 2S resamples at a finer rate.
    "HumanoidBenchH1BalanceSimple": 0.967609,  # mean 0.8579/255
    "HumanoidBenchH1HandBalanceSimple": 0.963911,  # mean 0.9218/255
    "HumanoidBenchH1HandRun": 0.970247,  # mean 0.8162/255
    "HumanoidBenchH1HandSitHard": 0.963375,  # mean 1.0008/255
    "HumanoidBenchH1HandSitSimple": 0.964001,  # mean 0.9794/255
    "HumanoidBenchH1HandSlide": 0.967253,  # mean 0.8828/255
    "HumanoidBenchH1HandSpoon": 0.959507,  # mean 1.2295/255
    "HumanoidBenchH1HandStair": 0.967337,  # mean 0.8099/255
    "HumanoidBenchH1HandStand": 0.970247,  # mean 0.8162/255
    "HumanoidBenchH1HandWalk": 0.970247,  # mean 0.8162/255
    "HumanoidBenchH1Run": 0.972082,  # mean 0.7091/255
    "HumanoidBenchH1SitHard": 0.966482,  # mean 0.8928/255
    "HumanoidBenchH1SitSimple": 0.967659,  # mean 0.8467/255
    "HumanoidBenchH1Slide": 0.969963,  # mean 0.7793/255
    "HumanoidBenchH1Stair": 0.971107,  # mean 0.6661/255
    "HumanoidBenchH1Stand": 0.972082,  # mean 0.7091/255
    "HumanoidBenchH1Walk": 0.972082,  # mean 0.7091/255
    "HumanoidmazeGiantNavigate": 0.712624,  # mean 1.1539/255
    "HumanoidmazeGiantStitch": 0.712624,  # mean 1.1539/255
    "HumanoidmazeLargeNavigate": 0.725895,  # mean 0.9690/255
    "HumanoidmazeLargeStitch": 0.725895,  # mean 0.9690/255
    "HumanoidmazeMediumNavigate": 0.723503,  # mean 0.6133/255
    "HumanoidmazeMediumStitch": 0.723503,  # mean 0.6133/255
    "Intersection": 0.958981,  # mean 4.5107/255  size re-frames; recorded, not asserted
    #: Jumanji renders through matplotlib at ``dpi = size / 10``, so 2S is a genuinely DIFFERENT
    #: rasterization rather than a supersample of S: text is laid out in points and re-hinted at the new
    #: dpi, and patch edges are re-antialiased at a different sub-pixel coverage. Maze is the exception
    #: at 1.000000 because its viewer is a single nearest-neighbour ``imshow`` whose cell boundaries
    #: land on exact pixel multiples at both sizes, so the box filter inverts the upscale exactly. The
    #: glyph-bearing ids (Sudoku, Minesweeper) and the sprite/patch ones (RubiksCube, Sokoban) sit at
    #: 0.71-0.86, which is a property of the reference and not of this renderer.
    #: JumanjiConnector is deliberately ABSENT: its cell borders are 1 POINT wide, which quantises to
    #: fully-black pixels at dpi 20 and to a two-cell blend at dpi 40, so the per-cell model is false at
    #: 400 px and the env raises ValueError rather than render there. This test skips it for that
    #: reason -- see docs/research/jumanji.md.
    "JumanjiMaze": 1.000000,  # mean 0.0000/255
    "JumanjiMinesweeper": 0.832503,  # mean 9.0860/255
    "JumanjiRubiksCube": 0.855150,  # mean 4.1042/255
    "JumanjiRubiksCubePartlyScrambled": 0.855150,  # mean 4.0577/255
    "JumanjiSokoban": 0.711994,  # mean 0.7419/255
    "JumanjiSudoku": 0.831450,  # mean 7.5042/255
    "JumanjiSudokuVeryEasy": 0.822840,  # mean 7.8610/255
    "Merge": 0.945075,  # mean 5.8377/255  size re-frames; recorded, not asserted
    "MiniGridDoorKey": 0.997700,  # mean 0.0228/255
    "MiniGridEmpty": 0.998932,  # mean 0.0048/255
    "MiniGridFourRooms": 0.999811,  # mean 0.0008/255
    "MiniGridKeyCorridor": 0.982485,  # mean 0.2123/255
    "MiniGridLavaCrossing": 0.974085,  # mean 0.1360/255
    "MiniGridPOVFetch": 0.997429,  # mean 0.0121/255
    "MiniHack": 1.000000,  # mean 0.0000/255
    #: The six VISUAL-tier Gymnasium locomotion domains, which are here rather than skipped by
    #: PLATE_REPRESENTATIVES because they bake no plate: they are re-rasterized whole by nvdiffrast at
    #: whatever resolution is asked for. What sets the number is the minified checker floor —
    #: doubling the resolution halves the footprint every floor pixel filters over, so the mip level
    #: changes and the 2S floor is a genuinely finer sampling of the texture rather than a supersample
    #: of the S one. It tracks how much floor each camera sees, which is why Swimmer (94.9% floor,
    #: MAGNIFIED rather than minified, so no mip level changes) is 0.909 while Humanoid is 0.682.
    #: The four PARITY envs of the same family are not in this table: they carry PLATE_PATHS.
    "MujocoAnt": 0.771469,  # mean 1.6888/255
    "MujocoHalfCheetah": 0.758811,  # mean 2.0481/255
    "MujocoHopper": 0.694054,  # mean 2.5325/255
    "MujocoHumanoid": 0.681649,  # mean 2.7766/255
    "MujocoSwimmer": 0.909481,  # mean 0.7384/255
    "MujocoWalker2d": 0.755514,  # mean 1.9871/255
    #: Overcooked sits with MiniGrid at the top of the tile families and for the same reason: the
    #: atlas is RE-RENDERED by jaxmarl at the requested tile size with its own 3x supersample, so 2S is
    #: close to a supersample of S rather than a resample of fixed source art (which is what puts
    #: griddly and Procgen at the other end). The residual is the round tiles -- an onion at 3x96 has a
    #: different subpixel boundary from one at 3x192 -- and it is 0.13-0.18/255 in the mean.
    #: MEASURED under this table's own protocol (`conformance.downsample_protocol_states`:
    #: 8 states at render_worker.STATE_SEED, default appearance, states sha256 055d4057...),
    #: alloc 6443480 on gpu2264, RTX 3090, mujoco 3.12.0 / ogbench 1.2.1, MUJOCO_GL=egl. These
    #: two are a baked MuJoCo plate composited with MJWarp movers, so the 2S picture rebakes
    #: the plate at 2S rather than doubling it -- the plate agrees with itself to the same
    #: point sampling everything else in this table does.
    "OGBCube": 0.885782,  # mean 1.0329/255
    "OGBScene": 0.805306,  # mean 2.0163/255
    "OvercookedAsymmAdvantages": 0.985520,  # mean 0.1493/255
    "OvercookedCoordRing": 0.981630,  # mean 0.1787/255
    "OvercookedCounterCircuit": 0.987140,  # mean 0.1251/255
    "OvercookedCrampedRoom": 0.980990,  # mean 0.1720/255
    "OvercookedForcedCoord": 0.981630,  # mean 0.1787/255
    #: gymnasium-robotics PointMaze. Much higher than the OGBench mazes (0.708-0.726) for one
    #: reason: its camera looks along the ground at elevation -45, so the plate's re-bake at 2S
    #: re-renders a scene that is mostly flat sky and flat floor, where the OGBench top-down camera
    #: fills the frame with a checkerboard whose every cell edge moves.
    #: The one nvdiffrast cell outside the Gymnasium locomotion block, and the highest number in this
    #: part of the table because its scene is a rocket against a flat sky over a checker pad: most of
    #: the frame is a constant, and the disagreement is the hull silhouette and the pad's texture
    #: edges. MEASURED under this table's own protocol (8 states at render_worker.STATE_SEED, default
    #: appearance, states sha256 a5f128ad...), on the RTX 3090 at torch 2.14.0+cu126 / nvdiffrast
    #: 0.4.0, MUJOCO_GL unused: exact 0.9561372, mean 0.9923/255, max 255 -- the same number in the
    #: closure and closure2 sweeps. 0.956137 is 0.95614 to five decimals, which is the value the
    #: Step-7 carry-over recorded. There is no cpu row: PFRocketLanding raises on device="cpu" by
    #: design. Artifact: tests/fixtures/regression/join/step7/closure2/gate_a/cuda_rocket.json.
    "PFRocketLanding": 0.956137,  # mean 0.9923/255
    "PointMazeLarge": 0.913387,  # mean 0.2794/255  (measured on PointMazeMedium)
    "PointMazeMedium": 0.913387,  # mean 0.2794/255
    "PointMazeOpen": 0.913387,  # mean 0.2794/255
    "PointMazeUMaze": 0.913387,  # mean 0.2794/255
    "PointmazeGiantNavigate": 0.708274,  # mean 1.1385/255
    "PointmazeGiantStitch": 0.708274,  # mean 1.1385/255
    "PointmazeLargeNavigate": 0.722781,  # mean 0.9528/255
    "PointmazeLargeStitch": 0.722781,  # mean 0.9528/255
    "PointmazeMediumNavigate": 0.723306,  # mean 0.6111/255
    "PointmazeMediumStitch": 0.723306,  # mean 0.6111/255
    "PointmazeTeleportNavigate": 0.717602,  # mean 1.1505/255
    "PointmazeTeleportStitch": 0.717602,  # mean 1.1505/255
    "Parking": 0.931908,  # mean 6.2932/255  size re-frames; recorded, not asserted
    #: Procgen is the low end of this table, and the mechanism is the reference's, not ours. A cell
    #: is 2.56 output px at 64 and 5.12 at 128, so the 2S frame takes a nearest-neighbour sample of
    #: the SAME 128x128 source art at twice the rate — it is a finer resampling of the texture, not a
    #: supersample of the S frame, and box-averaging four different source texels does not return the
    #: one texel the S frame picked. On sand and dirt, which are high-frequency noise by design,
    #: almost no pixel survives. procgen's own 512x512 human render differs from its 64x64
    #: observation for exactly this reason (and additionally turns antialiasing on). The floor is
    #: still a live detector: it is a fixed number for a fixed sampling grid and moves as soon as a
    #: rect lands somewhere else.
    "ProcgenChaser": 0.092380,  # mean 12.6856/255
    "ProcgenHeist": 0.061981,  # mean 7.7011/255
    "ProcgenMaze": 0.007202,  # mean 9.2941/255
    "ProcgenMiner": 0.092468,  # mean 9.7574/255
    "PushT": 0.897163,  # mean 2.2749/255
    "Racetrack": 0.983442,  # mean 1.7873/255  size re-frames; recorded, not asserted
    #: RoboCasa. These four were failing this test with "has no measured downsample bound recorded",
    #: and the failure was invisible in the main run because the suite's interpreter cannot import
    #: robocasa (it pins mujoco 3.3.1 against mujoco_warp's >= 3.11), so the whole family was skipped.
    #: Measured in ``~/venvs/se-robocasa``, three repeats each,
    #: identical to six decimals every time. They sit with the locomotion domains and for the same
    #: reason: a kitchen is 73 textures on 344 visible geoms, and doubling the resolution changes what
    #: each pixel of a photographic wood or granite surface filters over. The wrist camera is the
    #: highest because it fills the frame with a few large near surfaces rather than a room of small
    #: ones.
    "RoboCasaCounterToSink": 0.756603,  # mean 2.0074/255, max 205/255
    "RoboCasaCounterToSinkL45S33": 0.787102,  # mean 2.0138/255, max 216/255
    "RoboCasaCounterToSinkRight": 0.740753,  # mean 2.0634/255, max 233/255
    "RoboCasaCounterToSinkWrist": 0.871983,  # mean 0.7108/255, max 207/255
    "Roundabout": 0.974241,  # mean 2.7201/255  size re-frames; recorded, not asserted
    "Sokoban": 1.000000,  # mean 0.0000/255  16 px art replicated k x k, exactly
}
#: Envs whose renderer must be rebuilt from scratch at a new output size. Each additional (scene,
#: size) pair for the maze family constructs another ogbench env and another ``mujoco.Renderer``, so
#: covering all 25 cells costs ~22 more live EGL contexts and about a minute for a property that
#: belongs to the shared ray-caster rather than to the maze plate. Three representatives, one per
#: locomotion body, is what that property costs to establish.
#: DM Control adds two rather than one because its two camera geometries are different pictures, not
#: two sizes of the same one: cartpole looks along the ground under a speckled sky, point_mass looks
#: straight down at a floor that fills the frame, and the 2S re-bake costs them 0.282 and 0.730.
PLATE_REPRESENTATIVES = frozenset(
    {
        "PointMazeMediumNavigate",
        "AntMazeMediumNavigate",
        "HumanoidMazeMediumNavigate",
        "DMCCartpoleSwingup",
        "DMCPointMassEasy",
        # gymnasium-robotics PointMaze: a DIFFERENT camera geometry from
        # OGBench's (elevation -45 against straight down) and a different
        # bake path (gymnasium's viewer, not mujoco.Renderer), so one of
        # the four scenes has to stand for it in its own right.
        "PointMazeMedium",
    }
)


def _skip_if_size_is_not_resolution(env) -> None:
    """Skip envs where ``size`` re-frames the camera instead of re-sampling the same picture.

    Detected, not listed: build the scene AT two sizes and compare the world rectangle the rasterizer
    is given. highway-env's is ``(0, 0, W, H)`` in SCREEN pixels at a fixed ``scaling``, so doubling
    the size shows twice as much road — the 2S frame is a wider view, not a finer one, and
    box-downsampling it produces a picture of a different scene. Measured on Highway: mean 7.9/255
    against PushT's 2.3, and the difference is road that is simply not in the S frame.

    Comparing them anyway would produce a number that looks like an anti-aliasing bound and is
    nothing of the kind, and a floor written against it would be unfalsifiable.

    The probe asks ``scene(size=)``, which is how an env whose world is in OUTPUT PIXELS declares that
    it is: envs whose world is in world units do not take the argument, and not taking it is the
    answer. It used to render at 2S and then read ``env.scene()`` back — which worked only because
    that render LEAKED the size onto the env, i.e. the detector was reading the very defect this file
    reports two tests below. When that leak was fixed the probe would have gone silently blind, so it
    is now a pure question that mutates nothing.
    """
    import inspect

    dev = env.device
    st = torch.as_tensor(env.sample_states(1, rng=np.random.default_rng(0)), device=dev)
    if "size" not in inspect.signature(env.scene).parameters:
        return  # world units: size is a sampling rate, nothing to re-frame
    try:
        w1 = tuple(env.scene(st, size=int(env.size)).world)
        w2 = tuple(env.scene(st, size=2 * int(env.size)).world)
    except NotImplementedError:
        return  # no Scene2D to probe; the tile/raycast paths re-sample
    if w1 != w2:
        pytest.skip(
            f"{type(env).__name__}: size= re-frames the camera (world rect {w1} at S, {w2} at "
            f"2S), so a downsample comparison would be comparing two different views rather "
            f"than two resolutions"
        )


def test_resolution_applicability_is_read_from_the_declaration_and_moves_no_bound():
    """The band reader, on the two declaration shapes in the repo and on the cases that must not fire.

    Pure logic over a declaration and a state array — no renderer — because what has to be guaranteed
    is that an env WITHOUT a declaration is unaffected and an unparseable band widens nothing. A
    declaration that narrows the judged set is a claim about the REFERENCE, and getting it wrong in the
    permissive direction (judging states the reference cannot satisfy) shows up as a loud failure,
    while getting it wrong in the restrictive direction (silently dropping states) would not show up at
    all. So the restrictive direction is what these cases pin down.
    """
    names = ["x", "light_level"]
    st = np.array([[0.0, 0.1], [0.0, 0.45], [0.0, 0.5], [0.0, 1.0]], np.float32)

    # 1. no declaration at all: every state, and the caller may skip subsetting entirely.
    class Bare:
        state_names = names

    r = C.resolution_consistency_applicability(Bare(), st)
    assert r["applicable"] and r["indices"] is None, r

    # 2. the CraftaxClassic shape: one applicable band and one not.
    class Banded:
        state_names = names
        resolution_consistency_applicability = {
            "light_level >= 0.5": "asserted at the frozen bound",
            "light_level < 0.5": "not_applicable: the reference is itself inconsistent there",
            "evidence": "oracle.json",
        }

    r = C.resolution_consistency_applicability(Banded(), st)
    assert r["applicable"] and r["indices"] == [2, 3], r
    assert r["declaration"] == {"evidence": "oracle.json"}, "the cited evidence must survive"

    # 3. only the INAPPLICABLE band named: the complement is still judged, not dropped.
    class OnlyBad:
        state_names = names
        resolution_consistency_applicability = {
            "light_level < 0.5": "not_applicable: mechanism",
            "evidence": "oracle.json",
        }

    assert C.resolution_consistency_applicability(OnlyBad(), st)["indices"] == [2, 3]

    # 4. the CraftaxFull shape: the whole cell, answered without looking at a state.
    class AllBands:
        state_names = names
        resolution_consistency_applicability = {"all bands": "not_applicable: mechanism", "evidence": "oracle.json"}

    r = C.resolution_consistency_applicability(AllBands(), st)
    assert r["applicable"] is False and r["indices"] == [], r
    assert C.resolution_consistency_applicability(AllBands(), None)["applicable"] is False

    # 5. a band naming a dimension this env does not have must NOT narrow anything. A typo that
    #    silently dropped states would look like a pass.
    class Typo:
        state_names = names
        resolution_consistency_applicability = {"lightlevel >= 0.5": "asserted", "evidence": "oracle.json"}

    r = C.resolution_consistency_applicability(Typo(), st)
    assert r["applicable"] and r["indices"] is None and "could not be resolved" in r["why"], r


def test_resolution_downsample_agrees_within_the_measured_bound(env, inputs):
    """Rendering at 2S then box-downsampling, against rendering at S — to a measured, per-env bound.

    Deliberately not an equality. Two rasterizations at different sampling rates disagree at every
    edge, and the tile-atlas families disagree for a second reason: their atlas is resampled from the
    source art at the requested tile size, so 2S is a different picture rather than a supersampled
    one. Asserting equality would either fail on correct renderers or force a tolerance so loose it
    could not detect a shape moving.

    What is asserted is the recorded floor on exact-pixel agreement. A geometry regression — an
    off-by-half-pixel in the sampling grid, a scale that is applied in output pixels instead of world
    units — moves a shape, and that collapses ``exact_frac``. Anti-aliasing does not touch it.

    The default appearance is used rather than a sampled style, so the number is a property of the
    geometry and is reproducible from the recorded seed alone.
    """
    name = type(env).__name__
    if hasattr(type(env), "PLATE_PATHS") and name not in PLATE_REPRESENTATIVES:
        pytest.skip(f"{name}: covered by {sorted(PLATE_REPRESENTATIVES)} — see PLATE_REPRESENTATIVES")
    states, _ = inputs
    # A class may declare, with the OFFICIAL renderer's own oracle result cited, that the relation
    # itself does not hold in some or all of its state space -- Craftax draws its night noise field at
    # the frame's own resolution and Craftax Full resamples every texture PNG at each size, so 2S is a
    # different picture there for the reference too. This reads that declaration; it moves no bound.
    app = C.resolution_consistency_applicability(env, states)
    if not app["applicable"]:
        pytest.skip(f"{name}: {app['why']}")
    if app["indices"] is not None:
        states = np.ascontiguousarray(np.asarray(states)[np.asarray(app["indices"], int)], np.float32)
        print(f"\n{name}: judged on {len(states)} states in the applicable band(s) {app['bands']}")
    try:
        _skip_if_size_is_not_resolution(env)
        r = C.check_resolution_consistency(env, states)
    except ValueError as e:  # an env that renders at one size only says so
        pytest.skip(f"{name} cannot render at another size: {e}")
    if "skipped" in r:
        pytest.skip(f"{name}: {r['skipped']}")
    print(f"\n{name} 2S->S: exact {100 * r['exact_frac']:.3f}%  mean {r['mean_abs']:.4f}/255  max {r['max_abs']}/255")
    measured = DOWNSAMPLE_EXACT_FRAC.get(name)
    if measured is None:
        measured = DOWNSAMPLE_EXACT_FRAC.get(_recorded_name(name))
    assert measured is not None, (
        f"{name} has no measured downsample bound recorded. Run "
        f"stable_envs.testing.conformance.check_resolution_consistency on it and add the number to "
        f"DOWNSAMPLE_EXACT_FRAC with its provenance — a guessed bound is worse than no test, because "
        f"it either fails on a correct renderer or passes on a broken one"
    )
    assert r["exact_frac"] >= measured - DOWNSAMPLE_MARGIN, (
        f"{name}: only {100 * r['exact_frac']:.3f}% of pixels survive a 2x render-and-box-downsample "
        f"against a measured {100 * measured:.3f}% ({MEASURED_ON}). Anti-aliasing does not move this "
        f"number by half a point; a shape landing in a different place does: {r}"
    )


# ------------------------------------------------------------------ 4. input forms


def test_every_input_form_gives_the_same_frame(env, inputs):
    """numpy or torch, CPU or CUDA, float32 or float64, contiguous or strided — one answer.

    Every form is a call site somebody has. The strided ones are the reason this test exists rather
    than being folded into the others: ``frames[:, t]`` is how a rollout indexes time, it is a view
    with a gap between rows, and it has already produced a crash in this repo. float64 is exercised
    with values that were generated in float32 and widened, so the round trip is exact by construction
    and any difference is the code path rather than the arithmetic.
    """
    states, style = inputs
    r = C.check_input_form_invariance(env, states, style=style)
    assert not r["disagreements"], (
        f"{type(env).__name__}: these input forms did not give the frame that plain float32 numpy "
        f"gives: {r['disagreements']} — a caller whose states arrive as a CUDA tensor, as float64 out "
        f"of parquet, or as a strided slice of a trajectory buffer gets different pixels"
    )


def test_leading_dimensions_are_preserved(env, inputs):
    """``(n_traj, T, D)`` in, ``(n_traj, T, H, W, 3)`` out, with the frames in the right cells.

    ``render_states`` advertises this and a corpus is exactly this shape. A reshape that transposes
    time into batch produces a frame stack of the right shape and total size, containing the right
    frames in the wrong places, and no shape assertion anywhere would notice.
    """
    states, style = inputs
    flat = env.render(states, style=style).detach().cpu().numpy()
    nested = env.render(states.reshape(2, N // 2, -1), style=style).detach().cpu().numpy()
    assert nested.shape == (2, N // 2, *flat.shape[1:]), (
        f"{type(env).__name__}: leading dims not preserved — got {nested.shape}"
    )
    assert np.array_equal(nested.reshape(flat.shape), flat), (
        f"{type(env).__name__}: the nested render holds the same frames in different cells, i.e. the "
        f"leading dimensions were flattened in one order and restored in another"
    )


def test_out_buffer_receives_the_same_frames(env, inputs):
    """``render(..., out=buf)`` is the documented hot-loop idiom; it must fill ``buf`` with THE frames.

    An implementation that allocates internally and returns the new tensor passes every pixel check in
    this repo while leaving the caller's buffer untouched — and the caller, having been told the
    allocation is theirs, reads the buffer.
    """
    states, style = inputs
    ref = env.render(states, style=style).detach().cpu().numpy()
    buf = torch.zeros(ref.shape, dtype=torch.uint8, device=env.device)
    got = env.render(states, style=style, out=buf)
    assert np.array_equal(buf.detach().cpu().numpy(), ref), (
        f"{type(env).__name__}: render(out=buf) did not put the frames in buf — the documented "
        f"zero-allocation path hands back a buffer that is not the picture"
    )
    assert np.array_equal(got.detach().cpu().numpy().reshape(ref.shape), ref), (
        f"{type(env).__name__}: render(out=buf) returned something other than what it wrote"
    )


def test_a_non_contiguous_out_buffer_is_refused_rather_than_silently_corrupted(env, inputs):
    """``out=`` must write THE frames or refuse. It used to do neither.

    ``Tensor.view`` does not check contiguity when the shape it is given is the shape the tensor
    already has — it returns the strided tensor unchanged. Every backend here reaches its destination
    that way, so the kernel wrote ``B*H*W*3`` bytes linearly from the view's own offset and the frames
    landed across the neighbouring slices of the caller's buffer. Measured on PushT before the fix:
    ``env.render(st, out=buf[:, 1])`` raised nothing and left ``buf[:, 1]`` 255/255 away from the
    frames. A silent wrong answer is the worst of the three possible behaviours, and it was reachable
    from the idiom the README recommends as soon as the buffer has a time or camera axis in front of
    the frame axis — which is the ordinary shape for a rollout.

    Refusal is now central rather than per-family: ``StableEnv.__init_subclass__`` wraps every env's
    ``render``, so this holds for the ones that never touch the 2-D rasterizer too. MiniGrid used to be
    the only family that checked, which is what a per-env rule always decays to.
    """
    states, style = inputs
    ref = env.render(states[:1], style=None if style is None else style[:1])
    H, W = int(ref.shape[-3]), int(ref.shape[-2])
    n = min(4, N)
    buf = torch.zeros((n, 3, H, W, 3), dtype=torch.uint8, device=env.device)
    view = buf[:, 1]
    assert not view.is_contiguous(), "the probe buffer must be strided for this to mean anything"
    with pytest.raises((RuntimeError, ValueError), match="(?i)contiguous"):
        env.render(states[:n], style=None if style is None else style[:n], out=view)
    assert int(buf.max()) == 0, (
        f"{type(env).__name__}: render(out=<non-contiguous>) reported an error but had already "
        f"written into the caller's buffer — a refusal that scribbles first is not a refusal"
    )


# ------------------------------------------------------------------ 4b. the per-environment style contract


def test_rendering_a_sub_batch_refuses_environment_0_s_style(env, inputs):
    """A per-environment appearance must never be replaced by environment 0's when the batch shrinks.

    This is batch invariance for the APPEARANCE half of the state, and the existing batch-invariance
    checks cannot see it because they run under one shared style. Measured on PushT before the fix,
    with one random appearance per environment:

        env.set_state(st, style=sty); full = env.render()
        alone = env.render(st[3:4])                            # asked for row 3 only
        |alone - full[3]|.max()                        -> 229  # not row 3's appearance
        |alone - render(st[3:4], style=sty[0:1])|.max() ->   0  # it is row 0's

    ``render(st[3:4])`` hands the env a ``(1, state_dim)`` array; nothing in it says these are rows
    3..4 of the env's batch, so there is no appearance the env could correctly pick and every choice
    but refusing is a guess. The caller has the indices and passes them — which is what ``render_states``
    and both datasets already do, and why the corpus path never showed this.

    Asserted over the whole registry rather than on one env: the point of routing this through
    ``StableEnv.resolve_style`` is that no env can opt out, and only running all of them proves it.
    """
    if env.style_dim == 0:
        pytest.skip(f"{type(env).__name__} declares no style, so there is nothing to collapse")
    states, _ = inputs
    sty = C.uniform_style(env, N, groups=2)  # two distinct rows; see uniform_style on the cost
    env.set_state(states, style=sty)
    full = env.render().detach().cpu().numpy()

    with pytest.raises(ValueError, match="(?i)per-environment style"):
        env.render(states[3:4])

    alone = env.render(states[3:4], style=sty[3:4]).detach().cpu().numpy()[0]
    d = int(np.abs(alone.astype(np.int16) - full[3].astype(np.int16)).max())
    assert d == 0, (
        f"{type(env).__name__}: rendering row 3 on its own WITH row 3's style gave a frame {d}/255 "
        f"away from row 3 of the full batch — the indexed call is the one the error message tells "
        f"callers to make, so it has to be the right answer"
    )


def test_a_shared_style_broadcasts_to_any_batch(env, inputs):
    """The ergonomic half of the same rule: one appearance for everybody still works at any batch.

    Refusing a per-environment style at the wrong batch is only correct if the common case stays
    free — and the common case is an env at its default appearance, which stores B identical rows
    internally. Row 3 IS row 0 there, so broadcasting is not a collapse and must not raise. Both the
    stored-default path and an explicitly passed single row are checked, because they reach the rule
    from different sides.
    """
    states, _ = inputs
    env.set_style(None)
    ref = env.render(states).detach().cpu().numpy()
    for b in (1, 3, N):
        got = env.render(states[:b]).detach().cpu().numpy()
        assert np.array_equal(got, ref[:b]), (
            f"{type(env).__name__}: under the default appearance, rendering {b} of {N} rows differs "
            f"from those rows of the full batch — a shared style must broadcast unchanged"
        )
    if env.style_dim:
        one = env.sample_style(1, rng=np.random.default_rng(3))
        a = env.render(states, style=one).detach().cpu().numpy()
        b_ = env.render(states[:1], style=one).detach().cpu().numpy()[0]
        assert np.array_equal(b_, a[0]), (
            f"{type(env).__name__}: a single explicit style row did not broadcast identically at two batch sizes"
        )


def test_variation_overrides_do_not_drop_the_env_s_per_environment_style(env, inputs):
    """``render(variation={...})`` must keep the appearance the env is carrying for the other knobs.

    The second collapse inside the same function, and the one no test could have caught: the
    ``variation=`` branch of ``resolve_style`` rebuilt the style from ``self._style.values[:1]``
    unconditionally, so it substituted environment 0's appearance even when the batch MATCHED. Every
    variation test in this suite runs under the default appearance, where row 0 and row 3 are equal,
    which is exactly why it survived.
    """
    if env.style_dim == 0:
        pytest.skip(f"{type(env).__name__} declares no style")
    chosen = _continuous_leaf(env)
    if chosen is None:
        pytest.skip(f"{type(env).__name__} declares no continuous variation leaf")
    path, off, w, v = chosen
    states, _ = inputs
    sty = C.uniform_style(env, N, groups=2)  # two distinct rows; see uniform_style on the cost
    env.set_state(states, style=sty)

    want = np.array(sty, np.float32)
    want[:, off : off + w] = v
    a = env.render(states, variation={path: v}).detach().cpu().numpy()
    b = env.render(states, style=want).detach().cpu().numpy()
    assert np.array_equal(a, b), (
        f"{type(env).__name__}: with a per-environment style set on the env, "
        f"render(variation={{{path!r}: v}}) drew something other than the same rows with that one "
        f"column replaced (max {int(np.abs(a.astype(np.int16) - b.astype(np.int16)).max())}/255) — the "
        f"columns the override does not touch came from environment 0 instead of from each "
        f"environment's own row"
    )


# ------------------------------------------------------------------ 5. purity, idempotence, ordering


def test_render_is_pure_and_set_state_is_idempotent(env, inputs):
    """Rendering changes nothing, and restoring a state twice is the same as once.

    Both halves are the same claim from opposite sides, and both are silent when broken. A render that
    writes back a normalised state makes the second render of the same input differ from the first; a
    ``set_state`` that is not idempotent makes a dataset's second epoch differ from its first.
    """
    states, style = inputs
    r = C.check_render_is_pure(env, states, style=style)
    assert r["state_changed_by_render"] == 0.0, (
        f"{type(env).__name__}: render() mutated the env's state by {r['state_changed_by_render']} — "
        f"rendering is a read"
    )
    assert r["style_changed_by_render"] == 0.0, (
        f"{type(env).__name__}: render() mutated the env's style by {r['style_changed_by_render']}"
    )
    assert r["state_changed_by_double_set"] == 0.0, (
        f"{type(env).__name__}: set_state(get_state()) twice changed the state by "
        f"{r['state_changed_by_double_set']} — restoring a stored state is not idempotent"
    )
    assert r["frame_changed_by_double_set"] == 0, (
        f"{type(env).__name__}: set_state(get_state()) twice changed the FRAME by "
        f"{r['frame_changed_by_double_set']}/255 while leaving the state alone, so the frame depends "
        f"on something set_state carries besides the state"
    )


def test_set_state_copies_its_input(env):
    """``set_state`` must copy, and must write THROUGH the state buffer rather than rebind it.

    Two failures, both silent. A caller who keeps the array they passed and then reuses it would be
    mutating the env; and PushT captures its whole control step into a CUDA graph that reads the state
    buffer at a fixed address, so a ``set_state`` that rebound the attribute would leave the graph
    replaying against dead memory — every ``set_state`` silently ignored, the env apparently frozen.
    """
    r = C.check_set_state_copies(env)
    assert r["aliased_caller_buffer"] == 0.0, (
        f"{type(env).__name__}: zeroing the array that was handed to set_state changed the env's own "
        f"state by {r['aliased_caller_buffer']} — the env aliases the caller's memory"
    )
    assert not r["state_tensor_rebound"], (
        f"{type(env).__name__}: set_state REBOUND the state tensor instead of writing through it. Any "
        f"CUDA graph capturing this env's step reads the old address, so every later set_state would "
        f"be silently ignored"
    )


def test_batch_order_does_not_change_the_frames(env, inputs):
    """``render(x[perm]) == render(x)[perm]``, and a state appearing twice renders twice the same.

    A rasterizer reading geometry with the wrong batch stride, or a renderer that splits the batch
    into groups and scatters the results back through a stale index, is invisible while every
    environment in the batch is drawing the same picture. Run with two distinct style groups so the
    grouped path is exercised — two, not eight, because the plate-baking families pay a MuJoCo render
    per distinct static appearance.
    """
    states, _ = inputs
    style = C.uniform_style(env, N, groups=2)
    r = C.check_order_independence(env, states, style=style)
    assert r["permutation_max_abs_diff"] == 0, (
        f"{type(env).__name__}: permuting the batch did not permute the frames (max "
        f"{r['permutation_max_abs_diff']}/255) — one environment's pixels depend on its neighbours, "
        f"so a shuffled loader renders a different dataset"
    )
    assert r["duplicate_max_abs_diff"] == 0, (
        f"{type(env).__name__}: the same state twice in one batch rendered two different frames (max "
        f"{r['duplicate_max_abs_diff']}/255) — the output buffer is not fully written, or a slot "
        f"carries something from its position"
    )


def test_render_size_does_not_leak_into_later_calls(env, inputs):
    """``render(size=X)`` must not change what a later ``render()`` does.

    ``size`` is an argument, not a mode. An env that remembers the last requested size carries hidden
    state across calls, and the consequence is not cosmetic: a caller who renders one preview frame at
    a different resolution silently changes the resolution of the corpus they generate afterwards,
    while ``env.size`` still reports the original.

    This was a live defect in the highway family — ``render`` called ``_set_screen(size)``, which
    rebound ``self.H``/``self.W`` and the camera — and it is asserted here for all four of those envs
    rather than excused, because the screen is now derived inside the call and thrown away with it.
    """
    name = type(env).__name__
    if hasattr(type(env), "PLATE_PATHS") and name not in PLATE_REPRESENTATIVES:
        # A new output size rebuilds this family's whole reference scene and its MuJoCo renderer, so
        # covering all 25 cells buys ~22 more live EGL contexts for one shared code path. The
        # representatives already request 2S for the resolution check, so this reuses that scene and
        # costs nothing extra.
        pytest.skip(f"{name}: covered by {sorted(PLATE_REPRESENTATIVES)} — see PLATE_REPRESENTATIVES")
    states, style = inputs
    before = env.render(states, style=style).shape
    try:
        env.render(states[:1], size=2 * int(env.size), style=None if style is None else style[:1])
    except ValueError as e:
        pytest.skip(f"{name} renders at one size only, so no size can leak: {e}")
    after = env.render(states, style=style).shape
    # Put the env back before asserting. The module-scoped env is shared, so leaving a leaked size in
    # place turns one real defect into a cascade of unrelated failures in every later test that
    # compares a plain render against a sized one — which is exactly what it did before this line
    # existed: four dataset round-trip failures whose actual cause was three tests earlier.
    env.render(states[:1], size=int(env.size), style=None if style is None else style[:1])
    assert tuple(after) == tuple(before), (
        f"{name}: after one render(size={2 * int(env.size)}), a plain render() returns {tuple(after)} "
        f"instead of {tuple(before)} while env.size is still {env.size}. The size argument is being "
        f"stored on the env, so a single differently-sized call rewrites every later frame"
    )


@pytest.mark.parametrize("name", ["Highway", "Merge", "Roundabout", "Intersection"])
def test_highway_render_size_is_an_argument_not_a_mode(name):
    """The reproduction that was recorded as HIGHWAY_SIZE_LEAK, asserted verbatim.

    Kept beside the registry-wide check rather than folded into it because this family is where the
    defect lived and the shape of it is specific: the screen — height, width, camera offset, stripe
    count, line width — was derived from the requested size and STORED on the env by ``_set_screen``,
    so it survived the call that asked for it. Measured before the fix:

        env.render(st).shape       -> (4, 150, 600, 3)
        env.render(st, size=300)
        env.render(st).shape       -> (4, 300, 1200, 3)      # env.size is still 150

    A fresh env per parameter, so this states the property on its own and cannot be satisfied (or
    broken) by whatever the module-scoped fixture did before it.
    """
    if name not in se.registry.REGISTRY:
        pytest.skip(f"{name} is not registered")
    env = se.make(name, num_envs=4, device="cuda")
    st = env.sample_states(4, rng=np.random.default_rng(0))
    before = tuple(env.render(st).shape)
    big = tuple(env.render(st, size=2 * int(env.size)).shape)
    after = tuple(env.render(st).shape)
    assert big == (before[0], 2 * before[1], 2 * before[2], 3), (
        f"{name}: render(size=2S) returned {big}, which is not twice {before}"
    )
    assert after == before, (
        f"{name}: a plain render() returns {after} after one render(size={2 * int(env.size)}), against "
        f"{before} before it — the requested size outlived the call that requested it"
    )
    assert (int(env.size), int(env.H), int(env.W)) == (before[1], before[1], before[2]), (
        f"{name}: env.size/H/W report ({env.size}, {env.H}, {env.W}) but the frame is {before} — the "
        f"env's advertised geometry and the geometry it renders at have come apart"
    )


# ------------------------------------------------------------------ 6. serialization


def test_numpy_and_torch_serialization_round_trip(env, inputs, tmp_path):
    """States and style through ``np.save``/``np.load`` and ``torch.save``/``torch.load``, byte-exact.

    The cheapest possible corpus, and the one most people will actually write. float32 through either
    format is exact, so a difference here is a dtype being widened or narrowed on the way out — which
    a shape check would not catch and which moves pixels.
    """
    states, style = inputs
    ref = _digest(env.render(states, style=style))

    np.save(tmp_path / "s.npy", states)
    ns = np.load(tmp_path / "s.npy")
    nk = None
    if style is not None:
        np.save(tmp_path / "k.npy", style)
        nk = np.load(tmp_path / "k.npy")
    assert _digest(env.render(ns, style=nk)) == ref, (
        f"{type(env).__name__}: frames differ after a numpy save/load of the states"
    )

    torch.save(
        {"s": torch.as_tensor(states), "k": None if style is None else torch.as_tensor(style)}, tmp_path / "t.pt"
    )
    got = torch.load(tmp_path / "t.pt", weights_only=True)
    assert _digest(env.render(got["s"], style=got["k"])) == ref, (
        f"{type(env).__name__}: frames differ after a torch save/load of the states"
    )


def test_dataset_paths_regenerate_identical_frames(env, inputs):
    """``StateDataset``, its ``DataLoader`` and ``TrajectoryDataset.all_frames`` all give THE frames.

    These are the three surfaces a user actually consumes, and each adds machinery between the states
    and the kernel: a batched ``__getitems__``, a collate function that must be the identity on an
    already-assembled batch, a per-trajectory style that has to be expanded with ``repeat_interleave``
    rather than ``repeat``. That last one is not hypothetical — ``repeat`` gives trajectory j's
    appearance to trajectory i's states and renders without complaint.
    """
    states, style = inputs
    ref = env.render(states, style=style).detach().cpu().numpy()

    ds = StateDataset(env, states, style=style, size=env.size, device=str(env.device))
    assert np.array_equal(ds.__getitems__(list(range(N))).detach().cpu().numpy(), ref), (
        f"{type(env).__name__}: StateDataset renders different pixels from env.render"
    )

    # The caller's own DataLoader; ``ds.loader()`` was dropped. ``collate_batched`` is the one argument
    # a rendering dataset needs, because ``__getitems__`` hands back an assembled batch.
    from torch.utils.data import DataLoader

    loaded = next(iter(DataLoader(ds, batch_size=N, shuffle=False, num_workers=0, collate_fn=collate_batched)))
    assert np.array_equal(loaded.detach().cpu().numpy(), ref), (
        f"{type(env).__name__}: the DataLoader path renders different pixels from env.render"
    )

    # one trajectory of N steps, so the per-trajectory style has to be expanded over its own rows
    traj = TrajectoryDataset(
        env,
        states.reshape(1, N, -1),
        style=None if style is None else style[:1],
        window=1,
        size=env.size,
        device=str(env.device),
    )
    got = traj.all_frames().detach().cpu().numpy()
    want = env.render(states, style=None if style is None else np.repeat(style[:1], N, 0))
    assert np.array_equal(got, want.detach().cpu().numpy()), (
        f"{type(env).__name__}: TrajectoryDataset.all_frames does not reproduce env.render on the "
        f"same states and the trajectory's own style"
    )


def test_huggingface_round_trip_regenerates_identical_frames(env, inputs, tmp_path):
    """A corpus saved and reloaded must re-render bit-identically — states AND style.

    The assertion the README makes in prose. It is worth its own test because the failure is
    asymmetric: the numbers survive every serialization anyone checks, and the *appearance* is the
    half that quietly does not, leaving a reloaded corpus that renders a differently-coloured world
    and calls it reproduction.
    """
    pytest.importorskip("datasets", reason="the HuggingFace round-trip needs `datasets`")
    states, style = inputs
    ds = TrajectoryDataset(
        env,
        states.reshape(1, N, -1),
        style=None if style is None else style[:1],
        window=1,
        size=env.size,
        device=str(env.device),
    )
    ref = _digest(ds.all_frames())

    back = TrajectoryDataset.from_hf(ds.to_hf(), env=env, window=1, device=str(env.device))
    assert _digest(back.all_frames()) == ref, (
        f"{type(env).__name__}: an in-memory HuggingFace round-trip does not re-render the same frames"
    )

    path = str(tmp_path / "corpus")
    ds.save_to_disk(path)
    disk = TrajectoryDataset.load_from_disk(path, env=env, window=1, device=str(env.device))
    assert _digest(disk.all_frames()) == ref, (
        f"{type(env).__name__}: a corpus written with save_to_disk and read back renders different "
        f"frames — the stored corpus is not the corpus"
    )


def test_variation_dict_and_style_tensor_agree(env, inputs):
    """``render(variation={path: v})`` and ``render(style=<row with that column set>)`` are one thing.

    Users write the dict; datasets store the flat row. If the two disagree, every appearance ever
    written down with a dict comes back as something else, and the flat layout — which is what makes
    style storable at all — is not describing the same space the dict addresses.
    """
    states, _ = inputs
    chosen = _continuous_leaf(env)
    if chosen is None:
        pytest.skip(f"{type(env).__name__} declares no continuous variation leaf")
    path, off, w, v = chosen

    sty = np.tile(env.default_style(), (N, 1))
    sty[:, off : off + w] = v
    a = env.render(states, variation={path: v}).detach().cpu().numpy()
    b = env.render(states, style=sty).detach().cpu().numpy()
    assert np.array_equal(a, b), (
        f"{type(env).__name__}: render(variation={{{path!r}: v}}) and the equivalent flat style row "
        f"draw different pictures (max "
        f"{int(np.abs(a.astype(np.int16) - b.astype(np.int16)).max())}/255) — the dict API and the "
        f"stored form address different spaces"
    )


# ------------------------------------------------------------------ 7. CUDA graphs


def test_cuda_graph_step_matches_eager(env):
    """Where a step is captured into a CUDA graph, the graph and the eager path must agree exactly.

    A graph replays the same kernels on the same buffers, so anything other than bit-identical is a
    capture that did not close over what it was supposed to. The second assertion is the one that has
    teeth: a graph reads its inputs at FIXED ADDRESSES, so ``set_state`` after capture is only visible
    if the state buffer is written through rather than rebound. Get that wrong and the env silently
    ignores every state you give it while still producing plausible trajectories.
    """
    cls = type(env)
    if "step" not in cls.__dict__ or not getattr(cls, "use_step_graph", False):
        pytest.skip(f"{cls.__name__}: no CUDA-graph-captured step")
    graphed = se.make(cls.__name__, num_envs=N, device="cuda")
    eager = se.make(cls.__name__, num_envs=N, device="cuda")
    eager.use_step_graph = False

    st = env.sample_states(N, rng=np.random.default_rng(5))
    graphed.set_state(st)
    eager.set_state(st)
    rng = np.random.default_rng(6)
    adim = int(getattr(cls, "action_dim", 1))
    lo, hi = float(getattr(cls, "action_low", -1.0)), float(getattr(cls, "action_high", 1.0))
    worst = 0.0
    for _ in range(8):
        a = torch.as_tensor(rng.uniform(lo, hi, (N, adim)).astype(np.float32), device="cuda")
        # ``step`` is stateful and, for a migrated cell, returns a StepResult TensorDict
        # rather than the packed buffer. ``get_state()`` is the packed row on every path and is what
        # this comparison has always been about.
        graphed.step(a)
        eager.step(a)
        sg = graphed.get_state().detach().cpu().numpy().copy()
        se_ = eager.get_state().detach().cpu().numpy().copy()
        worst = max(worst, float(np.abs(sg - se_).max()))
    if getattr(graphed, "_step_graph", None) is None:
        pytest.skip(
            f"{cls.__name__}: graph capture did not happen "
            f"({getattr(graphed, '_step_graph_error', 'no reason recorded')})"
        )
    assert worst == 0.0, (
        f"{cls.__name__}: the CUDA-graph step and the eager step disagree by {worst} — a replayed "
        f"graph runs the same kernels on the same memory, so a difference means the capture did not "
        f"close over the same inputs"
    )

    # set_state must still reach a captured graph
    st2 = env.sample_states(N, rng=np.random.default_rng(77))
    graphed.set_state(st2)
    got = graphed.get_state().detach().cpu().numpy()
    assert np.abs(got - st2).max() == 0.0, (
        f"{cls.__name__}: set_state after graph capture did not take effect — the graph is reading a "
        f"buffer the setter no longer writes"
    )
    eager.set_state(st2)
    a = torch.as_tensor(rng.uniform(lo, hi, (N, adim)).astype(np.float32), device="cuda")
    graphed.step(a)
    eager.step(a)
    assert (
        np.abs(graphed.get_state().detach().cpu().numpy() - eager.get_state().detach().cpu().numpy()).max() == 0.0
    ), (
        f"{cls.__name__}: after set_state, the graphed step diverges from the eager one — the replay "
        f"is running on stale state"
    )


# ------------------------------------------------------------------ 8. the node itself


@pytest.mark.reference
def test_the_reference_renderer_on_this_machine_is_deterministic(env):
    """Before blaming the code: does THIS machine's offscreen rasterizer return the same pixels twice?

    Not decoration. Every EGL incident in this repo's history was first read as a state-sufficiency
    violation or an exhausted context budget, and at least one was neither — the node's own rasterizer
    was returning different pixels for the same scene, 25 envs failed in a row, and the interpreter
    then aborted with SIGABRT, which no ``try``/``except`` catches and which takes the whole suite down
    instead of one test. Two frames answer it, and the answer names the machine instead of the code.
    """
    cls = type(env)
    if not hasattr(cls, "reference_env") or env.reference_id is None:
        pytest.skip(f"{cls.__name__}: no official reference renderer")
    try:
        r = C.check_reference_renderer_is_stable(cls, env.size)
    except ImportError as e:
        pytest.skip(f"{cls.__name__}: reference env not installed ({e})")
    assert r["max_abs_diff"] == 0, (
        f"{cls.__name__}: the OFFICIAL renderer gave two different frames for the same states on this "
        f"machine (max {r['max_abs_diff']}/255), before any of our code was involved. This is a node "
        f"fault, not a repo fault — rerun elsewhere and exclude this host; every parity number "
        f"measured here is meaningless"
    )


# ------------------------------------------------------------------ 9. what a SCENE is a function of


def _visibility_capable_envs():
    """Envs whose RENDERER can carry a visibility knob, resolved without building anything.

    ``VISIBILITY_KNOB`` is a statement about the renderer, not about the scene: every fixed-camera DM
    Control cell sets it, and only the ones whose ``initialize_episode`` actually DRAWS a task object
    end up declaring a ``visibility.*`` leaf. The two are different questions and this is deliberately
    the cheap one -- answering the other needs each domain's compiled model, and building those at
    COLLECTION time would create an EGL context per domain for a run that may deselect every one of
    them. The test body asks the precise question once it has the env.
    """
    return [
        name
        for name, cls in sorted(se.registry.REGISTRY.items())
        if getattr(cls, "VISIBILITY_KNOB", False) and hasattr(cls, "_hideable")
    ]


def _declared_visibility_leaves(names) -> dict:
    """``{env name: [visibility leaf, ...]}`` for ``names``, measured from each class's own census.

    Pays for the compiled models, so it is called from a test body and never at collection.
    """
    out = {}
    for name in names:
        cls = se.registry.REGISTRY[name]
        try:
            out[name] = [f"visibility.{el[1]}" for el in cls._hideable()]
        except ImportError:  # the family is not installed here
            continue
    return out


def test_some_env_actually_declares_a_hideable_task_object():
    """The zero-coverage rule for the check below, which SKIPS per cell rather than failing.

    ``test_a_baked_scene_does_not_inherit_the_last_bake_s_visibility`` is swept over every env whose
    renderer can carry the knob -- 11 DM Control cells -- and seven of them draw no task object per
    episode, so they declare no leaf and there is nothing for it to hide. A per-cell skip is the right
    answer for those (it is what ``test_backend_agreement`` does for an env with no ``Scene2D``), and
    it is also how a sweep goes quietly vacuous: if the census ever emptied, all eleven would skip and
    the run would be green having asserted nothing. This is the line that fails instead.
    """
    names = _visibility_capable_envs()
    if not names:
        errs = se.import_errors()
        if "dmcontrol" in errs:
            pytest.skip(f"no env can carry a visibility knob here: dmcontrol did not import ({errs['dmcontrol']})")
        pytest.fail("no registered env sets VISIBILITY_KNOB, and every env module imported cleanly")
    declared = _declared_visibility_leaves(names)
    if not declared:
        pytest.skip(
            "the visibility-capable envs are registered but their models are not reachable in this interpreter"
        )
    with_leaf = {n: v for n, v in declared.items() if v}
    assert with_leaf, (
        f"{len(declared)} env(s) advertise VISIBILITY_KNOB and NOT ONE declares a visibility.* leaf, "
        f"so the scene-cache check below skips on every one of them and asserts nothing. Either "
        f"_episode_slots has stopped finding the per-episode draw or _hideable has stopped turning it "
        f"into a leaf; both are silent, and both make this file green: {declared}"
    )
    print(f"\nvisibility leaves declared by {len(with_leaf)} of {len(declared)} capable env(s): {with_leaf}")


@pytest.mark.parametrize("name", family_params(_visibility_capable_envs(), family="visibility"))
def test_a_baked_scene_does_not_inherit_the_last_bake_s_visibility(name):
    """A cached scene must be a function of ITS OWN appearance, not of whichever was baked last.

    The eighth entry in the defect list above, found in review of the ``visibility.*`` knob and fixed
    in the same branch. ``DMControlSuite`` caches three baked plates per appearance and casts the
    STATIC depth lazily, per (scene, supersample), because the supersample is a render-time argument.
    ``_static_t`` derived the static primitive set at that moment from the LIVE shared ``MjModel`` —
    and the live model carries whatever the LAST bake applied. So a scene baked with an element
    hidden, whose static depth was first cast after some other appearance had been baked, saw the
    element visible again, failed to find it in ``sc.movers``, and promoted it to a STATIC primitive
    at whatever ``geom_pos`` the last ``_fk`` had written.

    Measured on CPU over the 64 frozen Step-4 states before the fix: 18 px / max 215 / 3 rows on
    ``DMCReacherHard`` and 88 px / max 140 / 28 rows on ``DMCFingerTurnHard``.

    THE SUPERSAMPLE CHANGE IS THE TRIGGER and is why nothing else caught it: ``static_t`` is cached
    per (scene, ss), and at an env's default supersample it is always cast during that scene's own
    first render, while that scene's appearance is still on the model. Every check in this repository
    renders at the default. So the sequence below is the cheapest one that can fail — and the scene
    cache is cleared between the two halves because it is process-global and a previous test may
    already hold the entry this is about.
    """
    resolve_family(name, family="visibility", modules=("dm_control",), subject="the scene-cache visibility check")
    env = make_or_skip(name, num_envs=8, device="cuda")
    paths = [p for p in env.variation_space.names() if p.split(".")[0] == "visibility"]
    if not paths:
        # A property of the DOMAIN, not a defect: this cell's initialize_episode draws no task object
        # (measured by _episode_slots), so _hideable is empty and there is nothing to hide. Seven of
        # the eleven fixed-camera DM Control cells are in this position. The sweep is kept over all of
        # them so that a cell which STARTS drawing one is covered the day it does; that the family is
        # not entirely leaf-less is asserted by
        # test_some_env_actually_declares_a_hideable_task_object above.
        pytest.skip(
            f"{name} draws no task object per episode, so it declares no visibility.* leaf "
            f"and has nothing for this check to hide"
        )
    hidden = {p: 0 for p in paths}
    other = {p: [0.2, 0.9, 0.4] for p in env.variation_space.names() if p.split(".")[0] == "material"}
    assert other, f"{name} has no second appearance to bake in between"

    from stable_envs.envs import dmcontrol as D

    st = sampled_states(env, 8, seed=0)
    ss = env.ss + 1  # a supersample nothing has cached for this scene

    D._SCENE_CACHE.clear()
    alone = env.render(st, variation=hidden, supersample=ss).cpu().numpy()

    D._SCENE_CACHE.clear()
    env.render(st, variation=hidden)  # bake the hidden scene at the default ss
    env.render(st, variation=other)  # ... then bake a DIFFERENT appearance
    after = env.render(st, variation=hidden, supersample=ss).cpu().numpy()

    d = np.abs(alone.astype(np.int16) - after.astype(np.int16)).max(-1)
    assert int(d.max()) == 0, (
        f"{name}: the same hidden-target scene rendered at supersample {ss} differs by "
        f"{int(d.max())}/255 over {int((d > 0).sum())} pixel(s) in "
        f"{int((d.reshape(len(st), -1) > 0).any(1).sum())} of {len(st)} frames depending on whether "
        f"another appearance was baked first. A cached scene is reading the LIVE model instead of "
        f"what it was baked from; see _bake's `sc.statics`"
    )
