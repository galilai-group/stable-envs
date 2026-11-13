"""Determinism and reproducibility of rendered frames across the registry.

Each test checks one way stored states can fail to regenerate the same frames: a fresh process, seeded
sampling, batch size and chunking, resolution, input form, render purity, batch order, serialization,
and steps from a restored State.

Batches are 8 except where batch size is the subject, and no test uses more than two distinct static
style values, because plate-baking families re-render a MuJoCo plate per distinct appearance.
"""

from __future__ import annotations

import fnmatch
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

import _replay_state as RS
from _shared import family_params, make_or_skip, resolve_family, sampled_states

import stable_envs as se
from stable_envs.testing import _sampling as S
from stable_envs.testing import conformance as C
from stable_envs.testing._data import StateDataset, TrajectoryDataset, collate_batched
from tests._runner import render_worker as W
from tests.helpers.reference.certification import has_reference_renderer


def _recorded_name(name: str) -> str:
    """The name ``DOWNSAMPLE_EXACT_FRAC`` is keyed by for ``name``."""
    from tests.helpers.reference import frozen_from_canonical

    return frozen_from_canonical(name)


ENV_NAMES = sorted(se.registry.REGISTRY)

#: States per env.
N = 8

#: Batch sizes the batch-invariance check sweeps. Odd sizes catch masked-tail kernels; batched matmuls
#: can pick a different cuBLAS kernel (and summation order) per batch count.
BATCH_SIZES = (1, 3, 7, 8, 64)

#: Peak frame bytes one test may materialize; the large-batch check picks the biggest size from
#: ``BIG_BATCHES`` that fits.
BIG_FRAME_BUDGET = 1 << 30
BIG_BATCHES = (1024, 512, 256, 128, 64)


def _continuous_leaf(env):
    """The first continuous variation leaf, as ``(path, offset, width, midpoint)``, or None.

    The midpoint is a value inside the space that differs from the default.
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


@pytest.fixture(scope="module", params=ENV_NAMES)
def env(request):
    return se.make(request.param, num_envs=N, device="cuda")


@pytest.fixture(scope="module")
def inputs(env):
    """The (states, style) pair, drawn by ``render_worker.sample_inputs`` as the subprocess draws it.

    For an env without render rows the states are full States, which carry their Variation, so style is None.
    """
    states, style = W.sample_inputs(env, N)
    return states, (style if env.style_dim and S.has_render_rows(env) else None)


@pytest.fixture(scope="session")
def fresh_process(request, tmp_path_factory):
    """Render every collected env once in a single new interpreter via ``tests._runner.render_worker``.

    Only envs pytest collected are passed, so ``-k`` narrows the subprocess too.
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
        "tests._runner.render_worker",
        "--out-npz",
        str(npz),
        "--out-json",
        str(js),
        "--n",
        str(N),
        "--envs",
        ",".join(selected),
    ]
    # Output is streamed and each finished env is emitted as an event, so the supervisor's inactivity
    # watchdog can tell a long setup from a hung one.
    try:
        from tests._runner.safety_plugin import _emit
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
        f"{name} is registered in this process but the fresh process did not report it; the two "
        f"interpreters disagree on the registry"
    )
    if "error" in rec:
        pytest.fail(f"{name} could not be built or rendered in a fresh process, while it builds here:\n{rec['error']}")
    return rec


def _digest(x) -> str:
    return S.digest(x)


def _rows_only(env, why: str) -> None:
    """Skip an env without render rows, for a check of the render-row interface only."""
    if not S.has_render_rows(env):
        pytest.skip(f"{type(env).__name__} has no render rows: {why}")


def test_seeded_sampling_reproduces_in_a_fresh_process(env, fresh_process):
    """``sample_states(rng=)`` and ``sample_style(rng=)`` give the same bytes in a new interpreter.

    Kept separate from the frame check so a sampler fault is not reported as a renderer fault.
    """
    rec = _record(fresh_process, type(env).__name__)
    states, style = W.sample_inputs(env, N)
    assert _digest(states) == rec["states_sha"], (
        f"{type(env).__name__}.sample_states(rng=default_rng({W.STATE_SEED})) gave different numbers "
        f"in this process than in a fresh one; the seeded sampler reads something other than its "
        f"generator (a global RNG, the env's own advanced state, a hash seed)"
    )
    assert _digest(style) == rec["style_sha"], (
        f"{type(env).__name__}.sample_style(rng=default_rng({W.STYLE_SEED})) is not reproducible across processes"
    )


def test_frames_are_bit_identical_in_a_fresh_process(env, fresh_process):
    """The same states and style render identical bytes in a new interpreter.

    The states come from the subprocess's own npz, so this is independent of the seeded sampler. It
    catches process-accumulated state such as mutable cache keys, autotuner choices or dict order.
    """
    name = type(env).__name__
    rec = _record(fresh_process, name)
    _, arrays = fresh_process
    if not S.has_render_rows(env):
        states, styled = (
            S.state_from_arrays(
                {k.split("::", 2)[2]: arrays[k] for k in arrays.files if k.startswith(f"{name}::{part}::")}, env.device
            )
            for part in ("states", "styled")
        )
        frames = env.render(states)
        assert list(frames.shape) == rec["shape"], (
            f"{name}: frame shape {list(frames.shape)} here, {rec['shape']} in a fresh process"
        )
        assert _digest(frames) == rec["frames_sha"], (
            f"{name}: rendering the same States in a fresh process gave different pixels; the frame "
            f"depends on something this interpreter accumulated"
        )
        assert _digest(env.render(styled)) == rec["frames_styled_sha"], (
            f"{name}: frames agree under the default Variation but not under a sampled one across processes"
        )
        return
    states = arrays[f"{name}::states"]
    style = arrays[f"{name}::style"]
    frames = env._render(states)
    assert list(frames.shape) == rec["shape"], (
        f"{name}: frame shape {list(frames.shape)} here, {rec['shape']} in a fresh process"
    )
    assert _digest(frames) == rec["frames_sha"], (
        f"{name}: rendering the same states in a fresh process gave different pixels; the frame "
        f"depends on something this interpreter accumulated"
    )
    if env.style_dim:
        assert _digest(env._render(states, style=style)) == rec["frames_styled_sha"], (
            f"{name}: frames agree under the default appearance but not under a stored style across processes"
        )


def test_rollout_seed_reproduces_in_a_fresh_process(env, fresh_process):
    """A seeded rollout's states and actions are identical in a new interpreter.

    States and actions are compared separately to tell a simulator fault from a policy-seeding fault.
    """
    from tests.helpers.rollout import sample

    name = type(env).__name__
    rec = _record(fresh_process, name)
    if "rollout_states_sha" not in rec:
        pytest.skip(f"{name}: no batched GPU simulator, so there is no rollout to seed")
    if S.has_render_rows(env):
        tr = sample(env, steps=W.ROLLOUT_STEPS, batch=N, seed=W.ROLLOUT_SEED)
    else:
        # A replay State keeps what earlier episodes left after a reset; the worker rolls out on a new env too.
        with RS.fresh(env) as other:
            tr = sample(other, steps=W.ROLLOUT_STEPS, batch=N, seed=W.ROLLOUT_SEED)
    assert _digest(tr["states"]) == rec["rollout_states_sha"], (
        f"{name} sample(seed={W.ROLLOUT_SEED}) produced a different trajectory in a fresh process; "
        f"the rollout reads randomness the seed does not reach"
    )
    if tr.get("actions") is not None and "rollout_actions_sha" in rec:
        assert _digest(tr["actions"]) == rec["rollout_actions_sha"], (
            f"{name}: the seeded policy chose different actions in a fresh process"
        )


def _big_batch(env) -> int:
    """The largest batch from ``BIG_BATCHES`` whose frames fit ``BIG_FRAME_BUDGET``."""
    probe = S.render_any(env, S.draw_states(env, 1, np.random.default_rng(0)))
    per = int(np.prod(probe.shape[1:]))
    for b in BIG_BATCHES:
        if b * per <= BIG_FRAME_BUDGET:
            return b
    return BIG_BATCHES[-1]


def test_frame_i_is_independent_of_batch_size(env):
    """Row ``i`` renders the same alone and in batches of 3, 7, 8, 64 and the largest that fits.

    Odd sizes catch masked-tail kernels and int32 offsets; batched matmuls may sum in a different
    order per batch count.
    """
    big = _big_batch(env)
    states = S.draw_states(env, big, np.random.default_rng(11))
    r = C.check_batch_size_invariance(env, states, sizes=BATCH_SIZES + (big,))
    assert r["max_abs_diff"] == 0, (
        f"{type(env).__name__}: the same state renders differently depending on how many other "
        f"environments share the launch (max {r['max_abs_diff']}/255, {r['worst_case']}): {r}"
    )


def test_chunked_render_states_is_chunk_independent(env, inputs):
    """``render_states(chunk=c)`` gives the same frames for every ``c``.

    Catches stale output views, style rows sliced with the wrong index, and unsynchronized
    host-to-device copies at chunk boundaries.
    """
    states, style = inputs
    if S.has_render_rows(env):
        r = C.check_chunked_render_matches(env, states, style=style, chunks=(N, 1, 7, 4096))
    else:
        r = RS.check_chunked_render_matches(env, states, chunks=(N, 1, 7, 4096))
    assert r["max_abs_diff"] == 0, (
        f"{type(env).__name__}: render_states gives different pixels at different chunk sizes "
        f"({r['worst_case']}, max {r['max_abs_diff']}/255): {r}"
    )


#: Measured per-env floors on the fraction of pixels where rendering at 2S and box-downsampling equals
#: rendering at S. The two are not expected to be equal (point sampling against area averaging differs
#: at every edge), but a geometry regression lowers ``exact_frac``. Recorded with
#: ``conformance.check_resolution_consistency`` over 8 sampled states per env.
MEASURED_ON = (
    "H200, driver 590.48.01, torch 2.11+cu130, triton 3.6, mujoco 3.9; 8 states per env "
    "from sample_states(rng=default_rng(render_worker.STATE_SEED)), default appearance"
)
#: Allowed drop below the measured floor, for edge pixels rounding differently on another GPU. Moving a
#: shape by one pixel costs whole percent.
DOWNSAMPLE_MARGIN = 0.005
#: Why the floors differ by family:
#:   0.966       PushT: pygame polygon and circle edges.
#:   0.708-0.726 OGBench mazes: MuJoCo re-bakes the plate at 2S with its own multisampling.
#:   0.282-0.730 DM Control: the same re-bake over a speckled skybox and a checker floor.
#: Not listed: ``DMCManipulatorBringBall`` (a plate cell covered by ``PLATE_REPRESENTATIVES``).
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
    "DMCCartpoleSwingup": 0.282065,  # mean 1.4428/255  see DM Control note below
    "DMCPointMassEasy": 0.730424,  # mean 0.2418/255
    #: DM Control cells drawn entirely through MJWarp (no plate), each measured on its own; the low
    #: floors come from the skybox and checker textures. Measured on an H200, mujoco 3.17.0 +
    #: mujoco-warp 3.17.0, MUJOCO_GL=egl.
    "DMCCheetahRun": 0.278734,  # mean 2.0166/255
    #: Higher than the others: the camera is close to the body and sees less skybox.
    "DMCQuadrupedWalk": 0.660084,  # mean 0.7828/255
    "DMCHopperHop": 0.284917,  # mean 1.7307/255
    "DMCHumanoidWalk": 0.369827,  # mean 1.5763/255
    "DMCSwimmerSwimmer6": 0.590916,  # mean 1.1604/255
    "DMCWalkerWalk": 0.443576,  # mean 1.7291/255
    "HumanoidmazeGiantNavigate": 0.712624,  # mean 1.1539/255
    "HumanoidmazeGiantStitch": 0.712624,  # mean 1.1539/255
    "HumanoidmazeLargeNavigate": 0.725895,  # mean 0.9690/255
    "HumanoidmazeLargeStitch": 0.725895,  # mean 0.9690/255
    "HumanoidmazeMediumNavigate": 0.723503,  # mean 0.6133/255
    "HumanoidmazeMediumStitch": 0.723503,  # mean 0.6133/255
    #: A baked MuJoCo plate with MJWarp movers; the plate is re-baked at 2S. Measured on an RTX 3090,
    #: mujoco 3.12.0 / ogbench 1.2.1, MUJOCO_GL=egl.
    "OGBCube": 0.885782,  # mean 1.0329/255
    "OGBScene": 0.805306,  # mean 2.0163/255
    #: nvdiffrast; mostly flat sky, so only the hull silhouette and pad texture edges differ. Measured on
    #: an RTX 3090, torch 2.14.0+cu126, nvdiffrast 0.4.0.
    "PFRocketLanding": 0.956137,  # mean 0.9923/255
    "PointmazeGiantNavigate": 0.708274,  # mean 1.1385/255
    "PointmazeGiantStitch": 0.708274,  # mean 1.1385/255
    "PointmazeLargeNavigate": 0.722781,  # mean 0.9528/255
    "PointmazeLargeStitch": 0.722781,  # mean 0.9528/255
    "PointmazeMediumNavigate": 0.723306,  # mean 0.6111/255
    "PointmazeMediumStitch": 0.723306,  # mean 0.6111/255
    "PointmazeTeleportNavigate": 0.717602,  # mean 1.1505/255
    "PointmazeTeleportStitch": 0.717602,  # mean 1.1505/255
    #: Full States from render_worker.sample_inputs; the frames are the source pygame frames on any
    #: device. Measured on an RTX A6000, pygame 2.6.1.
    "PushT": 0.966525,  # mean 0.3170/255
    "PushTDiscrete": 0.966416,  # mean 0.3168/255
}
#: Plate-baking envs the size-changing checks run on. Each new output size rebuilds the plate env and
#: its MuJoCo renderer, so one maze per locomotion body and the two DM Control camera geometries
#: stand in for their families.
PLATE_REPRESENTATIVES = frozenset(
    {
        "PointMazeMediumNavigate",
        "AntMazeMediumNavigate",
        "HumanoidMazeMediumNavigate",
        "DMCCartpoleSwingup",
        "DMCPointMassEasy",
    }
)


def _skip_if_size_is_not_resolution(env) -> None:
    """Skip envs where ``size`` re-frames the camera instead of re-sampling the same picture.

    An env whose world is in output pixels (highway-env) takes ``scene(size=)``; if the world rectangle
    differs between S and 2S, the 2S frame is a wider view, not a finer one. Probing ``scene`` mutates
    nothing on the env.
    """
    import inspect

    if not S.has_render_rows(env):
        return  # a full State renders through env.render(size=), which re-samples the same picture
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
    """``resolution_consistency_applicability`` narrows the judged states only for a parseable band.

    An env without a declaration is unaffected, and a band naming an unknown dimension narrows nothing.
    """
    names = ["x", "light_level"]
    st = np.array([[0.0, 0.1], [0.0, 0.45], [0.0, 0.5], [0.0, 1.0]], np.float32)

    # 1. no declaration at all: every state, and the caller may skip subsetting entirely.
    class Bare:
        state_names = names

    r = C.resolution_consistency_applicability(Bare(), st)
    assert r["applicable"] and r["indices"] is None, r

    # 2. one applicable band and one not.
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

    # 4. the whole cell, answered without looking at a state.
    class AllBands:
        state_names = names
        resolution_consistency_applicability = {"all bands": "not_applicable: mechanism", "evidence": "oracle.json"}

    r = C.resolution_consistency_applicability(AllBands(), st)
    assert r["applicable"] is False and r["indices"] == [], r
    assert C.resolution_consistency_applicability(AllBands(), None)["applicable"] is False

    # 5. a band naming a dimension this env does not have narrows nothing.
    class Typo:
        state_names = names
        resolution_consistency_applicability = {"lightlevel >= 0.5": "asserted", "evidence": "oracle.json"}

    r = C.resolution_consistency_applicability(Typo(), st)
    assert r["applicable"] and r["indices"] is None and "could not be resolved" in r["why"], r


def test_resolution_downsample_agrees_within_the_measured_bound(env, inputs):
    """Rendering at 2S and box-downsampling matches rendering at S on at least the measured fraction of pixels.

    Uses the default appearance so the number depends on geometry and the recorded seed only.
    """
    name = type(env).__name__
    if hasattr(type(env), "PLATE_PATHS") and name not in PLATE_REPRESENTATIVES:
        pytest.skip(f"{name}: covered by {sorted(PLATE_REPRESENTATIVES)} — see PLATE_REPRESENTATIVES")
    states, _ = inputs
    # A class may declare state bands where the reference renderer itself is not resolution-consistent.
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
        f"DOWNSAMPLE_EXACT_FRAC"
    )
    assert r["exact_frac"] >= measured - DOWNSAMPLE_MARGIN, (
        f"{name}: only {100 * r['exact_frac']:.3f}% of pixels survive a 2x render-and-box-downsample "
        f"against a measured {100 * measured:.3f}% ({MEASURED_ON}); a shape has moved: {r}"
    )


def test_every_input_form_gives_the_same_frame(env, inputs):
    """numpy or torch, CPU or CUDA, float32 or float64, contiguous or strided inputs give one frame.

    float64 inputs are widened float32 values, so any difference comes from the code path.
    """
    states, style = inputs
    if S.has_render_rows(env):
        r = C.check_input_form_invariance(env, states, style=style)
    else:
        r = RS.check_input_form_invariance(env, states)
    assert not r["disagreements"], (
        f"{type(env).__name__}: these input forms did not give the frame that plain float32 numpy "
        f"gives: {r['disagreements']}"
    )


def test_leading_dimensions_are_preserved(env, inputs):
    """``(n_traj, T, D)`` in gives ``(n_traj, T, H, W, 3)`` out, with each frame in its own cell.

    A full State has one batch dimension, so ``render`` refuses a nested batch instead of reshaping it.
    """
    states, style = inputs
    if not S.has_render_rows(env):
        with pytest.raises(ValueError):
            env.render(states.reshape(2, N // 2))
        return
    flat = env._render(states, style=style).detach().cpu().numpy()
    nested = env._render(states.reshape(2, N // 2, -1), style=style).detach().cpu().numpy()
    assert nested.shape == (2, N // 2, *flat.shape[1:]), (
        f"{type(env).__name__}: leading dims not preserved, got {nested.shape}"
    )
    assert np.array_equal(nested.reshape(flat.shape), flat), (
        f"{type(env).__name__}: the nested render holds the same frames in different cells"
    )


def test_out_buffer_receives_the_same_frames(env, inputs):
    """``render(..., out=buf)`` writes the frames into ``buf`` and returns them."""
    states, style = inputs
    ref = S.render_any(env, states, style=style).detach().cpu().numpy()
    buf = torch.zeros(ref.shape, dtype=torch.uint8, device=env.device)
    got = S.render_any(env, states, style=style, out=buf)
    assert np.array_equal(buf.detach().cpu().numpy(), ref), (
        f"{type(env).__name__}: render(out=buf) did not put the frames in buf"
    )
    assert np.array_equal(got.detach().cpu().numpy().reshape(ref.shape), ref), (
        f"{type(env).__name__}: render(out=buf) returned something other than what it wrote"
    )


def test_a_non_contiguous_out_buffer_is_refused_rather_than_silently_corrupted(env, inputs):
    """``render(out=)`` with a non-contiguous buffer raises before writing anything.

    ``Tensor.view`` returns a strided tensor unchanged when the shape already matches, so a kernel
    writing linearly from it would scatter frames across the caller's buffer. The check is applied to
    every env through ``StableEnv.__init_subclass__``.
    """
    states, style = inputs
    ref = S.render_any(env, states[:1], style=None if style is None else style[:1])
    H, W = int(ref.shape[-3]), int(ref.shape[-2])
    n = min(4, N)
    buf = torch.zeros((n, 3, H, W, 3), dtype=torch.uint8, device=env.device)
    view = buf[:, 1]
    assert not view.is_contiguous(), "the probe buffer must be strided for this to mean anything"
    with pytest.raises((RuntimeError, ValueError), match="(?i)contiguous"):
        S.render_any(env, states[:n], style=None if style is None else style[:n], out=view)
    assert int(buf.max()) == 0, (
        f"{type(env).__name__}: render(out=<non-contiguous>) reported an error but had already "
        f"written into the caller's buffer"
    )


def test_rendering_a_sub_batch_refuses_environment_0_s_style(env, inputs):
    """Rendering a sub-batch under a per-environment style raises instead of using environment 0's style.

    ``render(st[3:4])`` does not say which rows of the batch it is, so the env cannot pick their style;
    passing ``style=sty[3:4]`` renders row 3 exactly as in the full batch.
    """
    if env.style_dim == 0:
        pytest.skip(f"{type(env).__name__} declares no style, so there is nothing to collapse")
    if not S.has_render_rows(env):
        # A full State carries each lane's Variation, so a sub-batch draws its own lanes' appearance.
        st = S.styled_inputs(env, N, C.uniform_style(env, N, groups=2))
        d = RS.max_diff(RS.frames(env, st[3:4])[0], RS.frames(env, st)[3])
        assert d == 0, f"{type(env).__name__}: State 3 rendered alone is {d}/255 away from row 3 of the batch"
        return
    states, _ = inputs
    sty = C.uniform_style(env, N, groups=2)  # two distinct rows; see uniform_style on the cost
    env._set_render_state(states, style=sty)
    full = env._render().detach().cpu().numpy()

    with pytest.raises(ValueError, match="(?i)per-environment style"):
        env._render(states[3:4])

    alone = env._render(states[3:4], style=sty[3:4]).detach().cpu().numpy()[0]
    d = int(np.abs(alone.astype(np.int16) - full[3].astype(np.int16)).max())
    assert d == 0, (
        f"{type(env).__name__}: rendering row 3 on its own with row 3's style gave a frame {d}/255 "
        f"away from row 3 of the full batch"
    )


def test_a_shared_style_broadcasts_to_any_batch(env, inputs):
    """A shared style (the stored default, or one explicit row) broadcasts to any batch without raising."""
    states, _ = inputs
    env.set_style(None)
    ref = S.render_any(env, states).detach().cpu().numpy()
    for b in (1, 3, N):
        got = S.render_any(env, states[:b]).detach().cpu().numpy()
        assert np.array_equal(got, ref[:b]), (
            f"{type(env).__name__}: under the default appearance, rendering {b} of {N} rows differs "
            f"from those rows of the full batch"
        )
    if env.style_dim and not S.has_render_rows(env):
        one = env.sample_style(1, rng=np.random.default_rng(3))
        st = S.styled_inputs(env, N, np.repeat(one, N, 0))
        assert np.array_equal(RS.frames(env, st[:1])[0], RS.frames(env, st)[0]), (
            f"{type(env).__name__}: a State under one shared Variation row draws differently at two batch sizes"
        )
    elif env.style_dim:
        one = env.sample_style(1, rng=np.random.default_rng(3))
        a = env._render(states, style=one).detach().cpu().numpy()
        b_ = env._render(states[:1], style=one).detach().cpu().numpy()[0]
        assert np.array_equal(b_, a[0]), (
            f"{type(env).__name__}: a single explicit style row did not broadcast identically at two batch sizes"
        )


def test_variation_overrides_do_not_drop_the_env_s_per_environment_style(env, inputs):
    """``render(variation={...})`` keeps each environment's own style row for the other columns.

    Uses two distinct style rows; under the default appearance all rows are equal and this cannot fail.
    """
    if env.style_dim == 0:
        pytest.skip(f"{type(env).__name__} declares no style")
    chosen = _continuous_leaf(env)
    if chosen is None:
        pytest.skip(f"{type(env).__name__} declares no continuous variation leaf")
    path, off, w, v = chosen
    if not S.has_render_rows(env):
        _assert_an_override_keeps_the_per_lane_variation(env, path, off, w, v)
        return
    states, _ = inputs
    sty = C.uniform_style(env, N, groups=2)  # two distinct rows; see uniform_style on the cost
    env._set_render_state(states, style=sty)

    want = np.array(sty, np.float32)
    want[:, off : off + w] = v
    a = env._render(states, variation={path: v}).detach().cpu().numpy()
    b = env._render(states, style=want).detach().cpu().numpy()
    assert np.array_equal(a, b), (
        f"{type(env).__name__}: with a per-environment style set on the env, "
        f"render(variation={{{path!r}: v}}) drew something other than the same rows with that one "
        f"column replaced (max {int(np.abs(a.astype(np.int16) - b.astype(np.int16)).max())}/255)"
    )


def _leaf_value(env, path, v):
    """``v`` from ``_continuous_leaf`` in the leaf's own shape and dtype, as ``set_variation`` takes it."""
    node = env.variation_space._walk(path)
    return np.asarray(v).reshape(node.shape).astype(node.dtype)


def _assert_an_override_keeps_the_per_lane_variation(env, path, off, w, v):
    """``set_variation({path: v})`` over a per-lane Variation equals installing those rows with one column replaced."""
    sty = C.uniform_style(env, N, groups=2)
    want = np.array(sty, np.float32)
    want[:, off : off + w] = v
    with RS.fresh(env) as a, RS.fresh(env) as b:
        a.set_variation(sty)
        rows_a = a.set_variation({path: _leaf_value(env, path, v)}).detach().cpu().numpy()
        rows_b = b.set_variation(want).detach().cpu().numpy()
        assert np.array_equal(rows_a, rows_b), (
            f"{type(env).__name__}: set_variation({{{path!r}: v}}) over a per-lane Variation installed other "
            f"rows than the same rows with that one column replaced"
        )
        fa, fb = RS.frames_under(a), RS.frames_under(b)
    assert np.array_equal(fa, fb), (
        f"{type(env).__name__}: after set_variation({{{path!r}: v}}) over a per-lane Variation the frames differ "
        f"from those of the same rows with that column replaced (max {RS.max_diff(fa, fb)}/255)"
    )


def test_render_is_pure_and_set_state_is_idempotent(env, inputs):
    """Rendering does not mutate the env, and ``set_state(get_state())`` twice equals once."""
    states, style = inputs
    if S.has_render_rows(env):
        r = C.check_render_is_pure(env, states, style=style)
    else:
        r = RS.check_render_is_pure(env, states)
    assert r["state_changed_by_render"] == 0.0, (
        f"{type(env).__name__}: render() mutated the env's state by {r['state_changed_by_render']}"
    )
    assert r["style_changed_by_render"] == 0.0, (
        f"{type(env).__name__}: render() mutated the env's style by {r['style_changed_by_render']}"
    )
    assert r["state_changed_by_double_set"] == 0.0, (
        f"{type(env).__name__}: set_state(get_state()) twice changed the state by {r['state_changed_by_double_set']}"
    )
    assert r["frame_changed_by_double_set"] == 0, (
        f"{type(env).__name__}: set_state(get_state()) twice changed the frame by "
        f"{r['frame_changed_by_double_set']}/255 while leaving the state alone"
    )


def test_set_state_copies_its_input(env):
    """``set_state`` copies its input and writes through the state buffer rather than rebinding it.

    CUDA-graph-captured steps read the state buffer at a fixed address. An env without render rows has
    no such buffer; its ``set_state`` must still copy the State it is given.
    """
    if not S.has_render_rows(env):
        r = RS.check_set_state_copies(env)
        assert not r["aliased_leaves"], (
            f"{type(env).__name__}: zeroing the State handed to set_state changed the env's own State at "
            f"{r['aliased_leaves']}; the env aliases the caller's memory"
        )
        return
    r = C.check_set_state_copies(env)
    assert r["aliased_caller_buffer"] == 0.0, (
        f"{type(env).__name__}: zeroing the array that was handed to set_state changed the env's own "
        f"state by {r['aliased_caller_buffer']}; the env aliases the caller's memory"
    )
    assert not r["state_tensor_rebound"], (
        f"{type(env).__name__}: set_state rebound the state tensor instead of writing through it"
    )


def test_batch_order_does_not_change_the_frames(env, inputs):
    """``render(x[perm]) == render(x)[perm]``, and a duplicated state renders identically.

    Uses two style groups so grouped renderers scatter results back through their index.
    """
    states, _ = inputs
    style = C.uniform_style(env, N, groups=2)
    if S.has_render_rows(env):
        r = C.check_order_independence(env, states, style=style)
    else:
        r = RS.check_order_independence(env, S.styled_inputs(env, N, style) if env.style_dim else states)
    assert r["permutation_max_abs_diff"] == 0, (
        f"{type(env).__name__}: permuting the batch did not permute the frames (max "
        f"{r['permutation_max_abs_diff']}/255)"
    )
    assert r["duplicate_max_abs_diff"] == 0, (
        f"{type(env).__name__}: the same state twice in one batch rendered two different frames (max "
        f"{r['duplicate_max_abs_diff']}/255)"
    )


def test_render_size_does_not_leak_into_later_calls(env, inputs):
    """``render(size=X)`` does not change the frame size of a later plain ``render()``."""
    name = type(env).__name__
    if hasattr(type(env), "PLATE_PATHS") and name not in PLATE_REPRESENTATIVES:
        pytest.skip(f"{name}: covered by {sorted(PLATE_REPRESENTATIVES)} — see PLATE_REPRESENTATIVES")
    states, style = inputs
    before = S.render_any(env, states, style=style).shape
    try:
        S.render_any(env, states[:1], size=2 * int(env.size), style=None if style is None else style[:1])
    except ValueError as e:
        pytest.skip(f"{name} renders at one size only, so no size can leak: {e}")
    after = S.render_any(env, states, style=style).shape
    # Restore the size before asserting so a leak does not fail later tests on the shared env.
    S.render_any(env, states[:1], size=int(env.size), style=None if style is None else style[:1])
    assert tuple(after) == tuple(before), (
        f"{name}: after one render(size={2 * int(env.size)}), a plain render() returns {tuple(after)} "
        f"instead of {tuple(before)} while env.size is still {env.size}"
    )


def test_numpy_and_torch_serialization_round_trip(env, inputs, tmp_path):
    """States and style saved and loaded with numpy and torch re-render byte-identical frames."""
    states, style = inputs
    if not S.has_render_rows(env):
        ref = _digest(env.render(states))
        np.savez(tmp_path / "s.npz", **S.state_arrays(states))
        with np.load(tmp_path / "s.npz") as f:
            back = S.state_from_arrays({k: f[k] for k in f.files}, env.device)
        assert _digest(env.render(back)) == ref, f"{type(env).__name__}: frames differ after a numpy save/load"
        torch.save({k: torch.as_tensor(v) for k, v in S.state_arrays(states).items()}, tmp_path / "t.pt")
        back = S.state_from_arrays(torch.load(tmp_path / "t.pt", weights_only=True), env.device)
        assert _digest(env.render(back)) == ref, f"{type(env).__name__}: frames differ after a torch save/load"
        return
    ref = _digest(env._render(states, style=style))

    np.save(tmp_path / "s.npy", states)
    ns = np.load(tmp_path / "s.npy")
    nk = None
    if style is not None:
        np.save(tmp_path / "k.npy", style)
        nk = np.load(tmp_path / "k.npy")
    assert _digest(env._render(ns, style=nk)) == ref, (
        f"{type(env).__name__}: frames differ after a numpy save/load of the states"
    )

    torch.save(
        {"s": torch.as_tensor(states), "k": None if style is None else torch.as_tensor(style)}, tmp_path / "t.pt"
    )
    got = torch.load(tmp_path / "t.pt", weights_only=True)
    assert _digest(env._render(got["s"], style=got["k"])) == ref, (
        f"{type(env).__name__}: frames differ after a torch save/load of the states"
    )


def test_dataset_paths_regenerate_identical_frames(env, inputs):
    """``StateDataset``, a ``DataLoader`` over it and ``TrajectoryDataset.all_frames`` match ``env.render``.

    The per-trajectory style must be expanded with ``repeat_interleave``, not ``repeat``.
    """
    _rows_only(env, "the datasets store render rows; full States are covered by the serialization test")
    states, style = inputs
    ref = env._render(states, style=style).detach().cpu().numpy()

    ds = StateDataset(env, states, style=style, size=env.size, device=str(env.device))
    assert np.array_equal(ds.__getitems__(list(range(N))).detach().cpu().numpy(), ref), (
        f"{type(env).__name__}: StateDataset renders different pixels from env.render"
    )

    # ``__getitems__`` returns an assembled batch, so ``collate_batched`` is the collate function.
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
    want = env._render(states, style=None if style is None else np.repeat(style[:1], N, 0))
    assert np.array_equal(got, want.detach().cpu().numpy()), (
        f"{type(env).__name__}: TrajectoryDataset.all_frames does not reproduce env.render on the "
        f"same states and the trajectory's own style"
    )


def test_huggingface_round_trip_regenerates_identical_frames(env, inputs, tmp_path):
    """A ``TrajectoryDataset`` round-tripped through HuggingFace (in memory and on disk) re-renders identically."""
    pytest.importorskip("datasets", reason="the HuggingFace round-trip needs `datasets`")
    _rows_only(env, "TrajectoryDataset stores render rows; full States are covered by the serialization test")
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
        f"{type(env).__name__}: a corpus written with save_to_disk and read back renders different frames"
    )


def test_variation_dict_and_style_tensor_agree(env, inputs):
    """``render(variation={path: v})`` equals ``render(style=<default row with that column set>)``."""
    states, _ = inputs
    chosen = _continuous_leaf(env)
    if chosen is None:
        pytest.skip(f"{type(env).__name__} declares no continuous variation leaf")
    path, off, w, v = chosen

    sty = np.tile(env.default_style(), (N, 1))
    sty[:, off : off + w] = v
    if not S.has_render_rows(env):
        with RS.fresh(env) as a, RS.fresh(env) as b:
            fa, fb = RS.frames_under(a, {path: _leaf_value(env, path, v)}), RS.frames_under(b, sty)
        assert np.array_equal(fa, fb), (
            f"{type(env).__name__}: set_variation({{{path!r}: v}}) and the equivalent flat Variation row "
            f"draw different pictures (max {RS.max_diff(fa, fb)}/255)"
        )
        return
    a = env._render(states, variation={path: v}).detach().cpu().numpy()
    b = env._render(states, style=sty).detach().cpu().numpy()
    assert np.array_equal(a, b), (
        f"{type(env).__name__}: render(variation={{{path!r}: v}}) and the equivalent flat style row "
        f"draw different pictures (max "
        f"{int(np.abs(a.astype(np.int16) - b.astype(np.int16)).max())}/255)"
    )


#: Default CUDA physics backends that are approximate: float32 and not bitwise reproducible, since contact
#: solving reduces with atomics. An env whose default is one of them also accepts ``simulation_backend="native"``,
#: an exact backend.
APPROXIMATE_BACKENDS = frozenset({"mjwarp"})
#: Steps to the restored State, then steps of the continued rollout, for approximate backends.
ROLLOUT_WARMUP, ROLLOUT_STEPS = 20, 300
#: Bounds on a continued rollout of an approximate backend, on the largest ``qpos`` difference per step between
#: the env that reached the State and the env restored to it: its rollout mean in the median lane, its largest
#: value in any lane, and the lanes whose final ``success`` may differ.
ROLLOUT_MEDIAN_QPOS, ROLLOUT_MAX_QPOS, ROLLOUT_SUCCESS_FLIPS = 0.01, 0.25, 1
#: The median lane's accumulated reward may differ between the two envs by this fraction of its size (at least 1).
ROLLOUT_REWARD_RTOL = 0.05
#: Wider bounds for tasks whose own source amplifies small differences. Slide's puck slides freely and can leave
#: the table: on CPU MuJoCo an action noise of 1e-6 alone moves it 3.3 cm on average over 200 steps.
#: MuJoCo Warp does not repeat a contact solve bit for bit, so bodies in contact are chaotic: two envs restored from
#: the same State differ from each other as much as from the original (largest gap over 300 steps, original against
#: restored and restored against restored: Hopper 0.85 and 1.12, Walker 0.59 and 1.04, HumanoidMaze 2.8 and 2.9,
#: OGBScene 0.85 and 0.61, OGBCube 0.72 and 0.11). One lane drifts far in qpos while the return and the success of
#: the median lane stay the same, which the checks below the qpos bounds cover. The DMC Humanoid falls under these
#: actions and every lane is in contact: over three runs, mean gap of the median lane 0.27 to 0.45, largest gap 2.69
#: to 2.74, returns equal. Against native CPU physics on 16 environments the same rollout gives 0.77 and 2.95.
ROLLOUT_QPOS_BOUNDS = {
    "FetchSlide*": (0.05, 2.5),
    "DMCHumanoid*": (1.0, 3.5),
    "DMC*": (0.01, 2.5),
    "HumanoidMaze*": (1.0, 3.5),
    "OGBScene*": (0.05, 2.0),
    "OGBCube*": (0.05, 2.0),
    # Under the smooth policy a Roadster leaves a roof, clips a parked car or touches down a step earlier or later,
    # and the paths separate by tens of metres: two envs reset alike and given the same actions differ by 50 m
    # (Rooftop) and 95 m (TrafficJam) on an RTX 4090, and on CPU a 1e-5 m/s change of the source's velocity moves
    # TrafficJam's car 271 m. Restored against original: median lane 1.7 m (Rooftop), largest 380 m. MuJoCo
    # Warp's deterministic mode allocates memory inside the captured step, so the step graph cannot use it.
    "Roadster*": (2.5, 600.0),
}


def _qpos(state):
    """The generalized positions of a State: a top-level ``qpos``, the ``physics`` or ``sim`` group, or the maze
    body parts."""
    keys = set(state.keys(True, True))
    for key in ("qpos", ("physics", "qpos"), ("sim", "qpos")):
        if key in keys:
            return state[key]
    parts = [("root", "position"), ("root", "orientation"), ("joints", "position")]
    return torch.cat([state[k] for k in parts if k in keys], dim=-1)


def _exact_backend(env) -> dict:
    """Construction arguments that put ``env``'s class on an exact physics backend."""
    approximate = getattr(env, "simulation_backend", None) in APPROXIMATE_BACKENDS
    return {"simulation_backend": "native"} if approximate else {}


def test_a_step_from_a_restored_state_matches_the_original(env):
    """Stepping an env restored to State ``s`` gives the same States as stepping the env that reached ``s``.

    Envs whose default CUDA physics is approximate run this on their exact backend; the rollout test below
    checks the approximate one.
    """
    cls = type(env)
    if not env.has_dynamics:
        pytest.skip(f"{cls.__name__}: no dynamics")
    kw = _exact_backend(env)
    with (
        se.make(cls.__name__, num_envs=N, device="cuda", **kw) as a,
        se.make(cls.__name__, num_envs=N, device="cuda", **kw) as b,
    ):
        rng = np.random.default_rng(6)
        b.reset(seed=11)
        b.set_state(a.reset(seed=5))
        for t in range(4):
            action = a.action_spec.sample(N, rng)
            sa = a.step(action, frames=None)["state"]
            sb = b.step(action, frames=None)["state"]
            for key in sa.keys(True, True):
                assert torch.equal(sa[key], sb[key]), f"{cls.__name__}: step {t} diverges at {key}"


def _smooth_policy(env):
    """Slow sinusoids inside the action bounds, one phase and frequency per lane and action."""
    spec = env.action_spec
    shape = (N, *spec.shape)
    low = np.broadcast_to(np.asarray(-1.0 if spec.low is None else spec.low, np.float64), shape)
    high = np.broadcast_to(np.asarray(1.0 if spec.high is None else spec.high, np.float64), shape)
    rng = np.random.default_rng(3)
    phase, freq = rng.uniform(0, 2 * np.pi, shape), rng.uniform(0.02, 0.08, shape)
    mid, half = 0.5 * (low + high), 0.25 * (high - low)

    def act(t, state):
        return torch.as_tensor(mid + half * np.sin(freq * t + phase), dtype=spec.dtype, device=env.device)

    return act


def _fetch_policy(env):
    """The scripted Fetch controller (reach, push, strike, grasp and carry), closed loop on the env's sites."""
    from tests.helpers.reference.fetch_policy import scripted_actions

    task, sim = env.SCENE, env._simulator
    phase, count = np.zeros(N, int), np.zeros(N, int)

    def act(t, state):
        if t == 0:
            phase[:], count[:] = 0, 0
        sites = torch.as_tensor(sim._site_xpos).cpu().numpy().astype(np.float64)
        grip, obj = sites[:, sim.task.grip_site], sites[:, sim.task.goal_site]
        goal = state["goal"].cpu().numpy()
        return torch.as_tensor(scripted_actions(task, grip, obj, goal, phase, count), device=env.device)

    return act


#: Family policies for the continued rollout; other families use ``_smooth_policy``.
ROLLOUT_POLICIES = {"fetch": _fetch_policy}


def test_an_approximate_backend_continues_a_restored_rollout(env):
    """An env restored to State ``s`` follows the env that reached ``s`` over a long rollout of the same actions.

    For approximate backends only, which do not reproduce a step bit for bit. The actions come from a policy run
    closed loop on the env that reached ``s``; the restored env replays them.
    """
    cls = type(env)
    if getattr(env, "simulation_backend", None) not in APPROXIMATE_BACKENDS:
        pytest.skip(f"{cls.__name__}: its default backend is exact; the bitwise restore test covers it")
    with se.make(cls.__name__, num_envs=N, device="cuda") as a, se.make(cls.__name__, num_envs=N, device="cuda") as b:
        limits = getattr(a._simulator, "_set_episode_limits", None)
        for e in (a, b):
            if limits is not None:
                e._simulator._set_episode_limits(np.full(N, ROLLOUT_WARMUP + ROLLOUT_STEPS + 1))
        policy = ROLLOUT_POLICIES.get(a.family, _smooth_policy)(a)
        state = a.reset(seed=5)
        for t in range(ROLLOUT_WARMUP):
            state = a.step(policy(t, state), frames=None)["state"]
        b.reset(seed=11)
        b.set_state(state)
        gaps, oa, ob = [], None, None
        returns = []
        for t in range(ROLLOUT_WARMUP, ROLLOUT_WARMUP + ROLLOUT_STEPS):
            action = policy(t, state)
            oa, ob = a.step(action, frames=None), b.step(action, frames=None)
            state = oa["state"]
            gaps.append((_qpos(state) - _qpos(ob["state"])).abs().max(-1).values)
            if "reward" in oa.keys():
                returns.append(torch.stack([oa["reward"], ob["reward"]]).reshape(2, N, -1).sum(-1))
        gaps = torch.stack(gaps).double()
        median_bound, max_bound = next(
            (b for pattern, b in ROLLOUT_QPOS_BOUNDS.items() if fnmatch.fnmatch(cls.__name__, pattern)),
            (ROLLOUT_MEDIAN_QPOS, ROLLOUT_MAX_QPOS),
        )
        median = float(gaps.mean(0).median())
        assert median <= median_bound, f"{cls.__name__}: mean qpos gap of the median lane {median:.3g}"
        assert float(gaps.max()) <= max_bound, f"{cls.__name__}: largest qpos gap {float(gaps.max()):.3g}"
        if returns:
            ra, rb = torch.stack(returns).sum(0).double()
            scale = max(1.0, float(ra.abs().median()))
            drift = float((ra - rb).abs().median())
            assert drift <= ROLLOUT_REWARD_RTOL * scale, f"{cls.__name__}: median lane return differs by {drift:.3g}"
        if "success" in oa.keys():
            final = lambda out: out["success"].reshape(N, -1)[:, -1]  # noqa: E731
            sa, sb = final(oa).double(), final(ob).double()
            flips = int(((sa != sb) & ~(sa.isnan() & sb.isnan())).sum())
            assert flips <= ROLLOUT_SUCCESS_FLIPS, f"{cls.__name__}: {flips} of {N} lanes changed task outcome"


@pytest.mark.reference
def test_the_reference_renderer_on_this_machine_is_deterministic(env):
    """The official reference renderer returns the same pixels twice for the same states on this machine.

    A failure here points at the machine's offscreen rasterizer rather than this repo's code.
    """
    cls = type(env)
    if not has_reference_renderer(cls):
        pytest.skip(f"{cls.__name__}: no official reference renderer")
    try:
        r = C.check_reference_renderer_is_stable(cls, env.size)
    except ImportError as e:
        pytest.skip(f"{cls.__name__}: reference env not installed ({e})")
    assert r["max_abs_diff"] == 0, (
        f"{cls.__name__}: the OFFICIAL renderer gave two different frames for the same states on this "
        f"machine (max {r['max_abs_diff']}/255). The machine's rasterizer is nondeterministic; parity "
        f"numbers measured here are unreliable"
    )


def _visibility_capable_envs():
    """Envs whose renderer supports a visibility knob, found without building anything.

    Whether an env actually declares a ``visibility.*`` leaf needs its compiled model, so that is
    checked in the test body rather than at collection.
    """
    return [
        name
        for name, cls in sorted(se.registry.REGISTRY.items())
        if getattr(cls, "VISIBILITY_KNOB", False) and hasattr(cls, "_hideable")
    ]


def _declared_visibility_leaves(names) -> dict:
    """``{env name: [visibility leaf, ...]}`` for ``names``; builds compiled models, so call it from a test."""
    out = {}
    for name in names:
        cls = se.registry.REGISTRY[name]
        try:
            out[name] = [f"visibility.{el[1]}" for el in cls._hideable()]
        except ImportError:  # the family is not installed here
            continue
    return out


def test_some_env_actually_declares_a_hideable_task_object():
    """At least one visibility-capable env declares a ``visibility.*`` leaf.

    The scene-cache check below skips envs with no leaf, so without this it could skip on every env.
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
        f"{len(declared)} env(s) advertise VISIBILITY_KNOB and none declares a visibility.* leaf, "
        f"so the scene-cache check skips on all of them. Check _episode_slots and _hideable: {declared}"
    )
    print(f"\nvisibility leaves declared by {len(with_leaf)} of {len(declared)} capable env(s): {with_leaf}")


@pytest.mark.parametrize("name", family_params(_visibility_capable_envs(), family="visibility"))
def test_a_baked_scene_does_not_inherit_the_last_bake_s_visibility(name):
    """A cached scene's static depth uses its own baked visibility, not the last bake's.

    The static depth is cast lazily per (scene, supersample) from the shared ``MjModel``, which holds
    whatever appearance was baked last. Rendering at a non-default supersample after baking another
    appearance forces that late cast. The process-global scene cache is cleared first.
    """
    resolve_family(name, family="visibility", modules=("dm_control",), subject="the scene-cache visibility check")
    env = make_or_skip(name, num_envs=8, device="cuda")
    paths = [p for p in env.variation_space.names() if p.split(".")[0] == "visibility"]
    if not paths:
        # This domain draws no per-episode task object; the family as a whole is covered by
        # test_some_env_actually_declares_a_hideable_task_object.
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
    alone = env._render(st, variation=hidden, supersample=ss).cpu().numpy()

    D._SCENE_CACHE.clear()
    env._render(st, variation=hidden)  # bake the hidden scene at the default ss
    env._render(st, variation=other)  # ... then bake a DIFFERENT appearance
    after = env._render(st, variation=hidden, supersample=ss).cpu().numpy()

    d = np.abs(alone.astype(np.int16) - after.astype(np.int16)).max(-1)
    assert int(d.max()) == 0, (
        f"{name}: the same hidden-target scene rendered at supersample {ss} differs by "
        f"{int(d.max())}/255 over {int((d > 0).sum())} pixel(s) in "
        f"{int((d.reshape(len(st), -1) > 0).any(1).sum())} of {len(st)} frames depending on whether "
        f"another appearance was baked first. A cached scene is reading the live model instead of "
        f"what it was baked from; see _bake's `sc.statics`"
    )
