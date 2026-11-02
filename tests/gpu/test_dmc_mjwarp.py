"""Explicit CUDA MJWarp gates. Missing CUDA or frozen evidence is a failure.

Use scripts/ci/run_dmc_mjwarp.py for the no-skips acceptance run. ``--smoke`` on
that runner exercises CUDA, exact snapshots and pixels without a source checkout.
The numerical gates consume a separately frozen calibration; they never learn a
tolerance from their own held-out trajectories.
"""

import os

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from tests.helpers.reference.dmc_mjwarp import (
    CELLS,
    CONSTRAINT_CELLS,
    CONTACT_CELLS,
    HELD_OUT_SEEDS,
    HORIZONS,
    Oracle,
    actions_for,
    assert_bounded,
    load_evidence,
    make_candidate,
    prepare_source,
    project_source_caps,
    public_step,
    replay_trial,
    require_cuda,
    source_modes,
    source_projection,
    source_trial,
)
from tests.helpers.reference.semantics import assert_same_state

pytestmark = pytest.mark.family("dmc")


@pytest.fixture(scope="module", autouse=True)
def cuda_required():
    require_cuda()


@pytest.fixture(scope="module")
def source_ready():
    return prepare_source(os.environ.get("SE_DMC_MJWARP_SOURCE_ROOT"))


@pytest.fixture(scope="module")
def evidence():
    return load_evidence(
        os.environ.get("SE_DMC_MJWARP_EVIDENCE"),
        allow_implementation_change=os.environ.get("SE_DMC_MJWARP_ALLOW_IMPLEMENTATION_CHANGE") == "1",
    )


@pytest.mark.parametrize("name", CELLS)
def test_cuda_stepping_snapshot_ownership_and_pixels(name, monkeypatch):
    """A forced backend, strided CUDA actions, foreign stream, and independent history."""
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream), make_candidate(name) as candidate:
        candidate.reset(seed=31)
        actions = torch.zeros(3, 2 * candidate.action_dim, device="cuda", dtype=torch.float32)[:, ::2]
        public_step(candidate, actions)  # compilation/warmup is outside the host-staging guard
        saved = candidate.get_state().clone()
        # A cached normal draw is a floating RNG leaf, not a numeric tolerance.
        saved["rng", "has_gauss"][:] = True
        saved["rng", "cached_gaussian"][:, 0] = torch.tensor(
            [0.123456789012345, -0.75, 1.25], device="cuda", dtype=torch.float64
        )
        if name == "DMCQuadrupedWalk":
            assert saved["activation"].shape[1] > 0
            saved["activation"][:] = torch.tensor([0.15, 0.45, 0.85], device="cuda", dtype=torch.float64)[:, None]
        candidate.set_state(saved)
        assert_same_state(candidate.get_state(), saved)
        pixels = candidate.render(saved).clone()

        def forbidden(*args, **kwargs):
            raise AssertionError("Host staging in a warmed CUDA MJWarp step")

        with monkeypatch.context() as patch:
            patch.setattr(torch.Tensor, "cpu", forbidden)
            patch.setattr(torch.Tensor, "numpy", forbidden)
            result = public_step(candidate, actions)
            held = result.clone()
            actions.fill_(0.25)
            public_step(candidate, actions)
        stream.synchronize()
        assert all(leaf.device.type == "cuda" for leaf in result.values(True, True))
        assert_same_state(result, held)
        candidate.set_state(saved)
        assert_same_state(candidate.get_state(), saved)
        assert_same_state(candidate.render(), pixels)
        assert_same_state(candidate.get_state(), saved)  # rendering must be observational

        passed = saved.clone()
        candidate.set_state(passed)
        passed["qpos"].add_(1)
        assert_same_state(candidate.get_state(), saved)  # set_state owns its input
        exported = candidate.get_state()
        exported["qvel"].add_(1)
        assert_same_state(candidate.get_state(), saved)  # get_state owns its output

        with make_candidate(name) as fresh:
            fresh.reset(seed=903)
            public_step(fresh, actions)
            fresh.set_state(saved)
            assert_same_state(fresh.get_state(), saved)
            assert_same_state(fresh.render(), pixels)

        # Pixels are exact for the State they depict. Resumed rollouts have a
        # separate bounded gate and are not required to yield identical pixels.
        sequence = torch.zeros(3, 4, candidate.action_dim, device="cuda", dtype=torch.float64)
        captured = candidate.step(sequence, states=slice(None), frames=slice(None))
        before_render = candidate.get_state().clone()
        for t in range(4):
            assert_same_state(candidate.render(captured["states"][:, t]), captured["frames"][:, t])
        assert_same_state(candidate.get_state(), before_render)


def _check_pairs(pairs, mode_caps, *, context, source=False):
    for horizon in HORIZONS:
        for step, (actual, expected) in enumerate(pairs[:horizon], 1):
            caps = mode_caps[str(horizon)]
            if source:
                caps = project_source_caps(caps, expected)
            assert_bounded(actual, expected, caps, context=f"{context} horizon={horizon} step={step}")


def _assert_source_fault_rejected(pairs, envelope, *, context):
    """Each horizon's own bounds must reject at least one faulty field/step."""
    assert len(pairs) >= max(HORIZONS), f"{context}: incomplete fault trajectory"
    for horizon in HORIZONS:
        rejected = False
        for step, (actual, expected) in enumerate(pairs[:horizon], 1):
            caps = project_source_caps(envelope[str(horizon)], expected)
            try:
                assert_bounded(actual, expected, caps, context=f"{context} horizon={horizon} step={step}")
            except AssertionError:
                rejected = True
                break
        assert rejected, f"{context} horizon={horizon}: frozen source envelope did not reject the physics fault"


@pytest.mark.parametrize("name", CELLS)
def test_held_out_independent_source_and_reset_boundaries(name, source_ready, evidence):
    del source_ready
    caps = evidence["environments"][name]["source_envelope"]
    with make_candidate(name) as candidate:
        for mode in source_modes(name):
            oracle = Oracle(name, varied=mode == "variation")
            try:
                for seed in HELD_OUT_SEEDS:
                    _, pairs = source_trial(
                        candidate,
                        oracle,
                        seed,
                        boundary=mode == "boundary",
                        contact=mode == "contact",
                        cached_reset=mode == "cached_reset",
                    )
                    context = f"{name} mode={mode} seed={seed}"
                    _check_pairs(pairs, caps, context=context, source=True)
                    if mode == "variation":
                        state_caps = project_source_caps(caps["1"], oracle.reset_pair[1])
                        assert_bounded(*oracle.reset_pair, state_caps, context=f"{context} horizon=1 reset")
                        if seed == HELD_OUT_SEEDS[0]:
                            varied_snapshot = candidate.get_state().clone()
                            varied_pixels = candidate.render(varied_snapshot).clone()
                            candidate.reset(seed=913)
                            candidate.set_state(varied_snapshot)
                            assert_same_state(candidate.get_state(), varied_snapshot)
                            assert_same_state(candidate.render(), varied_pixels)
                            with make_candidate(name) as fresh:
                                fresh.set_state(varied_snapshot)
                                assert_same_state(fresh.get_state(), varied_snapshot)
                                assert_same_state(fresh.render(), varied_pixels)
            finally:
                oracle.close()


@pytest.mark.parametrize("name", CELLS)
def test_held_out_resume_and_fault_detection(name, source_ready, evidence, monkeypatch):
    del source_ready
    caps = evidence["environments"][name]["replay"]
    with make_candidate(name) as candidate:
        for seed in HELD_OUT_SEEDS:
            _, pairs = replay_trial(candidate, seed)
            _check_pairs(pairs, caps, context=f"{name} mode=replay seed={seed}")
        actual, expected = pairs[-1]
        corrupted = actual.clone()
        corrupted["reward"][:] = expected["reward"] + 2 * caps["8"]["reward"]["atol"] + 1
        with pytest.raises(AssertionError, match="reward"):
            assert_bounded(corrupted, expected, caps["8"])
        corrupted = expected.clone()
        corrupted["state", "rng", "keys"][0, 0] ^= 1
        with pytest.raises(AssertionError, match="rng"):
            assert_bounded(corrupted, expected, caps["8"])
        for fault in ("missing_repeat", "controls"):
            # Alter the implementation, keeping the expected output untouched.
            candidate.reset(seed=HELD_OUT_SEEDS[0])
            snapshot = candidate.get_state().clone()
            actions_cpu = actions_for(HELD_OUT_SEEDS[0], candidate.action_dim)[: max(HORIZONS)]
            actions = torch.as_tensor(actions_cpu, device="cuda")
            oracle = Oracle(name)
            try:
                oracle.reset(HELD_OUT_SEEDS[0])
                expected = [oracle.step(action) for action in actions_cpu]
            finally:
                oracle.close()
            source_caps = evidence["environments"][name]["source_envelope"]
            context = f"{name} mode=source seed={HELD_OUT_SEEDS[0]} fault={fault}"
            baseline = [
                (source_projection(public_step(candidate, action), reference), reference)
                for action, reference in zip(actions, expected, strict=True)
            ]
            _check_pairs(baseline, source_caps, context=f"{context} baseline", source=True)
            candidate.set_state(snapshot)
            with monkeypatch.context() as patch:
                adapter = candidate._simulator
                if fault == "missing_repeat":
                    advance, calls = adapter._runtime.advance, []

                    def omit_second_round(controls):
                        calls.append(True)
                        if len(calls) % 2:
                            advance(controls)

                    patch.setattr(adapter._runtime, "advance", omit_second_round)
                else:
                    wrong_controls = adapter._controls(-torch.ones_like(actions[0]))
                    patch.setattr(adapter, "_controls", lambda actions: wrong_controls)
                faulty = [
                    (source_projection(public_step(candidate, action), reference), reference)
                    for action, reference in zip(actions, expected, strict=True)
                ]
            _assert_source_fault_rejected(faulty, source_caps, context=context)


@pytest.mark.parametrize(
    "name,fault",
    [
        *[(name, "contact") for name in sorted(CONTACT_CELLS)],
        *[(name, "constraint") for name in sorted(CONSTRAINT_CELLS)],
    ],
)
def test_contact_and_constraint_faults_are_detected(name, fault, source_ready, evidence, monkeypatch):
    """Real disabled-physics faults at source-owned, contact-bearing checkpoints."""
    del source_ready
    import mujoco

    oracle = Oracle(name)
    try:
        oracle.reset(HELD_OUT_SEEDS[0])
        source_snapshot = oracle.contact_checkpoint()
        with make_candidate(name) as candidate:
            snapshot = candidate.get_state()
            for key in source_snapshot.keys(True, True):
                snapshot[key] = source_snapshot[key].cuda()
            candidate.set_state(snapshot)
            actions_cpu = actions_for(HELD_OUT_SEEDS[0], candidate.action_dim)[: max(HORIZONS)]
            actions = torch.as_tensor(actions_cpu, device="cuda")
            expected = [oracle.step(action) for action in actions_cpu]
            source_caps = evidence["environments"][name]["source_envelope"]
            context = f"{name} mode=contact seed={HELD_OUT_SEEDS[0]} fault={fault}"
            baseline = [
                (source_projection(public_step(candidate, action), reference), reference)
                for action, reference in zip(actions, expected, strict=True)
            ]
            _check_pairs(baseline, source_caps, context=f"{context} baseline", source=True)
            candidate.set_state(snapshot)
            runtime = candidate._simulator._runtime
            bit = mujoco.mjtDisableBit.mjDSBL_CONTACT if fault == "contact" else mujoco.mjtDisableBit.mjDSBL_CONSTRAINT
            with monkeypatch.context() as patch:
                patch.setattr(runtime._m.opt, "disableflags", runtime._m.opt.disableflags | int(bit))
                # Capture must be rebuilt for the changed scalar option, and
                # forward must drop the old constrained acceleration first.
                runtime._graph = None
                runtime.load(snapshot)
                faulty = [
                    (source_projection(public_step(candidate, action), reference), reference)
                    for action, reference in zip(actions, expected, strict=True)
                ]
            _assert_source_fault_rejected(faulty, source_caps, context=context)
    finally:
        oracle.close()


@pytest.mark.parametrize("name", CELLS)
def test_lane_isolation_and_cached_gaussian_reset(name, evidence):
    import stable_envs as se

    caps = evidence["environments"][name]
    with make_candidate(name) as candidate:
        snapshot = candidate.reset(seed=HELD_OUT_SEEDS[0]).clone()
        action = torch.full((3, candidate.action_dim), 0.25, device="cuda", dtype=torch.float64)
        expected = public_step(candidate, action)
        changed = snapshot.clone()
        changed["qpos"][0, 0] += 0.2
        candidate.set_state(changed)
        different_action = action.clone()
        different_action[0] = -1
        actual = public_step(candidate, different_action)
        assert_bounded(
            actual[1:],
            expected[1:],
            caps["replay"]["1"],
            context=f"{name} mode=lane_isolation seed={HELD_OUT_SEEDS[0]} horizon=1",
        )
        assert not torch.equal(actual["state", "qpos"][0], expected["state", "qpos"][0])

        snapshot["rng", "has_gauss"][:] = True
        snapshot["rng", "cached_gaussian"][:, 0] = torch.tensor(
            [0.123456789012345, -0.75, 1.25], device="cuda", dtype=torch.float64
        )
        snapshot["lifecycle", "reset_next_step"][:, 0] = torch.tensor([False, True, False], device="cuda")
        # The native backend is only an interoperability oracle here. The source
        # fidelity gates above remain independent SWM/dm_control implementations.
        with se.make(name, num_envs=3, device="cuda", simulation_backend="native") as native:
            native.set_state(snapshot)
            candidate.set_state(snapshot)
            assert_same_state(candidate.get_state(), snapshot)
            expected = public_step(native, action)
            actual = public_step(candidate, action)
            assert_bounded(
                actual,
                expected,
                project_source_caps(caps["source_envelope"]["1"], expected),
                context=f"{name} mode=cached_reset_native seed={HELD_OUT_SEEDS[0]} horizon=1",
            )


@pytest.mark.parametrize("name", CELLS)
def test_capture_selection_and_repeat_ownership(name, evidence):
    """Sequence dispatch follows repeated public steps within the frozen resume bounds."""
    caps = evidence["environments"][name]["replay"]["8"]
    with make_candidate(name) as candidate:
        candidate.reset(seed=HELD_OUT_SEEDS[0])
        snapshot = candidate.get_state().clone()
        rng = np.random.default_rng(20991)
        actions = torch.tensor(rng.uniform(-1, 1, (3, 8, candidate.action_dim)), device="cuda")
        expected = [public_step(candidate, actions[:, t]) for t in range(8)]
        candidate.set_state(snapshot)
        actual = candidate.step(actions, states=slice(None, None, -2), frames=slice(0, 0))
        assert actual["frames"].shape[1] == 0
        assert actual["states"].batch_size == torch.Size([3, 4])
        task_caps = {key: value for key, value in caps.items() if not key.startswith("state/")}
        state_caps = {key: value for key, value in caps.items() if key.startswith("state/")}
        for t in range(8):
            wanted = expected[t].exclude("state")
            task = TensorDict({key: actual[key][:, t] for key in wanted.keys(True, True)}, [3])
            assert_bounded(task, wanted, task_caps)
        for selected, t in enumerate((7, 5, 3, 1)):
            selected_state = TensorDict({"state": actual["states"][:, selected]}, [3])
            assert_bounded(selected_state, expected[t].select("state"), state_caps)
        assert_bounded(actual.select("state"), expected[-1].select("state"), state_caps)
