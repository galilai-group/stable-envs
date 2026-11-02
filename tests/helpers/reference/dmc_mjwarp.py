"""Independent DMC CUDA calibration, with no built-in numerical acceptance caps.

Run ``python -m tests.helpers.reference.dmc_mjwarp --source-root CHECKOUT
--output NEW_DIRECTORY`` on CUDA. The twelve SWM cells use the pinned SWM
wrappers; the five siblings use dm_control directly. Neither oracle uses the
candidate's serializer. Calibration and held-out seeds are disjoint.

Bounds are per task, source/replay comparison, horizon and leaf: 1.5 * measured maximum
absolute error + eps(float32) * max(1, maximum absolute source value). The
second term is the explicitly recorded float32 floor, not an observed error.
Counters, masks, RNG, model/physics parameters and time remain exact. This
measures a candidate; it does not bless a large observed error automatically.
The source envelope pools nominal, boundary, variation, reached-contact and
cached-reset trials. Per-mode measurements and caps remain diagnostic evidence.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict

from stable_envs.testing.evidence import EvidenceRun, sha256, verify_seal
from stable_envs.testing.provenance import pinned_source_identity, verify_recorded_code
from tests.helpers.reference.bridges import load_source
from tests.helpers.reference.dmc_semantics import PHYSICS_LEAVES, DMCReference
from tests.helpers.reference.dmc_suite import SuiteReference
from tests.helpers.reference.profiles import PINS, PROFILES
from tests.helpers.reference.saved import SWM_PIN
from tests.helpers.reference.semantics import assert_same_state, compare_state

SWM_CELLS = {cell.name: cell.domain for cell in PROFILES["dmc"]}
SUITE_CELLS = {
    "DMCCartpoleBalance": ("cartpole", "balance"),
    "DMCFingerSpin": ("finger", "spin"),
    "DMCFingerTurnEasy": ("finger", "turn_easy"),
    "DMCPointMassEasy": ("point_mass", "easy"),
    "DMCSwimmerSwimmer6": ("swimmer", "swimmer6"),
}
CELLS = tuple(sorted((SWM_CELLS.keys() | SUITE_CELLS.keys()) - {"DMCReacher"}))
VARIED_CELLS = frozenset(SWM_CELLS) | {"DMCCartpoleBalance"}
LOCOMOTION_CELLS = frozenset({"DMCCheetahRun", "DMCHopperHop", "DMCWalkerWalk", "DMCQuadrupedWalk", "DMCHumanoidWalk"})
CONTACT_CELLS = LOCOMOTION_CELLS | {"DMCFingerSpin", "DMCFingerTurnEasy", "DMCFingerTurnHard"}
CONSTRAINT_CELLS = LOCOMOTION_CELLS | {"DMCBallInCupCatch", "DMCManipulatorBringBall"}
HORIZONS = (1, 8)
CALIBRATION_SEEDS = tuple(range(50))
HELD_OUT_SEEDS = tuple(range(10010, 10020))
BATCH = 3
PREFIX = 5
MARGIN = 1.5
SCHEMA = "stable-envs-dmc-mjwarp-calibration/v3"
ROOT = Path(__file__).resolve().parents[3]


def require_cuda():
    """Explicit gates fail when CUDA is missing; a skipped CPU run is not evidence."""
    if not torch.cuda.is_available():
        raise RuntimeError("DMC MJWarp validation requires an actual CUDA device")
    import mujoco_warp  # noqa: F401
    import warp as wp

    device = wp.get_device(f"cuda:{torch.cuda.current_device()}")
    if not device.is_cuda:
        raise RuntimeError("Warp did not select CUDA")


def machine_identity():
    require_cuda()
    import warp as wp

    index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    return {
        "device": properties.name,
        "compute_capability": list(torch.cuda.get_device_capability(index)),
        "uuid": str(getattr(properties, "uuid", "unavailable")),
        "total_memory": properties.total_memory,
        "torch_cuda": torch.version.cuda,
        "warp_driver": list(wp.get_cuda_driver_version() or ()),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "numpy", "mujoco", "mujoco-warp", "warp-lang", "dm-control")
        },
    }


def candidate_identity():
    """Hash implementation files, including untracked work."""
    files = sorted((ROOT / "stable_envs").rglob("*.py"))
    return {path.relative_to(ROOT).as_posix(): sha256(path) for path in files}


def recipe_identity():
    """Oracle serializers, bridges, calibration and acceptance tests remain frozen."""
    files = [
        *sorted((ROOT / "tests/helpers").rglob("*.py")),
        *sorted((ROOT / "tests/gpu").glob("test_dmc_mjwarp*.py")),
        *sorted((ROOT / "tests/unit").glob("test_dmc_mjwarp*.py")),
        ROOT / "tests/unit/test_dmc_sparse_outputs.py",
        ROOT / "tests/unit/test_dmc_capture_selection.py",
        ROOT / "scripts/ci/run_dmc_mjwarp.py",
    ]
    return {path.relative_to(ROOT).as_posix(): sha256(path) for path in files}


def prepare_source(root):
    if not root:
        raise ValueError("Set SE_DMC_MJWARP_SOURCE_ROOT or pass --source-root to the pinned SWM checkout")
    identity = pinned_source_identity(root, SWM_PIN, PINS["dmc"])
    load_source(root, "dmc")
    return identity


class Oracle:
    """Normalize independently serialized source results to public task outputs."""

    def __init__(self, name, batch=BATCH, *, varied=False):
        self.name, self.batch, self.varied = name, batch, varied
        self.is_swm = name in SWM_CELLS or (varied and name == "DMCCartpoleBalance")
        self.reset_options = None
        if varied and name == "DMCCartpoleBalance":
            # Existing independent SWM bridge: only the task is replaced with
            # dm_control's Balance; SWM still recompiles each varied model.
            from tests.helpers.reference.gates.dmc_physics import _balance_wrapper

            self.source = DMCReference("cartpole", batch, wrapper=_balance_wrapper())
        elif self.is_swm:
            self.source = DMCReference(SWM_CELLS[name], batch)
        else:
            self.source = SuiteReference(*SUITE_CELLS[name], batch)

    def reset(self, seed):
        self.seed = seed
        if self.varied:
            values = {}
            for lane, wrapper in enumerate(self.source.envs):
                selected = {}
                for leaf in PHYSICS_LEAVES[self.source.domain]:
                    node = wrapper.variation_space
                    for part in leaf.split("."):
                        node = node[part]
                    init = np.asarray(node.init_value).reshape(-1)
                    low, high = float(np.asarray(node.low).item()), float(np.asarray(node.high).item())
                    value = float(init[0]) if lane == 0 else (low if lane == 1 else high)
                    selected[leaf] = np.asarray([value], dtype=init.dtype)
                    values.setdefault(leaf, []).append(selected[leaf])
                wrapper.reset(seed=seed + lane, options={"variation_values": selected})
            self.reset_options = {"variation_values": {leaf: np.stack(rows) for leaf, rows in values.items()}}
            return self.source.get_state()
        return self.source.reset(seed)

    def near_limit(self):
        # Modify the independent source's lifecycle directly, never from candidate State.
        for lane, wrapper in enumerate(self.source.envs):
            env = wrapper.env._env if self.is_swm else wrapper._env
            env._step_count = self.source.step_limit - 3 + lane
        return self.source.get_state()

    def cached_reset(self):
        """Source-owned cached normal draws and a first-round reset in lane 1."""
        state = self.source.get_state().clone()
        state["rng", "has_gauss"][:] = True
        state["rng", "cached_gaussian"][:, 0] = torch.tensor([0.123456789012345, -0.75, 1.25], dtype=torch.float64)
        state["lifecycle", "reset_next_step"][:, 0] = torch.tensor([False, True, False])
        self._restore(state)
        actual = self.source.get_state()
        compare_state(actual, state)
        return actual

    def step(self, actions):
        result = self.source.step(actions)
        if self.is_swm:
            return result
        state, reward, observation = result
        output = TensorDict(
            {
                "state": state,
                "reward": reward,
                "observation": observation,
                "score": torch.tensor(self.source.cumulative.copy(), dtype=torch.float64) / 1000,
                "terminated": torch.zeros(self.batch, dtype=torch.bool),
                "truncated": torch.zeros(self.batch, dtype=torch.bool),
            },
            [self.batch],
        )
        if self.name != "DMCCartpoleBalance":
            rows = []
            for wrapper in self.source.envs:
                physics = wrapper.physics
                row = {"qpos": physics.data.qpos.copy(), "qvel": physics.data.qvel.copy()}
                if self.source.domain == "finger":
                    row.update(
                        hinge_velocity=np.asarray(physics.hinge_velocity()),
                        target_position=physics.target_position().copy(),
                        tip_position=physics.tip_position().copy(),
                    )
                rows.append(TensorDict(row, []))
            output["info"] = torch.stack(rows)
            output["success"] = torch.full((self.batch,), float("nan"), dtype=torch.float64)
        return output

    def close(self):
        self.source.close()

    def _restore(self, state):
        if self.is_swm:
            self.source.set_state(state)
            return
        for lane, (wrapper, row) in enumerate(zip(self.source.envs, state.cpu().unbind(0), strict=True)):
            physics, env = wrapper.physics, wrapper._env
            for key, (field, index) in self.source.fields.items():
                getattr(physics.named.model, field)[index] = row[key].numpy()
            physics.data.qpos[:] = row["qpos"].numpy()
            physics.data.qvel[:] = row["qvel"].numpy()
            physics.data.time = row["time"].item()
            physics.forward()
            physics.data.qacc_warmstart[:] = row["warmstart"].numpy()
            env._step_count = row["lifecycle", "step_count"].item()
            env._reset_next_step = row["lifecycle", "reset_next_step"].item()
            self.source.cumulative[lane] = row["task", "cumulative_reward"].item()
            env.task.random.set_state(
                (
                    "MT19937",
                    row["rng", "keys"].numpy().astype(np.uint32),
                    row["rng", "position"].item(),
                    int(row["rng", "has_gauss"].item()),
                    row["rng", "cached_gaussian"].item(),
                )
            )

    def _physics(self):
        return [wrapper.env.physics if self.is_swm else wrapper.physics for wrapper in self.source.envs]

    def contact_witness(self):
        import mujoco

        rows = []
        for physics in self._physics():
            force, strongest = np.zeros(6), 0.0
            for contact in range(physics.data.ncon):
                if self.name.startswith("DMCFinger"):
                    pair = physics.data.contact[contact]
                    names = {
                        mujoco.mj_id2name(physics.model.ptr, mujoco.mjtObj.mjOBJ_GEOM, int(geom))
                        for geom in (pair.geom1, pair.geom2)
                    }
                    if "fingertip" not in names or not names.intersection({"cap1", "cap2"}):
                        continue
                mujoco.mj_contactForce(physics.model.ptr, physics.data.ptr, contact, force)
                strongest = max(strongest, float(np.linalg.norm(force)))
            required_type = {"DMCBallInCupCatch": 4, "DMCManipulatorBringBall": 0}.get(self.name)
            selected = physics.data.efc_force
            if required_type is not None:
                selected = selected[physics.data.efc_type == required_type]
            rows.append(
                dict(
                    ncon=int(physics.data.ncon),
                    nefc=int(physics.data.nefc),
                    contact_force=strongest,
                    constraint_force=float(np.max(np.abs(physics.data.efc_force), initial=0)),
                    required_constraint_force=float(np.max(np.abs(selected), initial=0)),
                    required_constraint_type=-1 if required_type is None else required_type,
                )
            )
        return rows

    def contact_checkpoint(self):
        """Independent native fixtures with active forces after reconstruction.

        Locomotion settles for 200 actions. Finger reaches spinner contact from
        a collision-free joint-limit-valid pose using one bounded source action.
        Cup starts just inside its rope's native length limit and advances under
        gravity; Manipulator advances until its tendon equality carries force.
        """
        width = self.source.action_dim if self.is_swm else self.source.envs[0].action_spec().shape[0]
        if self.name in LOCOMOTION_CELLS:
            for _ in range(200):
                self.source.step(np.zeros((self.batch, width)))
        if self.name.startswith("DMCFinger"):
            for physics in self._physics():
                for joint, value in (("proximal", -0.5), ("distal", 0.2), ("hinge", 0.0)):
                    if joint != "hinge":
                        lo, hi = physics.named.model.jnt_range[joint]
                        assert lo <= value <= hi
                    physics.named.data.qpos[joint] = value
                physics.data.qvel[:] = 0
                physics.data.qacc_warmstart[:] = 0
                physics.forward()
                assert physics.data.ncon == 0, "Finger pre-contact fixture must begin collision-free"
            self.source.step(np.tile([0.0, 1.0], (self.batch, 1)))
        elif self.name == "DMCBallInCupCatch":
            for physics in self._physics():
                cup, ball = physics.named.data.site_xpos["cup"].copy(), physics.named.data.site_xpos["ball"].copy()
                length = float(physics.model.tendon_range[0, 1]) - 0.001
                physics.named.data.qpos["ball_x"] += cup[0] - ball[0]
                physics.named.data.qpos["ball_z"] += cup[2] - length - ball[2]
                physics.data.qvel[:] = 0
                physics.data.qacc_warmstart[:] = 0
                physics.forward()
                assert not np.any(physics.data.efc_force[physics.data.efc_type == 4]), "Cup fixture must start slack"
            self.source.step(np.zeros((self.batch, width)))
        elif self.name == "DMCManipulatorBringBall":
            self.source.step(np.tile(np.linspace(-0.8, 0.9, width), (self.batch, 1)))
        for _ in range(1024):
            state = self.source.get_state()
            self._restore(state)
            witness = self.contact_witness()
            key = "contact_force" if self.name in CONTACT_CELLS else "required_constraint_force"
            if all(row[key] > 0 for row in witness):
                self.contact_evidence = witness
                return state
            self.source.step(np.zeros((self.batch, width)))
        raise AssertionError(f"{self.name}: no active reconstructed contact/constraint in every source lane")


def make_candidate(name, batch=BATCH, **kwargs):
    import stable_envs as se

    env = se.make(name, num_envs=batch, device="cuda", simulation_backend="mjwarp", **kwargs)
    if env.simulation_backend != "mjwarp":
        env.close()
        raise AssertionError(f"{name}: explicit MJWarp request fell back to {env.simulation_backend}")
    return env


def public_step(env, action):
    return env.step(action, states=None, frames=None).exclude("valid")


def source_projection(value, expected):
    """Only sibling State's absent physics subtree is excluded, explicitly."""
    actual = value.clone()
    if "state" in expected.keys():
        if "physics" not in expected["state"].keys():
            actual["state"] = actual["state"].exclude("physics")
    elif "physics" not in expected.keys():
        actual = actual.exclude("physics")
    return actual


def exact_path(path, tensor):
    parts = path.split("/")
    if parts[0] == "state":
        parts = parts[1:]
    return (
        not tensor.is_floating_point()
        or parts[0] in {"rng", "lifecycle", "physics", "model", "time"}
        or (parts == ["task", "target_pos"])
        or parts == ["success"]
    )


def leaves(value):
    return {"/".join((key,) if isinstance(key, str) else key): value[key] for key in value.keys(True, True)}


def differences(actual, expected):
    """Measure continuous drift; structural and logical mismatches fail immediately."""
    a, e = leaves(actual), leaves(expected)
    assert actual.batch_size == expected.batch_size
    assert a.keys() == e.keys(), f"Output keys differ: {a.keys() ^ e.keys()}"
    result = {}
    for path in sorted(e):
        got, want = a[path].detach().cpu(), e[path].detach().cpu()
        assert got.shape == want.shape and got.dtype == want.dtype, path
        exact = exact_path(path, want)
        if exact:
            torch.testing.assert_close(got, want, atol=0, rtol=0, equal_nan=True, msg=path)
            error, scale = 0.0, 0.0
        else:
            assert torch.equal(torch.isnan(got), torch.isnan(want)), f"{path}: NaN mask differs"
            assert not torch.isinf(got).any() and not torch.isinf(want).any(), f"{path}: infinite result"
            finite = torch.isfinite(want)
            error = float((got[finite] - want[finite]).abs().max()) if finite.any() else 0.0
            scale = float(want[finite].abs().max()) if finite.any() else 0.0
        result[path] = {"max_abs_error": error, "reference_absmax": scale, "exact": exact}
    return result


def project_source_caps(envelope, expected):
    """Select declared source output paths; a missing cap is always an error.

    Cartpole Balance's varied SWM oracle includes physics metadata absent from
    its nominal Suite oracle. The envelope includes the union; each comparison
    still requires an exact match between actual, expected and selected caps.
    """
    paths = leaves(expected)
    missing = paths.keys() - envelope.keys()
    assert not missing, f"Source envelope has no caps for {sorted(missing)}"
    return {path: envelope[path] for path in paths}


def assert_bounded(actual, expected, caps, *, context=""):
    try:
        measured = differences(actual, expected)
        assert measured.keys() == caps.keys(), "Frozen cap coverage differs from actual output fields"
        for path, value in measured.items():
            cap = caps[path]
            assert value["exact"] == cap["exact"], path
            assert value["max_abs_error"] <= cap["atol"], (
                f"{path}: absolute error {value['max_abs_error']:.9g} exceeds frozen cap {cap['atol']:.9g}"
            )
    except AssertionError as error:
        if context:
            raise AssertionError(f"{context}: {error}") from error
        raise


def actions_for(seed, width):
    return np.random.default_rng(seed + 91717).uniform(-1, 1, (PREFIX + max(HORIZONS), BATCH, width))


def source_trial(env, oracle, seed, *, boundary=False, contact=False, cached_reset=False):
    assert sum((boundary, contact, cached_reset)) <= 1
    mode = (
        "boundary"
        if boundary
        else "contact"
        if contact
        else "cached_reset"
        if cached_reset
        else "variation"
        if oracle.varied
        else "source"
    )
    try:
        return _source_trial(env, oracle, seed, boundary=boundary, contact=contact, cached_reset=cached_reset)
    except AssertionError as error:
        raise AssertionError(f"{oracle.name} mode={mode} seed={seed}: {error}") from error


def _source_trial(env, oracle, seed, *, boundary, contact, cached_reset):
    expected = oracle.reset(seed)
    actual = env.reset(seed=seed, options=oracle.reset_options)
    # Varied masses are analytically updated by the candidate but recompiled by
    # SWM. Existing native physics gates measure their roundoff separately;
    # episode-drawn fields/parameters are exact, physical reset values measured.
    if oracle.varied:
        oracle.reset_pair = (
            TensorDict({"state": source_projection(actual, expected).cpu()}, [BATCH]),
            TensorDict({"state": expected}, [BATCH]),
        )
        differences(*oracle.reset_pair)
    else:
        compare_state(source_projection(actual, expected), expected)
    if contact or cached_reset:
        expected = oracle.contact_checkpoint() if contact else oracle.cached_reset()
        snapshot = env.get_state()
        for key in expected.keys(True, True):
            snapshot[key] = expected[key].cuda()
        env.set_state(snapshot)
        compare_state(source_projection(env.get_state(), expected), expected)
    if boundary:
        expected = oracle.near_limit()
        snapshot = env.get_state()
        snapshot["lifecycle", "step_count"][:] = expected["lifecycle", "step_count"].cuda()
        env.set_state(snapshot)
    actions = actions_for(seed, env.action_dim)[: max(HORIZONS)]
    pairs = []
    for action in actions:
        expected = oracle.step(action)
        actual = public_step(env, torch.as_tensor(action, device="cuda"))
        pairs.append((source_projection(actual, expected).cpu(), expected))
    return actions, pairs


def source_modes(name):
    return (
        ["source", "boundary", "cached_reset"]
        + (["variation"] if name in VARIED_CELLS else [])
        + (["contact"] if name in CONTACT_CELLS | CONSTRAINT_CELLS else [])
    )


def replay_trial(env, seed):
    env.reset(seed=seed)
    actions = actions_for(seed, env.action_dim)
    for action in actions[:PREFIX]:
        public_step(env, torch.as_tensor(action, device="cuda"))
    snapshot = env.get_state().clone()
    expected = [public_step(env, torch.as_tensor(a, device="cuda")).cpu() for a in actions[PREFIX:]]
    env.reset(seed=seed + 200000)
    env.set_state(snapshot)
    assert_same_state(env.get_state(), snapshot)
    actual = [public_step(env, torch.as_tensor(a, device="cuda")).cpu() for a in actions[PREFIX:]]
    return actions, list(zip(actual, expected, strict=True))


def measure_pairs(pairs, horizon):
    aggregate = {}
    for actual, expected in pairs[:horizon]:
        for path, value in differences(actual, expected).items():
            previous = aggregate.setdefault(path, dict(value))
            for key in ("max_abs_error", "reference_absmax"):
                previous[key] = max(previous[key], value[key])
    return aggregate


def freeze_caps(trials):
    result = {}
    for horizon in HORIZONS:
        aggregate = {}
        for trial in trials:
            for path, value in trial[str(horizon)].items():
                previous = aggregate.setdefault(path, dict(value))
                assert previous["exact"] == value["exact"], f"Inconsistent exact classification: {path}"
                for key in ("max_abs_error", "reference_absmax"):
                    previous[key] = max(previous[key], value[key])
        for value in aggregate.values():
            floor = 0.0 if value["exact"] else np.finfo(np.float32).eps * max(1.0, value["reference_absmax"])
            value["float32_floor"] = float(floor)
            value["atol"] = MARGIN * value["max_abs_error"] + float(floor)
        result[str(horizon)] = aggregate
    return result


def load_evidence(directory, *, allow_implementation_change=False):
    if not directory:
        raise ValueError("Set SE_DMC_MJWARP_EVIDENCE to a sealed CUDA calibration run; no fallback caps exist")
    root = Path(directory)
    verify_seal(root)
    record = json.loads((root / "calibration.json").read_text())
    assert record["schema"] == SCHEMA and record["batch"] == BATCH
    assert record["seeds"] == list(CALIBRATION_SEEDS) and record["horizons"] == list(HORIZONS)
    assert record["held_out_seeds"] == list(HELD_OUT_SEEDS)
    assert record["source_envelope_modes"] == {name: source_modes(name) for name in record["environments"]}
    assert record["margin"] == MARGIN and record["source"]["commit"] == SWM_PIN
    assert record["machine"] == machine_identity(), "Calibration runtime/GPU differs; recalibrate explicitly"
    verify_recorded_code(record["recipe_files"], ROOT)
    if not allow_implementation_change:
        verify_recorded_code(record["candidate_files"], ROOT)
    return record


def calibrate(names, source_root, output):
    require_cuda()
    source = prepare_source(source_root)
    evidence = EvidenceRun(output)
    record = dict(
        schema=SCHEMA,
        batch=BATCH,
        seeds=list(CALIBRATION_SEEDS),
        horizons=list(HORIZONS),
        margin=MARGIN,
        source=source,
        candidate_files=candidate_identity(),
        recipe_files=recipe_identity(),
        machine=machine_identity(),
        environments={},
        status="calibrated_not_accepted",
        held_out_seeds=list(HELD_OUT_SEEDS),
        source_envelope_modes={name: source_modes(name) for name in names},
    )
    for name in names:
        samples, caps = {}, {}
        with make_candidate(name) as env:
            for mode in [*source_modes(name), "replay"]:
                oracle = Oracle(name, varied=mode == "variation")
                try:
                    trials, arrays = [], {}
                    for seed in CALIBRATION_SEEDS:
                        actions, pairs = (
                            replay_trial(env, seed)
                            if mode == "replay"
                            else source_trial(
                                env,
                                oracle,
                                seed,
                                boundary=mode == "boundary",
                                contact=mode == "contact",
                                cached_reset=mode == "cached_reset",
                            )
                        )
                        measured = {str(h): measure_pairs(pairs, h) for h in HORIZONS}
                        if mode == "variation":
                            reset_errors = differences(*oracle.reset_pair)
                            for horizon in HORIZONS:
                                for path, value in reset_errors.items():
                                    for key in ("max_abs_error", "reference_absmax"):
                                        measured[str(horizon)][path][key] = max(
                                            measured[str(horizon)][path][key], value[key]
                                        )
                            for side, state in zip(("candidate", "reference"), oracle.reset_pair, strict=True):
                                for path, value in leaves(state).items():
                                    arrays.setdefault("reset/" + side + "/" + path, []).append(value.numpy())
                        trials.append(measured)
                        if mode == "contact":
                            for key in (
                                "ncon",
                                "nefc",
                                "contact_force",
                                "constraint_force",
                                "required_constraint_force",
                                "required_constraint_type",
                            ):
                                arrays.setdefault("contact_witness/" + key, []).append(
                                    np.asarray([row[key] for row in oracle.contact_evidence])
                                )
                        arrays.setdefault("actions", []).append(actions)
                        for side, index in (("candidate", 0), ("reference", 1)):
                            for path in leaves(pairs[0][index]):
                                rows = [leaves(pair[index])[path].numpy() for pair in pairs]
                                arrays.setdefault(side + "/" + path, []).append(np.stack(rows))
                    caps[mode], samples[mode] = freeze_caps(trials), trials
                    evidence.arrays(f"{name}/{mode}.npz", **{k: np.stack(v) for k, v in arrays.items()})
                    print(f"Calibrated {name} {mode}: {len(trials)} seeds x {BATCH} lanes", flush=True)
                finally:
                    oracle.close()
        # Pool measured errors and reference magnitudes before applying the
        # unchanged margin/floor formula. Every source gate and physical fault
        # detector uses this same task/field/horizon envelope.
        caps["source_envelope"] = freeze_caps([trial for mode in source_modes(name) for trial in samples[mode]])
        evidence.json(f"{name}/measurements.json", samples)
        evidence.json(f"{name}/caps.json", caps)
        record["environments"][name] = caps
    verify_recorded_code(record["candidate_files"], ROOT)
    verify_recorded_code(record["recipe_files"], ROOT)
    verify_recorded_code(record["source"]["files"], source_root)
    evidence.json("calibration.json", record)
    evidence.seal()
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default=os.environ.get("SE_DMC_MJWARP_SOURCE_ROOT"))
    parser.add_argument("--output", required=True, help="new directory; existing evidence is never overwritten")
    parser.add_argument("--envs", nargs="+", choices=CELLS, default=list(CELLS))
    args = parser.parse_args()
    calibrate(args.envs, args.source_root, args.output)


if __name__ == "__main__":
    main()
