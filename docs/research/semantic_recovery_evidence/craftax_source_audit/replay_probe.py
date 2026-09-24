"""Research-only named numeric snapshots of the real pinned SWM/Craftax source."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/craftax-mpl")
import dataclasses
import importlib
import importlib.util
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch
from tensordict import TensorDict

HERE = Path(__file__).resolve().parent
SWM = Path("/Users/samib/stable-worldmodel/stable_worldmodel/envs/craftax/craftax.py")
spec = importlib.util.spec_from_file_location("pinned_craftax_oracle", SWM)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def leaves(obj, path=()):
    if dataclasses.is_dataclass(obj):
        for f in dataclasses.fields(obj):
            yield from leaves(getattr(obj, f.name), path + (f.name,))
    elif isinstance(obj, tuple):
        for i, x in enumerate(obj):
            yield from leaves(x, path + (str(i),))
    elif obj is not None:
        yield path, np.asarray(obj)


def pack(env):
    values = {("engine", *k): torch.from_numpy(v.copy()).unsqueeze(0) for k, v in leaves(env._state)}
    values["rng", "wrapper_key"] = torch.from_numpy(np.asarray(env._key).copy()).unsqueeze(0)
    return TensorDict(values, batch_size=[1])


def decode(template, td, path=()):
    if dataclasses.is_dataclass(template):
        return template.replace(
            **{f.name: decode(getattr(template, f.name), td, path + (f.name,)) for f in dataclasses.fields(template)}
        )
    if isinstance(template, tuple):
        return tuple(decode(v, td, path + (str(i),)) for i, v in enumerate(template))
    if template is None:
        return None
    return jnp.asarray(td[("engine", *path)][0].numpy())


def restore(env, td):
    env._state = decode(env._state, td)
    env._key = jnp.asarray(td["rng", "wrapper_key"][0].numpy())


def equal(a, b, label):
    aa = dict(leaves(a))
    bb = dict(leaves(b))
    if not aa and isinstance(a, dict):
        assert a.keys() == b.keys(), label
        for k in a:
            equal(a[k], b[k], label + ":" + str(k))
        return
    if not aa:
        aa = {(): np.asarray(a)}
        bb = {(): np.asarray(b)}
    assert aa.keys() == bb.keys(), label
    for k in aa:
        if not np.array_equal(aa[k], bb[k]):
            np.savez_compressed(HERE / "replay-mismatch.npz", a=aa[k], b=bb[k])
            raise AssertionError((label, k, aa[k].shape, bb[k].shape))


def td_equal(a, b):
    assert set(a.keys(True, True)) == set(b.keys(True, True))
    assert all(torch.equal(a[k], b[k]) for k in a.keys(True, True))


def fixture(s, classic, kind, params):
    c = importlib.import_module("craftax." + ("craftax_classic" if classic else "craftax") + ".constants")
    y, x = map(int, s.player_position)
    level = 0 if classic else int(s.player_level)
    m = (
        s.map.at[y - 2 : y + 3, x - 2 : x + 3].set(c.BlockType.GRASS.value)
        if classic
        else s.map.at[level, y - 2 : y + 3, x - 2 : x + 3].set(c.BlockType.GRASS.value)
    )
    state = s.replace(map=m, player_direction=jnp.asarray(2, jnp.int32), mob_map=jnp.zeros_like(s.mob_map))
    groups = (
        ["zombies", "cows", "skeletons", "arrows"]
        if classic
        else ["melee_mobs", "passive_mobs", "ranged_mobs", "mob_projectiles", "player_projectiles"]
    )
    state = state.replace(
        **{k: getattr(state, k).replace(mask=jnp.zeros_like(getattr(state, k).mask)) for k in groups}
    )

    def cell(v):
        return state.map.at[y, x + 1].set(v) if classic else state.map.at[level, y, x + 1].set(v)

    if kind == "mine_tree":
        state = state.replace(map=cell(c.BlockType.TREE.value))
        action = 5
    elif kind == "craft":
        state = state.replace(
            map=cell(c.BlockType.CRAFTING_TABLE.value),
            inventory=state.inventory.replace(wood=jnp.asarray(9, jnp.int32)),
        )
        action = 11
    elif kind == "combat":
        name = "zombies" if classic else "melee_mobs"
        mob = getattr(state, name)
        ix = 0 if classic else (level, 0)
        mob = mob.replace(
            position=mob.position.at[ix].set(jnp.array([y, x + 1])),
            health=mob.health.at[ix].set(1),
            mask=mob.mask.at[ix].set(True),
        )
        mm = state.mob_map.at[y, x + 1].set(True) if classic else state.mob_map.at[level, y, x + 1].set(True)
        state = state.replace(**{name: mob, "mob_map": mm})
        action = 5
    elif kind == "damage":
        name = "zombies" if classic else "melee_mobs"
        mob = getattr(state, name)
        ix = 0 if classic else (level, 0)
        mob = mob.replace(
            position=mob.position.at[ix].set(jnp.array([y, x + 1])),
            health=mob.health.at[ix].set(5),
            mask=mob.mask.at[ix].set(True),
            attack_cooldown=mob.attack_cooldown.at[ix].set(0),
        )
        mm = state.mob_map.at[y, x + 1].set(True) if classic else state.mob_map.at[level, y, x + 1].set(True)
        state = state.replace(**{name: mob, "mob_map": mm})
        action = 0
    elif kind == "night":
        state = state.replace(timestep=jnp.asarray(150, jnp.int32), light_level=jnp.asarray(0.05, jnp.float32))
        action = 0
    elif kind == "terminal":
        state = state.replace(player_health=jnp.zeros_like(state.player_health))
        action = 0
    elif kind == "near_limit":
        state = state.replace(timestep=jnp.asarray(params.max_timesteps - 1, jnp.int32))
        action = 0
    else:
        action = 5
    return state, action


def main():
    results = {}
    for classic in [True, False]:
        for pixels in [False, True]:
            clsname = "Craftax" + ("Classic" if classic else "") + ("Pixels" if pixels else "Symbolic") + "Wrapper"
            cls = getattr(mod, clsname)
            a = cls(backend="cpu", render_mode="rgb_array")
            b = cls(backend="cpu", render_mode="rgb_array")
            row = {"status": "RUNNING", "cases": []}
            results[clsname] = row
            t0 = time.perf_counter()
            print("BEGIN", clsname, flush=True)
            for seed in [0, 17, 42]:
                obs, _ = a.reset(seed=seed)
                b.reset(seed=991)
                snap = pack(a)
                frozen = snap.clone()
                restore(b, snap)
                td_equal(snap, pack(b))
                for action in [0, 1, 2, 3, 4, 5, 6]:
                    ra = a.step(action)
                    rb = b.step(action)
                    np.testing.assert_array_equal(ra[0], rb[0])
                    assert ra[1:4] == rb[1:4]
                    for k in ra[4]:
                        if k != "env_name":
                            np.testing.assert_array_equal(ra[4][k], rb[4][k])
                    td_equal(pack(a), pack(b))
                td_equal(snap, frozen)
                restore(a, snap)
                restore(b, snap)
                np.testing.assert_array_equal(a.reset()[0], b.reset()[0])
                td_equal(pack(a), pack(b))
                row["cases"].append(
                    {
                        "seed": seed,
                        "reset_replay": True,
                        "independent_different_history": True,
                        "seven_steps": True,
                        "snapshot_independent": True,
                    }
                )
            a.reset(seed=17)
            base = a._state
            for kind in [
                "mine_tree",
                "craft",
                "combat",
                "damage",
                "night",
                "terminal",
                "near_limit",
                "stochastic_grass",
            ]:
                a._state, action = fixture(base, classic, kind, a._params)
                a._key = jax.random.PRNGKey(2026)
                snap = pack(a)
                b.reset(seed=42)
                b.step(3)
                restore(b, snap)
                ra = a.step(action)
                rb = b.step(action)
                np.testing.assert_array_equal(ra[0], rb[0])
                assert ra[1:4] == rb[1:4]
                td_equal(pack(a), pack(b))
                changed = ["/".join(k) for k in snap.keys(True, True) if not torch.equal(snap[k], pack(a)[k])]
                if kind == "mine_tree":
                    assert int(a._state.inventory.wood) > int(base.inventory.wood)
                if kind == "craft":
                    assert int(a._state.inventory.wood) < 9
                    assert ra[1] > 0
                if kind == "combat":
                    assert ra[1] > 0
                if kind == "damage":
                    assert float(a._state.player_health) < float(base.player_health)
                if kind in ["terminal", "near_limit"]:
                    assert ra[2] and not ra[3]
                row["cases"].append(
                    {
                        "fixture": kind,
                        "action": action,
                        "reward": ra[1],
                        "terminated": ra[2],
                        "truncated": ra[3],
                        "changed": changed,
                        "exact": True,
                    }
                )
                arrays = {"/".join(k): v.numpy() for k, v in snap.items(True, True)}
                arrays["expected/output"] = ra[0]
                np.savez_compressed(HERE / (clsname + "-" + kind + ".npz"), **arrays)
            # Separate pure output recomputation must be measured, not assumed identical across XLA boundaries.
            pure = jax.jit(a._env.get_obs)(a._state)
            jax.block_until_ready(pure)
            row["standalone_output_equals_step_output"] = bool(np.array_equal(np.asarray(pure), a._last_obs))
            row["engine_numeric_leaves"] = len(list(leaves(a._state)))
            row["weak_leaves"] = [
                list(path)
                for path, v in jax.tree_util.tree_flatten_with_path(a._state)[0]
                if getattr(v, "weak_type", False)
            ]
            row["symbolic_or_pixels_shape"] = list(a._last_obs.shape)
            row["status"] = "PASS"
            row["seconds"] = time.perf_counter() - t0
            (HERE / "replay-results.json").write_text(json.dumps(results, indent=2, default=str) + "\n")
            print("PASS", clsname, "cases", len(row["cases"]), "seconds", row["seconds"], flush=True)
    print("CRAFTAX_SOURCE_REPLAY_PASS", flush=True)


if __name__ == "__main__":
    main()
