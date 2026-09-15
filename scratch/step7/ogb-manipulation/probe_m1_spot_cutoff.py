"""M1, the half that needs no GPU: does MuJoCo's own renderer honour a spot cutoff above 90 degrees?

`mujoco_warp` attenuates a spot as `cos_theta < cos(cutoff)` -> unlit, then `pow(cos_theta, exp)`.
`swm/OGBCube-v0`'s spot is cutoff 100 deg, exponent 10. OpenGL's `GL_SPOT_CUTOFF` accepts [0, 90] or
the sentinel 180 and rejects everything else with `GL_INVALID_VALUE`, leaving the default 180 = "not
a spotlight" -- so if MuJoCo goes through that path, cutoff 100 and cutoff 180 must render the SAME
frame while a valid 89 must render a different one. That is a statement about MuJoCo alone and is
therefore measurable on any machine, which is why it is not in the GPU job.
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np


def main():
    import gymnasium as gym
    import mujoco
    import stable_worldmodel  # noqa: F401
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--env", default="swm/OGBCube-v0")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    env = gym.make(a.env); u = env.unwrapped
    env.reset(seed=21); m, d = u.model, u.data
    # Seed the ACTION space too. `action_space.sample()` draws from its own RNG, which
    # `reset(seed=)` does not touch, so an unseeded walk gives a different state every run and the
    # numbers in the report would not be the numbers in the JSON.
    env.action_space.seed(21)
    for _ in range(8):
        env.step(env.action_space.sample())
    spot = [i for i in range(m.nlight) if int(m.light_type[i]) == 0]
    rec = {"env": a.env, "spot_ids": spot,
           "cutoff": [float(m.light_cutoff[i]) for i in spot],
           "exponent": [float(m.light_exponent[i]) for i in spot],
           "diffuse": [m.light_diffuse[i].tolist() for i in spot],
           "pos": [m.light_pos[i].tolist() for i in spot],
           "dir": [m.light_dir[i].tolist() for i in spot],
           "cutoff_sweep": [], "exponent_sweep": []}
    base = u.render().copy()
    for cut in (100.0, 180.0, 90.0, 89.0, 45.0, 10.0):
        for i in spot:
            m.light_cutoff[i] = cut
        f = u.render().copy()
        rec["cutoff_sweep"].append({
            "cutoff": cut,
            "frac_differing_from_100": float((f != base).any(-1).mean()),
            "max_abs": int(np.abs(f.astype(np.int16) - base.astype(np.int16)).max()),
            "frame_mean": float(f.mean())})
    # ---- exponent sweep, in its own FRESH env ------------------------------------------------
    # The first version ran this on the model the cutoff sweep had been mutating and reported the
    # exponent as worth 2.40 % of pixels, contradicting the light-forms table on the same claim by
    # the same probe. A fresh env is the only restoration that is obviously complete; the verdict
    # below asserts the two tables now agree before either is believed.
    env3 = gym.make(a.env); u3 = env3.unwrapped
    env3.reset(seed=21); m3, d3 = u3.model, u3.data
    env3.action_space.seed(21)
    for _ in range(8):
        env3.step(env3.action_space.sample())
    spot3 = [i for i in range(m3.nlight) if int(m3.light_type[i]) == 0]
    base2 = u3.render().copy()
    _rep3 = u3.render().copy()
    _n3 = np.abs(_rep3.astype(np.int16) - base2.astype(np.int16)).max()
    rec["exponent_sweep_noise_max_abs"] = int(_n3)
    for ex in (10.0, 0.0, 1.0, 50.0):
        sv_e3 = m3.light_exponent.copy()
        for i in spot3:
            m3.light_exponent[i] = ex
        f = u3.render().copy()
        m3.light_exponent[:] = sv_e3
        rec["exponent_sweep"].append({
            "exponent": ex,
            "frac_differing_from_10": float((f != base2).any(-1).mean()),
            "max_abs": int(np.abs(f.astype(np.int16) - base2.astype(np.int16)).max())})
    env3.close()

    # ---- what "discarded" actually means: the forms this light can take --------------------
    # The obvious reading of "MuJoCo discards an out-of-range cutoff" is "so it is a point light".
    # That is measurably false, and this is the table that says so.
    #
    # A FRESH env, on purpose. Two earlier versions of this block ran on the model the sweeps above
    # had been mutating and compared every form against a baseline nine levels of mean away from the
    # cutoff table's own "cutoff = 100" row -- restoring the fields by hand did not reproduce the
    # starting state. Building a new env is the only restoration that is obviously complete, and the
    # verdict below asserts the two tables agree before either is believed.
    env2 = gym.make(a.env); u2 = env2.unwrapped
    env2.reset(seed=21); m2, d2 = u2.model, u2.data
    env2.action_space.seed(21)
    for _ in range(8):
        env2.step(env2.action_space.sample())
    spot2 = [i for i in range(m2.nlight) if int(m2.light_type[i]) == 0]
    base2 = u2.render().copy()
    # This host's renderer is not bit-stable frame to frame on this scene, so "no change" has to be
    # judged against the noise floor rather than against zero. Measured here, from two renders of an
    # unchanged MjData, and reported so the reader can see what "inert" was allowed to mean.
    _rep = u2.render().copy()
    _nd = np.abs(_rep.astype(np.int16) - base2.astype(np.int16)).max(-1)
    rec["render_noise_floor"] = {"frac_differing": float((_nd > 0).mean()),
                                 "max_abs": int(_nd.max())}
    rec["light_forms_baseline_mean"] = float(base2.mean())
    rec["light_forms"] = []

    def form(tag, mutate):
        undo = mutate()
        f = u2.render().copy()
        for k, v in undo.items():
            getattr(m2, k)[:] = v
        dd = np.abs(f.astype(np.int16) - base2.astype(np.int16)).max(-1)
        rec["light_forms"].append({
            "form": tag, "frame_mean": float(f.mean()),
            "frac_differing_from_shipped": float((dd > 0).mean()),
            "max_abs": int(dd.max())})

    def _set(field, value):
        def f_():
            u_ = {field: getattr(m2, field).copy()}
            for i in spot2:
                getattr(m2, field)[i] = value
            return u_
        return f_

    form("shipped (SPOT cutoff=100 exponent=10)", lambda: {})
    form("exponent 10 -> 0, cutoff still 100", _set("light_exponent", 0.0))
    form("cutoff -> 90", _set("light_cutoff", 90.0))
    form("cutoff -> 45", _set("light_cutoff", 45.0))
    form("light disabled", _set("light_active", 0))
    form("re-typed to POINT", _set("light_type", 2))
    env2.close()

    _noise = rec["render_noise_floor"]
    rec["verdict_exponent_is_inert_at_illegal_cutoff"] = bool(
        rec["light_forms"][1]["max_abs"] <= max(1, _noise["max_abs"])
        and rec["light_forms"][1]["frac_differing_from_shipped"] <= max(1e-3, 2 * _noise["frac_differing"]))
    rec["verdict_point_equals_disabled"] = bool(
        abs(rec["light_forms"][5]["frame_mean"] - rec["light_forms"][4]["frame_mean"]) < 1e-6)
    # Self-consistency WITHIN the table, not across the two. The two tables use different env
    # instances and `CubeEnv.reset` resamples its default variations from an RNG the gym seed does
    # not control (phase 0, G1d), so two envs seeded identically still place the cube differently and
    # their frame means legitimately differ by a few levels. What must hold is that the table's own
    # "shipped" row agrees with the baseline it is measured against, to within this host's render
    # noise -- that is the check the first two versions of this block failed.
    # The two tables must now agree about the exponent, which is the claim the shared correction's
    # docstring cites. Judged against each table's own render-noise floor.
    _e0 = [r for r in rec["exponent_sweep"] if r["exponent"] == 0.0]
    rec["verdict_exponent_tables_agree"] = bool(
        _e0 and _e0[0]["max_abs"] <= max(1, rec.get("exponent_sweep_noise_max_abs", 0))
        and rec["light_forms"][1]["max_abs"] <= max(1, rec["render_noise_floor"]["max_abs"]))
    rec["verdict_shipped_row_matches_its_baseline"] = bool(
        rec["light_forms"][0]["max_abs"] <= max(1, rec["render_noise_floor"]["max_abs"]))
    rec["note_cross_table_means_differ"] = (
        "light_forms and cutoff_sweep use different env instances; CubeEnv.reset resamples its "
        "default variations unseeded, so their absolute frame means are not comparable. Each table "
        "is internally consistent against its own baseline.")

    c100 = [r for r in rec["cutoff_sweep"] if r["cutoff"] == 100.0][0]
    c180 = [r for r in rec["cutoff_sweep"] if r["cutoff"] == 180.0][0]
    c89 = [r for r in rec["cutoff_sweep"] if r["cutoff"] == 89.0][0]
    rec["verdict_100_equals_180"] = bool(c180["frac_differing_from_100"] < 1e-3)
    rec["verdict_89_is_honoured"] = bool(c89["frac_differing_from_100"] > 0.5)
    (out / "m1_mujoco_spot_cutoff.json").write_text(json.dumps(rec, indent=1, sort_keys=True))
    for r in rec["light_forms"]:
        print(f"  {r['form']:38s} mean {r['frame_mean']:8.4f}  "
              f"differs {r['frac_differing_from_shipped']*100:7.3f}%  max {r['max_abs']:3d}")
    print("  exponent inert at illegal cutoff:", rec["verdict_exponent_is_inert_at_illegal_cutoff"])
    print("  exponent tables agree:", rec["verdict_exponent_tables_agree"],
          "| exponent_sweep:", [(r["exponent"], r["max_abs"]) for r in rec["exponent_sweep"]],
          "| noise", rec.get("exponent_sweep_noise_max_abs"))
    print("  POINT == disabled:", rec["verdict_point_equals_disabled"])
    print(json.dumps({k: rec[k] for k in ("cutoff", "exponent", "verdict_100_equals_180",
                                          "verdict_89_is_honoured")}, indent=1))
    for r in rec["cutoff_sweep"]:
        print(f"  cutoff {r['cutoff']:6.1f}  differs {r['frac_differing_from_100']*100:6.2f}%  "
              f"max {r['max_abs']:3d}  mean {r['frame_mean']:.3f}")


if __name__ == "__main__":
    main()
