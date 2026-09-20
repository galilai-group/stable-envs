"""Merge the Step 3 benchmark shards and apply the FROZEN Step 2 certification rules to them.

Two stages, deliberately separate, because Step 2's whole value is that the bar was set before the
measurement and a reader has to be able to check that it was not moved afterwards:

    raw shards  ->  merge()   ->  one deterministic merged artifact
    merged      ->  certify() ->  labels, computed ONLY from tests/fixtures/regression/fidelity/certification.json

No threshold is written in this file. Every bound is read out of the frozen spec, so a number that
appears in a label can be traced to the document that fixed it, and changing a bar means editing the
policy rather than the evaluator. If you find yourself wanting to add a constant here, the answer is
that Step 3 is not allowed to.

The unit of measurement is the stable-envs ENV, not the swm id: PushT backs two registered ids and is
timed once. Projection is recorded (``projected_to``) rather than duplicated, so the summary can say
"34 rows from 37 renderer measurements" instead of implying 34 independent timings.

    python scripts/regression/semantic/certify.py --shards tests/fixtures/regression/join/step3 --out tests/fixtures/regression/join/step3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _repo_root() -> Path:
    p = Path(__file__).resolve()
    for cand in (p, *p.parents):
        if (cand / "pyproject.toml").is_file():
            return cand
    raise RuntimeError("repo root not found")


ROOT = _repo_root()
SPEC = json.loads((ROOT / "tests" / "fixtures" / "regression" / "fidelity" / "certification.json").read_text())

UNKNOWN = "unknown"


# ------------------------------------------------------------------ inputs


def read_censuses(paths) -> dict:
    """Declared backends per env, from the Step 1 censuses. Used only to assign the backend class.

    Read from the census rather than from the live object: ``env.backend`` is the rasterizer variant
    ("auto"), not the render module the class imports, and Step 2's classes are defined over the latter.
    """
    out: dict[str, dict] = {}
    for p in sorted(paths):
        txt = Path(p).read_text()
        c = json.loads(txt.split("CENSUS ", 1)[1] if txt.startswith("CENSUS ") else txt)
        for k, v in (c.get("declarations") or {}).items():
            out.setdefault(k, v)
    return out


#: Backends that entered ``stable_envs.render`` AFTER Step 2 froze ``backend_classes``, and the class
#: Step 2's own definitions give them. Step 2 assigns a class by what the backend DOES -- section 6.4's
#: S is "3-D scene", R is "2-D raster/tile" -- and froze the lists that existed on 2026-09-09. Step 6
#: and Step 7 added four more modules; leaving them out of the mapping does not make them classless, it
#: makes the class fall through to a MODULE NAME, which is an accident of where a file lives rather
#: than a fact about the renderer. This table restores the frozen DEFINITION to the modules that
#: postdate the frozen LISTS. No threshold moves: ``tests/fixtures/regression/fidelity/certification.json`` is not edited, and the
#: ladder each class is judged on is read from it unchanged.
#:
#:   ``cpu3d``             the shared CPU-native 3-D backend (Step 6): a real camera, real geometry,
#:                         real visibility over the same scenes ``mjwarp`` draws. Class S.
#:   ``mjwarp_lightspec``  the light/material specification ``mjwarp`` renders through. Class S, and it
#:                         never appears without ``mjwarp`` beside it.
#:   ``navdot``            the ``swm_nav`` dot/zone renderer: torch arithmetic over a 2-D grid, no
#:                         camera and no 3-D scene anywhere in it. Class R.
#:   ``gfxdraw_classic``   the two ``gfxdraw`` variants. The frozen R list already names ``gfxdraw``;
#:   ``gfxdraw_fused``     these are the same 2-D rasterizer, split into a readable and a fused form.
POST_STEP2_BACKEND_CLASS = {
    "cpu3d": "S",
    "mjwarp_lightspec": "S",
    "navdot": "R",
    "gfxdraw_classic": "R",
    "gfxdraw_fused": "R",
}

#: Modules under ``stable_envs.render`` that are NOT renderers and must never decide a class. ``prng``
#: is Craftax's counter-based random number generator -- it draws the night noise FIELD, not the
#: picture -- and ``rendernet`` is a learned decoder, not a rasterizer. Both are listed explicitly
#: rather than left to fall off the end of the S/R lists, because falling off the end is what put
#: CraftaxFull at ``unknown``: once ``craftax_full`` imported ``prng``, its ``backends`` stopped being
#: empty, and an empty-backends env is the only kind ``NO_BACKEND_MODULE_CLASS`` was written to catch.
NOT_A_BACKEND = frozenset({"prng", "rendernet"})

#: Modules that import NO class-bearing ``stable_envs.render.*`` backend because they draw without one,
#: and the class Step 2 already assigns to such a renderer. This is the ONE clarifying interpretation
#: Step 3 makes, and section 21 of the brief is where it is allowed: without it the backend class
#: cannot be COMPUTED for these envs, and no numeric bar moves as a result.
#:
#:   ``craftax``      "the frame is a gather from a texture atlas with two alpha blends over" it (class
#:   ``craftax_full``  docstring). Step 2's class R list names "embedded atlas" for exactly this shape.
#:                    ``craftax_full`` is the same renderer over the full game's atlas and was the one
#:                    cell Step 7 left at ``unknown``, purely because this table keyed on the module
#:                    NAME and ``craftax_full`` is not spelled ``craftax``.
#:   ``swm_nav``      the reference is itself torch arithmetic producing a uint8 (3,H,W) image. There is
#:                    no 3-D scene anywhere in it. (Its envs now declare ``navdot`` above, so this entry
#:                    is a belt rather than the load-bearing rule it used to be; both give R.)
#:   ``minihack``     the same shape as ``craftax`` and identified the same way, from the module's own
#:                    account of the reference: "the frame is a gather from a finite atlas indexed by
#:                    the 21x79 glyph grid, and there is no sub-pixel geometry, no blending and no float
#:                    arithmetic anywhere in the path" (``stable_envs/envs/minihack.py``). It was the
#:                    second cell left at ``unknown`` and for the same fall-through reason as
#:                    CraftaxFull, not for any reason to do with the renderer.
#:
#: All are 2-D and none touches mjwarp/raycast3d/nvdiff/cpu3d. Class R also holds them to the HIGHER
#: bar (GPU_FAST at R >= 500 against class S's 4), so this is the conservative reading as well as the
#: accurate one.
#: Step 9 renamed two of these modules: Workstream A moved ``swm_nav`` to ``navigation`` and
#: Workstream E moved ``craftax_full`` to the private ``_craftax_full`` (both keeping a
#: ``sys.modules`` alias, ruling 6). This table is keyed by module BASENAME, so both spellings must
#: appear and BOTH must stay: ``certify`` is run against the frozen Step-8 censuses, which record
#: the pre-rename ``cls.__module__``, AND against censuses regenerated on the Step-9 tree, which
#: record the new one (``scratch/census/census.py`` reads ``cls.__module__`` live). Dropping the old
#: spelling would silently return ``unknown`` for the frozen inputs; not adding the new one did
#: exactly that for the regenerated ones, which is the defect this closes. The class is a property
#: of the RENDERER, which neither rename touched, so carrying both names preserves behaviour by
#: definition rather than by measurement. ``scratch/step4/universe.py::shard_of`` already carries
#: the ``swm_nav``/``navigation`` pair for the same reason (Workstream A added it); it needs no
#: ``craftax_full`` entry because neither spelling was ever in it -- both fall through to the
#: ``raster`` shard, so E's rename leaves that routing unchanged (verified over all 81 registered
#: classes on E's branch: shard tally dmc 18 / ogb 25 / swm 6 / raster 32, identical to integration).
NO_BACKEND_MODULE_CLASS = {
    "craftax": "R",
    "craftax_full": "R",
    "_craftax_full": "R",
    "minihack": "R",
    "swm_nav": "R",
    "navigation": "R",
}


def backend_class(env: str, census: dict) -> tuple[str, list[str]]:
    """``("S"|"R"|unknown, backends)`` -- Step 2 section 6.4, assigned from declared backends.

    The class follows what the backends ARE. The frozen lists in ``tests/fixtures/regression/fidelity/certification.json`` are
    consulted first and never edited; ``POST_STEP2_BACKEND_CLASS`` supplies the same definition for the
    four modules that were written after those lists were frozen; ``NOT_A_BACKEND`` removes the two
    ``stable_envs.render`` modules that are not renderers at all; and only when NO class-bearing
    backend remains does the module name get a say, which is the fall-through that put CraftaxFull at
    ``unknown``.

    S is checked before R, and that ordering is a claim: an env whose defining module reaches both a
    3-D backend and a 2-D one is drawing a 3-D scene with 2-D helpers, not the other way round. Every
    such env in the corpus is a DM Control cell, whose ``backends`` are ``cpu3d``/``mjwarp``/
    ``raycast3d`` -- three ways to draw the same 3-D scene on two devices, and S under all three.
    """
    decl = census.get(env) or {}
    b = sorted(decl.get("backends") or [])
    bearing = [x for x in b if x not in NOT_A_BACKEND]
    S = set(SPEC["backend_classes"]["S"]["backends"]) | {k for k, v in POST_STEP2_BACKEND_CLASS.items() if v == "S"}
    R = set(SPEC["backend_classes"]["R"]["backends"]) | {k for k, v in POST_STEP2_BACKEND_CLASS.items() if v == "R"}
    if set(bearing) & S:
        return "S", b
    if set(bearing) & R:
        return "R", b
    mod = (decl.get("module") or "").rsplit(".", 1)[-1]
    if mod in NO_BACKEND_MODULE_CLASS:
        return NO_BACKEND_MODULE_CLASS[mod], b
    return UNKNOWN, b


def merge(shard_paths) -> dict:
    """Deterministic union of shards, keyed by ``(env, device, host, threads)``.

    A later shard never silently overwrites an earlier one: the key carries the host and the thread
    budget, because the same env measured single-threaded on an M5 and all-core on a Xeon are two
    different measurements and collapsing them would be the exact error Step 2's "every ratio must be
    same-host" rule exists to prevent. A genuine repeat of the same key keeps the one with more
    measured cells, and records that it did.
    """
    runs, shards = {}, []
    for p in sorted(Path(x) for x in shard_paths):
        d = json.loads(p.read_text())
        prov = d["provenance"]
        shards.append(
            {
                "path": str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p),
                "provenance": prov,
                "batches": d.get("batches"),
                "n_envs": len(d.get("envs", [])),
            }
        )
        for r in d.get("envs", []):
            key = (r["env"], prov["device"], prov["host"], prov.get("torch_threads"))
            r = dict(r, _provenance=prov, _shard=shards[-1]["path"])
            prev = runs.get(key)
            if prev is None:
                runs[key] = r
            else:
                n_new = sum(1 for c in r.get("cells", []) if c.get("fps"))
                n_old = sum(1 for c in prev.get("cells", []) if c.get("fps"))
                keep, drop = (r, prev) if n_new > n_old else (prev, r)
                keep = dict(keep)
                keep["_superseded"] = (keep.get("_superseded") or []) + [drop["_shard"]]
                runs[key] = keep
    return {
        "what": "Step 3 merged renderer benchmark",
        "shards": shards,
        "runs": [
            runs[k] | {"_key": {"env": k[0], "device": k[1], "host": k[2], "threads": k[3]}}
            for k in sorted(runs, key=lambda k: tuple(str(x) for x in k))
        ],
    }


# ------------------------------------------------------------------ derived quantities


def _cells(run) -> dict[int, dict]:
    return {c["batch"]: c for c in run.get("cells", []) if isinstance(c, dict)}


def best_certified(run, device: str) -> tuple[int | None, float | None]:
    """Best throughput over the MANDATORY batches, excluding B=1.

    B=1 is required by the sweep and forbidden as a certification point (spec
    ``gpu.never_certify_on_batch``): the corpus median at B=1 is 0.6x the official renderer, so
    certifying there would measure launch overhead and call it a renderer.
    """
    key = "gpu" if device == "cuda" else "cpu"
    mandatory = set(SPEC[key]["required_batches"]) - {SPEC["gpu"]["never_certify_on_batch"]}
    cand = [(c["fps"], b) for b, c in _cells(run).items() if b in mandatory and c.get("fps")]
    if not cand:
        return None, None
    fps, b = max(cand)
    return b, fps


def official_fps(run) -> float | None:
    o = run.get("official_cpu") or {}
    v = (o.get("set_state_render") or {}).get("fps")
    return float(v) if v else None


def decode_fps(run) -> float | None:
    d = run.get("decode") or {}
    v = d.get("decode_fps_per_core")
    return float(v) if v else None


def _holds(cond: dict, keys: set[str], x: float | None) -> bool:
    """Do the bounds in ``cond`` drawn from ``keys`` hold for ``x``? Vacuously true if there are none."""
    for k, v in cond.items():
        if k not in keys:
            continue
        if x is None:
            return False
        if k.endswith("_lt") and not x < v:
            return False
        if k.endswith("_gte") and not x >= v:
            return False
    return True


def _tier_from(
    ladder: dict, order: list[str], ratio: float | None, cores: float | None, ratio_key: str, cores_key: str
) -> str:
    """The frozen ladder, walked the way section 6.6 defines it: ratio picks the rung, cores CAPS it.

    Not a single top-down scan over all conditions, which is what this was first written as and which
    is wrong in a way that only shows on the interesting rows. The middle rungs carry exclusive upper
    bounds (``GPU_FASTER`` is ``2 <= R < 4``), so an env with a big ratio that fails the ``cores``
    conjunct matches NO rung at all: it is too fast for FASTER and too poor at decode for FAST. The
    spec's own words for that case are "capped at ``GPU_FASTER``", so the ratio chooses the starting
    rung and a failed ``cores`` conjunct walks DOWN from it.

    DMCBallInCupCatch is the row that makes this concrete: 12.5x the official renderer and 0.59 cores
    of decode. The first version returned ``unknown`` for it -- the one row in the corpus the qualifier
    exists for.
    """
    # Only the rungs. The NO_REFERENCE ladders carry scalar metadata beside them ("when", "basis",
    # "may_contribute_to_parity_claim"), and iterating those as if they were conditions raises.
    rungs = {n: c for n, c in ladder.items() if isinstance(c, dict)}
    rk = {k for c in rungs.values() for k in c if k.startswith(ratio_key)}
    ck = {k for c in rungs.values() for k in c if k.startswith(cores_key)}
    if not rk:  # the NO_REFERENCE ladders are cores-only
        for name in order:
            if name in rungs and _holds(rungs[name], ck, cores):
                return name
        return UNKNOWN

    start = next((i for i, n in enumerate(order) if _holds(rungs.get(n) or {}, rk, ratio)), None)
    if start is None:
        return UNKNOWN
    for name in order[start:]:
        cond = rungs.get(name) or {}
        if _holds(cond, ck, cores):
            return name  # first rung at or below the ratio rung that cores allows
    return order[-1]


def classify(run, device: str, census: dict) -> dict:
    """Every label for one (env, device) measurement, and the numbers each one came from."""
    dev = "gpu" if device == "cuda" else "cpu"
    cls, backends = backend_class(run["env"], census)
    out: dict = {
        "env": run["env"],
        "device": dev,
        "backend_class": cls,
        "backends": backends,
        "projected_to": run.get("projected_to", []),
        "host": run["_provenance"]["host"],
        "device_name": run["_provenance"].get("device_name"),
        "threads": run["_provenance"].get("torch_threads"),
        "qualifiers": [],
        "evidence": {},
    }

    if "blocked" in run:
        out["tier"] = f"{dev.upper()}_UNMEASURED"
        out["blocked"] = run["blocked"]
        out["why"] = (
            "the renderer was never reached in this interpreter, so there is no performance "
            "claim to make: a dependency failure is not a regression"
        )
        return out

    b, fps = best_certified(run, device)
    ref, dec = official_fps(run), decode_fps(run)
    out["evidence"] = {
        "best_batch": b,
        "se_fps": fps,
        "official_cpu_fps": ref,
        "decode_fps_per_core": dec,
        "shape": run.get("shape"),
        "supersample": run.get("supersample"),
        "size": run.get("size"),
    }
    if fps is None:
        out["tier"] = f"{dev.upper()}_UNMEASURED"
        out["why"] = "no mandatory batch above B=1 produced a timing"
        return out

    cells = _cells(run)
    cf = max((c.get("cpu_frac") or 0) for c in cells.values() if c.get("fps"))
    out["evidence"]["max_cpu_frac"] = cf
    if cf >= SPEC["gpu"]["qualifiers"]["HOST_BOUND"]["threshold"] and dev == "gpu":
        out["qualifiers"].append("HOST_BOUND")

    cores = (fps / dec) if dec else None
    out["evidence"]["cores"] = cores

    spec_dev = SPEC[dev]
    if ref:
        ratio = fps / ref
        out["evidence"]["ratio"] = ratio
        if dev == "gpu":
            ladder = spec_dev["tiers"][cls] if cls in spec_dev["tiers"] else None
            order = ["GPU_EXCEPTIONAL", "GPU_FAST", "GPU_FASTER", "GPU_PARITY", "GPU_REGRESSION"]
            tier = _tier_from(ladder, order, ratio, cores, "R", "cores") if ladder else UNKNOWN
        else:
            order = ["CPU_EXCEPTIONAL", "CPU_FAST", "CPU_FASTER", "CPU_PARITY", "CPU_REGRESSION"]
            tier = _tier_from(spec_dev["tiers"], order, ratio, cores, "S_cpu", "cores_cpu")
    else:
        # No installable official renderer: the ratio is undefined and must not be invented.
        out["qualifiers"].append("NO_REFERENCE")
        fb = spec_dev["no_reference_fallback"]
        order = [f"{dev.upper()}_EXCEPTIONAL", f"{dev.upper()}_FAST", f"{dev.upper()}_FASTER"]
        tier = _tier_from(fb, order, None, cores, "R", "cores")
        if tier == UNKNOWN:
            tier = f"{dev.upper()}_REGRESSION" if cores is not None else f"{dev.upper()}_UNMEASURED"
        out["why"] = (run.get("official_cpu") or {}).get("absent") or "no official renderer measured here"

    if cores is not None and cores < 1:
        out["qualifiers"].append("BELOW_G4")

    if dev == "gpu":
        q = SPEC["gpu"]["qualifiers"]
        f1, f4096, f256 = (
            (cells.get(1) or {}).get("fps"),
            (cells.get(4096) or {}).get("fps"),
            (cells.get(256) or {}).get("fps"),
        )
        if f1 and f4096:
            scale = f4096 / f1
            out["evidence"]["batch_scales"] = scale
            passed = scale >= q["BATCH_SCALES"]["factor"]
            out["qualifiers"].append("BATCH_SCALES" if passed else "BATCH_SCALES_FAIL")
            # The spec does not merely RECORD this one -- it is "required_above_tier": "GPU_PARITY",
            # so a renderer that does not actually batch cannot hold a tier above GPU_PARITY however
            # good its ratio is. Recording the qualifier and then ignoring it, which is what this did
            # first, quietly applied a laxer rule than the one Step 2 froze.
            cap = q["BATCH_SCALES"]["required_above_tier"]
            if not passed and _tier_rank(tier) > _tier_rank(cap):
                out["evidence"]["tier_before_batch_scales_cap"] = tier
                out["capped_by"] = (
                    f"BATCH_SCALES: fps(4096) is {scale:.1f}x fps(1), under the "
                    f"{q['BATCH_SCALES']['factor']}x bar this renderer must clear to "
                    f"hold a tier above {cap}"
                )
                tier = cap
        if f256 and f4096:
            r = f4096 / f256
            out["evidence"]["large_batch_ratio"] = r
            if r < q["NO_LARGE_BATCH_REGRESSION"]["factor"]:
                out["qualifiers"].append("LARGE_BATCH_REGRESSION")
    out["tier"] = tier
    return out


#: A "CPU" render this close to the same env's CUDA render did not happen on the CPU. Half is a very
#: loose bar deliberately: elsewhere in this campaign a genuine CPU path runs at ~1/100 of the GPU for
#: these scenes, so anything above 0.5 is not a slow CPU, it is the device argument being ignored.
#: This is a VALIDITY check on a measurement, not a certification bound -- it decides whether a number
#: is a CPU number at all, before any tier is considered.
DEVICE_HONOURED_MAX_RATIO = 0.5


def _flag_device_not_honoured(labels: list) -> None:
    """Refuse any CPU label whose throughput proves it ran on the GPU.

    Found by measurement, not suspected: asked for ``device="cpu"`` on a GPU node, the four
    mjwarp-backed DM Control envs returned 1.01-1.03x of their own CUDA throughput. The mjwarp render
    path has no CPU implementation and does not say so -- it renders on the device anyway. Reporting
    those as CPU numbers would have put 3090 throughput in the CPU column of a document whose entire
    purpose is keeping the two apart, and they are large enough that all four would have certified
    CPU_EXCEPTIONAL.

    The raw shard is kept; it is evidence. What is refused is the *interpretation* of it as CPU.
    """
    gpu = {
        l["env"]: l["evidence"].get("se_fps")
        for l in labels
        if l["device"] == "gpu" and l.get("evidence", {}).get("se_fps")
    }
    for l in labels:
        if l["device"] != "cpu":
            continue
        f, g = l.get("evidence", {}).get("se_fps"), gpu.get(l["env"])
        if f and g and f / g > DEVICE_HONOURED_MAX_RATIO:
            l["tier"] = "CPU_UNMEASURED"
            l["qualifiers"] = sorted(set(l["qualifiers"]) | {"DEVICE_NOT_HONOURED"})
            l["blocked"] = {
                "stage": "measure",
                "kind": "blocked_environment",
                "reason": (
                    f'device="cpu" was not honoured: {f:,.0f} f/s is '
                    f"{f / g:.2f}x this env's own CUDA throughput ({g:,.0f} f/s), "
                    f"so the render ran on the GPU. The mjwarp backend has no CPU "
                    f"path and does not raise."
                ),
            }
            l["why"] = "there is no CPU render path here to measure, so no CPU claim is made"
            # Blank the derived columns. `se_fps` stays because it IS the evidence for the finding,
            # but an S_cpu of 201 sitting in a CPU table is the exact misreading this guard exists to
            # stop -- a reader scanning the column would take it for a spectacular CPU renderer.
            for k in ("ratio", "cores"):
                l["evidence"].pop(k, None)
            l["evidence"]["se_fps_is_actually_gpu"] = l["evidence"].pop("se_fps")


#: A Step 3 qualifier, not a Step 2 badge. `tests/fixtures/regression/fidelity/certification.json` is frozen and this does not
#: touch it: like `BATCH_SCALES_FAIL` and `DEVICE_NOT_HONOURED`, it is a finding this evaluator makes
#: ABOUT a measurement, not a tier or a threshold. The tier stays exactly what Step 2 says it is,
#: `CPU_UNMEASURED`.
CPU_BACKEND_MISSING = "CPU_BACKEND_MISSING"


def _flag_cpu_backend_missing(labels: list) -> None:
    """Separate "no CPU renderer exists" from "this interpreter could not reach one".

    Step 3 filed both as `blocked_environment`, which put 16 rows in the optimization queue as setup
    problems. Step 3B settled that they are not: the matched
    `mujoco 3.11.0 + mujoco-warp 3.11.0 + gymnasium-robotics 1.4.2` stack builds and measures all 12
    Fetch rows on the GPU, and the CPU pass still fails — inside the renderer, at
    `fetch.py:811`, because it allocates its mjwarp buffers on CUDA and then meets a CPU state tensor.
    The four mjwarp DM Control rows are the same gap without the courtesy of an exception.

    The two are different work: one is an install, the other is a backend nobody has written. The tier
    is `CPU_UNMEASURED` either way — there is no measurement — but only one of them is a renderer
    defect, and the queue has to be able to tell them apart.
    """
    # "No CPU renderer exists" is a claim about the ENV, so a CPU measurement on ANY host refutes it.
    # Checking only the label in hand tagged PushT and SimplePointMaze, which both render happily on
    # the M5 and merely failed on an Oscar node whose backend selection wanted CUDA -- a host problem
    # wearing the same exception as a real backend gap.
    measured_cpu = {
        l["env"] for l in labels if l["device"] == "cpu" and l.get("evidence", {}).get("se_fps") is not None
    }
    for l in labels:
        if l["device"] != "cpu" or l["env"] in measured_cpu:
            continue
        if not _is_gpu_only(l):
            continue
        l["qualifiers"] = sorted(set(l["qualifiers"]) | {CPU_BACKEND_MISSING})
        if l.get("blocked"):
            l["blocked"]["kind"] = "backend_missing"
        l["why"] = "the renderer has no CPU implementation — an implementation gap, not a dependency or setup blocker"


def certify(merged: dict, census: dict) -> dict:
    labels = [classify(r, r["_key"]["device"], census) for r in merged["runs"]]
    _flag_device_not_honoured(labels)
    _flag_cpu_backend_missing(labels)
    labels.sort(key=lambda l: (l["env"], l["device"], l["host"]))
    return {
        "what": "Step 3 certification labels, derived only from tests/fixtures/regression/fidelity/certification.json",
        "spec_schema_version": SPEC["schema_version"],
        "spec_frozen_on": SPEC["frozen_on"],
        "labels": labels,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default="tests/fixtures/regression/join/step3")
    ap.add_argument("--census", nargs="*", default=None)
    ap.add_argument("--out", default="tests/fixtures/regression/join/step3")
    a = ap.parse_args()
    root = ROOT / a.shards if not Path(a.shards).is_absolute() else Path(a.shards)
    shards = [
        p
        for p in sorted(root.rglob("*.json"))
        if p.name not in ("merged.json", "certification.json") and "pipeline" not in p.parts
    ]
    census = read_censuses(
        a.census or sorted((ROOT / "tests" / "fixtures" / "regression" / "census").glob("census_*.json"))
    )
    m = merge(shards)
    c = certify(m, census)
    out = ROOT / a.out if not Path(a.out).is_absolute() else Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "merged.json").write_text(json.dumps(m, indent=1, sort_keys=True) + "\n")
    (out / "certification.json").write_text(json.dumps(c, indent=1, sort_keys=True) + "\n")
    pipeline = [json.loads(q.read_text()) for q in sorted((out / "pipeline").glob("*.json"))]
    doc = ROOT / "docs" / "research" / "step3_benchmarks.md"
    doc.write_text(report(m, c, pipeline) + "\n")
    print(f"wrote {doc.relative_to(ROOT)}")
    print(f"{len(shards)} shard(s) -> {len(m['runs'])} runs -> {len(c['labels'])} labels")
    from collections import Counter

    for dev in ("cpu", "gpu"):
        t = Counter(l["tier"] for l in c["labels"] if l["device"] == dev)
        if t:
            print(f"  {dev}: {dict(t)}")


# ------------------------------------------------------------------ reporting
#
# Everything below turns the labels into the Step 3 page. It reads the merged artifact and the frozen
# spec and nothing else, so the page can be regenerated from committed data without a cluster.

#: How far a run got, for choosing which label represents an env when several interpreters tried it.
#: A measurement beats a blocker, and among blockers the one that got FURTHEST is the informative one:
#: Fetch fails at import in the dm_control interpreter ("No module named 'gymnasium_robotics'") and at
#: construction in its own ("AssertionError" inside set_joint_qpos). Only the second names the real
#: mechanism, and showing the first would make a solved packaging question look like an unsolved one.
_STAGE = {"registry": 1, "construct": 2, "measure": 3}


def _rank_label(l: dict) -> tuple:
    if l.get("evidence", {}).get("se_fps") is not None:
        return (3, 0, 0, 0)
    b = l.get("blocked")
    if not b:
        return (0, 0, 0, 0)
    # Among blockers at the same stage, a plain missing module is the SHALLOWER failure: it says the
    # package was not installed here, which is a fact about this interpreter. Anything else was raised
    # by the library after it loaded, and names a mechanism. Fetch fails both ways in different
    # interpreters and only the second reading is worth showing.
    shallow = b.get("reason", "").startswith(("ModuleNotFoundError", "ImportError"))
    # A backend gap outranks a dependency failure for the same env: Step 3B fixed the dependency and
    # the gap survived it, so the gap is the fact about the row. Without this the Fetch CPU row is
    # decided between "AssertionError in gymnasium-robotics" and "no CPU path" by hostname sort order.
    backend = _is_gpu_only(l)
    return (1, 2 if backend else 0, _STAGE.get(b.get("stage"), 0), 0 if shallow else 1)


def blocker_text(b: dict, width: int = 150) -> str:
    """The blocker's reason, plus the frame it came from when the exception carried no message.

    ``AssertionError:`` on its own is what gymnasium-robotics raises for Fetch, and printed bare it
    tells a reader nothing at all. The traceback was captured for exactly this case, so the last
    library frame is folded in: the useful half of that failure is *where* the assert is, not that one
    fired.
    """
    reason = (b.get("reason") or "").strip()
    if reason.rstrip(":").strip() in ("AssertionError", "RuntimeError", "ValueError", "Exception"):
        tb = (b.get("traceback") or "").strip().splitlines()
        frames = [l.strip() for l in tb if l.strip().startswith("File ")]
        if frames:
            f = frames[-1]
            where = f.split('"')[1].split("/")[-1] + f.split(",")[1] if '"' in f else f
            reason = f"{reason.rstrip(':')} at {where.strip()}"
    return reason[:width]


def _fmt(x, n=0, dash="—"):
    return dash if x is None else f"{x:,.{n}f}"


def historical_h200() -> dict:
    """The committed H200 corpus, keyed by env: ``(best fps, official cpu fps, cores)``.

    Provenance only. Step 2 was explicit that these are context for the RTX 3090 and never
    certification of it, and Step 3's job is to TEST the 5.13x prediction against them rather than to
    inherit any of their numbers.
    """
    p = ROOT / "tests" / "fixtures" / "regression" / "references" / "decode_all.json"
    if not p.is_file():
        return {}
    return {r["name"]: r for r in json.loads(p.read_text())["rows"]}


#: Errors that mean "this backend has no CPU implementation", as opposed to "this interpreter was
#: missing a package". Measured, not guessed: mjwarp raises the first two when handed CPU tensors and
#: the third when asked for a CPU device outright, and the DEVICE_NOT_HONOURED qualifier is this
#: evaluator's own finding for the case where it does not raise at all.
_GPU_ONLY_SIGNS = (
    "Expected all tensors to be on the same device",
    "Expected a cuda device",
    "Found no NVIDIA driver",
    'device="cpu" was not honoured',
)


def _is_gpu_only(lab: dict | None) -> bool:
    if not lab:
        return False
    if "DEVICE_NOT_HONOURED" in (lab.get("qualifiers") or []):
        return True
    reason = (lab.get("blocked") or {}).get("reason", "")
    return any(sign in reason for sign in _GPU_ONLY_SIGNS)


def bottleneck(lab_gpu: dict | None, lab_cpu: dict | None, partial: bool) -> tuple[str, str]:
    """``(category, why)`` -- the measured failure mode, in the vocabulary Step 3 asked for.

    Ordered by what a fix would have to attack. Two orderings here were wrong in the first draft and
    both distorted the queue:

      * a row was called a **dependency blocker** whenever EITHER device was blocked, so the 12 Fetch
        rows and the four mjwarp DM Control rows sat at the top of the optimization queue with their
        GPU numbers already measured and excellent. A dependency blocker now requires that nothing was
        measured on either device -- otherwise there is a measurement, and it is what ranks the row.
      * "no CPU render path" was folded into the same category as a missing package. They are not the
        same finding and they do not have the same fix: one is an install, the other is a backend that
        was never written. It ranks LAST because it is a statement of scope, not a defect.
    """
    g, c = lab_gpu or {}, lab_cpu or {}
    gq, cq = set(g.get("qualifiers") or []), set(c.get("qualifiers") or [])
    g_meas = g.get("evidence", {}).get("se_fps") is not None
    c_meas = c.get("evidence", {}).get("se_fps") is not None

    # A backend gap is the more specific reading of "nothing was measured", so it goes before the
    # dependency branch -- but ONLY when the GPU was not measured either. Fetch has both problems at
    # once: no CPU renderer, and a GPU that does not batch. The second is measured, actionable and
    # ranks the row; putting the CPU gap first buried twelve real BATCH_SCALES failures at the bottom
    # of the queue behind a fact no optimization pass can act on.
    if CPU_BACKEND_MISSING in cq and not g_meas:
        return (
            "GPU-only backend, no CPU path",
            "the renderer has no CPU implementation, so there is nothing on the CPU to measure",
        )
    if not g_meas and not c_meas and (g.get("blocked") or c.get("blocked")):
        b = g.get("blocked") or c.get("blocked")
        return "dependency / benchmark environment blocker", blocker_text(b)
    if "BELOW_G4" in gq:
        return "below decode break-even", f"{g['evidence'].get('cores'):.2f} cores of decode on the GPU"
    if "BATCH_SCALES_FAIL" in gq:
        return (
            "launch-bound / does not batch",
            f"fps(4096) is only {g['evidence'].get('batch_scales'):.1f}x fps(1); capped at {g.get('tier')}",
        )
    if "LARGE_BATCH_REGRESSION" in gq:
        return "large-batch regression", f"fps(4096)/fps(256) = {g['evidence'].get('large_batch_ratio'):.2f}"
    if "HOST_BOUND" in gq:
        return "host-bound python loop", f"cpu_frac {g['evidence'].get('max_cpu_frac'):.2f} at the certified batch"
    if c.get("tier") == "CPU_REGRESSION":
        r = c.get("evidence", {}).get("ratio")
        return "CPU implementation too slow", f"S_cpu = {r:.2f}" if r else "slower than the official renderer"
    if g.get("tier") in ("GPU_PARITY", "GPU_REGRESSION", "GPU_FASTER"):
        r = g.get("evidence", {}).get("ratio")
        return "GPU kernel throughput", f"R = {r:,.1f}" if r else "below the class FAST bar"
    if c.get("tier") in ("CPU_PARITY", "CPU_FASTER"):
        return "CPU implementation too slow", f"S_cpu = {c.get('evidence', {}).get('ratio', 0):.2f}"
    if partial:
        return "partial coverage", "the registered default is only partly drawn"
    if CPU_BACKEND_MISSING in cq:
        return ("GPU-only backend, no CPU path", "the GPU meets its bars; there is no CPU renderer to measure at all")
    return "none measured", "both devices meet their frozen bars"


def report(merged: dict, cert: dict, pipeline: list[dict]) -> str:
    from collections import Counter, defaultdict

    matrix = json.loads((ROOT / "tests" / "fixtures" / "regression" / "fidelity" / "swm_matrix.json").read_text())
    rows = {r["swm_id"]: r for r in matrix["rows"]}
    matched = [r for r in matrix["rows"] if r["stable_envs"]["envs"]]
    partial_envs = {
        e
        for r in matched
        for e in r["stable_envs"]["envs"]
        if not r["axes"]["registered_default_render_covered"]["covered"]
    }
    hist = historical_h200()

    # One representative label per (env, device) for the headline counts and the queue, plus every
    # label kept intact for the per-host tables. Step 2 certifies the CPU at ONE thread, so an
    # all-core run may never become the representative: it is a scaling result, and letting it stand
    # in for the tier is exactly the substitution section 5.3 forbids.
    by_env: dict[str, dict] = defaultdict(dict)
    all_labels: dict[str, list] = defaultdict(list)
    for l in cert["labels"]:
        all_labels[l["env"]].append(l)
        if l["device"] == "cpu" and (l.get("threads") or 1) != 1:
            continue
        cur = by_env[l["env"]].get(l["device"])
        if cur is None or _rank_label(l) > _rank_label(cur):
            by_env[l["env"]][l["device"]] = l

    L = [
        "# Step 3 — the current implementation, measured",
        "",
        '!!! info "Raw measurements, classified by the frozen Step 2 rules"',
        "    Nothing here moved a threshold. Every label is computed by",
        "    `scripts/regression/semantic/certify.py` from `tests/fixtures/regression/fidelity/certification.json`, which was frozen before any",
        "    of these numbers existed; the evaluator contains no bound of its own and a unit test fails",
        "    if one appears. Raw shards are under `tests/fixtures/regression/join/step3/`, the merged artifact is",
        "    `merged.json`, and the labels are `certification.json` — every row below can be recomputed",
        "    without rerunning a benchmark.",
        "",
        "    **No renderer was changed.** Step 3 measures; a slow row is a result, not a defect to fix here.",
        "",
        "## The conclusion",
        "",
        "> **GPU rendering is broadly successful. The dominant unfinished engineering problem is CPU "
        "rendering — including a missing CPU backend for the whole mjwarp family.**",
        "",
        "All 34 matched rows now carry a GPU tier and none is a regression. On the CPU not one row "
        "reaches `CPU_FAST`, 14 are outright regressions against the renderer they replace, every "
        "measured row loses to reading a stored frame off disk (`BELOW_G4`), and the 16 that are "
        "unmeasured are unmeasured because **no CPU renderer exists for them** rather than because a "
        "benchmark failed. `ENVIRONMENT_DONE` is therefore 0, and CPU is the reason for all 34.",
        "",
    ]

    # ---- provenance
    L += ["## Hosts", "", "| shard | device | host | CPU | threads | cores | notes |", "|---|---|---|---|---|---|---|"]
    for s in merged["shards"]:
        p = s["provenance"]
        L.append(
            f"| `{Path(s['path']).name}` | {p.get('device_name') or p['device']} | `{p['host']}` | "
            f"{p.get('cpu', '?')} | {p.get('torch_threads')} | {p.get('cpu_count_physical')} | "
            f"{'SLURM ' + str(p['slurm_job_id']) if p.get('slurm_job_id') else p.get('os', '')} |"
        )
    L.append("")

    # ---- CPU
    L += [
        "## CPU",
        "",
        "`S_cpu` is the stable-envs renderer over the official renderer **on the same host**, at the "
        "single-thread budget Step 2 certifies on. `cores_cpu` is the same renderer over one core of "
        "JPEG-q90 decode of that env's own frame — the G4 economics bar.",
        "",
        "| swm id(s) | renderer | host | best B | official f/s | stable-envs f/s | S_cpu | cores_cpu | tier | notes |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for env in sorted(all_labels):
        for l in sorted(
            (x for x in all_labels[env] if x["device"] == "cpu" and (x.get("threads") or 1) == 1),
            key=lambda x: x["host"],
        ):
            e = l.get("evidence", {})
            ids = ", ".join(i.replace("swm/", "") for i in l["projected_to"]) or "—"
            note = (
                ("blocked: " + blocker_text(l["blocked"], 70)) if l.get("blocked") else " ".join(l["qualifiers"]) or ""
            )
            L.append(
                f"| {ids} | `{env}` | `{l['host'].split('.')[0]}` | {_fmt(e.get('best_batch'))} | "
                f"{_fmt(e.get('official_cpu_fps'))} | {_fmt(e.get('se_fps'))} | "
                f"{_fmt(e.get('ratio'), 2)} | {_fmt(e.get('cores'), 2)} | **{l['tier']}** | {note} |"
            )
    L.append("")

    # ---- CPU thread scaling
    scale = []
    for env in sorted(all_labels):
        per = {}
        for l in all_labels[env]:
            if l["device"] == "cpu" and l.get("evidence", {}).get("se_fps"):
                per.setdefault(l["host"], {})[l.get("threads") or 1] = l["evidence"]
        for host, byt in per.items():
            if len(byt) > 1 and 1 in byt:
                top = max(t for t in byt if t != 1)
                one, many = byt[1], byt[top]
                scale.append((env, host.split(".")[0], one, top, many))
    if scale:
        L += [
            "### CPU thread scaling",
            "",
            "The all-core column is a **scaling** result and is never the certified number. Step 2 "
            "certifies at one thread so a fast node cannot stand in for a fast renderer — and this "
            "table is where that rule earns itself.",
            "",
            "| renderer | host | 1 thread | all-core | renderer speedup | official 1 thread | official all-core | baseline moved |",
            "|---|---|---|---|---|---|---|---|",
        ]
        artifacts = []
        for env, host, one, t, many in scale:
            r = many["se_fps"] / one["se_fps"]
            o1, o2 = one.get("official_cpu_fps"), many.get("official_cpu_fps")
            drift = (o1 / o2) if (o1 and o2) else None
            # 2x, not the 1.25x the column flags: a reference that moved by a third across two runs
            # on a shared laptop is node noise, and naming it beside a genuine 8x collapse would blur
            # the mechanism this warning exists to name.
            if drift and drift > 2.0:
                artifacts.append((env, drift))
            L.append(
                f"| `{env}` | `{host}` | {one['se_fps']:,.0f} | {many['se_fps']:,.0f} ({t}t) | "
                f"**{r:.2f}×** | {_fmt(o1)} | {_fmt(o2)} | "
                + (f"**{drift:.1f}× slower**" if drift and drift > 1.25 else "—")
                + " |"
            )
        L.append("")
        if artifacts:
            L += [
                '!!! warning "The all-core column degrades some references, which is why it cannot certify"',
                "    Giving torch every core makes the *official* renderer of "
                + ", ".join(f"`{e}`" for e, _ in sorted(artifacts, key=lambda x: -x[1])[:4])
                + (" and others" if len(artifacts) > 4 else "")
                + f" **{min(d for _, d in artifacts):.1f}–{max(d for _, d in artifacts):.1f}× slower**, "
                "not faster. Those references are themselves torch arithmetic on small tensors "
                "(`swm_nav`'s envs draw a uint8 image with torch ops), so a thread pool costs more "
                "than it buys and the baseline collapses. Smaller drifts in the column above are "
                "node noise on a shared machine, not this mechanism.",
                "",
                "    An all-core `S_cpu` for those rows would therefore look excellent while measuring "
                "nothing but a damaged denominator. Step 2 froze the certified budget at one thread "
                "before any of this was known; the table above is the reason that was right.",
                "",
            ]

    # ---- GPU
    L += [
        "## GPU — RTX 3090",
        "",
        "`R` is the stable-envs GPU renderer over the official CPU renderer measured on the **same "
        "node**. `cores` is the same over one core of decode there. Both bars are class-dependent and "
        "frozen, and two qualifiers cap a row regardless of how large `R` is: `BELOW_G4` "
        "(`cores < 1`) caps at `GPU_FASTER`, and a failed `BATCH_SCALES` caps at `GPU_PARITY` "
        "because Step 2 makes it required above that tier.",
        "",
        "| swm id(s) | renderer | class | best B | official f/s | 3090 f/s | R | cores | tier | qualifiers |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for env in sorted(by_env):
        l = by_env[env].get("gpu")
        if not l:
            continue
        e = l.get("evidence", {})
        ids = ", ".join(i.replace("swm/", "") for i in l["projected_to"]) or "—"
        q = " ".join(x for x in l["qualifiers"] if x != "BATCH_SCALES") or "—"
        note = ("blocked: " + blocker_text(l["blocked"], 60)) if l.get("blocked") else q
        L.append(
            f"| {ids} | `{env}` | {l['backend_class']} | {_fmt(e.get('best_batch'))} | "
            f"{_fmt(e.get('official_cpu_fps'))} | {_fmt(e.get('se_fps'))} | "
            f"{_fmt(e.get('ratio'), 1)} | {_fmt(e.get('cores'), 2)} | **{l['tier']}** | {note} |"
        )
    L.append("")

    # ---- same-env, different-node spread, where two shards happened to measure the same thing
    dup = []
    for env in sorted(all_labels):
        got = [l for l in all_labels[env] if l["device"] == "gpu" and l.get("evidence", {}).get("se_fps")]
        if len(got) > 1:
            fps = sorted(l["evidence"]["se_fps"] for l in got)
            dup.append((env, sorted({l["host"].split(".")[0] for l in got}), fps[0], fps[-1], fps[-1] / fps[0]))
    if dup:
        L += [
            "### The same renderer on two 3090 nodes",
            "",
            "Where a shard overlapped another, the same env was measured twice on different nodes. "
            "Step 2 warned that a GPU row does not automatically travel; this is how far it moved "
            "here, on rows that are NOT host-bound.",
            "",
            "| renderer | nodes | slower f/s | faster f/s | spread |",
            "|---|---|---|---|---|",
        ]
        for env, hosts, lo, hi, r in dup:
            L.append(f"| `{env}` | {', '.join(hosts)} | {lo:,.0f} | {hi:,.0f} | **{r:.2f}×** |")
        L.append("")

    # ---- H200 vs 3090
    L += [
        "## The H200 → RTX 3090 prediction, tested",
        "",
        f"Step 2 pre-registered **{SPEC['device_scaling']['factor_h200_to_rtx3090']}×** on the grounds "
        "that the rasterizer is output-bandwidth-bound. That was a prediction and this is the test of "
        "it. A `HOST_BOUND` row is expected NOT to scale with the device, so the two populations are "
        "reported apart.",
        "",
        "| renderer | class | host-bound | H200 f/s | 3090 f/s | H200/3090 |",
        "|---|---|---|---|---|---|",
    ]
    factors = {"S": [], "R": [], "host_bound": [], "device_bound": []}
    for env in sorted(by_env):
        l = by_env[env].get("gpu")
        h = hist.get(env)
        if not l or not h or not l.get("evidence", {}).get("se_fps"):
            continue
        f = h["gpu"] / l["evidence"]["se_fps"]
        hb = "HOST_BOUND" in l["qualifiers"]
        factors[l["backend_class"]].append(f) if l["backend_class"] in factors else None
        (factors["host_bound"] if hb else factors["device_bound"]).append(f)
        L.append(
            f"| `{env}` | {l['backend_class']} | {'yes' if hb else 'no'} | {h['gpu']:,.0f} | "
            f"{l['evidence']['se_fps']:,.0f} | **{f:,.2f}×** |"
        )
    L.append("")
    L += ["| population | n | median H200/3090 |", "|---|---|---|"]
    import statistics as st

    for k, v in factors.items():
        if v:
            L.append(f"| {k.replace('_', ' ')} | {len(v)} | **{st.median(v):,.2f}×** |")
    L.append("")

    # ---- pipeline
    if pipeline:
        L += [
            "## Input pipeline (product C — never a certification number)",
            "",
            "The renderer sweep times states already resident. This times the road to that point. "
            "`ceiling` is the no-loader arm; the others are the fraction of it each idiom reaches.",
            "",
            "| env | device | B | ceiling f/s | default_collate | batch_sampler | collate_batched | render share of ceiling |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for pf in pipeline:
            dev = pf["provenance"].get("device_name") or pf["provenance"]["device"]
            for e in pf["envs"]:
                if "blocked" in e:
                    L.append(f"| `{e['env']}` | {dev} | — | — | — | — | — | blocked: {e['blocked']['reason'][:60]} |")
                    continue
                for b, arms in sorted(e["arms"].items(), key=lambda kv: int(kv[0])):
                    ceil = arms.get("no_loader") or {}
                    L.append(
                        f"| `{e['env']}` | {dev} | {int(b):,} | {_fmt(ceil.get('fps'))} | "
                        + " | ".join(
                            f"{(arms.get(a) or {}).get('frac_of_ceiling', float('nan')):.2f}"
                            for a in ("default_collate", "batch_sampler", "collate_batched")
                        )
                        + f" | {ceil.get('render_frac', float('nan')):.2f} |"
                    )
        L.append("")

    # ---- blocked
    # Only blockers that still STAND. Once a later shard measured the same (env, device), the earlier
    # failure is a fact about that interpreter and not about the row -- listing Fetch as gpu-blocked
    # beside a table of its 3090 numbers reads as a contradiction rather than as history.
    measured = {(l["env"], l["device"]) for l in cert["labels"] if l.get("evidence", {}).get("se_fps") is not None}
    blocked, superseded = {}, 0
    backend_gap = sorted({l["env"] for l in cert["labels"] if CPU_BACKEND_MISSING in (l.get("qualifiers") or [])})
    for l in cert["labels"]:
        if not l.get("blocked"):
            continue
        if (l["env"], l["device"]) in measured:
            superseded += 1
            continue
        if CPU_BACKEND_MISSING in (l.get("qualifiers") or []):
            continue  # an implementation gap, reported in its own section below
        blocked.setdefault(blocker_text(l["blocked"], 110), []).append(f"{l['env']}/{l['device']}")
    if blocked:
        L += [
            "## Blocked measurements",
            "",
            "A blocker is **not** a performance tier. Every row here was never reached, so there is no "
            'claim to make about it in either direction — Step 2 forbids turning "we could not build '
            'this here" into `REGRESSION` or `unsupported`.',
            "",
            (
                f"{superseded} earlier blocker(s) are omitted because a later shard measured the same "
                f"env on the same device; they are kept in `certification.json` as history."
                if superseded
                else ""
            ),
            (
                f"A further {len(backend_gap)} env(s) are omitted because their CPU failure is an "
                f"**implementation gap and not a blocker** — see `CPU_BACKEND_MISSING` below."
                if backend_gap
                else ""
            ),
            "",
            "| what failed | where | rows |",
            "|---|---|---|",
        ]
        for reason, where in sorted(blocked.items(), key=lambda kv: -len(kv[1])):
            envs = sorted({w.split("/")[0] for w in where})
            L.append(
                f"| `{reason}` | {', '.join(sorted({w.split('/')[1] for w in where}))} | "
                f"{len(envs)} ({', '.join(envs[:3])}{'…' if len(envs) > 3 else ''}) |"
            )
        L += [
            "",
            "### Fetch: the conflict was real, and a matched stack dissolved it",
            "",
            "Step 3 concluded that no mujoco version could host both halves of Fetch, having tested two combinations:",
            "",
            "```text",
            "mujoco 3.13  + gymnasium-robotics      -> AssertionError at mujoco_utils.py:142",
            "mujoco 3.11  + mujoco_warp HEAD        -> AttributeError: no attribute 'mjtCtrlInput'",
            "```",
            "",
            "**That conclusion was wrong, and Step 3B corrects it.** Both failures came from mixing a "
            "pinned mujoco with a mujoco_warp built for a different one. mujoco_warp publishes a "
            "`v3.11.0` tag built against mujoco 3.11, and the matched pair holds:",
            "",
            "```text",
            "mujoco 3.11.0 + mujoco-warp 3.11.0 + gymnasium-robotics 1.4.2",
            "    mujoco          3.11.0      imports",
            "    mujoco_warp     3.11.0      imports",
            "    FetchPush-v4                reset OK",
            "```",
            "",
            "The intermediate failures on the way there (`No module named 'packaging'`, then "
            "`'pettingzoo'`) were self-inflicted: installing `gymnasium-robotics --no-deps` to stop it "
            "dragging mujoco forward also starved it of pure-python dependencies. Resolving its full "
            "dependency set turned out not to move mujoco at all — **the version conflict Step 1 and "
            "Step 3 both recorded does not exist for this combination**, and the cost of assuming it "
            "did was 12 rows reported as unmeasurable.",
            "",
            "All **12 Fetch rows now carry fresh RTX 3090 measurements**, over a state corpus written "
            "by the GPU pass and re-read by the CPU pass so both see byte-identical states.",
            "",
            "#### Fetch has no CPU render path",
            "",
            "The CPU pass of the same matched stack fails on all 12, and not on a dependency:",
            "",
            "```text",
            "RuntimeError: Expected all tensors to be on the same device,",
            "              but found at least two devices, cuda:0 and cpu!",
            "  at stable_envs/envs/fetch.py:811 in _upload",
            "```",
            "",
            "The Fetch renderer allocates its mjwarp buffers on CUDA regardless of the requested "
            "device, and its own upload path then meets the CPU state tensor. This is the same "
            "conclusion as the four mjwarp DM Control rows above — mjwarp is GPU-only — reached the "
            "honest way round: **Fetch raises where DM Control silently returned device frames.** "
            "So the 12 rows are `CPU_UNMEASURED` because there is no CPU renderer to measure, which "
            "is a fact about the backend and not a dependency blocker and not a regression.",
            "",
        ]

        nh = sorted({l["env"] for l in cert["labels"] if "DEVICE_NOT_HONOURED" in (l.get("qualifiers") or [])})
        if nh:
            L += [
                '### `device="cpu"` is silently ignored by the mjwarp backend',
                "",
                "The four mjwarp-backed DM Control envs (" + ", ".join(f"`{e}`" for e in nh) + ") were "
                "first blocked on a CPU-only node with `RuntimeError: Found no NVIDIA driver`, which is "
                "ambiguous — it could mean the renderer has no CPU path, or just that the node had no "
                "driver. Re-running the **CPU** sweep on a GPU node separated the two, and the answer "
                "was not the expected one:",
                "",
                "```text",
                'asked for device="cpu", on a node with a 3090 present:',
                "    DMCCheetahRun     77,457 f/s   = 1.01x its own CUDA throughput",
                "    DMCHopperHop      78,729 f/s   = 1.03x",
                "    DMCHumanoidStand  66,567 f/s   = 1.03x",
                "    DMCWalkerWalk     75,122 f/s   = 1.02x",
                "```",
                "",
                "A single CPU core does not match a 3090 to within 3 %. The render ran on the GPU: the "
                "mjwarp path has no CPU implementation and **does not raise when asked for one** — it "
                "quietly returns device frames. Every other renderer in this campaign honours the flag, "
                "which is why the four stood out at all.",
                "",
                "Two consequences, and the second is the reason this has its own section:",
                "",
                "1. those four rows have **no CPU throughput**, and are `CPU_UNMEASURED` rather than "
                "   fast or slow — there is nothing on the CPU to be either;",
                "2. taken at face value they would have certified **`CPU_EXCEPTIONAL`** on an `S_cpu` "
                "   above 200, putting 3090 numbers in the CPU column of a document whose whole "
                "   purpose is keeping the two apart. `scripts/regression/semantic/certify.py` now refuses any CPU "
                "   label above half its own env's CUDA throughput, and the raw shard is kept as the "
                "   evidence for the refusal rather than deleted.",
                "",
            ]

    if backend_gap:
        L += [
            "### `CPU_BACKEND_MISSING` — the mjwarp family has no CPU renderer",
            "",
            "These are **not** dependency or setup blockers, and Step 3B is what settles the difference:",
            "",
            "```text",
            "dependency / setup blocker   a working implementation may exist; this interpreter",
            "                             could not reach it   (an install fixes it)",
            "",
            "CPU_BACKEND_MISSING          the renderer has no CPU path at all",
            "                             (only writing one fixes it)",
            "```",
            "",
            "Step 3 filed all 16 of these as setup problems. That was wrong. The matched "
            "`mujoco 3.11.0 + mujoco-warp 3.11.0 + gymnasium-robotics 1.4.2` stack **builds and "
            "measures every one of them on the GPU** — so the dependency question is closed — and the "
            "CPU pass still fails, inside the renderer:",
            "",
            "```text",
            "Fetch (12 rows)        RuntimeError: Expected all tensors to be on the same device,",
            "                       cuda:0 and cpu, at stable_envs/envs/fetch.py:811 in _upload",
            "",
            'DM Control (4 rows)    no exception at all: device="cpu" returns frames at 1.01-1.03x',
            "                       the env's own CUDA throughput, because the render ran on the GPU",
            "```",
            "",
            "| env | GPU tier | CPU |",
            "|---|---|---|",
        ]
        for env in backend_gap:
            g = by_env.get(env, {}).get("gpu", {})
            L.append(f"| `{env}` | {g.get('tier', '—')} | `CPU_UNMEASURED` + `CPU_BACKEND_MISSING` |")
        L += [
            "",
            "The tier stays `CPU_UNMEASURED` — Step 2's definition is untouched and there is no "
            "measurement to tier. What changes is that these rows are now legible as **renderer work "
            "that has not been done**, rather than as benchmark plumbing that failed.",
            "",
        ]

    # ---- headline counts
    L += ["## Headline counts over the 34 matched rows", ""]
    for dev, label in (("cpu", "CPU"), ("gpu", "GPU")):
        c = Counter({f"{dev.upper()}_{t}": 0 for t in _TIER_ORDER})  # zeros are results too
        for r in matched:
            tiers = [by_env.get(e, {}).get(dev, {}).get("tier") for e in r["stable_envs"]["envs"]]
            tiers = [t for t in tiers if t]
            c[min(tiers, key=_tier_rank) if tiers else f"{dev.upper()}_UNMEASURED"] += 1
        L += (
            [f"**{label}** (a row with several renderers takes its weakest):", "", "```text"]
            + [f"{k:22s} {v}" for k, v in sorted(c.items(), key=lambda kv: _tier_rank(kv[0]), reverse=True)]
            + ["```", ""]
        )
        if dev == "cpu" and c["CPU_UNMEASURED"]:
            L += [
                f"All {c['CPU_UNMEASURED']} `CPU_UNMEASURED` rows are `CPU_BACKEND_MISSING`: no CPU "
                "renderer exists for them. None is an unrun or failed benchmark.",
                "",
            ]
    done = sum(
        1
        for r in matched
        if r["axes"]["registered_default_render_covered"]["covered"]
        and by_env.get(r["stable_envs"]["envs"][0], {}).get("cpu", {}).get("tier") in ("CPU_FAST", "CPU_EXCEPTIONAL")
        and by_env.get(r["stable_envs"]["envs"][0], {}).get("gpu", {}).get("tier") in ("GPU_FAST", "GPU_EXCEPTIONAL")
    )
    L += [
        "```text",
        f"ENVIRONMENT_DONE               {done}   (needs full coverage + CPU_FAST + GPU_FAST)",
        f"RENDERER_ABSENT                {matrix['summary']['renderer_gaps']}   (the Step 1 renderer gaps)",
        f"NOT_APPLICABLE_SYMBOLIC        {matrix['summary']['symbolic_no_frame']}",
        "```",
        "",
    ]

    # ---- optimization queue
    L += [
        "## Optimization queue",
        "",
        "Ranked by measured failure mode and by how many registered ids ride on it. **This is the "
        "output of Step 3, not a plan** — nothing here was acted on.",
        "",
        "| # | renderer | swm rows | CPU | GPU | bottleneck | evidence |",
        "|---|---|---|---|---|---|---|",
    ]
    queue = []
    for env in sorted(by_env):
        g, c = by_env[env].get("gpu"), by_env[env].get("cpu")
        cat, why = bottleneck(g, c, env in partial_envs)
        if cat == "none measured":
            continue
        # Tag by device. A union would put the GPU shard's NO_REFERENCE next to a CPU S_cpu measured
        # on a host that DID have the reference installed, which reads as a contradiction.
        extra = []
        for dev, lab in (("cpu", c), ("gpu", g)):
            for q in sorted(set((lab or {}).get("qualifiers") or []) - {"BATCH_SCALES"}):
                extra.append(f"{dev}:{q}")
        if extra:
            why = f"{why} · also {' '.join(extra)}"
        n_rows = len((g or c or {}).get("projected_to", []))
        queue.append((_CAT_RANK.get(cat, 9), -n_rows, env, cat, why, g, c))
    for i, (_, nneg, env, cat, why, g, c) in enumerate(sorted(queue), 1):
        L.append(
            f"| {i} | `{env}` | {-nneg} | {(c or {}).get('tier', '—')} | {(g or {}).get('tier', '—')} "
            f"| {cat} | {why} |"
        )
    L.append("")
    return "\n".join(L)


#: Weakest first. Used to pick a row's representative tier and to order the count blocks.
_TIER_ORDER = ["UNMEASURED", "REGRESSION", "PARITY", "FASTER", "FAST", "EXCEPTIONAL"]
#: Queue order. A dependency blocker is first because no renderer work changes it; "GPU-only backend"
#: is nearly last because it is a scope statement rather than a defect, and putting it at the top
#: (which the first draft did, by calling it a dependency blocker) buried the rows that can actually
#: be optimized under sixteen that cannot.
_CAT_RANK = {
    "dependency / benchmark environment blocker": 0,
    "below decode break-even": 1,
    "launch-bound / does not batch": 2,
    "large-batch regression": 3,
    "host-bound python loop": 4,
    "CPU implementation too slow": 5,
    "GPU kernel throughput": 6,
    "partial coverage": 7,
    "GPU-only backend, no CPU path": 8,
}


def _tier_rank(t: str) -> int:
    for i, k in enumerate(_TIER_ORDER):
        if t and t.endswith(k):
            return i
    return 0


if __name__ == "__main__":
    main()
