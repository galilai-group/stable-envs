"""Build ``tests/fixtures/regression/join/step8/performance/tiers.json``: the final Step-8 performance
accounting, one row per registered ``stable_worldmodel`` id, assembled ONLY from committed
Step-6/Step-7 artifacts at this HEAD.

WHAT THIS FILE IS AND IS NOT

  * It **re-measures nothing**. Step 8 does not reopen optimisation (coordinator plan section 2),
    so no benchmark is run here and no threshold is moved. Every number is READ out of a committed
    artifact, and every artifact is named with its sha256 in the output.
  * Tiers are **copied**, never decided: the tier column is the Step-7 record's tier for that id
    (``tests/fixtures/regression/join/step7/closure2/coverage_matrix.json``), which itself copied it from a class
    ``performance_record_*`` declaration, the generated Step-6 tier table, or the Step-7 dashboard
    workstream row. What this file adds is the *evidence underneath* that tier -- the ratio, the
    cores, the batch cells, the host/job/version provenance -- read from the shard the Step-6 table
    names or, for the Step-7 cells, from the tier artifact the workstream committed.
  * The tier implied by those numbers is computed too, with ``scripts/regression/semantic/certify.py`` -- Step 3's
    own evaluator against Step 2's frozen ladder, imported rather than reimplemented -- and recorded
    beside the copied one as ``tier_recomputed``. Where the two disagree the row carries
    ``AGREES_WITH_STEP7: false`` and the disagreement is listed in ``cross_checks``. Nothing is
    silently reconciled: the doc explains each one.
  * ``status`` says only whether a measurement EXISTS, in the three words the closure criteria use:
    ``MEASURED`` / ``UNMEASURED`` / ``MISSING-UNSUPPORTED_BY_DESIGN``. A tier is a grade, not a gate:
    ``CPU_REGRESSION`` is a perfectly well-measured tier, and ``CPU_MISSING [RAISES_BY_DESIGN]`` is a
    supported, honest answer (the env raises on ``device="cpu"`` rather than falling back to CUDA),
    not a gap.

WHAT IT REFUSES TO DO

  * No projection stands in for a measurement. A cell with no committed number is ``UNMEASURED`` and
    says why.
  * Quadruped CPU stays ``CPU_UNMEASURED [LOAD_QUARANTINED]``: the only CPU run is the quarantined
    one (load 34-44 against a load-12 guard) and this file does not launch another.
  * The frozen ``BATCH_SCALES`` bar (``fps(4096) >= 20 * fps(1)``) is applied exactly as Step 2 froze
    it. Rocket and Fetch fail it with strong absolute throughput; the classification is kept and the
    edge case is documented in ``docs/research/step8_performance.md``, not patched here.

Run:  MUJOCO_GL=glfw python scripts/regression/support/build_performance.py [--check]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path


def _repo_root() -> Path:
    p = Path(__file__).resolve()
    for cand in (p, *p.parents):
        if (cand / "pyproject.toml").is_file():
            return cand
    raise RuntimeError("repo root not found")


ROOT = _repo_root()
OUT = ROOT / "tests" / "fixtures" / "regression" / "performance_tiers.json"

STEP6_DOC = "docs/research/step6_optimization.md"
STEP7_MATRIX = "tests/fixtures/regression/join/step7/closure2/coverage_matrix.json"
STEP7_DECLARATIONS = "tests/fixtures/regression/join/step7/closure2/declarations/ogbmanip.json"
STEP3_LABELS = "tests/fixtures/regression/join/step3/certification.json"
SUPPORT_MATRIX = "tests/fixtures/regression/support_matrix.json"
SPEC_JSON = "tests/fixtures/regression/fidelity/certification.json"

#: Oscar scratch root. Committed artifacts quote absolute ``/oscar`` paths inside provenance
#: strings; the repo contract is that committed files carry ``$SCR`` instead, so every string that
#: reaches the output goes through :func:`scrub`.
SCR = "/oscar/scratch/sboughan"

#: The Step-7 cells, whose tier evidence is not in the Step-6 table because they did not exist then.
#: Hand-entered, and this is the ONLY hand-entered table here: each entry names a committed artifact
#: and a reader. The numbers themselves are read from the artifact, never typed.
STEP7_EVIDENCE = {
    ("CraftaxClassic", "cpu"): {
        "reader": "declaration",
        "path": STEP7_DECLARATIONS,
        "key": "CraftaxClassic",
        "why": "the class carries performance_record_cpu, measured in Step 7 against the JITTED "
        "official renderer; it SUPERSEDES the Step-6 tilemap tier, which was measured on an "
        "observation that could not represent night and had no reference at all",
    },
    ("CraftaxClassic", "gpu"): {
        "reader": "declaration",
        "path": STEP7_DECLARATIONS,
        "key": "CraftaxClassic",
        "why": "as the CPU side: performance_record_gpu supersedes the Step-6 NO_REFERENCE tier",
    },
    ("CraftaxFull", "cpu"): {
        "reader": "declaration",
        "path": STEP7_DECLARATIONS,
        "key": "CraftaxFull",
        "why": "cell created in Step 7 (W2); the class carries the measured record",
    },
    ("CraftaxFull", "gpu"): {
        "reader": "declaration",
        "path": STEP7_DECLARATIONS,
        "key": "CraftaxFull",
        "why": "cell created in Step 7 (W2); the class carries the measured record",
    },
    ("DMCManipulatorBringBall", "cpu"): {
        "reader": "manipulator_cpu",
        "path": "tests/fixtures/regression/join/step7/dmc-manipulator/cputier/cpu_tier_branch_node2315.json",
        "why": "W3's one-thread CPU tier run, job 6396735 node2315, gates USABLE",
    },
    ("DMCManipulatorBringBall", "gpu"): {
        "reader": "economics",
        "path": "tests/fixtures/regression/join/step7/dmc-manipulator/economics/bench_vs_decode_se-step6-3090.json",
        "key": "DMCManipulatorBringBall",
        "why": "the single-job economics run: ours, the official dm_control renderer and the one-core "
        "decode all timed in ONE process on ONE node, which is what makes the ratio legitimate",
    },
    ("DMCQuadrupedWalk", "cpu"): {
        "reader": "quarantine",
        "path": "tests/fixtures/regression/join/step7/dmc-quadruped/phase1/tier_cpu/QUARANTINED_LOAD_node2301_6398084.json",
        "why": "the ONLY CPU run for this cell, and it measured nothing: 73 polls over 6 h never saw "
        "the node below the pre-registered load-12 guard. Step 8 runs no benchmark, so the "
        "quarantine stands",
    },
    ("DMCQuadrupedWalk", "gpu"): {
        "reader": "label_file",
        "key": "DMCQuadrupedWalk",
        "path": "tests/fixtures/regression/join/step7/dmc-quadruped/phase1/tier_gpu/tiers_B_branch_gpu2109_6398085.json",
        "why": "W4's GPU tier labels, produced by scripts/regression/semantic/certify.py on the paired shard, "
        "job 6398085 gpu2109",
    },
    ("OGBCube", "gpu"): {
        "reader": "ogb_remeasure",
        "key": "OGBCube",
        "path": "tests/fixtures/regression/join/step7/ogb-manipulation/m3/fidelity/remeasure.json",
        "why": "the W5 closure re-measurement on the gate instrument; it supersedes m4/m4_tiers.json",
    },
    ("OGBScene", "gpu"): {
        "reader": "ogb_remeasure",
        "key": "OGBScene",
        "path": "tests/fixtures/regression/join/step7/ogb-manipulation/m3/fidelity/remeasure.json",
        "why": "as OGBCube",
    },
    ("OGBCube", "cpu"): {
        "reader": "ogb_cpu_missing",
        "key": "OGBCube",
        "path": "tests/fixtures/regression/join/step7/ogb-manipulation/m3/fidelity/remeasure.json",
        "why": "device='cpu' raises by design; the artifact records the raise and its mechanism",
    },
    ("OGBScene", "cpu"): {
        "reader": "ogb_cpu_missing",
        "key": "OGBScene",
        "path": "tests/fixtures/regression/join/step7/ogb-manipulation/m3/fidelity/remeasure.json",
        "why": "as OGBCube",
    },
    ("PFRocketLanding", "gpu"): {
        "reader": "rocket_gates",
        "path": "tests/fixtures/regression/join/step7/rocket-landing/tier/job6410105/gpu2106.json",
        "why": "W6's final tier job 6410105 on gpu2106; the earlier jobs 6406061/6408817 are "
        "superseded and disagree only in the third digit",
    },
    ("PFRocketLanding", "cpu"): {
        "reader": "none",
        "why": "device='cpu' raises by design. The measured CPU feasibility probe "
        "(tests/fixtures/regression/join/step7/rocket-landing/phase0/cpu_raster_probe.json) is a "
        "feasibility number for a renderer that does not exist, not a tier",
    },
}

#: Measurements that a LATER committed measurement supersedes. Listed rather than deleted: a
#: superseded number is still a number somebody may find and quote, and the reader deserves to know
#: which one the tier rests on. Each entry names both artifacts.
SUPERSEDED = [
    {
        "cell": "OGBCube",
        "device": "gpu",
        "superseded": "tests/fixtures/regression/join/step7/ogb-manipulation/m4/m4_tiers.json",
        "by": "tests/fixtures/regression/join/step7/ogb-manipulation/m3/fidelity/remeasure.json",
        "why": "m4 measured the renderer before the W5 closure follow-on (PlateRig, float32 round-trip "
        "states, role-split reference cache); it reads R 2.995 / cores 1.714 / BATCH_SCALES 5.02 "
        "against the re-measured 3.201 / 1.875 / 5.362. Same tier, different digits",
    },
    {
        "cell": "OGBScene",
        "device": "gpu",
        "superseded": "tests/fixtures/regression/join/step7/ogb-manipulation/m4/m4_tiers.json",
        "by": "tests/fixtures/regression/join/step7/ogb-manipulation/m3/fidelity/remeasure.json",
        "why": "as OGBCube",
    },
    {
        "cell": "PFRocketLanding",
        "device": "gpu",
        "superseded": "tests/fixtures/regression/join/step7/rocket-landing/tier/gpu2109.json (job 6406061) and "
        "tests/fixtures/regression/join/step7/rocket-landing/tier/job6408817/gpu2109.json",
        "by": "tests/fixtures/regression/join/step7/rocket-landing/tier/job6410105/gpu2106.json",
        "why": "three tier jobs on the same code path; the earliest read R 248.38 / BATCH_SCALES 14.56 "
        "and the last R 241.51 / 14.17. All three fail the 20x bar and all three land GPU_PARITY",
    },
]

#: Cells whose CPU path raises by design rather than falling back to CUDA. Kept explicit so the
#: three of them can never be confused with a cell that simply has no measurement yet.
CPU_RAISES_BY_DESIGN = {"OGBCube", "OGBScene", "PFRocketLanding"}


def scrub(x):
    """Replace the absolute Oscar scratch root with ``$SCR`` everywhere in a JSON-able value."""
    if isinstance(x, str):
        return x.replace(SCR, "$SCR")
    if isinstance(x, dict):
        return {k: scrub(v) for k, v in x.items()}
    if isinstance(x, list):
        return [scrub(v) for v in x]
    return x


def relocate(rel: str) -> str:
    """Map historical result/probe paths onto the maintained regression fixtures."""
    rel = str(rel)
    mapping = (
        ("examples/results/step8/support_matrix.json", "tests/fixtures/regression/support_matrix.json"),
        ("examples/results/step", "tests/fixtures/regression/join/step"),
        ("scratch/census/", "tests/fixtures/regression/census/"),
        ("scratch/certification.json", "tests/fixtures/regression/fidelity/certification.json"),
    )
    for old, new in mapping:
        if rel == old or rel.startswith(old):
            return new + rel[len(old) :]
    return rel


class Sources:
    """Every file this build reads, with its sha256, recorded in the output."""

    def __init__(self) -> None:
        self._seen: dict[str, dict] = {}

    def add(self, rel: str, role: str) -> str:
        rel = relocate(str(rel))
        p = ROOT / rel
        e = self._seen.get(rel)
        if e is None:
            e = {"path": rel, "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "roles": []}
            self._seen[rel] = e
        if role not in e["roles"]:
            e["roles"].append(role)
        return rel

    def load(self, rel: str, role: str) -> dict:
        rel = self.add(rel, role)
        return json.loads((ROOT / rel).read_text())

    def as_list(self) -> list[dict]:
        return [
            {"path": e["path"], "sha256": e["sha256"], "roles": sorted(e["roles"])}
            for e in sorted(self._seen.values(), key=lambda e: e["path"])
        ]


def load_certify():
    """Step 3's own evaluator, imported. The frozen ladder is applied by ITS code, not a copy."""
    spec = importlib.util.spec_from_file_location(
        "step3_certify", ROOT / "scripts" / "regression" / "semantic" / "certify.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------ the Step-6 tier table


def step6_tier_table(src: Sources) -> tuple[dict, dict]:
    """``{cell: {"cpu": (tier, src_key), "gpu": (tier, src_key)}}`` plus ``{src_key: path}``.

    Parsed out of the generated page rather than re-derived, because that page is itself generated
    by ``scripts/regression/optimization/report.py`` from the Step-6 artifacts and each row names the artifact key its
    tier came from. Reading it copies a tier; it does not decide one.
    """
    text = (ROOT / src.add(STEP6_DOC, "Step-6 generated tier tables + src legend")).read_text()
    tiers: dict[str, dict] = {}
    legend: dict[str, str] = {}
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        m = re.fullmatch(r"`(s\d+)`", cells[0]) if cells else None
        if m and len(cells) >= 2:
            legend[m.group(1)] = relocate(cells[1].strip("`"))
            continue
        m = re.fullmatch(r"`([A-Za-z0-9_]+)`", cells[0]) if cells else None
        if not m or len(cells) < 4:
            continue
        row = tiers.setdefault(m.group(1), {"cpu": None, "gpu": None})
        for dev, pref in (("cpu", "CPU_"), ("gpu", "GPU_")):
            if row[dev]:
                continue
            for i, c in enumerate(cells[1:], start=1):
                t = c.replace("`", "")
                if t.startswith(pref):
                    nxt = cells[i + 1].replace("`", "") if i + 1 < len(cells) else ""
                    row[dev] = (t, nxt if re.fullmatch(r"s\d+", nxt) else None)
                    break
    return tiers, legend


# ------------------------------------------------------------------ readers


def _per_batch(run: dict) -> dict:
    return {str(c["batch"]): c.get("fps") for c in run.get("cells", []) if isinstance(c, dict)}


def _numbers(dev: str, ev: dict, fps: dict, cpu_frac=None) -> dict:
    """One normalised numbers block, whatever artifact shape it came from."""
    ratio = ev.get("ratio")
    out = {
        "best_batch": ev.get("best_batch"),
        "fps_best": ev.get("se_fps"),
        "fps_1": fps.get("1"),
        "fps_4096": fps.get("4096"),
        "fps_256": fps.get("256"),
        "official_fps": ev.get("official_cpu_fps"),
        "decode_fps_per_core": ev.get("decode_fps_per_core"),
        "per_batch_fps": {k: fps[k] for k in sorted(fps, key=lambda s: int(s))} if fps else {},
        "shape": ev.get("shape"),
        "supersample": ev.get("supersample"),
    }
    if dev == "cpu":
        out["S_cpu"] = ratio
        out["cores_cpu"] = ev.get("cores")
    else:
        out["R"] = ratio
        out["cores"] = ev.get("cores")
        out["batch_scales_ratio"] = ev.get("batch_scales")
        out["no_large_batch_regression_ratio"] = ev.get("large_batch_ratio")
        out["host_bound_cpu_frac"] = ev.get("max_cpu_frac") if cpu_frac is None else cpu_frac
    return out


def read_shard(src: Sources, path: str, cell: str, dev: str, certify, census) -> dict | None:
    """A Step-3-format benchmark shard: classified by the frozen evaluator itself."""
    path = src.add(path, f"tier evidence: {cell} {dev}")
    merged = certify.merge([ROOT / path])
    runs = [
        r
        for r in merged["runs"]
        if r["env"] == cell and r["_provenance"]["device"] == ("cuda" if dev == "gpu" else "cpu")
    ]
    if not runs:
        return None
    run = sorted(runs, key=lambda r: (str(r["_provenance"].get("host")), str(r["_provenance"].get("torch_threads"))))[
        0
    ]
    label = certify.classify(run, "cuda" if dev == "gpu" else "cpu", census)
    prov = run["_provenance"]
    return {
        "numbers": _numbers(dev, label.get("evidence") or {}, _per_batch(run)),
        "tier_recomputed": label.get("tier"),
        "qualifiers_from_measurement": label.get("qualifiers") or [],
        "capped_by": label.get("capped_by"),
        "blocked": label.get("blocked"),
        "provenance": {
            "artifact": path,
            "host": prov.get("host"),
            "device_name": prov.get("device_name"),
            "threads": prov.get("torch_threads"),
            "job": prov.get("slurm_job") or prov.get("job"),
            "loadavg": prov.get("loadavg"),
            "versions": prov.get("versions"),
            "evaluator": "scripts/regression/semantic/certify.py:classify (Step-3 evaluator, frozen Step-2 ladder)",
        },
    }


def read_label_file(src: Sources, path: str, cell: str, dev: str) -> dict | None:
    """An artifact that already IS a ``certify.classify`` label (or a list of them)."""
    doc = src.load(path, f"tier evidence: {cell} {dev}")
    want = "gpu" if dev == "gpu" else "cpu"
    cands = []
    if isinstance(doc.get("labels"), list):
        cands = [l for l in doc["labels"] if l.get("env") == cell and l.get("device") == want]
    elif doc.get("after"):
        a = doc["after"]
        if a.get("env") == cell and a.get("device") == want:
            cands = [a]
    if not cands:
        return None
    # Prefer a label that reached the renderer; among blocked ones prefer the most informative.
    cands.sort(
        key=lambda l: (bool(l.get("blocked")), not l.get("qualifiers"), str(l.get("host")), str(l.get("threads")))
    )
    label = cands[0]
    fps = {k: v for k, v in (label.get("per_batch_fps") or {}).items()}
    return {
        "numbers": _numbers(dev, label.get("evidence") or {}, fps),
        "tier_recomputed": label.get("tier"),
        "qualifiers_from_measurement": label.get("qualifiers") or [],
        "capped_by": label.get("capped_by"),
        "blocked": label.get("blocked"),
        "provenance": {
            "artifact": path,
            "host": label.get("host"),
            "device_name": label.get("device_name"),
            "threads": label.get("threads"),
            "job": None,
            "shard": label.get("shard"),
            "why": label.get("why"),
            "evaluator": doc.get("evaluator")
            or "scripts/regression/semantic/certify.py:classify (Step-3 evaluator, frozen Step-2 ladder)",
        },
    }


def read_declaration(src: Sources, path: str, key: str, dev: str) -> dict | None:
    """A class ``performance_record_{cpu,gpu}`` declaration, as the Step-7 closure recorded it."""
    doc = src.load(path, f"class declaration: {key} {dev}")
    rec = (doc.get(key) or {}).get(f"performance_record_{dev}")
    if not isinstance(rec, dict):
        return None
    low = {k.lower(): v for k, v in rec.items()}
    ev = {
        "best_batch": low.get("mixed_default_batch") or low.get("mixed_batch"),
        "se_fps": low.get("mixed_default_fps") or low.get("mixed_fps"),
        "official_cpu_fps": low.get("official_jitted_fps"),
        "decode_fps_per_core": low.get("decode_bar_fps"),
        "ratio": low.get("s_cpu") if dev == "cpu" else low.get("r"),
        "cores": low.get("cores_cpu") if dev == "cpu" else low.get("cores"),
        "max_cpu_frac": None,
        "large_batch_ratio": low.get("no_large_batch_regression"),
    }
    bs = low.get("batch_scales")
    ev["batch_scales"] = bs if isinstance(bs, (int, float)) else None
    q = rec.get("qualifier") or rec.get("qualifiers") or []
    return {
        "numbers": _numbers(dev, ev, {}),
        "tier_recomputed": rec.get("tier"),
        "qualifiers_from_measurement": [q] if isinstance(q, str) else list(q),
        "capped_by": None,
        "blocked": None,
        "provenance": {
            "artifact": path,
            "host": rec.get("host"),
            "device_name": rec.get("card"),
            "threads": None,
            "job": rec.get("job"),
            "conditions": rec.get("reference_conditions") or rec.get("condition"),
            "declaration": f"{key}.performance_record_{dev} (class declaration, quoted by the "
            f"Step-7 closure into {path})",
        },
        "declaration_extras": {k: v for k, v in rec.items() if k not in ("tier", "qualifier", "qualifiers")},
    }


def read_manipulator_cpu(src: Sources, path: str, dev: str) -> dict:
    d = src.load(path, "tier evidence: DMCManipulatorBringBall cpu")
    fps = {str(c["batch"]): c.get("fps") for c in d.get("cells", [])}
    ev = {
        "best_batch": d.get("certified_batch"),
        "se_fps": d.get("fps"),
        "official_cpu_fps": (d.get("official") or {}).get("fps"),
        "decode_fps_per_core": (d.get("decode") or {}).get("decode_fps_per_core"),
        "ratio": d.get("S_cpu"),
        "cores": d.get("cores_cpu"),
        "shape": f"{d.get('size')}x{d.get('size')}",
        "supersample": 1,
    }
    return {
        "numbers": _numbers("cpu", ev, fps),
        "tier_recomputed": None,
        "qualifiers_from_measurement": [],
        "capped_by": None,
        "blocked": None,
        "provenance": {
            "artifact": path,
            "host": d.get("host"),
            "device_name": None,
            "threads": d.get("torch_threads"),
            "job": d.get("slurm_job"),
            "loadavg": d.get("loadavg_before"),
            "versions": {"mujoco": d.get("mujoco"), "dm_control": d.get("dm_control")},
            "tree": d.get("tree"),
            "note": "B=1 is the certified batch here because every larger batch is "
            "SLOWER on this cell; Step 2 forbids certifying a GPU row on B=1 and "
            "this is a CPU row, where the sweep's own maximum is the certified "
            "point",
        },
    }


def read_economics(src: Sources, path: str, key: str) -> dict:
    d = src.load(path, f"tier evidence: {key} gpu")
    e = (d.get("envs") or {}).get(key) or {}
    best = e.get("gpu_best") or {}
    fps = {str(c["batch"]): c.get("fps") for c in e.get("gpu") or []}
    official = ((e.get("cpu") or {}).get("set_state_render") or {}).get("fps")
    decode = (e.get("decode") or {}).get("decode_fps_per_core")
    f1, f4096 = fps.get("1"), fps.get("4096")
    ev = {
        "best_batch": best.get("batch"),
        "se_fps": best.get("fps"),
        "official_cpu_fps": official,
        "decode_fps_per_core": decode,
        "ratio": (best.get("fps") / official) if official and best.get("fps") else None,
        "cores": (best.get("fps") / decode) if decode and best.get("fps") else None,
        "max_cpu_frac": best.get("cpu_frac"),
        "batch_scales": (f4096 / f1) if f1 and f4096 else None,
        "large_batch_ratio": (f4096 / fps["256"]) if f4096 and fps.get("256") else None,
        "shape": f"{e.get('size')}x{e.get('size')}",
        "supersample": best.get("supersample"),
    }
    return {
        "numbers": _numbers("gpu", ev, fps),
        "tier_recomputed": None,
        "qualifiers_from_measurement": [],
        "capped_by": None,
        "blocked": None,
        "provenance": {
            "artifact": path,
            "host": None,
            "device_name": d.get("device"),
            "threads": d.get("cpu_count"),
            "job": None,
            "loadavg": best.get("loadavg"),
            "cpu": d.get("cpu"),
            "versions": {"torch": d.get("torch")},
            "note": "the artifact does not record its host or SLURM job; the Step-7 "
            "workstream record (tests/fixtures/regression/join/step7/dmc-manipulator/"
            "findings_phase1.md section 10, and the dashboard W3 row) names "
            "gpu2107 and job 6395157. Flagged HOST_FROM_PROSE",
        },
    }


def read_ogb_remeasure(src: Sources, path: str, key: str) -> dict:
    d = src.load(path, f"tier evidence: {key} gpu")
    c = (d.get("cells") or {}).get(key) or {}
    t = c.get("tiers") or {}
    prov = (c.get("corpus") or {}).get("provenance") or {}
    ev = {
        "best_batch": None,
        "se_fps": t.get("fps_best"),
        "official_cpu_fps": t.get("official_fps_same_node"),
        "decode_fps_per_core": t.get("decode_fps_per_core_same_node"),
        "ratio": t.get("R_ours_over_official"),
        "cores": t.get("cores"),
        "batch_scales": t.get("batch_scales"),
        "max_cpu_frac": None,
    }
    return {
        "numbers": _numbers("gpu", ev, {"1": t.get("fps_1")}),
        "tier_recomputed": None,
        "qualifiers_from_measurement": ([] if t.get("batch_scales_pass") else ["BATCH_SCALES_FAIL"]),
        "capped_by": None
        if t.get("batch_scales_pass")
        else "BATCH_SCALES: recorded in the artifact as batch_scales_pass=false against the frozen 20x bar",
        "blocked": None,
        "provenance": {
            "artifact": path,
            "host": prov.get("host"),
            "device_name": prov.get("gpu"),
            "threads": None,
            "job": prov.get("slurm_job"),
            "versions": {
                k: prov.get(k)
                for k in ("mujoco", "mujoco-warp", "ogbench", "torch", "warp-lang", "dm-control", "python")
            },
            "note": "all three comparators (ours, the official on hardware EGL, the "
            "one-core decode) timed on the same node",
        },
    }


def read_ogb_cpu_missing(src: Sources, path: str, key: str) -> dict:
    d = src.load(path, f"cpu raise record: {key}")
    t = ((d.get("cells") or {}).get(key) or {}).get("tiers") or {}
    return {
        "numbers": {},
        "tier_recomputed": None,
        "qualifiers_from_measurement": ["RAISES_BY_DESIGN"],
        "capped_by": None,
        "blocked": {"kind": "raises_by_design", "reason": t.get("cpu_error")},
        "provenance": {
            "artifact": path,
            "host": None,
            "device_name": None,
            "threads": None,
            "job": None,
            "note": "device='cpu' raises rather than silently rendering on CUDA; the "
            "artifact records cpu_raises and the mechanism",
        },
    }


_ROCKET_TIER = re.compile(
    r"(?P<tier>[A-Z_]+)\s+\[(?P<quals>[^\]]*)\]\s+R=(?P<R>[\d.]+)\s+\(ours\s+(?P<fps>[\d,.]+)\s+f/s"
    r"\s+at\s+B=(?P<batch>\d+)\s+vs\s+official\s+(?P<off>[\d,.]+)\s+f/s[^)]*\)\s+cores=(?P<cores>[\d.]+)"
    r"\s+\(vs\s+(?P<dec>[\d,.]+)\s+f/s[^)]*\)\s+cpu_frac=(?P<cf>[\d.]+)"
)
_ROCKET_RATIO = re.compile(r"=\s*(?P<v>[\d.]+)\s*\(bar")


def read_rocket_gates(src: Sources, path: str) -> dict:
    """W6's tier job document: three gate rows carry the whole tier, as structured evidence strings."""
    d = src.load(path, "tier evidence: PFRocketLanding gpu")
    gates = d.get("gates") or {}
    m = _ROCKET_TIER.search((gates.get("tier_frozen_definitions_same_node") or {}).get("evidence", ""))
    if not m:
        raise SystemExit(f"{path}: tier_frozen_definitions_same_node evidence did not parse")
    num = lambda s: float(s.replace(",", ""))  # noqa: E731
    bs = _ROCKET_RATIO.search((gates.get("BATCH_SCALES") or {}).get("evidence", ""))
    nlbr = _ROCKET_RATIO.search((gates.get("NO_LARGE_BATCH_REGRESSION") or {}).get("evidence", ""))
    fps_best, scale = num(m.group("fps")), float(bs.group("v")) if bs else None
    ev = {
        "best_batch": int(m.group("batch")),
        "se_fps": fps_best,
        "official_cpu_fps": num(m.group("off")),
        "decode_fps_per_core": num(m.group("dec")),
        "ratio": float(m.group("R")),
        "cores": float(m.group("cores")),
        "max_cpu_frac": float(m.group("cf")),
        "batch_scales": scale,
        "large_batch_ratio": float(nlbr.group("v")) if nlbr else None,
    }
    fps = {"1": (fps_best / scale) if scale else None, str(m.group("batch")): fps_best}
    return {
        "numbers": _numbers("gpu", ev, {k: v for k, v in fps.items() if v}),
        "tier_recomputed": m.group("tier"),
        "qualifiers_from_measurement": [q.strip().strip("'") for q in m.group("quals").split(",") if q.strip()],
        "capped_by": None
        if (gates.get("BATCH_SCALES") or {}).get("pass")
        else f"BATCH_SCALES: fps(4096) is {scale}x fps(1), under the 20x bar",
        "blocked": None,
        "provenance": {
            "artifact": path,
            "host": d.get("host"),
            "device_name": d.get("gpu"),
            "threads": d.get("threads"),
            "job": d.get("slurm_job"),
            "loadavg": d.get("load_average"),
            "versions": d.get("versions"),
            "note": "fps(1) is derived from the committed BATCH_SCALES ratio and the "
            "committed fps at B=4096; the artifact records the ratio, not the "
            "B=1 cell",
        },
    }


def read_quarantine(src: Sources, path: str) -> dict:
    d = src.load(path, "cpu quarantine record: DMCQuadrupedWalk")
    return {
        "numbers": {},
        "tier_recomputed": d.get("tier"),
        "qualifiers_from_measurement": ["LOAD_QUARANTINED"],
        "capped_by": None,
        "blocked": {
            "kind": "load_quarantined",
            "reason": f"{d.get('polls')} polls, load1 {d.get('load_min')}-{d.get('load_max')}, rule: {d.get('rule')}",
        },
        "provenance": {
            "artifact": path,
            "host": d.get("node"),
            "device_name": None,
            "threads": None,
            "job": d.get("job"),
            "note": "nothing was measured: the pre-registered load guard never opened. "
            "Step 8 runs no benchmark, so this stays UNMEASURED",
        },
    }


def read_step3_labels(src: Sources, cell: str, dev: str) -> dict | None:
    doc = src.load(STEP3_LABELS, f"tier evidence: {cell} {dev}")
    want = "gpu" if dev == "gpu" else "cpu"
    cands = [l for l in doc["labels"] if l.get("env") == cell and l.get("device") == want]
    if not cands:
        return None
    cands.sort(
        key=lambda l: (bool(l.get("blocked")), not l.get("qualifiers"), str(l.get("host")), str(l.get("threads")))
    )
    label = cands[0]
    return {
        "numbers": _numbers(dev, label.get("evidence") or {}, {}),
        "tier_recomputed": label.get("tier"),
        "qualifiers_from_measurement": label.get("qualifiers") or [],
        "capped_by": label.get("capped_by"),
        "blocked": label.get("blocked"),
        "provenance": {
            "artifact": STEP3_LABELS,
            "host": label.get("host"),
            "device_name": label.get("device_name"),
            "threads": label.get("threads"),
            "job": None,
            "why": label.get("why"),
            "evaluator": "scripts/regression/semantic/certify.py:classify (Step-3 evaluator)",
        },
        "alternatives_in_artifact": len(cands),
    }


# ------------------------------------------------------------------ assembly


def evidence_for(cell: str, dev: str, src: Sources, s6: dict, legend: dict, certify, census):
    """(evidence-or-None, how) for one (cell, device), preferring the Step-7 record where one exists."""
    spec = STEP7_EVIDENCE.get((cell, dev))
    if spec:
        r = spec["reader"]
        if r == "declaration":
            ev = read_declaration(src, spec["path"], spec["key"], dev)
        elif r == "manipulator_cpu":
            ev = read_manipulator_cpu(src, spec["path"], dev)
        elif r == "economics":
            ev = read_economics(src, spec["path"], spec["key"])
        elif r == "ogb_remeasure":
            ev = read_ogb_remeasure(src, spec["path"], spec["key"])
        elif r == "ogb_cpu_missing":
            ev = read_ogb_cpu_missing(src, spec["path"], spec["key"])
        elif r == "rocket_gates":
            ev = read_rocket_gates(src, spec["path"])
        elif r == "label_file":
            ev = read_label_file(src, spec["path"], spec["key"], dev)
        elif r == "quarantine":
            ev = read_quarantine(src, spec["path"])
        elif r == "none":
            ev = None
        else:  # pragma: no cover
            raise SystemExit(f"unknown reader {r}")
        if ev is not None:
            ev["measurement_source"] = f"Step-7 workstream artifact ({spec['reader']})"
            ev["why_this_source"] = spec["why"]
            return ev, "step7"
        return None, spec["why"]
    entry = (s6.get(cell) or {}).get(dev)
    if not entry:
        return None, (
            "no committed tier artifact names this cell for this device; the Step-6 generated table has no row for it"
        )
    tier, key = entry
    path = legend.get(key or "")
    if not path:
        return None, (
            f"the Step-6 table gives the tier {tier} for this cell but names no src "
            f"artifact key, so no underlying numbers can be quoted"
        )
    if path == STEP3_LABELS:
        ev = read_step3_labels(src, cell, dev)
    elif path.endswith(("tier_cpu_final.json", "tier_gpu_final.json")):
        ev = read_label_file(src, path, cell, dev)
    else:
        ev = read_shard(src, path, cell, dev, certify, census)
    if ev is None:
        return None, f"{path} (named by the Step-6 table as `{key}`) has no row for {cell}/{dev}"
    ev["measurement_source"] = f"Step-6 tier table src `{key}` -> {path}"
    ev["why_this_source"] = "the generated Step-6 tier table names this artifact as the evidence for this cell's tier"
    return ev, "step6"


def reference_label(dev: str, ev: dict | None) -> dict:
    """The frozen reference rule: an official renderer timed on the same host, else NO_REFERENCE."""
    if not ev or not (ev.get("numbers") or {}).get("fps_best"):
        return {
            "label": "NO_MEASUREMENT",
            "why": "no throughput was measured for this row, so the reference question does not arise",
        }
    n = ev["numbers"]
    quals = ev.get("qualifiers_from_measurement") or []
    if n.get("official_fps"):
        prov = ev.get("provenance") or {}
        same = True
        why = (
            "the official renderer was timed in the same artifact, on the host that artifact "
            "records, so the ratio is same-host as Step 2 requires"
        )
        job = str(prov.get("job") or "")
        host = str(prov.get("host") or "")
        if "/" in host or "," in job:
            same = False
            why = (
                f"the declaration records more than one host/job for the two arms "
                f"(host {host!r}, job {job!r}): ours and the official reference were NOT timed "
                f"in the same job, and the row is flagged REFERENCE_SPLIT_JOB"
            )
        return {
            "label": "OFFICIAL_SAME_HOST" if same else "OFFICIAL_SPLIT_JOB",
            "official_fps": n["official_fps"],
            "why": why,
        }
    return {
        "label": "NO_REFERENCE",
        "why": (
            "no official renderer was timed for this row, so the ratio is undefined and the "
            "row certifies on the cores ladder (certification.md 5.5 / 6.8)"
            if "NO_REFERENCE" in quals
            else "the artifact carries no official-renderer fps for this row"
        ),
    }


def recompute_tier(certify, dev: str, cls: str | None, n: dict) -> tuple[str | None, str]:
    """Walk the FROZEN ladder for a row whose artifact carries numbers but no label.

    ``certify._tier_from`` and ``certify.SPEC`` are Step 3's own; this function only feeds them the
    numbers the artifact recorded and then applies the two frozen GPU caps the same file applies.
    It exists to CHECK the copied tier, never to replace it.
    """
    ratio = n.get("R") if dev == "gpu" else n.get("S_cpu")
    cores = n.get("cores") if dev == "gpu" else n.get("cores_cpu")
    if cores is None and ratio is None:
        return None, "the artifact carries neither a ratio nor a cores number"
    spec = certify.SPEC["gpu" if dev == "gpu" else "cpu"]
    up = "GPU" if dev == "gpu" else "CPU"
    order = [f"{up}_EXCEPTIONAL", f"{up}_FAST", f"{up}_FASTER", f"{up}_PARITY", f"{up}_REGRESSION"]
    if ratio is None:
        tier = certify._tier_from(spec["no_reference_fallback"], order[:3], None, cores, "R", "cores")
        why = "NO_REFERENCE ladder (no official renderer timed for this row)"
        if tier == "unknown":
            return None, why + "; the cores number clears no rung"
    elif dev == "gpu":
        ladder = (spec["tiers"] or {}).get(cls)
        if not ladder:
            return None, (
                f"backend class {cls!r} has no frozen GPU ladder: the Step-3 census at "
                f"this HEAD carries no entry for this cell (a Step-7 carryover for the "
                f"cells Step 7 added), so the evaluator cannot pick class S or class R"
            )
        tier = certify._tier_from(ladder, order, ratio, cores, "R", "cores")
        why = f"class {cls} ladder on R={ratio}, cores={cores}"
    else:
        tier = certify._tier_from(spec["tiers"], order, ratio, cores, "S_cpu", "cores_cpu")
        why = f"CPU ladder on S_cpu={ratio}, cores_cpu={cores}"
    if dev == "gpu" and tier and tier != "unknown":
        q = certify.SPEC["gpu"]["qualifiers"]
        scale = n.get("batch_scales_ratio")
        cap = q["BATCH_SCALES"]["required_above_tier"]
        if (
            scale is not None
            and scale < q["BATCH_SCALES"]["factor"]
            and certify._tier_rank(tier) > certify._tier_rank(cap)
        ):
            why += f"; capped at {cap} by BATCH_SCALES {scale}x < {q['BATCH_SCALES']['factor']}x"
            tier = cap
    return (None if tier == "unknown" else tier), why


def _blocked_from_record(tier: str | None, flags: list[str], t7: dict, how: str) -> dict | None:
    """Why a row with no numbers has none, taken from the Step-7 record rather than invented."""
    if not tier or tier.endswith(("_FAST", "_FASTER", "_PARITY", "_REGRESSION", "_EXCEPTIONAL")):
        return None
    note = (t7.get("detail") or {}).get("note") or (t7.get("detail") or {}).get("why")
    kind = (
        "raises_by_design"
        if "RAISES_BY_DESIGN" in flags
        else "load_quarantined"
        if "LOAD_QUARANTINED" in flags
        else "no_measurement"
    )
    return {"kind": kind, "reason": note or how, "recorded_in": t7.get("source") or STEP7_MATRIX}


def status_of(tier: str | None, quals: list[str]) -> str:
    if not tier:
        return "UNMEASURED"
    if tier.endswith("_MISSING"):
        return "MISSING-UNSUPPORTED_BY_DESIGN" if "RAISES_BY_DESIGN" in quals else "UNMEASURED"
    if tier.endswith("_UNMEASURED") or tier == "unknown":
        return "UNMEASURED"
    return "MEASURED"


def support_matrix_check(src: Sources, rows: list[dict]) -> dict:
    """Compare every id's tier and status against Workstream E's support matrix, row by row.

    The matrix is the other authoritative Step-8 table of the same 42 ids and it reaches its tier
    column by a different route (the Step-7 coverage matrix, read through its own generator). Two
    tables built from the same evidence by different code disagreeing is exactly the drift both
    exist to prevent, so the comparison belongs in the artifact, not in prose.
    """
    if not (ROOT / SUPPORT_MATRIX).is_file():
        return {"status": "ABSENT", "why": f"{SUPPORT_MATRIX} is not present at this HEAD; the comparison was not run"}
    sm = src.load(SUPPORT_MATRIX, "cross-check: Workstream E support matrix (tier columns)")
    by_id = {r["swm_id"]: r for r in sm.get("rows", [])}
    tier_diffs, status_diffs, missing, placeholders = [], [], [], []
    placeholder = {"n/a", "unknown", None, ""}
    for row in rows:
        other = by_id.get(row["swm_id"])
        if other is None:
            missing.append(row["swm_id"])
            continue
        for dev in ("cpu", "gpu"):
            theirs = other.get(f"{dev}_tier") or {}
            if row[dev]["tier"] != theirs.get("tier"):
                d = {
                    "swm_id": row["swm_id"],
                    "device": dev,
                    "here": row[dev]["tier"],
                    "support_matrix": theirs.get("tier"),
                }
                # Neither side names a tier: the two tables spell "there is no tier here"
                # differently. That is not a disagreement about a measurement and is kept apart
                # from one, so the disagreement list means what it says.
                (placeholders if {d["here"], d["support_matrix"]} <= placeholder else tier_diffs).append(d)
            if row[dev]["status"] != theirs.get("status"):
                status_diffs.append(
                    {
                        "swm_id": row["swm_id"],
                        "device": dev,
                        "here": row[dev]["status"],
                        "support_matrix": theirs.get("status"),
                    }
                )
    return {
        "status": "COMPARED",
        "ids_compared": len(rows) - len(missing),
        "ids_missing_from_support_matrix": missing,
        "tier_disagreements": tier_diffs,
        "no_tier_placeholder_differences": placeholders,
        "status_word_differences": status_diffs,
        "note": "a tier disagreement would be a real disagreement about a measurement; a status "
        "word difference is a vocabulary difference. Both are listed, neither is "
        "resolved here, and the doc explains each one",
    }


def build(census_dir: Path | None = None) -> dict:
    src = Sources()
    certify = load_certify()
    src.add(SPEC_JSON, "frozen Step-2 certification spec (ladders, bars, qualifiers)")
    src.add("docs/research/certification.md", "frozen Step-2 tier definitions (prose)")
    census_dir = census_dir or (ROOT / "tests" / "fixtures" / "regression" / "census")
    census_paths = sorted(census_dir.glob("census_*.json"))
    census = certify.read_censuses(census_paths)
    for p in census_paths:
        rel = p.relative_to(ROOT) if p.is_relative_to(ROOT) else p
        if p.is_relative_to(ROOT):
            src.add(rel, "backend-class census (Step-1 declarations)")
        else:
            src._seen.setdefault(
                str(p),
                {
                    "path": str(p),
                    "roles": ["backend-class census (OUT OF TREE, --census-dir)"],
                    "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                },
            )
    s6, legend = step6_tier_table(src)
    matrix = src.load(STEP7_MATRIX, "Step-7 record: per-id tiers, cells, pixel/symbolic")

    rows, disagreements, superseded_by_step7 = [], [], []
    for (cell, dev), spec in sorted(STEP7_EVIDENCE.items()):
        old6 = (s6.get(cell) or {}).get(dev)
        if not old6:
            continue
        superseded_by_step7.append(
            {
                "cell": cell,
                "device": dev,
                "superseded": f"{STEP6_DOC} (generated Step-6 tier table: {old6[0]}, src `{old6[1]}`)",
                "by": spec.get("path", "n/a"),
                "why": spec["why"],
            }
        )
    for r in sorted(matrix["rows"], key=lambda r: r["swm_id"]):
        sid = r["swm_id"]
        cells = list(r.get("stable_envs_envs") or [])
        primary = cells[0] if cells else None
        row = {
            "swm_id": sid,
            "pixel_or_symbolic": "pixel" if r.get("pixel_producing") else "symbolic",
            "stable_envs_cells": cells,
            "primary_cell": primary,
            "backend_class": certify.backend_class(primary, census)[0] if primary else "n/a",
            "backends": certify.backend_class(primary, census)[1] if primary else [],
        }
        for dev in ("cpu", "gpu"):
            if not r.get("pixel_producing"):
                row[dev] = {
                    "tier": "n/a",
                    "status": "UNSUPPORTED_BY_DESIGN",
                    "flags": ["SYMBOLIC"],
                    "tier_source": STEP7_MATRIX,
                    "why": "this id produces no frame by design (the upstream env is symbolic), so "
                    "there is no renderer to tier; the live probe recorded the raise verbatim",
                    "numbers": {},
                    "reference": {"label": "UNSUPPORTED_BY_DESIGN"},
                    "provenance": {},
                    "tier_recomputed": None,
                    "agrees_with_step7": None,
                }
                continue
            t7 = r.get(f"{dev}_tier") or {}
            tier = t7.get("tier")
            quals = list(t7.get("qualifiers") or [])
            ev, how = evidence_for(primary, dev, src, s6, legend, certify, census)
            flags = list(quals)
            for q in (ev or {}).get("qualifiers_from_measurement", []):
                if q not in flags:
                    flags.append(q)
            if dev == "cpu" and primary in CPU_RAISES_BY_DESIGN and "RAISES_BY_DESIGN" not in flags:
                flags.append("RAISES_BY_DESIGN")
            ref = reference_label(dev, ev)
            if ref["label"] == "OFFICIAL_SPLIT_JOB" and "REFERENCE_SPLIT_JOB" not in flags:
                flags.append("REFERENCE_SPLIT_JOB")
            if ref["label"] == "NO_REFERENCE" and "NO_REFERENCE" not in flags and ev:
                flags.append("NO_REFERENCE")
            if ev and (ev.get("provenance") or {}).get("note", "").find("does not record its host") >= 0:
                flags.append("HOST_FROM_PROSE")
            if how == "step6" and not quals:
                flags.append("TIER_COPIED_FROM_GENERATED_TABLE")
            rec = (ev or {}).get("tier_recomputed")
            rec_why = "the artifact is itself a label produced by the frozen Step-3 evaluator"
            if rec == "unknown":
                # The label in the artifact says `unknown` because the evaluator could not assign a
                # backend class, not because the numbers are missing. Walk the ladder here instead.
                rec, rec_why = None, "the committed label reads `unknown`"
            if ev and rec is None and ev.get("numbers"):
                cls = row["backend_class"]
                if cls in (None, "unknown"):
                    cls = ((ev.get("declaration_extras") or {}).get("backend_class")) or cls
                rec, rec_why = recompute_tier(certify, dev, cls, ev["numbers"])
            if rec is None and ev and ev.get("numbers") and "census" in (rec_why or ""):
                flags.append("TIER_RECOMPUTE_BLOCKED_BY_CENSUS")
            agrees = None
            if rec and tier:
                agrees = rec == tier
                if not agrees:
                    flags.append("TIER_DISAGREES_WITH_EVIDENCE")
                    disagreements.append(
                        {
                            "swm_id": sid,
                            "cell": primary,
                            "device": dev,
                            "step7_record_tier": tier,
                            "tier_from_committed_evidence": rec,
                            "how_recomputed": rec_why,
                            "evidence_artifact": (ev.get("provenance") or {}).get("artifact"),
                            "resolution": "NOT resolved here. The Step-7 record's tier is kept and the "
                            "disagreement is explained in docs/research/step8_performance.md",
                        }
                    )
            row[dev] = {
                "tier": tier or "unknown",
                "status": status_of(tier, flags),
                "flags": sorted(set(flags)),
                "tier_source": t7.get("source"),
                "tier_source_note": ((t7.get("detail") or {}).get("note") or (t7.get("detail") or {}).get("why")),
                "measurement_source": (ev or {}).get("measurement_source"),
                "why_this_source": (ev or {}).get("why_this_source") or (None if ev else how),
                "numbers": (ev or {}).get("numbers") or {},
                "reference": ref,
                "provenance": (ev or {}).get("provenance") or {},
                "blocked": (ev or {}).get("blocked") or _blocked_from_record(tier, flags, t7, how),
                "capped_by": (ev or {}).get("capped_by"),
                "tier_recomputed": rec,
                "tier_recomputed_how": rec_why,
                "agrees_with_step7": agrees,
            }
        rows.append(row)

    counts = {}
    for dev in ("cpu", "gpu"):
        c = {}
        for row in rows:
            c[row[dev]["status"]] = c.get(row[dev]["status"], 0) + 1
        counts[f"{dev}_status"] = dict(sorted(c.items()))
        t = {}
        for row in rows:
            t[row[dev]["tier"]] = t.get(row[dev]["tier"], 0) + 1
        counts[f"{dev}_tier"] = dict(sorted(t.items()))
    counts["rows"] = len(rows)
    counts["pixel"] = sum(1 for r in rows if r["pixel_or_symbolic"] == "pixel")
    counts["symbolic"] = sum(1 for r in rows if r["pixel_or_symbolic"] == "symbolic")

    sm_check = support_matrix_check(src, rows)
    sources = src.as_list()
    return scrub(
        {
            "what_this_is": "Step-8 Workstream G: the final CPU/GPU tier accounting for all 42 "
            "registered stable_worldmodel ids, assembled from committed Step-6/Step-7 "
            "artifacts. Nothing was re-measured and no threshold was moved.",
            "schema_version": 1,
            "generated_by": "scripts/regression/support/build_performance.py",
            "step8_baseline_sha": _baseline_sha(),
            "status_vocabulary": {
                "MEASURED": "a tier was measured for this device and it is a GRADE, not a gate: "
                "CPU_REGRESSION is measured",
                "UNMEASURED": "no usable measurement exists (never run, blocked, or load-quarantined)",
                "MISSING-UNSUPPORTED_BY_DESIGN": "the renderer raises on this device by design rather "
                "than falling back to the other one",
                "UNSUPPORTED_BY_DESIGN": "this id produces no frame at all (symbolic)",
            },
            "rules_applied": {
                "tiers": "COPIED from the Step-7 record; recomputed only as a cross-check",
                "reference": "an official renderer timed on the same host makes the ratio (S_cpu / R); "
                "otherwise NO_REFERENCE and the row certifies on the cores ladder",
                "batch_scales": "fps(4096) >= 20 * fps(1), frozen in Step 2 and NOT changed in Step 8",
                "never_certify_on_batch_1": "GPU rows certify on the best batch above B=1",
                "no_projections": "a projection never stands in for a measurement",
            },
            "sources": sources,
            "counts": counts,
            "cross_checks": {
                "against": STEP7_MATRIX,
                "support_matrix": sm_check,
                "disagreements": disagreements,
                "superseded_measurements": SUPERSEDED + superseded_by_step7,
                "note": "a disagreement is recorded, never resolved here",
            },
            "rows": rows,
        }
    )


def _baseline_sha() -> str:
    txt = (ROOT / "tests/fixtures/regression/join/step8/dashboard.yaml").read_text()
    m = re.search(r"baseline_sha:\s*\"?([0-9a-f]{7,40})", txt)
    return m.group(1) if m else "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--check", action="store_true", help="regenerate and compare against the committed file instead of writing it"
    )
    ap.add_argument(
        "--census-dir",
        type=Path,
        default=None,
        help="read the backend-class census from another tree (a cross-check only: the "
        "committed tiers.json is always built from this worktree's regression census)",
    )
    ap.add_argument("--stdout", action="store_true", help="print the document instead of writing it")
    args = ap.parse_args()
    doc = json.dumps(build(args.census_dir), indent=1, sort_keys=True, ensure_ascii=False) + "\n"
    if args.stdout:
        print(doc, end="")
        return
    if args.check:
        if not OUT.is_file():
            sys.exit(f"{OUT} does not exist")
        if OUT.read_text() != doc:
            sys.exit(f"{OUT} is NOT byte-identical to a regeneration")
        print(f"{OUT.relative_to(ROOT)}: byte-identical to a regeneration")
        return
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(doc)
    d = json.loads(doc)
    print(f"wrote {OUT.relative_to(ROOT)}: {d['counts']['rows']} rows")
    print("cpu status:", d["counts"]["cpu_status"])
    print("gpu status:", d["counts"]["gpu_status"])
    print("disagreements:", len(d["cross_checks"]["disagreements"]))
    for x in d["cross_checks"]["disagreements"]:
        print(
            f"  {x['swm_id']:36s} {x['device']}: step7={x['step7_record_tier']} "
            f"evidence={x['tier_from_committed_evidence']}"
        )


if __name__ == "__main__":
    main()
