"""check_resolution_consistency for EVERY config with a DOWNSAMPLE_EXACT_FRAC bound, on both state sets.

The evidence Sami needs before the certifier moves the judged number from the frozen corpus to the
bound's own protocol: for each of the 18 bounded configs, the number on the protocol draw
(``render_worker.sample_inputs(env, 8)``, default appearance -- ``MEASURED_ON``) and on the 64-state
frozen Step-4 corpus, against the recorded floor and margin. The check is reference-free, so any
interpreter that can BUILD the env is the right interpreter; run once per venv and merge with
``--merge``. A config the venv cannot build is recorded as blocked, never as a number.

    python scratch/step6/texture-fidelity/downsample_all.py --out <dir>/<venv>.json
    python scratch/step6/texture-fidelity/downsample_all.py --merge <dir>/*.json --out <dir>/table.md
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
N = 8                         # tests/gpu/test_reproducibility.py::N -- the fixture the bound was measured by


def bounds() -> tuple[dict, float, str]:
    src = (ROOT / "tests" / "gpu" / "test_reproducibility.py").read_text()
    tree = ast.parse(src)
    table, margin, on = {}, None, ""
    for n in tree.body:
        if isinstance(n, (ast.Assign, ast.AnnAssign)):
            t = n.targets[0] if isinstance(n, ast.Assign) else n.target
            if isinstance(t, ast.Name) and t.id == "DOWNSAMPLE_EXACT_FRAC":
                table = ast.literal_eval(n.value)
            if isinstance(t, ast.Name) and t.id == "DOWNSAMPLE_MARGIN":
                margin = ast.literal_eval(n.value)
            if isinstance(t, ast.Name) and t.id == "MEASURED_ON":
                on = ast.literal_eval(n.value)
    return table, margin, on


def bounded_configs() -> list[str]:
    cert = json.loads((ROOT / "examples" / "results" / "step4" / "certification.json").read_text())
    return sorted(k for k, v in cert["configs"].items()
                  if v["checks"].get("check_resolution_consistency", {}).get("status") in ("pass", "fail"))


def measure(out: Path) -> None:
    import torch
    import stable_envs as se
    from stable_envs.testing import conformance as C
    from stable_envs.testing import render_worker as W
    table, margin, on = bounds()
    doc = {"venv": sys.prefix, "device": "cuda", "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "host": os.uname().nodename,
           "protocol": {"n": N, "state_seed": W.STATE_SEED, "measured_on": on}, "margin": margin, "configs": {}}
    states_dir = ROOT / "examples" / "results" / "step4" / "states"
    for cfg in bounded_configs():
        t0 = time.perf_counter()
        row = {"bound": table.get(cfg)}
        if cfg not in se.REGISTRY:
            row.update(status="blocked_environment", why="not registered in this interpreter")
        else:
            try:
                env = se.make(cfg, batch=N, device="cuda")
                if env.style_dim:
                    env.set_style(None)
                proto, _ = W.sample_inputs(env, N)
                proto = np.ascontiguousarray(proto, np.float32)
                frozen = np.ascontiguousarray(np.load(states_dir / f"{cfg}.npz", allow_pickle=False)["states"], np.float32)
                r_p = C.check_resolution_consistency(env, proto)
                r_f = C.check_resolution_consistency(env, frozen)
                r_8 = C.check_resolution_consistency(env, frozen[:N])
                if "skipped" in r_p or "skipped" in r_f:
                    row.update(status="not_applicable", why=r_p.get("skipped") or r_f.get("skipped"))
                else:
                    f = table[cfg] - margin
                    row.update(status="measured", protocol=r_p, frozen_corpus=r_f, frozen_first_n=r_8,
                               protocol_states_sha256=C.frames_digest(proto),
                               protocol_pass=bool(r_p["exact_frac"] >= f), frozen_pass=bool(r_f["exact_frac"] >= f),
                               protocol_minus_bound=r_p["exact_frac"] - table[cfg],
                               frozen_minus_bound=r_f["exact_frac"] - table[cfg])
            except ValueError as e:
                row.update(status="not_applicable", why=f"ValueError: {str(e)[:200]}")
            except (ImportError, ModuleNotFoundError, MemoryError, torch.cuda.OutOfMemoryError) as e:
                row.update(status="blocked_environment", why=f"{type(e).__name__}: {str(e)[:200]}")
            except BaseException as e:                                      # noqa: BLE001
                row.update(status="error", why=f"{type(e).__name__}: {str(e)[:300]}", traceback=traceback.format_exc()[-1200:])
        row["seconds"] = round(time.perf_counter() - t0, 1)
        doc["configs"][cfg] = row
        print(f"[{cfg:28s}] {row['status']:20s} " + (f"protocol={row['protocol']['exact_frac']:.6f} frozen={row['frozen_corpus']['exact_frac']:.6f} bound={row['bound']:.6f}" if row["status"] == "measured" else str(row.get("why", ""))[:100]), flush=True)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(doc, indent=1, sort_keys=True))
    print("wrote", out)


def merge(files: list[str], out: Path) -> None:
    table, margin, on = bounds()
    rows, prov = {}, {}
    for f in files:
        d = json.loads(Path(f).read_text())
        for cfg, r in d["configs"].items():
            if r["status"] == "measured" and rows.get(cfg, {}).get("status") != "measured":
                rows[cfg] = dict(r, venv=Path(d["venv"]).name, gpu=d.get("gpu"), job=d.get("slurm_job_id"))
            elif cfg not in rows:
                rows[cfg] = dict(r, venv=Path(d["venv"]).name)
        prov[Path(d["venv"]).name] = {"gpu": d.get("gpu"), "host": d.get("host"), "job": d.get("slurm_job_id")}
    lines = ["| config | recorded bound | floor (bound - margin) | protocol (8 states, STATE_SEED) | frozen corpus (64) | corpus first 8 | protocol - bound | corpus - bound | verdict on protocol | verdict on corpus | venv |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for cfg in bounded_configs():
        r = rows.get(cfg, {"status": "not run"})
        if r["status"] != "measured":
            lines.append(f"| {cfg} | {table.get(cfg):.6f} | | {r['status']}: {str(r.get('why',''))[:80]} | | | | | | | {r.get('venv','')} |")
            continue
        lines.append(f"| {cfg} | {r['bound']:.6f} | {r['bound']-margin:.6f} | {r['protocol']['exact_frac']:.6f} | {r['frozen_corpus']['exact_frac']:.6f} | {r['frozen_first_n']['exact_frac']:.6f} | {r['protocol_minus_bound']:+.6f} | {r['frozen_minus_bound']:+.6f} | {'pass' if r['protocol_pass'] else 'FAIL'} | {'pass' if r['frozen_pass'] else 'FAIL'} | {r['venv']} |")
    head = [f"# check_resolution_consistency on the bound's protocol vs the frozen corpus, all bounded configs", "",
            f"Bound: `tests/gpu/test_reproducibility.py::DOWNSAMPLE_EXACT_FRAC`, margin {margin}. Protocol: `{on}`.",
            f"Interpreters: {json.dumps(prov)}", ""]
    out.write_text("\n".join(head + lines) + "\n")
    (out.with_suffix(".json")).write_text(json.dumps({"margin": margin, "protocol": on, "interpreters": prov, "configs": rows}, indent=1, sort_keys=True))
    print("\n".join(lines)); print("wrote", out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--merge", nargs="*", default=None)
    a = ap.parse_args()
    if a.merge is not None:
        merge(a.merge, Path(a.out))
    else:
        measure(Path(a.out))


if __name__ == "__main__":
    main()
