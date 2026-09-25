"""Finish clone ownership and fresh-process deserialization diagnostics."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cloudpickle
from clone_diagnostic import capture, identity, make


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--engine", required=True)
    p.add_argument("--case", default="free")
    p.add_argument("--method", default="copy")
    p.add_argument("--load", type=Path)
    p.add_argument("--child", action="store_true")
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    if a.load:
        result = {}
        try:
            e = cloudpickle.loads(a.load.read_bytes())
            result["identity"] = identity(e, a.engine)
            import numpy as np

            e.step(np.zeros(7))
            capture(e, a.engine)
            e.render()
            result["step_usable"] = True
        except Exception as exc:
            result.update(step_usable=False, error=repr(exc))
        print("OWNERSHIP " + json.dumps(result))
        return
    if a.child:
        source, actions = make(a.engine, a.case)
        if a.method == "copy":
            clone = copy.copy(source)
        elif a.method == "deepcopy":
            clone = copy.deepcopy(source)
        else:
            payload = cloudpickle.dumps(source)
            clone = cloudpickle.loads(payload)
        result = dict(
            engine=a.engine,
            case=a.case,
            method=a.method,
            source_identity=identity(source, a.engine),
            clone_identity=identity(clone, a.engine),
        )
        if a.method == "cloudpickle":
            folder = Path(tempfile.mkdtemp(prefix="se-clone-portability-"))
            path = folder / "candidate.pkl"
            path.write_bytes(payload)
            proc = subprocess.run(
                [sys.executable, __file__, "--engine", a.engine, "--load", str(path)],
                capture_output=True,
                text=True,
                timeout=60,
            )
            lines = [x.removeprefix("OWNERSHIP ") for x in proc.stdout.splitlines() if x.startswith("OWNERSHIP ")]
            result["fresh_process"] = (
                json.loads(lines[-1]) if lines else dict(exit_code=proc.returncode, error=proc.stderr[-1000:])
            )
            result["diagnostic_pickle"] = dict(
                path=str(path), sha256=hashlib.sha256(payload).hexdigest(), bytes=len(payload), not_public_state=True
            )
        source.close()
        try:
            clone.step(actions[0])
            capture(clone, a.engine)
            clone.render()
            result["clone_usable_after_source_close"] = True
        except Exception as exc:
            result.update(clone_usable_after_source_close=False, error=repr(exc))
        # Repeat the reverse direction with a fresh source/clone pair; post-terminal
        # Python-only step success is not proof that a native world remains usable.
        source, actions = make(a.engine, a.case)
        clone = (
            copy.copy(source)
            if a.method == "copy"
            else copy.deepcopy(source)
            if a.method == "deepcopy"
            else cloudpickle.loads(cloudpickle.dumps(source))
        )
        clone.close()
        try:
            source.step(actions[0])
            capture(source, a.engine)
            source.render()
            result["source_usable_after_clone_close"] = True
        except Exception as exc:
            result["source_usable_after_clone_close"] = False
            result["reverse_close_error"] = repr(exc)
        try:
            source.close()
        except Exception:
            pass
        print("OWNERSHIP " + json.dumps(result))
        return
    a.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for case in ["free", "first_contact", "combined", "terminal"]:
        for method in ["copy", "deepcopy", "cloudpickle"] if a.engine == "rocket" else ["copy"]:
            proc = subprocess.run(
                [sys.executable, __file__, "--engine", a.engine, "--case", case, "--method", method, "--child"],
                capture_output=True,
                text=True,
                timeout=180,
            )
            lines = [x.removeprefix("OWNERSHIP ") for x in proc.stdout.splitlines() if x.startswith("OWNERSHIP ")]
            rows.append(
                json.loads(lines[-1])
                if lines
                else dict(case=case, method=method, exit_code=proc.returncode, error=proc.stderr[-1200:])
            )
            (a.output / "result.json").write_text(
                json.dumps(
                    dict(
                        records=rows,
                        scope="Supplemental failed ownership/portability checks; no independent clone established",
                    ),
                    indent=2,
                )
                + "\n"
            )
            print(a.engine, case, method, flush=True)


if __name__ == "__main__":
    main()
