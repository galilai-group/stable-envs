"""Validate independent automatic-renderer gates and summarize resident-State FPS."""

import argparse
import csv
import hashlib
import itertools
import json
import math
import statistics
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--source", type=Path, required=True, help="Frozen stage's code directory")
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = json.loads((args.source.parent / "source_manifest.json").read_text())
    for rel, digest in manifest["files"].items():
        assert hashlib.sha256((args.source / rel).read_bytes()).hexdigest() == digest, rel
    reference = args.source.parent / "reference/manipulation_renderer.py"
    assert hashlib.sha256(reference.read_bytes()).hexdigest() == manifest["reference_renderer_sha256"]
    rows = []
    for family, device in itertools.product(("Cube", "Scene"), ("cpu", "cuda")):
        path = args.results / family / f"{device}.json"
        r = json.loads(path.read_text())
        assert r["status"] == "PASS" and r["domain"] == family and r["device"] == device, path
        assert r["source_sha256"] == manifest["files"], path
        assert r["reference_sha256"] == manifest["reference_renderer_sha256"], path
        assert r["owned_frames"] == "PASS", path
        assert r["opengl"]["software"] == (device == "cpu"), path
        if device == "cuda":
            assert "nvidia" in r["opengl"]["vendor"].lower() and "3090" in r["gpu"], path
        native_size = 224 if family == "Cube" else 200
        expected = {("render", varied, size, compact) for varied, size, compact in
                    itertools.product((False, True), (64, native_size), (False, True))}
        selections = ["default", "None", "slice(None, None, None)", "slice(None, None, -2)", "slice(9, 12, None)"]
        expected |= {("step", varied, horizon, selection) for varied, horizon, selection in
                     itertools.product((False, True), (1, 8), selections)}
        cases = set()
        for c in r["cases"]:
            assert c["mse"] == 0, path
            key = (("render", c["varied"], c["size"], c["compact"]) if c["kind"] == "render"
                   else ("step", c["varied"], c["horizon"], c["selection"]))
            assert key not in cases, path
            cases.add(key)
        assert cases == expected and len(cases) == 28, path
        assert len(r["timings"]) == 2 and {c["mode"] for c in r["timings"]} == {"reference", "auto"}, path
        for c in r["timings"]:
            assert c["batch"] == 2 and c["size"] == native_size, path
            assert len(c["seconds"]) == 6 and all(math.isfinite(t) and t > 0 for t in c["seconds"]), path
            rates = [2 / t for t in c["seconds"]]
            assert math.isclose(c["fps_mean"], statistics.mean(rates))
            assert math.isclose(c["fps_sd"], statistics.stdev(rates))
            rows.append(dict(env="swm/OGB" + family + "-v0", device=device, mode=c["mode"], batch=2,
                             size=native_size, batch_render_ms_mean=1000 * statistics.mean(c["seconds"]),
                             batch_render_ms_sd=1000 * statistics.stdev(c["seconds"]),
                             resident_state_render_fps_mean=c["fps_mean"],
                             resident_state_render_fps_sd=c["fps_sd"], frame_mse_vs_frozen_renderer=0,
                             cases=28, host=r["host"], gpu=r["gpu"], job=r["job"], report=str(path)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print("Validated four independent gates (112 cases); wrote eight resident-State timing rows")


if __name__ == "__main__":
    main()
