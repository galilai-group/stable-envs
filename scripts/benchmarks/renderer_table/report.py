"""Render the complete table and its measurement interpretation from verified results."""

import argparse
import json
from pathlib import Path

from .verify import validate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    summary = json.loads((args.results / "summary.json").read_text())
    validate(summary)
    rows = summary["registrations"]
    nframes = sum(r["matched_frames"] for r in rows)
    text = f"""# Randall’s matched-State renderer table

Measured on Brown Oscar, 27 September 2026. This is a descriptive measurement campaign at `636bc5863af9a7cb599cce14ece57a605cc5a254` on `repair/environment-semantics`, worktree `/private/tmp/stable-envs-semantic-recovery`. The matrix remains **39 VERIFIED / 3 INCOMPLETE**. No renderer, simulator, shared contract, threshold, implementation map, or accepted evidence was changed. No push occurred.

The table has exactly 42 registration rows and {nframes} matched source/candidate trajectory checkpoints collected on CPU. `VERIFIED` is the existing semantic status; these measurements make no acceptance decision. The two PushT rows have no legitimate matched-State corpus. The two symbolic Craftax registrations have no RGB public renderer. All other rows retain actual paired image measurements. GPU quality uses only checkpoints whose numeric State matches the CPU oracle; its exact count is retained in each metric aggregate.

Latency columns are **median warm milliseconds, batch size 1**. MSE columns are **mean / p95 normalized RGB MSE across matched frames**. CPU and GPU MSE stay separate because some renderers produce different frames. Exact CPU/GPU image equality is recorded per row where the whole corpus supports it. `UNSUPPORTED` means the public backend explicitly rejects rendering; `NONCOMPARABLE` means the required State/configuration comparison is unavailable; `NOT IMPLEMENTED` identifies the symbolic registrations’ explicit no-RGB API. None is a synthetic zero.

## Complete table

"""
    text += (args.results / "table.md").read_text()
    text += """
## How the measurements were made

Every numeric value comes from retained Oscar measurements. Local Mac execution was limited to editing, static inspection, artifact verification and report assembly. Candidate and original measurements within every measured row used the same Oscar host; isolated subprocesses prevented GPU EGL settings from leaking into the source CPU process. Each record contains the actual host, CPU/GPU model, package versions and paths, thread settings and environment. The campaign manifest binds those records to SLURM terminal evidence, NVIDIA driver output, source archives, job recipes and harness overlays.

CPU MuJoCo rendering used Xvfb/GLX with `LIBGL_ALWAYS_SOFTWARE=1`, `GALLIUM_DRIVER=llvmpipe` and a retained live `GL_RENDERER` proof. This is software OpenGL on the CPU. Rocket’s original uses PyBullet TinyRenderer. Native 2-D originals use their software rasterizers. Craftax originals run JAX with the CPU backend: the actual pixel-wrapper `render()` converts its cached observation to RGB. Consequently Craftax’s original render latency does **not** include the upstream observation-generation work performed during reset/step. Our public renderer recomputes pixels from State. Both are the requested actual public calls; the latency difference should be interpreted with that cache boundary in mind.

At each checkpoint the candidate and pinned authoritative source received the same seed, explicit options and identical valid action prefix. The harness either compared every exposed numeric State leaf exactly or invoked the already accepted independent family witness. The latter covers Fetch controller/cache/lifecycle/RNG fields, Cube/Scene’s engine/model/entropy ownership, Craftax engine/PRNG leaves, and Rocket’s continuation capture. Per-frame records retain the witness, State hash, action dtype and prefix hash, exact render kwargs, camera, shape and RGB hashes. Saved numeric States and full action arrays allow every prefix to be checked independently. Source rendering and repeated candidate rendering must preserve supplied/live State, continuation caches, lifecycle and RNG. Replay and State capture are outside timing.

Frames use seeds 0, 17 and 42, early/middle/late or terminal checkpoints, and registered appearance cases. Native continuous tasks, Fetch trajectories, OGBench and Rocket cover float32 and float64 commands; discrete tasks use integer commands. DMC uses its accepted canonical float64 action boundary. The sampling recipes give the complete checkpoint lists and variations. Cube/Scene additionally replay accepted contact controllers (cube, button, drawer, window), camera/light/size/color and lock cases; two reset applications reproduce the source’s appearance lifecycle. Craftax trajectories stop at terminal States. No image is cropped, aligned, recolored, shifted or rescaled by the harness. Fetch’s authoritative wrapper is configured with both its output resolution and underlying raster dimensions at 224×224, so its built-in resize has identical input/output dimensions.

For every supported registration/backend combination there are at least 10 untimed warmups and 100 recorded calls. Warm samples are pooled across that backend’s matched checkpoints. Each timed candidate call includes the public State→RenderState projection and renderer, including real public transfers. No reset, step, get_state, set_state, metrics or PNG encoding is inside the timer. GPU timing synchronizes before and after each call and requires an actual CUDA output tensor. Its primary latency retains native GPU output. The supplemental measurement times the complete same public call plus CPU materialization; it is not a transfer-only subtraction. First-call timing is recorded separately at every checkpoint, with the first checkpoint capturing initial renderer setup/compilation. It is never mixed into the warm distribution.

Metrics use float64 over the complete H×W×3 RGB output. Primary MSE is `mean((ours/255 - original/255)**2)`. Raw RGB-scale MSE, RGB MAE, mean per-pixel maximum-channel absolute error, exact-pixel fraction, within-8 fraction, SSIM and maximum absolute channel error are retained per frame. SSIM uses float64 luminance, a 7×7 uniform window with nearest boundary handling, and constants `(0.01*255)^2` and `(0.03*255)^2`. Fractions require all three channels to satisfy the condition. Each metric includes count, mean, median, p95 and maximum; timings additionally retain every sample, population standard deviation and minimum. Raw RGB arrays were independently rehashed and all metrics recomputed on a CPU compute node before compact export.

## Scope limits and fidelity findings

**PickAndPlace:** all three registrations have independent reset-only measurements: three seeds × six default/object/table-color configurations. Their source renderer calls `mj_forward` after action steps and changes continuation-relevant gripper-position/quaternion caches. Independent float32/float64 probes cover all three registrations, three seeds, zero/random actions and six checkpoints. Dynamic frames therefore fail the required render nonmutation rule and are excluded; no cache repair or State restoration is hidden around rendering. The reset-only measurements have empty action prefixes and do not claim post-action dtype coverage.

**Craftax:** the public State signature includes CPU versus GPU platform. CPU States cannot be transplanted into a GPU environment. The harness instead independently replays the recorded public reset/actions on GPU and compares every numeric leaf except the platform signature. Some checkpoints differ in numeric engine fields, such as `light_level`; these receive `NONCOMPARABLE` records and no GPU image or timing. Only exact semantic matches are rendered/timed. Both full States, all leaf hashes, differing fields and magnitudes are retained. No signature or engine field is edited, and no numerical tolerance admits a mismatch. CPU quality covers the entire CPU corpus; GPU quality covers its explicitly counted subset. Both symbolic registrations independently exercise their source CPU, candidate CPU and candidate GPU no-RGB paths.

**PushT:** current continuous and discrete implementations were independently probed under the source-aligned pymunk 7.2.0 profile. Reset poses differ for seeds 0/17/42 before any action; the current absolute-action implementation also does not match SWM’s relative-action interface, and the discrete quantizer/replay witness is absent. Neither row claims matched frames or renderer latency. No pose teleport or new replay implementation was introduced.

**Appearance fidelity:** semantic State equality does not imply identical renderer appearance. OGBMaze’s historical registered source defaults to state observations while the current renderer was calibrated for pixel appearance; its large image discrepancy remains visible. Fetch uses identical explicit supported reset options and its actual public style defaults/overrides, without sampling unmatched source-only visual knobs. Existing source material-unbinding behavior and renderer approximations remain part of the measured error. These are findings, not repairs or changes to accepted thresholds.

**Rocket:** all 204 canonical checkpoints are included, covering both action dtypes, three seeds, fall/mixed/success/horizon cases, reset/appearance cases and the high-ceiling horizon-600 failure. Its `INCOMPLETE` status is unchanged. The earlier frozen 204-frame artifact was independently rehashed and recomputed separately; it was not overwritten or used to substitute timings.
"""
    rocket = next(r for r in rows if r["name"] == "PyFlytRocketLanding")
    witness = next(f for f in rocket["frames"] if f["index"] == 19)
    q = witness["gpu"]["gpu_quality"]
    text += f"\nThe fresh seed-0 float32 horizon-600 witness (frame 19) has normalized RGB MSE **{q['mse_normalized']:.10g}**, raw RGB MSE **{q['mse_uint8']:.10g}**, and mean per-pixel maximum-channel absolute difference **{q['mean_max_channel_abs']:.10g}**. The last value is the retained acceptance metric; it is not MSE. Its pair and unamplified absolute-difference image are committed.\n"
    text += "\n## Largest image errors\n\n| Registration | Backend | Frames | MSE mean | MSE p95 | MSE maximum |\n|---|---|---:|---:|---:|---:|\n"
    errors = [
        (q["mse_normalized"]["mean"], r, b, q["mse_normalized"])
        for r in rows
        for b, q in r["quality"].items()
        if "status" not in q
    ]
    for _, r, b, q in sorted(errors, key=lambda x: x[0], reverse=True)[:10]:
        text += (
            f"| {r['registration']} | {b} | {q['count']} | {q['mean']:.7g} | {q['p95']:.7g} | {q['maximum']:.7g} |\n"
        )
    text += "\n## Major latency outliers\n\nThese are absolute B=1 public-call costs on each row’s recorded hardware, not a cross-hardware speed ranking.\n\n| Registration | Backend | Median ms | p95 ms | Calls |\n|---|---|---:|---:|---:|\n"
    costs = [
        (c["statistics_ms"]["median"], r, b, c)
        for r in rows
        for b, c in r["timings"].items()
        if c["status"] == "MEASURED"
    ]
    for _, r, b, c in sorted(costs, key=lambda x: x[0], reverse=True)[:10]:
        s = c["statistics_ms"]
        text += f"| {r['registration']} | {b} | {s['median']:.5g} | {s['p95']:.5g} | {s['count']} |\n"
    text += "\n## Detailed warm and supplemental latency\n\nCells show **median / p95 milliseconds (calls)**. Full statistics, cold/first-call samples, backend matched counts and every warm sample are in [summary.json](summary.json).\n\n| Registration | Ours CPU | Ours GPU | Original CPU | Public GPU→CPU output |\n|---|---|---|---|---|\n"

    def timing(c):
        if c["status"] != "MEASURED":
            return c["status"]
        s = c["statistics_ms"]
        return f"{s['median']:.5g} / {s['p95']:.5g} ({s['count']})"

    for r in rows:
        cells = [timing(r["timings"][b]) for b in ["cpu", "gpu", "original"]]
        cells.append(timing(r.get("supplemental_gpu_to_cpu", r["timings"]["gpu"])))
        text += "| " + r["registration"] + " | " + " | ".join(cells) + " |\n"
    text += """
## Retained witness images

These are unmodified renderer outputs and unamplified absolute RGB differences. No visualization affects scoring.

| Witness | Original | Candidate | Difference |
|---|---|---|---|
| Rocket retained horizon-600 | [RGB](../randall_renderer_measurements/swm_PFRocketLanding-v0/retained-horizon600-original.png) | [RGB](../randall_renderer_measurements/swm_PFRocketLanding-v0/retained-horizon600-candidate.png) | [difference](../randall_renderer_measurements/swm_PFRocketLanding-v0/retained-horizon600-absolute-difference.png) |
| OGBMaze worst | [RGB](../randall_renderer_measurements/swm_OGBMaze-v0/worst-original.png) | [RGB](../randall_renderer_measurements/swm_OGBMaze-v0/worst-candidate.png) | [difference](../randall_renderer_measurements/swm_OGBMaze-v0/worst-absolute-difference.png) |
| FetchPush worst | [RGB](../randall_renderer_measurements/swm_FetchPush-v3/worst-original.png) | [RGB](../randall_renderer_measurements/swm_FetchPush-v3/worst-candidate.png) | [difference](../randall_renderer_measurements/swm_FetchPush-v3/worst-absolute-difference.png) |
| Finger worst | [RGB](../randall_renderer_measurements/swm_FingerDMControl-v0/worst-original.png) | [RGB](../randall_renderer_measurements/swm_FingerDMControl-v0/worst-candidate.png) | [difference](../randall_renderer_measurements/swm_FingerDMControl-v0/worst-absolute-difference.png) |
| Scene worst | [RGB](../randall_renderer_measurements/swm_OGBScene-v0/worst-original.png) | [RGB](../randall_renderer_measurements/swm_OGBScene-v0/worst-candidate.png) | [difference](../randall_renderer_measurements/swm_OGBScene-v0/worst-absolute-difference.png) |

## Artifacts and reproduction

- [CSV: exactly 42 rows](summary.csv) and [JSON: exactly 42 registration objects](summary.json), including per-frame quality/hashes, exact raw timings, profiles and recipes.
- [Verification result](verification.json), [campaign manifest](campaign-manifest.json), [compact evidence hashes](../randall_renderer_measurements/manifest.json) and [explicit source selection](../randall_renderer_measurements/remote-selection.json).
- [Starting checkpoint](../randall_renderer_start/start.json), [frozen Rocket independent recomputation](../randall_renderer_start/frozen-rocket-independent-metrics.json), [Fetch float32 diagnostic](../randall_renderer_start/fetch-original-render-mutation-float32.json), [Fetch float64 diagnostic](../randall_renderer_start/fetch-original-render-mutation-float64.json), and [submitted recipes/immutable harness overlays](../randall_renderer_start/submission/).

From the repository root, use a CPU Python environment with NumPy, SciPy, Torch, TensorDict and Pillow:

```bash
python -m scripts.benchmarks.renderer_table.verify \\
  docs/research/semantic_recovery_evidence/randall_renderer_results/summary.json \\
  --evidence docs/research/semantic_recovery_evidence/randall_renderer_measurements \\
  --output /tmp/randall-verification.json
python -m pytest scripts/benchmarks/renderer_table/test_verify.py -q
```

The verifier rejects missing/duplicate registrations, incorrect State/frame/camera/resolution pairing, missing GPU synchronization, GPU-backed CPU labels, missing definitions/counts, invalid metric aggregation, nonfinite numbers and unexplained unavailable cells. It verifies retained State/action-prefix hashes and the independently exported artifact hashes. Strict Craftax replay checks reject any semantic leaf difference in an admitted GPU frame. Production/matrix/profile hashes are checked against the starting checkpoint.

To reproduce measurements, use the selected job recipes in the campaign manifest: reconstruct the exact starting Git tree plus its recorded harness overlay, use the pinned SWM oracle archive and specified isolated profile, then run on Oscar. Every attempt has a distinct output directory; failed or superseded attempts remain identified and are excluded from the numeric table. Full raw RGB NPY files remain in the hashed Oscar scratch paths. Compact committed evidence contains numeric States/actions, metrics/timings, a small set of representative worst-case pairs and unamplified differences. The containing Git commit is discoverable with `git log -1 --format=%H -- <this report>`; no self-referential commit hash is embedded in its own bytes.
"""
    (args.results / "report.md").write_text(text)


if __name__ == "__main__":
    main()
