# Step 6 findings (pre-implementation)

Recorded during profiling, before any renderer edit. Nothing here has been fixed: section 0 gates
implementation on the completing Step 5 commit, and a finding is worth more dated to before the
work than after it.

## 1. The Scene2D CPU path is unreachable on any host that has Triton

`stable_envs/render/raster2d.py:670`

```python
if backend == "torch" or (backend == "auto" and not HAVE_TRITON):
    ...torch path...
return render_triton(scene, out_size, supersample, out=out)
```

The `auto` dispatch keys off `HAVE_TRITON`, which is a property of the HOST, and never consults the
device the scene lives on. On a machine where Triton imports, a `device="cpu"` scene is routed into
`render_triton`, whose first act is `with torch.cuda.device(scene.kind.device)` (line 646) —
raising `ValueError: Expected a cuda device, but got: cpu`.

- **affected**: the two configurations declaring the `scene` backend — `PushT`, `SimplePointMaze`.
- **not affected**: `TwoRoom`, `TwoRoomTarget`, `Piecewise`, `PiecewiseZones` (swm_nav torch
  arithmetic, no Scene2D), and everything on a host without Triton.
- **evidence**: job 6318867 on `node2305` (batch partition, no GPU, Triton present in the venv);
  both envs fail identically. The same envs render on CPU on the laptop, which has no Triton.
- **why it was invisible until now**: Step 3 measured every 1-thread CPU row for these two envs on
  the laptop. The CPU tier they carry could not have been produced on the certification cluster.
- **contract it breaks**: section 19 CPU — "`device="cpu"` is actually honored". It fails loudly
  rather than silently running the GPU renderer, which is the better of the two failure modes, but
  it is still no CPU path.
- **shape of the fix** (NOT applied): the dispatch predicate needs the scene's device, not the
  host's capability. Owner: `step6/raster2d`.

## 2. `gfxdraw` and `raster2d` share one root cause across two workstreams

Sections 10.4 and 10.5 assign these to separate agents. The profiles say the bottleneck is the
same: **each primitive is evaluated over the whole frame**.

| env | frame | ops/render | full-frame `where` | per-frame cost |
|---|---|---|---|---|
| Acrobot (gfxdraw) | 500px | 27,073 | 1,396 | 4.3 ms @B=64 |
| SimplePointMaze (scene) | 500px | 10,247 | 182 | 104 ms @B=64 |
| PushT (scene) | 96px | 6,682 | 126 | 2.7 ms @B=64 |

SimplePointMaze against PushT is the cleanest evidence: `aten::where` costs 10.4 ms/call against
0.39 ms/call, a ratio of 26.7 where the pixel-count ratio is 27.1. Per-call cost is proportional to
frame area and nothing else — these are B x H x W temporaries per primitive.

The two differ in mix, and a fix should be sized accordingly: `gfxdraw` is roughly half Python
dispatch at B=1 (27k operator calls, ~4.9 ms, matching ~0.18 us/op) and half whole-frame compute;
`raster2d`/SimplePointMaze is ~97% compute. A bounded/tiled primitive path addresses both; removing
dispatch alone would leave SimplePointMaze essentially where it is.

SimplePointMaze also renders a **static maze** — `state_dim` 2, four distinct colours in the frozen
frame — from scratch on every call, which is section 7's "failure to pre-bake static geometry".

## 3. A hypothesis that profiling killed

The first smoke test (PushT, laptop, B=1) showed 6,677 operator calls and suggested a
dispatch-bound path. The batch sweep refuted it: PushT's per-frame cost falls 4.96 -> 2.7 ms from
B=1 to B=64, so fixed dispatch is ~2.3 ms against ~2.7 ms of real work — worth removing, not the
headline. Recorded because section 8 exists to make exactly this happen before a patch rather than
after one.
