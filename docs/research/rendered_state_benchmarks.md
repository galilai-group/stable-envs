# Rendered end-to-end State benchmark extension

This research extension preserves the earlier measurements and their immutable
report. It adds a sixteen-step composed rollout at B1 with four render cadences:
every step, every fourth step, final frame only, and no rendering. The four
workloads are ordinary step/render, step/get/render, restore then step/render,
and snapshot/eight divergent steps/restore/rendered continuation. Each uses the
same starting State and prescribed actions. Preparation is outside the timer;
complete branch work is inside. Ten warmups and fifty timed samples retain
materialized outputs and synchronize GPU/JAX work.

`rendered_state_ops.py --derive <prior-summary.json> --output <fresh-directory>`
first derives the requested ratios for all 37 renderable registrations. Two
limits are explicit: the earlier non-Craftax CPU dynamics ran on Apple M5 while
GPU rendering ran on Oscar, so adding those medians is cross-host arithmetic,
not a measured latency prediction. Also, `render(full State)` already includes
projection. The literal requested sum and the version without double-counting
projection are both recorded. No old artifact is rewritten.

The composed collector measures components again on the **same host/profile**.
Where public projection exists, it times `to_render_state` followed by
`render(RenderState)` exactly once. Otherwise public `render(State)` includes its
private projection and separate projection cost is zero, explicitly classified
as included rather than an absent computational cost. The main additive estimate
uses these matched, nonoverlapping components. Both extra-get and extra-set
ratios divide by step + projection + render; their one-frame slowdown factors
are one plus those ratios. Rollout predictions scale step/get by the horizon,
render/projection by actual cadence, and set by the number of restores.

The collector separately compares a step after restore with the same action at
a fresh reset/prefix checkpoint, and project/render immediately after restore
with project/render on the warm fixed snapshot. These measurements reveal
restoration-related initialization and invalidation costs. Ratios and sample
variance are descriptive; no semantic acceptance gate or performance threshold
is introduced. Full rollout measurements are compared with sums of component
medians, which need not equal a median of a composed call.

CPU Craftax and all available GPU renderer registrations use the accepted
family dependency profiles. Scripts refuse existing output directories and
retain failed runs. The profile dispatcher uses a clean base PYTHONPATH before
launching each family's interpreter. Raw per-case JSON is consolidated in the
final report rather than duplicating previous native archives.
