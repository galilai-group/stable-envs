# Quadruped renderer-history diagnostic

Job 6967610 runs a diagnostic on gpu2109, the original failing RTX 3090 host.
The frozen exporter gate and its exact assertions are unchanged. The driver
wraps the rendering boundary to record compact State/style/pristine-input hashes,
MuJoCo scene geometry/cameras/lights, kinematics and images. It groups repeated
byte-identical renderer inputs and reports which fields differ when pixels do.
Snapshots own their arrays; a local smoke verified this and camera-field change
detection. No rendering flags or production code are modified.

The original failure and its first diagnostic remain untouched in
`../quadruped_export_diagnostic_20261003/`. This run writes to a fresh scratch
directory. A reproduced assertion should leave the job FAILED with diagnostic
artifacts; that is evidence to inspect, not acceptance. Instrumentation could
affect reproducibility, and non-reproduction alone cannot close the original gate.
Quadruped automatic process selection remains disabled.

Remote stage: `/oscar/scratch/sboughan/se-semantic-recovery/quadruped-render-history-v1-20261003`.

V1 job 6967610 failed before any captured frame: the diagnostic used `.items()`
on StyleTensor, which exposes named-path indexing instead. Its raw reports and
terminal status are preserved under `failed-v1/`; this says nothing about
production correctness. The original submitted source remains unchanged here.

V2 job 6967633 uses only the style paths consumed by the renderer. Its local
capture smoke uses an actual StyleTensor and MuJoCo scene/data, with stub image
output: three identical-input renders and two injected pixel differences are
correctly classified without scene or kinematic differences. This verifies
instrumentation, not GL output. The corrected driver, hashes, launcher and smoke
record are in `v2/`. It runs on the same gpu2109 using gpu/norm-gpu and writes to
`/oscar/scratch/sboughan/se-semantic-recovery/quadruped-render-history-v2-20261003`.
