# DMC direct-source transport prerequisite

All twelve DMC registrations independently pass four local CPU B2/P2 cases:
T1/T8 from seeded reset and from an explicit internal reset boundary. The total
is 48 exact comparisons between unmodified pinned source wrappers, their
persistent shared-memory worker execution and the accepted local adapters.
This is neither a public Env/render gate nor performance evidence.

The source is [stable-worldmodel 176d54532b3c0a251a4f509b74f2fdacc143a63d](https://github.com/galilai-group/stable-worldmodel/tree/176d54532b3c0a251a4f509b74f2fdacc143a63d).
`source_manifest.json` hashes registration, spaces/utilities, the common DMC
wrapper, all twelve concrete classes and four custom-task implementations.
Namespaces skip unrelated package initializers without modifying source logic.
The existing independent reference classes handle untimed complete-State
snapshot/restoration. Timed-source infrastructure calls actual wrapper.step;
it never inserts reference.get_state into source stepping.

| Source registrations | Concrete source wrappers / State oracle |
| --- | --- |
| Acrobot, Pendulum, Cartpole DMControl-v0 | Corresponding *DMControlWrapper classes / DMCFirstFamilyReference |
| ReacherDMControl-v0 | ReacherDMControlWrapper / ReacherReference |
| Cheetah, Hopper, Walker DMControl-v0 | Corresponding *DMControlWrapper classes / DMCContinuationReference |
| BallInCup, Finger, Manipulator, Quadruped DMControl-v0 | Corresponding *DMControlWrapper classes / DMCAlignedReference |
| HumanoidDMControl-v0 | HumanoidDMControlWrapper / DMCHumanoidReference |

All IDs carry the swm/ prefix. The registrations directly select these classes;
none adds a registered TimeLimit. The shared wrapper repeats each accepted
action twice through the source task/action-scale wrapper, aggregates reward,
updates cumulative score, and constructs task observations/info. Dynamics are
scalar CPU MuJoCo; raw mj_step omits this logic and is not a valid substitute.
Public terminated/truncated stay false for these pinned defaults, including
internal dm_control reset-next-step boundaries. The comparison retains that
behavior and the internal counters, MT19937 state and cached Gaussian value.
Target model fields and activation state are restored where required.

Source render(width, height, camera_id) delegates to dm_control physics.render.
Its default dimensions are 224x224; the planned comparison explicitly requests
64x64 on both sides to match the completed online DMC cohort. The source default
camera is 0, except Quadruped's camera 2. Source physics has no native GPU/batch
API, but rendering can run on hardware OpenGL. Thus source GPU is a real required
baseline, unlike Navigation/pygame. Software GL and NVIDIA GL must run in separate
interpreters and their actual context identities must be checked, including
source workers. The helper queries dm_control's own context on its dedicated
render executor. That GL helper is **not validated by these CPU-only results**.

The benchmark-only source shard reuses the existing Gymnasium AsyncVectorEnv
transport: spawn, shared_memory=True, copy=True, disabled autoreset. Each worker
owns its source worlds. Numeric task outputs/images use shared buffers; parent
outputs own their storage before the next worker call. The first four accepted
adapters expose score without the source's extra numeric info fields. Their
benchmark projection matches that existing API; upstream still computes its
ordinary complete info. Remaining adapters retain all accepted numeric info,
including NaN success fields with matching NaN positions. No production field
is added, removed or changed.

The gate checks full endpoint State and every adapter task leaf exactly, source
serial versus source workers exactly, and output ownership after a different
subsequent action. Boundary cases use step_count at the source limit and
reset_next_step, with has_gauss=True and cached_gaussian=0.125. Local versions:
Torch 2.14.0, NumPy 2.5.3, dm-control 1.0.45, MuJoCo 3.12.0, Gymnasium 1.3.0,
TensorDict 0.14.2. No timing or rendering is claimed from these tests.

A preliminary attempt to use public Env construction encountered the Mac's
CoreGraphics initialization failure because DMC construction itself creates a
renderer. The final prerequisite deliberately constructs only adapters and
source worlds. It does not stand in for the public CPU/CUDA acceptance tests.
The initial failed report remains at the local scratch path; the committed
reports here are the final adapter-only runs. No shared renderer workaround or
production change was introduced.

Reproduce from base 739c4470 plus the two committed benchmark scripts; verify
candidate_manifest.json and the source manifest. From the checkout, run:

```sh
PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python scripts/benchmarks/online_stepping/dmc_source_transport_gate.py \
  --source-root /path/to/pinned/stable-worldmodel \
  --source-manifest docs/research/semantic_recovery_evidence/dmc_source_transport_20261003/source_manifest.json \
  --output /new/path/dmc-source-transport.json
```

Next: integrate these source workers into a same-host public online benchmark,
verify CPU/software-GL and CUDA/NVIDIA-GL independently, gate matched-state images,
and measure serial source, parallel source and the ordinary library route with
six samples per batch/horizon/selection. Retain source readback and required
output transfer in timing. No source speedup is established yet. Quadruped's
independent renderer-history issue still prevents promotion of its process
default; passing this transport prerequisite does not close that gate.
