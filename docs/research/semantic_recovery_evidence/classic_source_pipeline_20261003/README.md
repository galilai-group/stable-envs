# Direct-source classic-control online comparison

This benchmark measures all five accepted classic registrations against unchanged
SWM wrappers from commit `176d54532b3c0a251a4f509b74f2fdacc143a63d`. The seven
source files are extracted directly from that Git object, hashed, and validated
before importing them. Namespace loading bypasses unrelated policy/data imports;
it does not replace any source environment, reset, physics or rendering method.
Runtime Gymnasium sources and exact package versions are also recorded.

The timed source path calls the scalar wrappers' `step` and `render` directly.
It never calls the correctness reference's `step` or `get_state` in timing.
Initial restoration, final complete-State comparison, task-output equality,
image error calculation, startup and warmup are outside timing. Source and ours
materialize every task output and selected frame on the requested destination.
Ours uses `states=None` but still returns its mandatory complete final State;
the upstream API does not provide that output. This asymmetry favors source and
is explicit. Historical all-State-retaining adapter sweeps are different workloads.

For each registration and output device, the planned sweep is B1/64/256, T1/8,
frames none/final/all, six samples in reversed source/candidate order. The
candidate uses ordinary public construction and its current default renderer.
Only Acrobot selects processes at eligible large batches. Independent component
measurements cover candidate State export, candidate saved-State rendering, and
source rendering at the same final State. Component times are not additive.

Source Pygame rendering and physics are CPU only. GPU-source FPS is unavailable.
The CUDA-output source path includes CPU-to-GPU output delivery. The CSV also
includes source CPU throughput with native CPU output, separately, to avoid
calling host rendering plus upload a GPU renderer. MSE is mean squared RGB error
on [0,1], per frame; mean/SD/max and the <=0.001 flag are retained. A measurement
PASS does not imply that the image threshold passes.

This is explicitly a **serial-source baseline**. Both paths receive the same
allocation, but source uses one simulation worker while eligible Acrobot uses
up to eight. No result establishes superiority over a parallelized source
implementation. That additional baseline remains required before a full
parallel-source claim. Rates are transitions/s and emitted images/s; with T8
and final-only frames they differ by eight. B256/T8/all checks 2,048 images per
device. B1024 at native 500x500 would require over 6 GB for each RGB history
alone; this first sweep does not claim that larger-batch memory acceptance.

Local CPU checks at B2 pass independently for all five registrations: six
State/task checks, twelve timing cells and three component cells each, with
zero source image MSE for all checked frames. Four samples are retained in
these local smoke reports; the cluster gate requires six. Local results do not
establish CUDA or large-batch acceptance. The isolated source bundle is also
smoke-tested without relying on the developer checkout.

The launcher uses an RTX 3090, eight physical CPUs, 32 GiB RAM and five array
tasks (maximum two concurrent). Source/candidate manifests, overrides and local
reports are preserved here; mutable accepted evidence is not rerun. The strict
summarizer requires every device/batch/horizon/selection cell, complete
State/task checks, exact candidate hashes, and all component records before
emitting the comparison CSV. Same-device source baselines for other families,
parallel-source scaling, and the full 35-registration table remain open.
