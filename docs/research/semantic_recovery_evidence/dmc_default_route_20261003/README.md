# Isolated DMC default-routing acceptance candidate

Parent 4514ff79bbca9eeeb5419dfd673d6ba2bc4e6ffd. The repository's public DMC
constructors remain unchanged and local. Only the source-hashed Oscar candidate
activates make_dmc_adapter in five constructor modules. Their exact replacements
are in oscar/source_overrides.json; reconstruct the submitted 180-file source by
applying those overrides to this checkpoint and checking every digest.

Evidence supports testing processes at Linux batches >=64 with >=4 allocated
CPUs, bounded at eight owners, with local fallback on small batches, restricted
CPU allocations, non-Linux hosts and daemon workers. Eleven registrations are
eligible. Quadruped is explicitly excluded pending its separate exporter frame
failure. No public worker controls or step API changes are introduced.

84 local routing cases pass, including the Quadruped exclusion and fallback
constructor arguments. All 84 collect from the isolated bundle. An attempted
Mac public no-frame probe failed before simulation because the constructor
bakes a GL scene and the sandbox has no CoreGraphics connection. That failed
report and exact driver are preserved; it is not counted as an acceptance pass.

RTX 3090 array 6966465 runs one task per eligible registration, eight physical
CPUs and 64 GB each, up to four concurrent tasks. It exercises the UNPATCHED
public candidate constructor at B64, verifies eight process owners were selected,
and independently runs CPU/software-GL and CUDA/EGL State/task/frame/RNG/lifecycle/
variation/ownership checks with one/eight actions and five slice selections.
The forced-local reference is constructed in its own temporary patch context;
no patch surrounds the candidate constructor. Physical source done flags retain
their historical behavior across internal episode resets.

Each gate requires exact frames and observed overlap, and records its actual
batch and process count. summarize_dmc_process.py --production requires these
provenance fields and all explicitly selected eligible domains; --timings is a
separate mode. Pass --domains acrobot pendulum cartpole reacher cheetah hopper
walker ball_in_cup finger manipulator humanoid. Do not activate any constructor
from this bundle until its required registration/device gates pass. Subsequent
component, larger-batch, source and combined pipeline benchmarks remain open.

The reconstruction overrides also retain the unrelated OGBench benchmark
from before its CPU-backend correction; this does not alter the DMC job source.
