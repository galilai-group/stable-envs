# Cube/Scene process harness retry

Array 6966296 failed for both registrations after each independently passed its
four cluster process tests. The CPU gate selected backend=auto, which routes
Cube/Scene to a mesh renderer requiring CUDA. It failed before comparing frames;
no mismatch, throughput or GPU acceptance is inferred. Failed JSON/logs and both
passing JUnit reports are preserved in failed-v1/.

The only source change in V2 is an explicit backend="opengl" argument for both
controls. This is the same supported CPU/software-GL and GPU/EGL renderer used
by the accepted exporter gates. All production source and unit tests are byte-
identical to V1. The already-passing process tests are not repeated; their
source hashes and terminal reports remain available.

Fresh array 6966557 uses ogbench-manipulation-process-v2-20261003 and new result
paths. Each family independently reruns its CPU/GPU public frame gates and
local/P4/P8 timing cells. This is not a retry of any live job: both V1 tasks were
confirmed FAILED first. v2/ preserves the frozen 175-file manifest, launcher and
one coordinator override needed to reconstruct it from this checkpoint.

This does not change the default CPU renderer or introduce a new backend API.
The selected OpenGL backend must be carried forward when interpreting these
gates and eventual pipeline measurements. Default process selection remains local.

The CPU auto-backend API limitation is now explicitly tracked in the execution
backlog. Correcting this harness does not fix that default route; it requires
its own production change and independent CPU/GPU renderer regression gates.
