# Cube exporter acceptance closure

Array task `6964281_1` completed successfully. Cube independently passes
CPU/software-GL and CUDA/EGL exact-export tests and ten public State/task/frame
cases per device (MSE 0 versus the old library exporter), plus all sixteen
paired B1/16, T1/8 timing cells per device with six samples each.

Production/reference/benchmark hashes match commit
`c39880f92f436adda94050535e06c4ef240cdccf`; the exporter was introduced in
`6ce4e829`. The later shared masking candidate is not part of this measurement.
Source manifest and launcher remain under
`../ogbench_online_closure_20261002/column-v3/`. CPU proof uses Mesa llvmpipe;
GPU proof uses NVIDIA RTX A6000. This is not labeled RTX 3090 evidence.

Both reports are PASS and all expected cells/gates/hashes were checked,
independently of scheduler success. Logs retain harmless GLFW/EGL destructor
warnings at interpreter shutdown. Maze's previous acceptance is preserved
separately; Scene is still running. No new upstream fidelity claim or process
default follows from this exporter gate.
