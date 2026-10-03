# OGBench column export: all six independent device gates complete

Array 6964281 independently passed Maze, Cube and Scene on CPU software-GL and
CUDA/EGL. Every report has ten exact State/task/RNG/lifecycle/ownership/frame
cases and all sixteen timing cells with six samples each. The strict combined
CSV has 48 paired comparisons. MSE is zero against the previous library exporter;
this does not replace upstream image-fidelity evidence.

Actual hardware: Maze RTX 3090; Cube RTX A6000; Scene RTX A5000. Each ratio pairs
controls on the same host/device. Do not compare absolute rates across these
hosts as task-cost measurements. Physics remains CPU-based for both output devices.

Frozen source hashes resolve at c39880f92f436adda94050535e06c4ef240cdccf. The
exporter itself was introduced in 6ce4e82931a629b68218bbe5331296bdc0a946d2.
Run summarize_ogbench_state_export.py results NEW_OUTPUT --source FROZEN_CHECKOUT.
The source-provenance file records all verified paths. Earlier committed Maze
and Cube evidence is preserved; this folder consolidates it with terminal Scene
evidence. Process/default/pipeline/source-baseline work remains separate.
