# Classic-control exporter closure

All five tasks in array 6964284 completed on RTX 3090 hosts. Each registration
independently passes ten exact public State/task/RNG/lifecycle/ownership/frame
cases on CPU and CUDA; all 48 timing cells per registration contain six samples.
The strict summarizer verifies all ten gates and writes 120 paired comparison
rows. Source hashes resolve at c39880f92f436adda94050535e06c4ef240cdccf.

Run summarize_classic_state_export.py results NEW_OUTPUT --source FROZEN_CHECKOUT.
Each rate compares the previous library exporter with column export, paired on
the same host. CPU physics is used in both device configurations. CUDA-output
speedup is largely reduced field transfer/assembly overhead, not GPU simulation.
At B256/T8 with all State history and frames=None, ratios range 2.80–6.40x for
CPU output and 5.48–13.39x for CUDA output. The CSV retains individual means/SDs,
other batch/horizon cells and hardware. Regression frame MSE is zero; this is
not a new upstream fidelity or source-throughput comparison.

No process default or completed online scaling benchmark is implied. Worker
routing, larger batch budgets, source baselines and combined render pipelines
remain separate work. Existing local evidence is unchanged.
