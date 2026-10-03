# Sustained CPU-output follow-up

Array 6971014 uses the same measurement driver, frozen adapters and production
candidate as the completed sustained CUDA diagnostic. Only the summarizer adds
explicit device/batch arguments. Each task requests one RTX 3090 host, eight
physical CPUs, 32 GiB and two hours; actual rendering/output is CPU. Four cells
use B256, T1/8, six windows of 128 consecutive transitions per lane, normal GC,
independent strategy/cell processes and exact selected-output trajectory hashes.
The batch is bounded because CPU rendering dominates these longer windows.
This does not establish sustained CPU performance at B1024.

Reconstruct the sibling sustained-diagnostic source, replace its summarizer
with this copy, and verify all source_manifest.json hashes. Run run.sh.
The candidate and measurement-driver hashes are unchanged. No production
activation or CPU sustained performance result is claimed while pending.
