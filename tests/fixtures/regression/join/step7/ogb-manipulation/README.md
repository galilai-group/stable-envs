# Layout of this directory, so the two copies of some measurements are not read as duplication

| directory | what is in it |
|---|---|
| `census/` | the phase-0 model census, taken on the **M5 laptop** under GLFW |
| `probe/` | the phase-0 purity, determinism, bars, plate, interior and footprint probes, **M5 laptop** |
| `oscar/` | the SAME phase-0 probes re-run on **Oscar** under EGL. `oscar/census` and `census/` differ, and that is the point: the Linux/EGL run is the certification-host measurement and the laptop run is what it is checked against. Neither is redundant. |
| `m1/`, `m1b/` | the attribution passes (which term the MJWarp-vs-MuJoCo gap is) |
| `m2/` | the shared-helper proofs: registry sweeps, byte-identity replays, the coverage limit |
| `m3/` | the cells — design gate, fidelity measurement (`m3/fidelity`), corpus and Gate F |
| `m4/` | the tier measurement |
| `m5/` | the Q11 memo for Sami and the button-bake residual |
| `states/` | the FROZEN corpora in Step-4 schema, and the pool each was selected from |

SLURM logs live beside the artifact they produced, gzipped, one copy each.
