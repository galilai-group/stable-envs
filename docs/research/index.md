# Research archive

These notes are the investigations behind the shipped environments — including families we
measured and did not ship. They are evidence, not the getting-started path. Start with the
[user guide](../index.md) unless you are looking for a measurement or a no-go.

A no-go is a result, not a gap to hide: [when does this pay off](when_does_this_pay_off.md)
states the gates before any renderer is written.

## Evergreen

Coverage and policy (what is in, what "done" means, which 3-D backend):

- [When does this pay off](when_does_this_pay_off.md)
- [Certification: what "done" means](certification.md)
- [stable_worldmodel coverage matrix](swm_matrix.md)
- [Coverage notes (why, not just the counts)](swm_coverage.md)
- [Choosing a 3-D backend](backend3d.md)
- [nvdiffrast backend](nvdiffrast.md)

Per-family studies:

- [Classic 2-D control](classic_control.md)
- [LunarLander](lunarlander.md)
- [MiniGrid and BabyAI](minigrid.md)
- [Crafter](crafter.md)
- [OGBench mazes](ogbench.md)
- [D4RL maze2d / PointMaze](maze2d.md)
- [Overcooked (JaxMARL)](overcooked.md)
- [Griddly](griddly.md)
- [Sokoban and Boxoban](sokoban.md)
- [CarRacing](carracing.md)
- [highway-env](highway.md)
- [Jumanji](jumanji.md)
- [PGX board games](pgx.md)
- [Pixel-native families](pixel_native.md)
- [GPU-native families](gpu_native.md)
- [RoboCasa](robocasa.md)
- [DM Control](dmcontrol.md)
- [Fetch (gymnasium-robotics)](fetch.md)
- [Gymnasium MuJoCo](mujoco_locomotion.md)
- [Meta-World](metaworld.md)
- [HumanoidBench](humanoidbench.md)

Families examined and rejected:

- [Adroit hand](adroit.md)
- [Franka Kitchen](kitchen.md)
- [MiniWorld](miniworld.md)

## Historical measurements

Milestone write-ups and generated tables. They use the project's internal names
(Step / Wave / workstream) on purpose. They are not the product documentation.

- [Implementation measured](step3_benchmarks.md)
- [Fidelity baseline](step4_fidelity_baseline.md)
- [Downstream semantics](step5_downstream_semantics.md)
- [Optimisation](step6_optimization.md)
- [Renderer coverage](step7_coverage.md)
- [Variation maps (new cells)](step8_variations_newcells.md)
- [Variation maps (DMC + classic control)](step8_variations_dmc_control.md)
- [Fetch variation maps](step8_variations_fetch.md)
- [Certification harness and provenance](step8_certification.md)
- [Support matrix (42 ids)](step8_support_matrix.md)
- [Reference versions and provenance](step8_reference_provenance.md)
- [Craftax semantic follow-up](step8_semantic_craftax.md)
- [Performance and pipeline accounting](step8_performance.md)
- [SWM integration matrix](step8_swm_integration.md)
- [SWM / upstream findings](step8_upstream_findings.md)
- [Functional-completion packet](step8_closure.md)
- [Functional-completion handoff](step8_handoff.md)
- [Takeover stop note](step8_takeover_stop_handoff.md)
- [Public-surface freeze](step9_closure.md)
- [Refactor handoff](step9_handoff.md)
- [Replacement-readiness (42 historical cells)](step9_replacement_readiness.md)
- [Historical research artifacts](artifacts/README.md)
- [Open items after the freeze](step9_open_items.md)
- [v0.1 release-surface contract](v0_release_surface.md)
- [Historical throughput tables](throughput.md)
- [Variation collapses](variation_collapses.md)
