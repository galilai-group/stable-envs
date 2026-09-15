# Which job wrote which Gate-A document

| file | device | job | node | cells | note |
|---|---|---|---|---|---|
| `cpu_craftax.json` | cpu | `6413322` | node2304 | CraftaxClassic, CraftaxFull | `se7-craftax` |
| `cpu_vis311.json` | cpu | `6413322` | node2304 | DMCManipulatorBringBall, DMCQuadrupedWalk | `se-vis311`, the declared mujoco 3.11.0 / dm_control 1.0.44 pair |
| `cpu_ogbmanip.json`, `cpu_rocket.json` | cpu | `6413321` | gpu2107 | OGBCube, OGBScene, PFRocketLanding | asked ON THE CARD on purpose: these three raise on `cpu` by design, and on a batch node `mujoco_warp` and `nvdiffrast` do not import at all, so a refusal and a missing module would be indistinguishable |
| `cuda_*.json` | cuda | `6413321` | gpu2107, RTX 3090 | all seven | one interpreter per cell family, each the one that cell's declaration names |

Two documents written by the superseded job `6413171` are NOT kept: it ran under `MUJOCO_GL=osmesa`,
which makes `stable_envs.envs.dmcontrol` fail to import on these nodes and silently removes every DM
Control cell from the registry. Its `cpu_vis311.json` recorded `NOT_REGISTERED_HERE` for two cells
that are registered, and its `cpu_ogbmanip.json` / `cpu_rocket.json` recorded `BLOCKED` where the
question was whether the class refuses. Its `cpu_craftax.json` reached the same verdicts as the
re-run (Craftax needs no GL) and is superseded rather than contradicted.
