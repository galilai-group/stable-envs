# `stable_envs.render.mjwarp`

MJWarp's batch raytracer behind a `render(qpos) -> (B,H,W,3) uint8` surface, plus the model pre-pass
without which it silently draws a different scene.

Four things have to be right and three of them are silent when wrong: the buffer is BGRA, MuJoCo renders
shadows by default, geom groups must match the reference's, and **MJWarp never reads alpha** — so a geom
with `rgba="0 1 0 0"` renders as opaque saturated colour. That last one is what made a RoboCasa kitchen
score 7.6% exact and read as a texture-path failure; the pre-pass here takes the same scene to interior
mean 4.96 with 89.4% of interior pixels within 8.

::: stable_envs.render.mjwarp
