# `stable_envs.render.rendernet`

A learned render corrector for the visual tier: a cheap MJWarp frame in, MuJoCo's own frame
approximated out — faster than the render it replaces *and* closer to the official one.

Three things in this module are locks rather than conveniences, and each exists because the failure it
prevents is silent. A checkpoint is **keyed to the (scene, camera) it was fitted on** and a mismatch
raises, because on the wrong camera a corrector is measured at 2.6× worse than not correcting at all
while looking perfectly reasonable. The **declaration lives inside the weights file**, because
`visual_fidelity` here describes a (renderer, checkpoint) pair and a header that does not belong to its
weights is a claim about something else. And the network is only ever handed **one batch shape**,
because Inductor's autotuner picks per shape and two convolution tile configurations round some pixel
the other way.

See [Fidelity tiers → the learned decoder](../concepts/fidelity-tiers.md#the-learned-decoder) for the
contract and [`examples/rendernet/`](https://github.com/galilai-group/stable-envs/tree/main/examples/rendernet)
for how to build one.

::: stable_envs.render.rendernet
