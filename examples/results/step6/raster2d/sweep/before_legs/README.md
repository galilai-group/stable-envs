# The `1103326` legs of the same two jobs

Not a second measurement of the parent directory's shards — the other half of each ratio. Same job,
same node, same states (`--states-in`), and `stable_envs_file` pointing at
`$SCR/se6/raster2d-preB/`, a copy of this tree with exactly two files replaced by their `1103326`
contents: `stable_envs/render/raster2d.py` and `stable_envs/envs/swm_nav.py`.

They live here rather than beside their `after` counterparts because `scratch/step3/certify.py:merge`
keys on `(env, device, host, threads)` and would collapse the pair, labelling a tier from whichever
shard happened to carry more cells.
