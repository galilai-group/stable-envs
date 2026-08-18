# `stable_envs.data`

States in, frames out — for the cases that are genuinely about rendering or about episode structure.
The recommended training idiom does not use this module: bring your own dataset and `DataLoader` over
states and call `env.set_state(batch).render()` in the forward pass. Slicing a `(N, state_dim)` array is
something torch and HuggingFace already do; stable-envs supplies the renderer.

What is here:

- `render_states` streams a corpus through the renderer with bounded peak memory.
- `StateDataset` stores states and yields **frames** — the drop-in for an image dataset, so a pipeline
  already built around an image folder can point at states without its training step changing.
- `TrajectoryDataset` adds episode structure: windows that never straddle a trajectory boundary
  (re-derived and asserted in the constructor), per-trajectory style, and a HuggingFace round-trip whose
  frames are asserted bit-identical across save and load (`examples/hf_roundtrip.py`).

Both datasets render inside `__getitems__`, so the object handed to `collate_fn` is already a batch:
pass `collate_fn=stable_envs.collate_batched`, together with `num_workers=0` while the dataset is the
one rendering. There is no `ds.loader()` — it wired that one argument and hid two decisions (batch size,
workers) that belong to the caller. Yielding per-item samples instead, so `default_collate` would work,
was measured and rejected: for `TrajectoryDataset` at B=1,024 it is 8.741 ms/batch against 0.202.

If you want a batched index with torch's own collate — the fastest bring-your-own recipe, and worth it
above about B=1,024 — pass a `BatchSampler` as the sampler and `batch_size=None`, which makes the loader
hand `ds[list]` a whole index list at once.

::: stable_envs.data
