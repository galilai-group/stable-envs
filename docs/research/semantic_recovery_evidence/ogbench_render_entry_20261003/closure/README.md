# Render-entry profile closure

Both jobs independently pass exact image comparisons with frozen unchanged
production source. Eighteen component rows preserve six synchronized samples
and SDs. Terminal accounting resolves source hashes, registration, job and host.

At B64, compact RenderState rendering costs 17.25 ms for Cube and 19.28 ms for
Scene. The public full-State path costs 272.23 and 273.65 ms. Separate full-State
CUDA validation costs 247.37 and 293.50 ms. These independent timings overlap;
they must not be added or treated as an exact additive partition.

cProfile identifies 8,896 Tensor.cpu calls for Cube and 8,448 for Scene in one
validation. It also records per-lane TensorDict unbinding/iteration and repeated
scalar space membership checks. A column-wise transfer of the entire State
costs only about 2.1–2.2 ms. Full rendering with explicit batched host conversion
included costs 144.71 and 153.41 ms, with zero image difference. This is a
diagnostic input-path comparison, not a production patch or upstream speedup.

The bounded next implementation should validate structural shapes/dtypes,
transfer each column once, keep every finite/range/RNG/profile check and own
its host buffers, then verify independent CPU/CUDA outputs and invalid-State
rejection. No renderer, public State schema or shared step API needs redesign.
Scalar validation after transfer remains another measured cost; start with
source-equivalent batching and reprofile before broadening the change.

SHA256SUMS.json paths are relative to the parent evidence directory.
