# GPU-native env families: MinAtar, Octax, PixelBrax, KAGE-Bench

Four families that already run on the GPU, gated against the criterion in
`when_does_this_pay_off.md`. They were measured because the RL ecosystem moved: the thesis here is
"store a small state, regenerate the frame on the GPU", and a family that *already* runs end to end on
the GPU is the case where that could be either redundant or exactly right. Three of the four turn out
to be redundant, each for a different reason, and the fourth is a sharper version of Craftax's
storage instruction.

Probes and raw JSON in `scratch/gpunative/`; venvs under
`~/venvs/gpunat-{minatar,octax,kage,pixelbrax}`, all clean (`python -m venv` without
`--system-site-packages`, because `jax`'s CUDA plugin discovery breaks otherwise) and every run
verified `jax.devices() == [cuda:0]` before a number was recorded.

| family | G1 purity | G2 render-state / frame | G3 decoder | G4 regen vs decode | verdict | gate that decided |
|---|---|---|---|---|---|---|
| **MinAtar** (via `minatar` 1.0.15 / `gymnax` 0.0.9) | ✅ 50,000 frames, 0 blank; 14,416 + 14,999 cross-history pairs, **0 differing pixels**; `render()` mutates nothing | ✅ **0.127–0.167×** on the collapsed index grid (38–50 B vs a 300 B frame); ❌ **1.33–3.33× BIGGER** as shipped | ✅ **4 palettes, 28 colours, 84 bytes for the whole family** — a colour LUT, not an atlas, because the frame is 1 px per cell | ❌ nothing to regenerate | **NO-GO** | **G4** |
| **Octax** (CHIP-8 in JAX, arXiv 2510.01764) | ✅ full-state round-trip into a **fresh instance: 0/12 mismatches** — the test ALE fails on 13.75 % of pairs | ❌ **3.05×** the display, **1.015×** the scale-1 RGB frame; the render slice **IS** the frame (1.0×) | ✅ vacuously — **2 RGB triples, 6 bytes**, identical for all 14 games | ❌ moot | **NO-GO** | **G2 (degenerate)** |
| **PixelBrax** (arXiv 2502.00021) | ✅ 158 distinct values, not blank; re-render **0**, different 7-step history **0** | ✅ **0.087–0.113×** (`pipeline_state` 1,836–2,384 B vs a 21,168 B frame); q+qd alone 0.0034× | ✅ `sys.geoms` identical across 4 resets, **`n_mesh = 0`** — but the map is a triangle rasteriser with per-pixel shading, not a gather | ❌ **1.43 cores** (9,392 f/s at B=2048 vs 6,570 f/s/core of 84 px decode); render is **98.8 %** of the step | **NO-GO** | **G4** |
| **KAGE-Bench** (arXiv 2601.14232, ICML 2026) | ✅ 0 blank on the rich config; re-render after a different 29-step rollout **0 differing pixels**; `step`'s obs == `render(state)` 50/50 | ❌ **5.68–6.69× BIGGER** as shipped; ✅ **0.051–0.060×** once the render window is bit-packed **and** the background is stored as an index | ✅ 128 backgrounds + 59 sprites / 27 skins = **6.46 MB**, constant across resets (next to MiniGrid's 6.08 MB) | ⚠️ **2.3 cores** at the benchmark's own visual settings, 3,388 at plain ones | **CONDITIONAL** | **G2** |

## MinAtar — the first family to clear G1, G2 and G3 and still fail

Worth stating first because it is the cleanest illustration that the four gates are not a formality.

MinAtar **does** have an official RGB renderer, which was the open question: `minatar/gym.py`'s
`BaseEnv.render(render_mode="rgb_array")` returns a real `(10, 10, 3)` array (5 distinct colours,
verified non-blank), and it is exactly `np.stack(cubehelix_palette)[argmax_channel]` — rebuilding it
from the observation and the palette gives **max abs diff 0.0**. So there is no need to *define* a
frame, which is the thing this repo will not do silently.

Everything structural passes. The decoder is **84 bytes for all five games** — four
`sns.color_palette("cubehelix", n)` tables plus black, byte-identical across seaborn 0.11.2/0.13.2 and
matplotlib 3.8.4/3.11.1. The render slice is a 38–50 byte index grid, 0.127–0.167× the frame, and an
attribute ablation over 200 live game objects finds **3–9 fields per game that never move a pixel**
(asterix: 9). A five-line gather runs at **0.89–1.17 billion f/s** against the shipped numpy
renderer's 164k–191k, a **5,437–6,643×** ratio with the **crossover at batch 8**.

And it is worthless, because **the frame is 300 bytes and it is not what the agent consumes.** The
observation *is* the state — a `(10, 10, C)` bool grid — so there is no state→pixel step in the loop
to accelerate, and a trajectory is `(seed, actions)` = 1 byte per step which the official emulator
replays at 28k–177k steps/s on one core. That is ALE's G4 entry reached from the opposite direction:
ALE has no compact state and no storage problem; MinAtar has a perfect compact state and no storage
problem. **Neither is a renderer question.**

One fidelity number worth keeping, since gymnax is the GPU MinAtar and therefore decides whether the
JAX path *is* the official path: breakout, sticky actions off, ball start aligned, **1,256 frames over
120 episodes, 0 mismatched — 100.0000 %**. Coverage is 4 of 5: `Seaquest-MinAtar` is present but
unregistered and broken on jax 0.10.2 (`IndexError: Slice entries must be static integers` — a dynamic
oxygen-gauge slice in `get_obs`).

## Octax — a genuine pure function of state, and that is the problem

Octax is a CHIP-8 emulator written in JAX (arXiv [2510.01764](https://arxiv.org/abs/2510.01764),
`riiswa/octax`). It is the interesting counterpoint to ALE, and it settles the question ALE raised:
**a GPU emulator whose state is explicit does pass G1.** Restoring a full `OctaxEnvState` into a fresh
instance and stepping reproduces the frame on **12/12** pairs, where ALE's 128-byte RAM fails on
13.75 %. Dropping the display and stepping changes 1.7–15.6 % of pixels on 12/12, so the framebuffer
is real XOR-accumulated state rather than a cache.

It fails G2 by being *too* pure. CHIP-8 has a framebuffer, so the state contains the picture:
`OctaxEnvState` is 6,239 B of which **2,048 B is the display**, against a 6,144 B RGB frame at scale
1 — **1.015×**, and 3.05× the display itself. The honest slice makes it worse rather than better: of
the 4,096 memory bytes, the number ever written over 300-step rollouts is **0** (tetris) / **1** (pong,
brix) / **2** (missile), so memory is the ROM and the dynamic non-display state is **74–76 B, 0.036×
the display** — leaving the render slice as the display, **ratio exactly 1.0**. The decoder is 2 RGB
triples, six bytes, identical for every game, which sounds like the strongest G3 in the set and is
actually the diagnosis: **the state→pixel map is the identity plus a 1-bit palette.** There is nothing
to build.

The throughput claim reproduces and is worth recording because it is the fastest emulator in this
document: **1.34 M env-steps/s at batch 65,536** on one H200 = 5.36 M frames/s at frame-skip 4 = 59 M
CHIP-8 instructions/s. The paper claims 350k steps/s on an RTX 3090 at ≤8,192 envs; this is 3.8×
better on better hardware. A rollout is again `(ROM, seed, actions)`.

Install note, since it costs an afternoon otherwise: `pip install octax` **does not import** —
`octax/__init__.py` unconditionally imports `octax.rendering`, which imports `cv2`, and
`opencv-python` is only in the `[gui]` extra. `pip install opencv-python-headless pillow` fixes it.
The 25 `.ch8` ROMs *are* shipped.

## PixelBrax — the one whose renderer is the whole cost, and is the wrong renderer

PixelBrax (arXiv [2502.00021](https://arxiv.org/abs/2502.00021), `trevormcinroe/pixelbrax`) bolts
JoeyTeng's `jaxrenderer` onto Brax: a tinyrenderer-style triangle rasteriser with a z-buffer,
Gouraud/Phong shading, an optional shadow pass, a tracking perspective camera and a
`texture_scaling=8192` checkerboard ground.

G1, G2 and G3 all pass — including the one that matters most, `sys.geoms` **identical across four
resets with `n_mesh = 0`**: 9 geoms for halfcheetah, 8 for walker2d, 18 for ant, all capsules, boxes
and a plane. That is OGBench's signature exactly, and OGBench is a 113–1,092× win.

G4 says no. Blocking on the **pixels** — a first pass that blocked on the proprioceptive observation
over-reported by 41 %, which is the trap this document keeps rediscovering — the whole step runs at
**9,392 f/s at batch 2048** at 84 px, against **6,570 f/s/core** of 84 px JPEG decode: **1.43 cores**,
inside RoboCasa's 0.37–5.1 band and an order of magnitude below every family the criterion calls a
win. The render is **98.8 %** of the step at that batch, so all the headroom is in the renderer and
none of it is realised by the renderer that exists.

The constructive reading: **the thing worth building for Brax-from-pixels is the OGBench treatment** —
`n_mesh = 0`, capsules and boxes, one baked plate, an analytic shader — and that is a new renderer
against a reference with no second implementation, not a port. No cross-parity number is possible
here, because PixelBrax *is* its own reference.

Installing it needs three undocumented fixes and none is in the README: the wheel does not ship
`brax` (`install_requires` is only `trimesh` and `typing-extensions`; the vendored brax 0.9.4 fork
sits at `pixelbrax/pixelbrax/brax/brax` with no `setup.py` and must go on `PYTHONPATH` by hand),
`jaxopt` is imported by the vendored `brax/generalized/constraint.py` and declared nowhere, and the
repo root is itself named `pixelbrax` without an `__init__.py` so running from the parent gives
`AttributeError: module 'pixelbrax' has no attribute 'make'`. Last commit **2025-04-28**, not on PyPI.

## KAGE-Bench — Craftax's storage instruction, needing two substitutions instead of one

KAGE-Bench (arXiv [2601.14232](https://arxiv.org/abs/2601.14232), ICML 2026,
`CognitiveAISystems/kage-bench`) is a JAX-native 2-D platformer built to vary appearance along known
axes. G1 and G3 are clean — the atlas is 128 pre-decoded backgrounds plus 59 sprite arrays across 27
skins, **6.46 MB**, constant across resets, the same class as MiniGrid's 6.08 MB.

G2 fails as shipped and passes on the slice, and the slice needs **two** substitutions where Craftax
needed one. `EnvState` is **279–329 kB against a 49,152 B frame (5.68–6.69×)**, dominated by
`layout_solid_mask` (128×2048 bool = 262,144 B) and — the entry worth carrying — **`bg_image`, a
128×128×3 array which is *exactly one frame stored inside the state*.** Bit-packing the on-screen mask
window and keeping the background as pixels still gives 52,102 B = **1.06×**; storing the background
as an *index into the 128-image library* as well gives **2,520–2,950 B = 0.051–0.060×**. So the
instruction is "store the render window **and** an asset index", and a family can carry a whole frame
inside its state without that being visible from the pytree's shape.

**Its "93 controllable parameters" map onto this repo's `variation_space` in kind but not in
contract.** The shipped `configs/default_config.yaml` has exactly 93 leaf keys, but **81 are visual and
12 are physics or task** (gravity, move_speed, jump_force, friction, air_resistance, max_fall_speed,
episode_length and five reward terms). A `variation_space` leaf must change the picture without
changing the control problem, so 12 of the 93 would not be admissible; and the defaults are not "the
reference appearance", because there is no second renderer to be exact against.

Two numbers to quote carefully, because the headline is real and conditional. The paper's **33 M
env-steps/s on a single GPU reproduces exactly** — at 2^15, plain visuals, **with the frame never
materialised** (33,273,886 sps; 38.1 M at 2^16). Materialise the frame and it is **13.2 M**; use the
benchmark's own visual config and it is **8,984 f/s**, flat in batch from 256 (7,860) to 16,384
(8,984), which is the signature of a long serial chain — the same shape this repo records for
procgen's SourceOver blends. Against 3,906 f/s/core of 128 px decode that is **3,388 cores** (plain,
frames live) or **2.3 cores** (shipped visuals). The README's separate claim of "2^15 … reaching
7,000,000 FPS" matches neither the paper nor either measurement.

One trap for anyone repeating this: **the python-default config renders a frame with 3 distinct pixel
values in total.** The shipped YAML gives 240. A parity number taken off the python defaults would be
measured against something very close to a blank.

## Gaps

- **PixelBrax's `distractor="videos"` mode is not gated.** It needs the 480p DAVIS-2017 dataset
  (~2 GB) and would be the decisive G3 test for that mode, since a per-episode *video* background is
  content by construction. Only `distractor=None` was measured; G4 fails either way.
- **MinAtar's `Seaquest` is not covered on the JAX path** — unregistered in gymnax 0.0.9 and broken on
  jax 0.10.2. The 100.0000 % gymnax-vs-official fidelity number is breakout only; the other three
  registered games were not diffed against the CPU reference.
- KAGE's rich config **OOMs at batch 32,768** with frames live (138 GiB), so its frame-materialised
  curve stops at 16,384.
