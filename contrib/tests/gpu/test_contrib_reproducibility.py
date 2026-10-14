"""The contrib cases of ``tests/gpu/test_reproducibility.py``.

The measured 2S->S downsample floors of the contrib families, run through the same test body as the
packaged families, and the highway family's size-leak reproduction.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.gpu

if not torch.cuda.is_available():
    pytest.skip("stable-envs GPU tests require a CUDA device", allow_module_level=True)

import test_reproducibility as _main

import stable_envs as se
from tests._runner import render_worker as W

CONTRIB_NAMES = sorted(n for n, c in se.registry.REGISTRY.items() if c.__module__.startswith("contrib."))


@pytest.fixture(scope="module", params=CONTRIB_NAMES)
def env(request):
    return se.make(request.param, num_envs=_main.N, device="cuda")


@pytest.fixture(scope="module")
def inputs(env):
    """The (states, style) pair, drawn exactly as ``tests/gpu/test_reproducibility.py`` draws it."""
    states, style = W.sample_inputs(env, _main.N)
    return states, (style if env.style_dim else None)


#: The contrib rows of ``DOWNSAMPLE_EXACT_FRAC`` in ``tests/gpu/test_reproducibility.py``; see
#: ``MEASURED_ON`` and ``DOWNSAMPLE_MARGIN`` there for the protocol. Costs by renderer family:
#:   1.000       gather from a bitmap atlas whose tile size scales exactly (MiniHack) — the
#:               2S frame IS the S frame at twice the size, so the box filter inverts it exactly.
#:   0.007-0.092 Procgen: a gather from a bitmap atlas whose sampling RATE scales instead, so the 2S
#:               frame resamples the same source art more finely. See the note on its entries.
#:   0.97-1.00   MiniGrid: same idea, but the atlas is re-rendered by MiniGrid at the requested tile
#:               size with its own 3x supersample, so the two are close rather than identical.
DOWNSAMPLE_EXACT_FRAC: dict[str, float] = {
    "BabyAIGoToLocal": 0.987213,  # mean 0.1315/255
    #: Crafter is the outlier of the tile families and the reason is arithmetic, not geometry.
    #: ``Env.render`` CENTRES the 9-tile view inside the frame — ``border = (size - (size // 9) * 9)
    #: // 2`` — so 64 px puts the view at offset 0 and 128 px puts it at offset 1. The 2S picture is
    #: therefore the S picture shifted by half an output pixel, and no box filter inverts a half-pixel
    #: shift; on top of that the atlas is resampled from 16x16 source art at each tile size, so 14 px
    #: tiles are not 7 px tiles doubled. Both are properties of the reference. The floor is low and it
    #: is still a live regression detector: it moves the moment a tile lands in a different place.
    "Crafter": 0.382263,  # mean 12.5190/255
    #: Griddly is the low end of the TILE families and the mechanism is the reference's, not ours.
    #: `size` here means `Observers.Sprite2D.TileSize`, and griddly RESAMPLES its 24 px sprite PNGs to
    #: whatever tile size is asked for -- so the 2S atlas is a finer resampling of the same source art
    #: rather than the S atlas doubled, and box-averaging four resampled texels does not return the one
    #: the S atlas picked. The wall art is high-frequency pixel art by design, which is why the
    #: wall-dense games sit lowest (Sokoban 0.22, Labyrinth 0.35) and the sparsest highest
    #: (Butterflies 0.70). Mean absolute difference stays at 1.1-2.9/255 throughout -- the two pictures
    #: agree, it is exact-pixel agreement that the resample costs. Still a live detector: the number is
    #: fixed for a fixed sampling grid and moves the moment a tile lands somewhere else.
    "GriddlyBait": 0.432180,  # mean 1.7029/255
    "GriddlyButterflies": 0.704126,  # mean 1.2416/255
    "GriddlyClusters": 0.392938,  # mean 1.4342/255
    "GriddlyCookMePasta": 0.298415,  # mean 1.0666/255
    "GriddlyLabyrinth": 0.354689,  # mean 2.1506/255
    "GriddlySokoban": 0.222717,  # mean 2.9219/255
    "GriddlySokoban2": 0.248369,  # mean 2.8664/255
    "Highway": 0.922090,  # mean 7.6935/255  size re-frames; recorded, not asserted
    #: HumanoidBench draws the whole scene through MJWarp with no plate (its camera is TRACKCOM,
    #: so no static plate exists), and it carries no PLATE_PATHS -- so every cell is measured in
    #: its own right. They sit together at 0.95-0.97 because the two things 2S resamples
    #: differently are both small here: the floor's texture is REPLACED by its dominant colour,
    #: so the only rate-dependent content left is
    #: the H1's mesh silhouettes and its shadow edge. Spoon is the low end because it is the one
    #: cell here with textured primitives on the table, which 2S resamples at a finer rate.
    "HumanoidBenchH1BalanceSimple": 0.967609,  # mean 0.8579/255
    "HumanoidBenchH1HandBalanceSimple": 0.963911,  # mean 0.9218/255
    "HumanoidBenchH1HandRun": 0.970247,  # mean 0.8162/255
    "HumanoidBenchH1HandSitHard": 0.963375,  # mean 1.0008/255
    "HumanoidBenchH1HandSitSimple": 0.964001,  # mean 0.9794/255
    "HumanoidBenchH1HandSlide": 0.967253,  # mean 0.8828/255
    "HumanoidBenchH1HandSpoon": 0.959507,  # mean 1.2295/255
    "HumanoidBenchH1HandStair": 0.967337,  # mean 0.8099/255
    "HumanoidBenchH1HandStand": 0.970247,  # mean 0.8162/255
    "HumanoidBenchH1HandWalk": 0.970247,  # mean 0.8162/255
    "HumanoidBenchH1Run": 0.972082,  # mean 0.7091/255
    "HumanoidBenchH1SitHard": 0.966482,  # mean 0.8928/255
    "HumanoidBenchH1SitSimple": 0.967659,  # mean 0.8467/255
    "HumanoidBenchH1Slide": 0.969963,  # mean 0.7793/255
    "HumanoidBenchH1Stair": 0.971107,  # mean 0.6661/255
    "HumanoidBenchH1Stand": 0.972082,  # mean 0.7091/255
    "HumanoidBenchH1Walk": 0.972082,  # mean 0.7091/255
    "Intersection": 0.958981,  # mean 4.5107/255  size re-frames; recorded, not asserted
    #: Jumanji renders through matplotlib at ``dpi = size / 10``, so 2S is a genuinely DIFFERENT
    #: rasterization rather than a supersample of S: text is laid out in points and re-hinted at the new
    #: dpi, and patch edges are re-antialiased at a different sub-pixel coverage. Maze is the exception
    #: at 1.000000 because its viewer is a single nearest-neighbour ``imshow`` whose cell boundaries
    #: land on exact pixel multiples at both sizes, so the box filter inverts the upscale exactly. The
    #: glyph-bearing ids (Sudoku, Minesweeper) and the sprite/patch ones (RubiksCube, Sokoban) sit at
    #: 0.71-0.86, which is a property of the reference and not of this renderer.
    #: JumanjiConnector is deliberately ABSENT: its cell borders are 1 POINT wide, which quantises to
    #: fully-black pixels at dpi 20 and to a two-cell blend at dpi 40, so the per-cell model is false at
    #: 400 px and the env raises ValueError rather than render there. This test skips it for that
    #: reason.
    "JumanjiMaze": 1.000000,  # mean 0.0000/255
    "JumanjiMinesweeper": 0.832503,  # mean 9.0860/255
    "JumanjiRubiksCube": 0.855150,  # mean 4.1042/255
    "JumanjiRubiksCubePartlyScrambled": 0.855150,  # mean 4.0577/255
    "JumanjiSokoban": 0.711994,  # mean 0.7419/255
    "JumanjiSudoku": 0.831450,  # mean 7.5042/255
    "JumanjiSudokuVeryEasy": 0.822840,  # mean 7.8610/255
    "Merge": 0.945075,  # mean 5.8377/255  size re-frames; recorded, not asserted
    "MiniGridDoorKey": 0.997700,  # mean 0.0228/255
    "MiniGridEmpty": 0.998932,  # mean 0.0048/255
    "MiniGridFourRooms": 0.999811,  # mean 0.0008/255
    "MiniGridKeyCorridor": 0.982485,  # mean 0.2123/255
    "MiniGridLavaCrossing": 0.974085,  # mean 0.1360/255
    "MiniGridPOVFetch": 0.997429,  # mean 0.0121/255
    "MiniHack": 1.000000,  # mean 0.0000/255
    #: The six VISUAL-tier Gymnasium locomotion domains, which are here rather than skipped by
    #: PLATE_REPRESENTATIVES because they bake no plate: they are re-rasterized whole by nvdiffrast at
    #: whatever resolution is asked for. What sets the number is the minified checker floor —
    #: doubling the resolution halves the footprint every floor pixel filters over, so the mip level
    #: changes and the 2S floor is a genuinely finer sampling of the texture rather than a supersample
    #: of the S one. It tracks how much floor each camera sees, which is why Swimmer (94.9% floor,
    #: MAGNIFIED rather than minified, so no mip level changes) is 0.909 while Humanoid is 0.682.
    #: The four PARITY envs of the same family are not in this table: they carry PLATE_PATHS.
    "MujocoAnt": 0.771469,  # mean 1.6888/255
    "MujocoHalfCheetah": 0.758811,  # mean 2.0481/255
    "MujocoHopper": 0.694054,  # mean 2.5325/255
    "MujocoHumanoid": 0.681649,  # mean 2.7766/255
    "MujocoSwimmer": 0.909481,  # mean 0.7384/255
    "MujocoWalker2d": 0.755514,  # mean 1.9871/255
    #: Overcooked sits with MiniGrid at the top of the tile families and for the same reason: the
    #: atlas is RE-RENDERED by jaxmarl at the requested tile size with its own 3x supersample, so 2S is
    #: close to a supersample of S rather than a resample of fixed source art (which is what puts
    #: griddly and Procgen at the other end). The residual is the round tiles -- an onion at 3x96 has a
    #: different subpixel boundary from one at 3x192 -- and it is 0.13-0.18/255 in the mean.
    "OvercookedAsymmAdvantages": 0.985520,  # mean 0.1493/255
    "OvercookedCoordRing": 0.981630,  # mean 0.1787/255
    "OvercookedCounterCircuit": 0.987140,  # mean 0.1251/255
    "OvercookedCrampedRoom": 0.980990,  # mean 0.1720/255
    "OvercookedForcedCoord": 0.981630,  # mean 0.1787/255
    #: gymnasium-robotics PointMaze. Much higher than the OGBench mazes (0.708-0.726) for one
    #: reason: its camera looks along the ground at elevation -45, so the plate's re-bake at 2S
    #: re-renders a scene that is mostly flat sky and flat floor, where the OGBench top-down camera
    #: fills the frame with a checkerboard whose every cell edge moves.
    "PointMazeLarge": 0.913387,  # mean 0.2794/255  (measured on PointMazeMedium)
    "PointMazeMedium": 0.913387,  # mean 0.2794/255
    "PointMazeOpen": 0.913387,  # mean 0.2794/255
    "PointMazeUMaze": 0.913387,  # mean 0.2794/255
    "Parking": 0.931908,  # mean 6.2932/255  size re-frames; recorded, not asserted
    #: Procgen is the low end of this table, and the mechanism is the reference's, not ours. A cell
    #: is 2.56 output px at 64 and 5.12 at 128, so the 2S frame takes a nearest-neighbour sample of
    #: the SAME 128x128 source art at twice the rate — it is a finer resampling of the texture, not a
    #: supersample of the S frame, and box-averaging four different source texels does not return the
    #: one texel the S frame picked. On sand and dirt, which are high-frequency noise by design,
    #: almost no pixel survives. procgen's own 512x512 human render differs from its 64x64
    #: observation for exactly this reason (and additionally turns antialiasing on). The floor is
    #: still a live detector: it is a fixed number for a fixed sampling grid and moves as soon as a
    #: rect lands somewhere else.
    "ProcgenChaser": 0.092380,  # mean 12.6856/255
    "ProcgenHeist": 0.061981,  # mean 7.7011/255
    "ProcgenMaze": 0.007202,  # mean 9.2941/255
    "ProcgenMiner": 0.092468,  # mean 9.7574/255
    "Racetrack": 0.983442,  # mean 1.7873/255  size re-frames; recorded, not asserted
    #: RoboCasa. These four were failing this test with "has no measured downsample bound recorded",
    #: and the failure was invisible in the main run because the suite's interpreter cannot import
    #: robocasa (it pins mujoco 3.3.1 against mujoco_warp's >= 3.11), so the whole family was skipped.
    #: Measured in ``~/venvs/se-robocasa``, three repeats each,
    #: identical to six decimals every time. They sit with the locomotion domains and for the same
    #: reason: a kitchen is 73 textures on 344 visible geoms, and doubling the resolution changes what
    #: each pixel of a photographic wood or granite surface filters over. The wrist camera is the
    #: highest because it fills the frame with a few large near surfaces rather than a room of small
    #: ones.
    "RoboCasaCounterToSink": 0.756603,  # mean 2.0074/255, max 205/255
    "RoboCasaCounterToSinkL45S33": 0.787102,  # mean 2.0138/255, max 216/255
    "RoboCasaCounterToSinkRight": 0.740753,  # mean 2.0634/255, max 233/255
    "RoboCasaCounterToSinkWrist": 0.871983,  # mean 0.7108/255, max 207/255
    "Roundabout": 0.974241,  # mean 2.7201/255  size re-frames; recorded, not asserted
    "Sokoban": 1.000000,  # mean 0.0000/255  16 px art replicated k x k, exactly
}
#: gymnasium-robotics PointMaze: a DIFFERENT camera geometry from
#: OGBench's (elevation -45 against straight down) and a different
#: bake path (gymnasium's viewer, not mujoco.Renderer), so one of
#: the four scenes has to stand for it in its own right.
PLATE_REPRESENTATIVES = frozenset({"PointMazeMedium"})


def test_resolution_downsample_agrees_within_the_measured_bound(env, inputs, monkeypatch):
    """``tests/gpu/test_reproducibility.py``'s downsample check, against the contrib rows above."""
    monkeypatch.setattr(_main, "DOWNSAMPLE_EXACT_FRAC", {**_main.DOWNSAMPLE_EXACT_FRAC, **DOWNSAMPLE_EXACT_FRAC})
    monkeypatch.setattr(_main, "PLATE_REPRESENTATIVES", _main.PLATE_REPRESENTATIVES | PLATE_REPRESENTATIVES)
    _main.test_resolution_downsample_agrees_within_the_measured_bound(env, inputs)


#: Moved with the test below from the record of closed defects in ``tests/gpu/test_reproducibility.py``.
#:
#: HIGHWAY_SIZE_LEAK. ``HighwayEnvBase``
#: derived ``self.H``/``self.W``/the camera from ``render(size=)`` and STORED them, so
#:     env._render(st).shape -> (4,150,600,3); env._render(st, size=300); env._render(st).shape
#: came back (4,300,1200,3) with ``env.size`` still 150 — and it cascaded into four unrelated dataset
#: round-trip failures. The screen is now a frozen ``_Screen`` value derived inside the call, so there
#: is nowhere left to store it; the four highway envs assert the property below like everyone else.


@pytest.mark.parametrize("name", ["Highway", "Merge", "Roundabout", "Intersection"])
def test_highway_render_size_is_an_argument_not_a_mode(name):
    """The reproduction that was recorded as HIGHWAY_SIZE_LEAK, asserted verbatim.

    Kept beside the registry-wide check rather than folded into it because this family is where the
    defect lived and the shape of it is specific: the screen — height, width, camera offset, stripe
    count, line width — was derived from the requested size and STORED on the env by ``_set_screen``,
    so it survived the call that asked for it. Measured before the fix:

        env._render(st).shape       -> (4, 150, 600, 3)
        env._render(st, size=300)
        env._render(st).shape       -> (4, 300, 1200, 3)      # env.size is still 150

    A fresh env per parameter, so this states the property on its own and cannot be satisfied (or
    broken) by whatever the module-scoped fixture did before it.
    """
    if name not in se.registry.REGISTRY:
        pytest.skip(f"{name} is not registered")
    env = se.make(name, num_envs=4, device="cuda")
    st = env.sample_states(4, rng=np.random.default_rng(0))
    before = tuple(env._render(st).shape)
    big = tuple(env._render(st, size=2 * int(env.size)).shape)
    after = tuple(env._render(st).shape)
    assert big == (before[0], 2 * before[1], 2 * before[2], 3), (
        f"{name}: render(size=2S) returned {big}, which is not twice {before}"
    )
    assert after == before, (
        f"{name}: a plain render() returns {after} after one render(size={2 * int(env.size)}), against "
        f"{before} before it — the requested size outlived the call that requested it"
    )
    assert (int(env.size), int(env.H), int(env.W)) == (before[1], before[1], before[2]), (
        f"{name}: env.size/H/W report ({env.size}, {env.H}, {env.W}) but the frame is {before} — the "
        f"env's advertised geometry and the geometry it renders at have come apart"
    )
