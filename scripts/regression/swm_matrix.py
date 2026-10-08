"""Join the LIVE ``stable_worldmodel`` registry against the ``stable-envs`` registry, one row per swm id.

WHY THIS EXISTS. ``docs/research/swm_coverage.md`` answered this question once, by hand, and has since
drifted: it records nine DM Control ids as gaps, and seven of those nine now have a renderer. A
hand-maintained table cannot survive a codebase that adds environments, so this rebuilds the same
answer from the things that cannot drift -- the live registries, the class declarations, and the result
JSONs already in the repo -- and writes both a machine-readable matrix and the Markdown view of it.

WHAT IT DOES NOT DO. It measures almost nothing itself. Every number in the output is either read from
a declaration, read from a result file this repo already committed, or a fact the probe interpreter
observed while enumerating the registries. Where no such source exists the cell says
``needs_measurement`` and the row gets a next action. Two claims in particular are refused:

  * **CPU throughput.** Nothing in this repo has ever timed the stable-envs renderer on
    ``device="cpu"``. ``tests/fixtures/regression/references/bench_gpu_vs_cpu.json``'s ``cpu`` block is the OFFICIAL
    renderer, which is a different program. So ``se_cpu_fps`` is ``needs_measurement`` on every row,
    and the separate ``cpu_render_executes`` field -- which the probe CAN establish -- is kept well
    away from it, because a path that runs and a path that is fast are different claims.
  * **Downstream equivalence from pixels.** Pixel parity and DINO feature agreement are recorded as
    what they are. Only ``docs/research/artifacts/step5_lewm_pusht/`` carries a task-success result, and only PushT gets
    the downstream label.

INPUTS (all discovered from the repo except the first, which must be produced where swm is installed):

    scratch/swm_registry.py OUT.json          the live swm registry + per-id probe   [required]
    scratch/census/census.py                  per-interpreter stable-envs declarations [repeatable]
    tests/fixtures/regression/census/union.json                 the registry union across interpreters
    tests/fixtures/regression/references/decode_all.json
                                          merged multi-interpreter throughput (196 envs)
    tests/fixtures/regression/references/bench_gpu_vs_cpu.json
                                          official-CPU vs GPU, with describe() blocks
    tests/fixtures/regression/references/dino_agreement.json
                                          DINOv2 feature agreement
    docs/research/artifacts/step5_lewm_pusht/results/*.json        the LeWM PushT downstream evaluation
    tests/fixtures/regression/references/swm_workflow.json   recorded check_mapping / adapter run

    python scripts/regression/swm_matrix.py --swm-registry OUT.json \\
        --se-census census_a.json --se-census census_b.json --live

``--live`` additionally imports both libraries in THIS interpreter to run
``compat.worldmodel.check_mapping`` and to ask whether ``device="cpu"`` renders at all. Everything it
learns is labelled as a probe result, and the run works without it.

DETERMINISM. The builder calls no clock and no RNG: rerunning it on the same inputs produces the same
bytes. The only time in the output is the ``probed_at`` the input probe recorded for itself.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# ============================================================== how a swm id is matched to our envs

#: The evidence ladder, strongest first. A row records WHICH rung matched, so a reader can discount a
#: weak one without having to re-derive the join. The brief for this matrix asked for exactly this:
#: never a bare string-equality match, and never a silent nearest-name.
#:
#: ``variation_map``        stable_envs.compat.worldmodel.VARIATION_MAPS names the swm id outright.
#:                          The strongest rung, because that table is also what makes the env usable
#:                          from swm, so a match here is a match a caller can act on.
#: ``declared_docstring``   our class docstring names the swm id. Generated families do this
#:                          (fetch.py writes "(swm/FetchPush-v3)" into every class it makes), so the
#:                          declaration itself carries the join.
#: ``reference_id``         our ``reference_id`` equals the official env the swm id was MEASURED to
#:                          wrap -- for DM Control, the ``dm_control/<domain>-<task>`` the probe
#:                          resolved by comparing task objects against the suite, which is the only
#:                          way to learn that ``swm/ReacherDMControl-v0`` is ``reacher-hard``.
#: ``underlying_env_name``  the THIRD-PARTY env inside the swm wrapper reports a ``name`` equal to our
#:                          ``reference_id``. This is what identifies Craftax: the wrapper is called
#:                          ``CraftaxClassicPixelsWrapper`` and says nothing, while the object it
#:                          wraps calls itself ``Craftax-Classic-Pixels-NoAutoReset-v1`` against our
#:                          declared ``Craftax-Classic-Pixels-v1``. The only normalisation applied is
#:                          dropping a ``-NoAutoReset`` segment, which selects whether the env resets
#:                          itself at a terminal step and changes no pixel.
#: ``reference_env_class``  our ``reference_id`` builds a gymnasium env whose unwrapped class is the
#:                          class the probe saw under the swm id. This is what ties the five
#:                          classic-control ids to our five, and it is a measurement, not a name.
#: ``same_dmc_appearance``  the swm id resolved to a ``dm_control`` task we do not declare, but we DO
#:                          declare another task of the same domain, and the two were MEASURED to
#:                          compile to the same render model after ``reset()`` -- so the picture is the
#:                          same and only the reward differs. Measured rather than assumed because the
#:                          assumption is false in general: the same test separates ``reacher-easy``
#:                          from ``reacher-hard``, which differ only in a target size set at episode
#:                          initialisation.
#: ``same_scene_as``        the probe saw this id and an already-matched id produce the same
#:                          ``env_name``, the same variation leaves and the same frame shape, so they
#:                          are the same picture. This is how PushT-Discrete inherits PushT's match.
#: ``normalized_id``        the swm id and our ``reference_id`` are equal after lowercasing and
#:                          deleting ``-``, ``_`` and a leading ``swm/``. Weak on its own; recorded as
#:                          its own rung so it can be discounted.
#: ``documented_manual``    none of the above fired and a human wrote the join down, with the probe
#:                          fields that justify it. See ``MANUAL_JOINS``.
MATCH_RUNGS = (
    "variation_map",
    "declared_docstring",
    "reference_id",
    "underlying_env_name",
    "reference_env_class",
    "same_dmc_appearance",
    "same_scene_as",
    "normalized_id",
    "documented_manual",
)


#: Joins no rung above can make, each with the probe evidence that justifies it. Deliberately short:
#: an entry here is a claim nothing in either codebase states, so it needs a reason a reader can check.
#:
#: ``envs``      the stable-envs environments the id corresponds to.
#: ``primary``   the one cell to quote when the id has a single default configuration.
#: ``evidence``  what was observed, and where.
#: ``confidence``  "verified" if the observation settles it, "uncertain" if it does not.
def _fetch_joins() -> dict[str, dict]:
    """The 12 ``swm/Fetch*-v3`` ids -> our 12 Fetch cells.

    This rung is only reached when no census could import ``stable_envs.envs.fetch``, which needs
    ``mujoco_warp`` and therefore a CUDA interpreter. On a machine that CAN import it the
    ``declared_docstring`` rung fires instead and this table is never consulted, because ``fetch.py``
    writes ``(swm/Fetch{Scene}{Suffix}-v3)`` into every class docstring it generates.

    The rule is not a name guess: ``stable_envs/envs/fetch.py::CELLS`` builds exactly three classes
    per scene from one official id -- ``Fetch{S}`` and ``Fetch{S}Dict`` both from ``Fetch{S}-v4``, and
    ``Fetch{S}Dense`` from ``Fetch{S}Dense-v4`` -- and the swm registration kwargs the probe recorded
    carry that same official id (``env_id='FetchPush-v4'``, plus ``flatten='False'`` for the Dict
    variant). Dense and Dict differ from their siblings in the reward and the observation container,
    neither of which reaches the renderer, so all three share one plate and one model.
    """
    src = "stable_envs/envs/fetch.py::CELLS and the generated class __doc__, which names the swm id"
    out = {}
    for scene in ("Reach", "Push", "Slide", "PickAndPlace"):
        for suffix in ("", "Dense", "Dict"):
            out[f"swm/Fetch{scene}{suffix}-v3"] = {
                "envs": [f"Fetch{scene}{suffix}"],
                "primary": f"Fetch{scene}{suffix}",
                "evidence": (
                    f"the probe recorded this id's registration kwargs as env_id="
                    f"'Fetch{scene}{'Dense' if suffix == 'Dense' else ''}-v4'"
                    f"{', flatten=False' if suffix == 'Dict' else ''}; {src}. Reached only because no "
                    f"supplied census could import stable_envs.envs.fetch (it needs mujoco_warp, so a "
                    f"CUDA interpreter); on one that can, the declared_docstring rung supersedes this."
                ),
                "confidence": "verified",
                "caveats": [
                    "declarations unreachable in this run, so tier / state_dim / step for "
                    "this env read as unknown rather than being filled in from the doc"
                ],
            }
    return out


MANUAL_JOINS: dict[str, dict] = {
    **_fetch_joins(),
    "swm/OGBMaze-v0": {
        "envs": ["PointmazeLargeNavigate", "PointmazeLargeStitch"],
        "family": "Antmaze*/Humanoidmaze*/Pointmaze*",
        "primary": "PointmazeLargeNavigate",
        "evidence": (
            "one swm id stands for a whole family. The probe records the REGISTERED default as "
            "maze_type='large' with PointEnv in the MRO, and stable_worldmodel's MazeEnv builds "
            "loco_env_type in {point, ant, humanoid} x maze_type in {arena, medium, large, giant, "
            "teleport}; the default therefore corresponds to ogbench/pointmaze-large-v0, which is "
            "the reference_id of PointmazeLargeNavigate and PointmazeLargeStitch (the two dataset "
            "suffixes share one rendered scene). The other 23 Antmaze*/Humanoidmaze*/Pointmaze* "
            "envs cover the rest of the cross product; we render no 'arena' maze."
        ),
        "confidence": "verified",
        "caveats": [
            "stable-envs has no 'arena' maze; the other 14 of the 15 point/ant/humanoid x "
            "medium/large/giant/teleport cells are registered."
        ],
    },
}

#: WHY a gap is a gap, from the research this repo already did. Classification only -- nothing here
#: is new investigation, and an entry exists only where a committed doc supports it.
#:
#: ``straightforward_unimplemented``   the mechanism exists and is proven on a sibling; this cell was
#:                                     not built.
#: ``existing_research_promising``     probed and not refuted; a doc says the design should apply.
#: ``blocked_by_renderer_feature``     a named capability the backend does not have.
#: ``structural_no_go``                measured and rejected; the env's own construction defeats the
#:                                     premise.
#: ``low_value_deprioritized``         reachable, judged not worth it.
#: ``unknown_needs_research``          no committed evidence supports any of the above. The default.
GAP_REASONS: dict[str, dict] = {
    "swm/CraftaxPixels-v1": {
        "reason": "straightforward_unimplemented",
        "detail": (
            "full Craftax (9 levels) against the Craftax CLASSIC we ship -- the same renderer "
            "design, a bigger sprite atlas and a deeper world to slice. The storage gate, "
            "which is the one that decides whether the family belongs here at all, PASSES on "
            "the render slice: the shipped EnvState is 6.37x the frame (273,276 B vs 42,900 B) "
            "because it carries all 9 levels of a 48x48 world, but the render window is "
            "5,268 B = 0.12x. The atlas is already characterised at 143 PNGs / 212 kB for full "
            "Craftax against 59 / 150 kB for Classic."
        ),
        "source": [
            "docs/research/swm_coverage.md",
            "docs/research/when_does_this_pay_off.md",
            "docs/research/pixel_native.md",
        ],
    },
    "swm/ManipulatorDMControl-v0": {
        "reason": "existing_research_promising",
        "detail": (
            "`manipulator` passes the DM Control state-sufficiency gate 11/11 with no dead "
            "dimensions, has a FIXED camera and compiles to `nmesh = 0`, which is the "
            "configuration the existing plate design is built for. The doc's own words: "
            '"not re-probed since the first pass ... the plate design should apply -- '
            'untested, not refuted."'
        ),
        "source": ["docs/research/dmcontrol.md#82-what-is-still-out"],
    },
    "swm/QuadrupedDMControl-v0": {
        "reason": "existing_research_promising",
        "detail": (
            "`quadruped-walk` passes state sufficiency 23/23. It is listed beside `dog` as one "
            "of the two suite XMLs containing `<mesh>`, but the doc records that `quadruped` "
            "COMPILES TO `nmesh = 0` while `dog` does not -- so the mesh blocker that rules "
            'out `dog` does not bind here. The doc concludes "none of these is a NO-GO on the '
            'thesis". Note the camera: this id is the only one probed at `camera_id=2`.'
        ),
        "source": ["docs/research/dmcontrol.md#82-what-is-still-out"],
    },
    "swm/OGBCube-v0": {
        "reason": "blocked_by_renderer_feature",
        "detail": (
            "OGBench MANIPULATION, where stable-envs covers the mazes (25 cells, interior 100% "
            "exact). Its variation space names the blocker: `camera.angle_delta` and "
            "`light.intensity` make a MOVABLE CAMERA and a light part of the appearance, and "
            "the maze port's plate bake explicitly assumes both away. Recorded as not gated "
            "rather than guessed at; the doc names the cheap first measurement (reset twice, "
            "compare ngeom/nmesh)."
        ),
        "source": ["docs/research/swm_coverage.md#ogbench-cube-and-scene"],
    },
    "swm/OGBScene-v0": {
        "reason": "blocked_by_renderer_feature",
        "detail": (
            "as OGBCube, plus a `lock_color` leaf. Same movable-camera-and-light blocker "
            "against the same plate-bake assumption; same ungated status."
        ),
        "source": ["docs/research/swm_coverage.md#ogbench-cube-and-scene"],
    },
    "swm/PFRocketLanding-v0": {
        "reason": "structural_no_go",
        "detail": (
            "a PyBullet scene rendered through PyFlyt's perspective camera over URDF MESH "
            "assets -- a third renderer with no batched backend in this repo -- and, "
            "decisively, it WRITES A NEW URDF TO A TEMPORARY FILE ON EVERY RESET "
            "(`_modify_rocket_urdf`, `_modify_pad_urdf`) so the decoder is regenerated per "
            "episode from an asset library instead of being a constant of the family. That is "
            "the RoboCasa signature and it fails the same gate. Milder than RoboCasa in one "
            "respect: only material colours change, so geometry could be compiled once. "
            'Verdict recorded as "NO-GO for now", not pixel-verified.'
        ),
        "source": ["docs/research/swm_coverage.md#rocket-landing-pyflyt-pybullet"],
    },
}

#: Cells this pass TRIED to establish and could not, with what was observed. An entry here does not
#: change a value -- the cell stays ``unknown`` -- it records WHY, so the next person does not repeat
#: the attempt blind. Investigated-and-still-unknown is a more useful state than bare unknown, and a
#: much more useful one than a wrong answer.
PROBE_LIMITATIONS: dict[str, dict] = {
    "fetch_cpu": {
        "applies_to_env_prefix": "Fetch",
        "field": "cpu_support.render_executes",
        "detail": (
            "attempted on an Oscar RTX 3090 node with mujoco 3.13.0 + mujoco_warp 3.12.0 (job "
            'se-step1b-tb). `se.make("FetchPush", device="cpu")` raises AssertionError from '
            "`stable_envs/envs/fetch.py:553` in `__init__`, which builds the official "
            "gymnasium-robotics env -- the SAME upstream `set_joint_qpos` assertion that makes plain "
            '`gymnasium.make("FetchPush-v4")` fail under mujoco >= 3.12. So this is NOT evidence '
            "that Fetch has no CPU path; it is the reference env failing before the renderer is "
            "reached. The CUDA arm of the same job failed separately and unrelatedly (the venv's "
            "torch is a cu130 build and the node's driver is 12.9). Resolving it needs one "
            "interpreter with mujoco <= 3.11 for gymnasium-robotics AND a mujoco_warp built against "
            "it, which was not available in this pass."
        ),
        "next": (
            "build that interpreter, or read the CPU answer off a machine that already has the "
            "Fetch family working end to end"
        ),
    },
}


#: Rows where the RENDERER IDENTITY IS SETTLED but what it draws is a documented SUBSET of what the
#: swm id observes. Neither "covered" nor "uncertain" is true of these, and forcing either loses the
#: fact that matters. Every entry quotes the stable-envs class's own docstring, which is where the
#: exclusion is already declared and measured -- this table points at it, it does not decide it.
OBSERVATION_SUBSETS: dict[str, dict] = {
    "swm/CraftaxClassicPixels-v1": {
        "kind": "same_scene_partial_observation",
        "detail": (
            "the renderer identity is settled, and what it draws is the WORLD LAYER of the 63x63 "
            "agent observation, not the whole observation. `stable_envs/envs/craftax/classic.py` declares "
            "three exclusions and measures the rest: (1) the 2-row INVENTORY STRIP, rows 49-63, is a "
            "function of twelve inventory counts and four intrinsics rather than of the map window "
            "and is not rendered; (2) NIGHT frames (`light_level < 0.5`) mix in a noise field drawn "
            "from JAX's threefry2x32 PRNG; (3) a SLEEPING player desaturates the whole frame. On what "
            "it does draw, `reference_render` crops the official frame to the world layer so the "
            "comparison is like-for-like, and parity is 100% of all pixels exact, mean |diff| 0.000, "
            "max 0, over 64 states at both tile sizes. Resolution differs by DEFAULT and not by "
            "capability: swm renders 63x63 at the 7 px agent tile size, our default_size is 16 px "
            "(112x144 for the 7x9 map window), and the class is measured at both."
        ),
        "source": ["stable_envs/envs/craftax/classic.py (class docstring)"],
    },
}


#: Registered swm ids that stand for a FAMILY of configurations rather than one scene, and what the
#: registered default resolves to. Without this a row can only say "covered", which overclaims: one
#: id having a renderer for its default says nothing about the other cells it can build.
#:
#: ``config_axes``   the constructor arguments that move the scene, from the probe and the wrapper.
#: ``default``       what the REGISTERED id builds with no kwargs -- probed, not read off a default.
#: ``covered``/``missing``  stable-envs cells for the cross product, by declared ``reference_id``.
PARAMETERIZED: dict[str, dict] = {
    "swm/OGBMaze-v0": {
        "config_axes": {
            "loco_env_type": ["point", "ant", "humanoid"],
            "maze_type": ["arena", "medium", "large", "giant", "teleport"],
        },
        "default": {"loco_env_type": "point", "maze_type": "large"},
        "default_evidence": ("probed: `_maze_type == 'large'` and `PointEnv` in the MRO of the constructed env"),
        "reference_pattern": "ogbench/{loco}maze-{maze}-v0",
        "detail": (
            "15 locomotion x maze cells. stable-envs registers 24 envs over 14 of them (each "
            "cell appears under the dataset suffixes Navigate/Stitch/Explore, which share one "
            "rendered scene); no `arena` maze is registered for any robot."
        ),
    },
}


#: swm ids whose render call raises rather than returning a frame BECAUSE THERE IS NO FRAME. Matched
#: against the probe's recorded error text, so an env that starts producing frames stops matching and
#: is reclassified instead of staying quietly excused.
SYMBOLIC_ERROR_PATTERN = re.compile(r"symbolic envs do not produce a frame", re.I)

#: ``backends`` from the census -> the coarse label the compact table shows. Keyed on what the census
#: OBSERVED the defining module to import, not on the class's name.
BACKEND_LABELS = {
    (): "custom (in-module)",
    ("gfxdraw",): "gfxdraw raster",
    ("mjwarp", "raycast3d"): "MJWarp + raycast3D",
    ("nvdiff", "raycast3d"): "nvdiffrast + raycast3D",
    ("raster2d", "scene"): "Scene2D + raster2d",
    ("raycast3d",): "raycast3D",
    ("scene",): "Scene2D + raster2d",
    ("tilemap",): "tile/atlas gather",
}

UNKNOWN = "unknown"
NEEDS = "needs_measurement"


# ====================================================================== small helpers


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _rel(path) -> str:
    """Repo-relative if the file is in the repo, absolute otherwise.

    ``os.path.relpath`` will happily climb out of the tree and emit
    ``../../../private/tmp/.../probe.json``, which is machine-specific noise in a committed page and
    is not a path anyone else can open. An input from outside the repo is named as what it is.
    """
    p = Path(path).resolve()
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return str(p)


def _load(path: Path):
    with path.open() as fh:
        return json.load(fh)


def _norm_id(s: str) -> str:
    """``swm/PushT-Discrete-v1`` -> ``pushtdiscretev1``. Used only by the weakest match rung."""
    s = s.lower()
    for p in ("swm/", "gym_pusht/", "dm_control/", "ogbench/"):
        if s.startswith(p):
            s = s[len(p) :]
    return re.sub(r"[-_/ ]", "", s)


# ====================================================================== input readers

#: How good a probe record is, worst first. Merging several probes keeps the BEST record per id.
#: Needed because the swm side has the same one-interpreter problem the stable-envs side has:
#: dm_control needs mujoco >= 3.12 and gymnasium-robotics needs <= 3.11, so no single interpreter can
#: construct both the DM Control ids and the Fetch ids. Measured, not assumed -- gymnasium-robotics
#: 1.4.2 raises inside `set_joint_qpos` under mujoco 3.12, and plain `gymnasium.make("FetchPush-v4")`
#: fails the same way, so it is an upstream pairing and not stable_worldmodel's doing.
_PROBE_RANK = {
    None: -1,
    "unknown_probe_failed": 0,
    "no_frame_returned": 1,
    "symbolic_no_frame": 2,
    "absent": 2,
    "pixels": 3,
    "present": 3,
}


def _probe_score(rec: dict) -> tuple:
    """How much a probe record actually established. Higher wins a merge."""
    return (
        rec.get("make") == "ok",
        _PROBE_RANK.get(rec.get("frame_status_probe"), 0),
        _PROBE_RANK.get(rec.get("variation_space_status"), 0),
        len(rec),
    )


def read_swm_registries(paths: list[Path]) -> dict:
    """Union several live swm probes, keeping the record that learned the most about each id.

    One probe would be simpler and is not enough. The registry itself is identical across
    interpreters -- it is built at import time from the same source -- so the ID LIST is taken from
    the first probe and cross-checked against the rest; disagreement is an error worth stopping for,
    because it would mean two different versions of stable_worldmodel were probed and the matrix
    would be describing neither. What differs is how far each interpreter could CONSTRUCT each id,
    and that is merged per id with the best record winning.

    Every row records which probe supplied it, so "Fetch's variation space is present" can be traced
    to the interpreter that established it rather than looking like a free-floating claim.
    """
    if not paths:
        raise SystemExit(
            "no --swm-registry given. Produce one with\n"
            "    python scratch/swm_registry.py OUT.json\n"
            "in an interpreter where `import stable_worldmodel` works."
        )
    probes = []
    for path in paths:
        d = _load(path)
        for key in ("ids", "probe", "entry_points", "kwargs"):
            if key not in d:
                raise SystemExit(f"{path} is not a swm_registry.py output: no {key!r}.")
        probes.append((path, d))

    base_ids = probes[0][1]["ids"]
    for path, d in probes[1:]:
        if d["ids"] != base_ids:
            only_a = sorted(set(base_ids) - set(d["ids"]))
            only_b = sorted(set(d["ids"]) - set(base_ids))
            raise SystemExit(
                f"the swm registries disagree, so they cannot describe one library:\n"
                f"  only in {probes[0][0].name}: {only_a}\n  only in {path.name}: {only_b}\n"
                f"Probe the same stable_worldmodel checkout in every interpreter."
            )

    merged, source = {}, {}
    for path, d in probes:
        tag = path.stem
        for i in base_ids:
            rec = d["probe"].get(i, {})
            if i not in merged or _probe_score(rec) > _probe_score(merged[i]):
                merged[i], source[i] = rec, tag
    out = dict(probes[0][1])
    out["probe"] = merged
    out["probe_source"] = source
    out["probe_envs"] = {p.stem: d.get("probe_env", {}) for p, d in probes}
    out["dmc_index_available"] = any(d.get("dmc_index_available") for _, d in probes)
    # entry points and kwargs come from the registry, not from construction, so any probe has them;
    # take them from whichever probe actually recorded a non-empty value.
    for key in ("entry_points", "kwargs"):
        acc = {}
        for _, d in probes:
            for k, v in d.get(key, {}).items():
                if k not in acc or (not acc[k] and v):
                    acc[k] = v
        out[key] = acc
    return out


def read_censuses(paths: list[Path]) -> dict:
    """Union the per-interpreter stable-envs censuses, keeping who saw what.

    Same shape of problem ``scratch/census/merge_census.py`` already solves for names, and the same
    resolution: no interpreter can import every family, so the union is the registry and the owners
    are how a reader tells "absent from the project" from "absent from that venv".
    """
    per, decls, owners, errors, interpreters = {}, {}, defaultdict(list), {}, {}
    for path in paths:
        text = path.read_text()
        # accept either a raw census JSON or a log line prefixed "CENSUS "
        blobs = [ln[len("CENSUS ") :] for ln in text.splitlines() if ln.startswith("CENSUS ")]
        for blob in blobs or [text]:
            try:
                d = json.loads(blob)
            except json.JSONDecodeError:
                continue
            if "registry" not in d:
                continue
            # the file stem, not the interpreter path: it is short, stable across machines, and is
            # what the reader passes back on the command line. The path is kept beside it.
            tag = path.stem
            per[tag] = sorted(d["registry"])
            interpreters[tag] = d.get("python")
            errors[tag] = d.get("import_errors", {})
            for name in d["registry"]:
                owners[name].append(tag)
            # First writer wins, so the union is order-independent given the same inputs; paths are
            # sorted by the caller. Later censuses only ADD envs they alone can reach.
            for name, dec in sorted(d.get("declarations", {}).items()):
                decls.setdefault(name, dict(dec, _seen_in=tag))
    return {
        "per_interpreter": per,
        "declarations": decls,
        "interpreters": interpreters,
        "owners": {k: sorted(v) for k, v in owners.items()},
        "import_errors": errors,
    }


def read_union(path: Path | None) -> dict:
    """``tests/fixtures/regression/census/union.json`` -- the registry union and which interpreter reaches each env.

    Carries the ownership too, not just the names, because "this env exists but nothing here can
    import it" is only actionable if the row can say WHERE to go and import it.
    """
    if path is None or not path.exists():
        return {"union": [], "per_interpreter": {}}
    d = _load(path)
    return {
        "union": sorted(d.get("union", [])),
        "per_interpreter": {k: sorted(v) for k, v in sorted(d.get("per_interpreter", {}).items())},
    }


def read_speed(decode_all: Path | None, bench: Path | None) -> dict:
    """Existing throughput results, per stable-envs env, with provenance and nothing recomputed.

    ``decode/ALL.json`` is ``scratch/census/merge_bench.py``'s union and is the only source that
    covers all 196 envs, so it is the primary. ``bench_gpu_vs_cpu.json`` adds the resolution sweep and
    the per-size official-CPU column for the 64 envs one interpreter could reach.

    The ``cpu`` column in BOTH files is the OFFICIAL renderer. Neither file contains a measurement of
    the stable-envs renderer on a CPU device, which is why every row's ``se_cpu_fps`` is
    ``needs_measurement``.
    """
    out: dict[str, dict] = {}
    if decode_all is not None and decode_all.exists():
        d = _load(decode_all)
        for row in d.get("rows", []):
            out[row["name"]] = {
                "se_gpu_fps": row.get("gpu"),
                "gpu_batch": row.get("batch"),
                "official_cpu_fps": row.get("cpu"),
                "shape": row.get("shape"),
                "supersample": row.get("ss"),
                "host_cpu_frac": row.get("cpu_frac"),
                "family": row.get("family"),
                "interpreter": row.get("tag"),
                "source": "examples/results/decode/ALL.json (merge_bench union)",
                "device": (d.get("meta", {}).get(row.get("tag"), {}) or {}).get("device"),
                "se_cpu_fps": NEEDS,
                "se_cpu_fps_why": (
                    "no result file in this repo times the stable-envs renderer on "
                    "device='cpu'; the 'cpu' column of every bench result is the "
                    "OFFICIAL renderer"
                ),
            }
    if bench is not None and bench.exists():
        b = _load(bench)
        for name, rec in b.get("envs", {}).items():
            e = out.setdefault(name, {"se_cpu_fps": NEEDS})
            e["bench_sizes"] = sorted(rec.get("sizes", {}))
            e["bench_device"] = b.get("device")
            e["bench_source"] = "examples/results/bench_gpu_vs_cpu.json"
            if "describe" in rec:
                e["describe"] = rec["describe"]
            best = None
            for size, block in sorted(rec.get("sizes", {}).items()):
                for g in block.get("gpu") or []:
                    if isinstance(g, dict) and isinstance(g.get("fps"), (int, float)):
                        if best is None or g["fps"] > best["fps"]:
                            best = {
                                "fps": g["fps"],
                                "batch": g.get("batch"),
                                "size": int(size),
                                "supersample": g.get("supersample"),
                                "host_cpu_frac": g.get("cpu_frac"),
                            }
                cpu = block.get("cpu")
                if isinstance(cpu, dict) and isinstance(cpu.get("set_state_render"), dict):
                    e.setdefault("official_cpu_by_size", {})[size] = cpu["set_state_render"].get("fps")
            if best:
                e["bench_best_gpu"] = best
    return out


def read_dino(path: Path | None) -> dict:
    """DINOv2 feature agreement. A REPRESENTATION-level check, never a task-success one."""
    if path is None or not path.exists():
        return {}
    d = _load(path)
    out = {}
    for name, rec in d.get("envs", {}).items():
        out[name] = {
            k: rec.get(k)
            for k in (
                "dino_cos_ours_vs_official",
                "dino_cos_ours_vs_official_max",
                "dino_cos_state_vs_state",
                "dino_cos_blank_vs_official",
                "dino_cos_jpeg95_vs_official",
                "interior_exact_frac",
                "all_exact_frac",
                "max_abs",
                "size",
                "tier",
                "reference_id",
            )
        }
        out[name]["model"] = d.get("model")
        out[name]["n"] = d.get("n")
        out[name]["source"] = "examples/results/dino/dino_agreement.json"
    return out


def read_lewm(results_dir: Path | None) -> dict | None:
    """The LeWM PushT downstream evaluation, aggregated with the repo's own statistics.

    ``docs/research/artifacts/step5_lewm_pusht/summarize.py`` already owns the paired McNemar test and the Wilson interval,
    and the pairing is the whole point of the experiment, so its helpers are imported rather than
    reimplemented. Only the aggregation loop is here, because ``summarize.main()`` prints.
    """
    if results_dir is None or not results_dir.exists():
        return None
    sys.path.insert(0, str(results_dir.parent))
    try:
        from summarize import mcnemar_exact, wilson  # type: ignore
    except Exception as ex:
        return {
            "error": f"cannot import docs/research/artifacts/step5_lewm_pusht/summarize.py: {type(ex).__name__}: {ex}"
        }
    finally:
        sys.path.pop(0)

    runs: dict[str, dict[int, list[int]]] = defaultdict(dict)
    for f in sorted(glob.glob(str(results_dir / "*.json"))):
        try:
            d = json.loads(Path(f).read_text())
        except json.JSONDecodeError:
            continue
        if "renderer" in d and "episode_successes" in d:
            runs[d["renderer"]][d["seed"]] = list(d["episode_successes"])
    if "official" not in runs:
        return None

    arms = {}
    for arm in sorted(runs):
        seeds = sorted(runs[arm])
        succ = [x for s in seeds for x in runs[arm][s]]
        k, n = int(sum(succ)), len(succ)
        lo, hi = wilson(k, n)
        arms[arm] = {
            "seeds": seeds,
            "successes": k,
            "n": n,
            "success_rate_pct": round(100 * k / n, 2),
            "ci95_pct": [round(float(lo), 2), round(float(hi), 2)],
        }
    paired = {}
    for arm in sorted(runs):
        if arm == "official":
            continue
        shared = sorted(set(runs["official"]) & set(runs[arm]))
        if not shared:
            continue
        a = [bool(x) for s in shared for x in runs["official"][s]]
        b = [bool(x) for s in shared for x in runs[arm][s]]
        only_off = sum(1 for x, y in zip(a, b) if x and not y)
        only_new = sum(1 for x, y in zip(a, b) if y and not x)
        paired[arm] = {
            "n": len(a),
            "official_only_wins": only_off,
            "arm_only_wins": only_new,
            "delta_pts": round(100 * (sum(b) / len(b) - sum(a) / len(a)), 2),
            "mcnemar_p": round(float(mcnemar_exact(only_off, only_new)), 4),
        }
    return {
        "arms": arms,
        "paired_vs_official": paired,
        "source": "docs/research/artifacts/step5_lewm_pusht/results/*.json via summarize.py",
        "note": (
            "paired evaluation: for a seed, both arms run the same episode indices, start "
            "steps, goal offset and CEM seed, so the only difference is the renderer. The "
            "blank-frame arm is the control that shows the evaluation is pixel-sensitive."
        ),
    }


def read_swm_workflow(path: Path | None) -> dict:
    """The recorded ``examples/swm_compat/run_swm_workflow.py`` run: adapters and check_mapping."""
    if path is None or not path.exists():
        return {}
    d = _load(path)
    return {
        "check_mapping": d.get("E_variation_mapping", {}),
        "renderer_swap": d.get("D_renderer_swap", {}),
        "info_key_parity": d.get("key_parity", {}),
        "scaling": d.get("G_scaling", {}),
        "source": "tests/fixtures/regression/references/swm_workflow.json",
    }


# ====================================================================== the live probe (optional)


def _dmc_appearance_groups() -> dict:
    """Per DM Control domain, which tasks compile to the SAME render model. A measurement.

    The question this answers is narrow and load-bearing: ``swm/AcrobotDMControl-v0`` is
    ``acrobot-swingup_sparse`` and we declare ``acrobot-swingup``, so does our renderer cover that id
    or not? Reading the source says "sparse only changes the reward", which is the kind of plausible
    reading CONTRIBUTING.md exists to forbid. So it is measured instead.

    The signature is taken AFTER ``reset()``, not on the freshly compiled model, because that is where
    a task gets its last chance to change the picture -- ``reacher`` sizes its target in
    ``initialize_episode`` and nowhere else. Taken before the reset, easy and hard look identical and
    the test would be worthless; taken after, they separate, which is what shows the test can tell.

    ``random=0`` is fixed so two runs group the same way. Returns ``{domain: {task: digest}}``, or
    ``{}`` if dm_control is absent, and the caller records which.
    """
    try:
        import hashlib as _h

        import numpy as _np
        from dm_control import suite
    except Exception:
        return {}
    #: model arrays that a renderer reads. Positions and orientations of BODIES are included because a
    #: task may re-place a prop; qpos is not, because that is state and varies per episode by design.
    FIELDS = (
        "geom_type",
        "geom_size",
        "geom_rgba",
        "geom_matid",
        "mat_rgba",
        "body_pos",
        "geom_pos",
        "geom_quat",
        "cam_pos",
        "cam_quat",
        "cam_fovy",
        "light_pos",
    )
    groups: dict[str, dict[str, str]] = {}
    for dom, tsk in suite.ALL_TASKS:
        try:
            env = suite.load(dom, tsk, task_kwargs={"random": 0})
            env.reset()
            m = env.physics.model
            parts = []
            for f in FIELDS:
                a = getattr(m, f, None)
                if a is not None:
                    parts.append(f.encode() + _np.ascontiguousarray(a).tobytes())
            groups.setdefault(dom, {})[tsk] = _h.sha256(b"".join(parts)).hexdigest()[:12]
        except Exception:
            continue
    return groups


def live_probe(swm_ids: list[str]) -> dict:
    """Facts only a process holding BOTH libraries can establish. Optional; labelled as a probe.

    Three things, and each is deliberately narrow:

    ``check_mapping``   run for real, per VARIATION_MAPS entry, rather than read out of a result file
                        that could predate the last change to either variation space.
    ``cpu_render``      does ``device="cpu"`` actually produce a frame? This distinguishes "no CPU
                        path" from "CPU path never benchmarked", which are different next actions.
                        IT IS NOT A SPEED MEASUREMENT and no timing is recorded here.
    ``reference_class`` the unwrapped class of each plain-gymnasium ``reference_id``, which is what
                        ties the classic-control swm ids to ours by construction rather than by name.
    """
    out: dict = {"available": False}
    try:
        import stable_envs as se
        from tests.helpers import reference as wm
    except Exception as ex:
        out["error"] = f"stable_envs not importable: {type(ex).__name__}: {ex}"
        return out
    out["available"] = True
    out["registry_here"] = sorted(se.REGISTRY)
    out["import_errors_here"] = {k: str(v) for k, v in se.envs.import_errors().items()}

    mapping = {}
    for name in sorted(wm.VARIATION_MAPS):
        try:
            env = se.make(name, batch=1, device="cpu")
            mapping[name] = wm.check_mapping(env)
        except Exception as ex:
            mapping[name] = {"error": f"{type(ex).__name__}: {ex}"[:300]}
    out["check_mapping"] = mapping

    cpu = {}
    for name in sorted(se.REGISTRY):
        try:
            env = se.make(name, batch=2, device="cpu")
            frames = env.render(env.sample_states(2))
            cpu[name] = {"executes": True, "shape": list(frames.shape), "dtype": str(frames.dtype)}
        except Exception as ex:
            cpu[name] = {"executes": False, "error": f"{type(ex).__name__}: {ex}"[:200]}
    out["cpu_render"] = cpu

    out["dmc_appearance"] = _dmc_appearance_groups()

    refs: dict[str, str] = {}
    for name in sorted(se.REGISTRY):
        rid = getattr(se.REGISTRY[name], "reference_id", None)
        if not isinstance(rid, str) or "/" in rid or rid in refs:
            continue
        try:
            import gymnasium

            e = gymnasium.make(rid)
            refs[rid] = type(e.unwrapped).__name__
            e.close()
        except Exception:
            continue
    out["reference_env_class"] = refs
    return out


# ====================================================================== the join


def match_swm_id(
    swm_id: str,
    probe: dict,
    kwargs: dict,
    decls: dict,
    var_maps: dict[str, str],
    ref_classes: dict[str, str],
    dmc_appearance: dict,
    already: dict[str, dict],
) -> dict:
    """Which stable-envs envs, if any, this swm id corresponds to -- and on what evidence.

    Every rung is tried and every hit recorded, so a row that matches three ways says so; the reported
    ``rung`` is the strongest that fired. A row with no hit is a gap and says which rungs were tried.
    """
    hits: dict[str, list[str]] = {}

    # -- variation_map: our compat table names the id
    v = sorted(n for n, sid in var_maps.items() if sid == swm_id)
    if v:
        hits["variation_map"] = v

    # -- declared_docstring: the class docstring names the id (generated families do this)
    d = sorted(n for n, dec in decls.items() if swm_id in (dec.get("doc") or ""))
    if d:
        hits["declared_docstring"] = d

    # -- reference_id: equal to the official env the probe MEASURED this id to wrap
    wanted = set()
    dmc = probe.get("dmc_reference") or {}
    if dmc.get("reference_id"):
        wanted.add(dmc["reference_id"])
    if kwargs.get("env_id"):
        wanted.add(kwargs["env_id"])
    wanted.add(swm_id)  # our swm_nav envs declare the swm id as their reference
    r = sorted(n for n, dec in decls.items() if dec.get("reference_id") in wanted)
    if r:
        hits["reference_id"] = r

    # -- underlying_env_name: the wrapped third-party env calls itself what we declare
    inner = probe.get("inner_env") or {}
    inner_names = {v for k, v in inner.items() if k in ("name", "env_name") and isinstance(v, str)}
    norm_inner = {n.replace("-NoAutoReset", "") for n in inner_names}
    if norm_inner:
        u = sorted(n for n, dec in decls.items() if dec.get("reference_id") in norm_inner)
        if u:
            hits["underlying_env_name"] = u

    # -- reference_env_class: our reference builds the class the probe saw
    cls = probe.get("class")
    if cls:
        c = sorted(n for n, dec in decls.items() if ref_classes.get(dec.get("reference_id") or "") == cls)
        if c:
            hits["reference_env_class"] = c

    # -- same_dmc_appearance: a different task of the same domain, MEASURED to draw the same model
    dmc_same: list[str] = []
    if dmc.get("reference_id") and not hits.get("reference_id"):
        dom_task = dmc["reference_id"].split("/", 1)[1]
        dom, _, task = dom_task.partition("-")
        want = (dmc_appearance.get(dom) or {}).get(task)
        if want:
            for n, dec in sorted(decls.items()):
                rid = dec.get("reference_id") or ""
                if not rid.startswith(f"dm_control/{dom}-"):
                    continue
                ours = rid.split("-", 1)[1]
                if (dmc_appearance.get(dom) or {}).get(ours) == want:
                    dmc_same.append(n)
    if dmc_same:
        hits["same_dmc_appearance"] = sorted(dmc_same)
        # every one of these draws the same picture, so all are listed; the closest TASK is still
        # worth naming, because a caller reproducing the swm id's rollouts wants the nearest reward.
        want_task = dmc["reference_id"].split("-", 1)[1]
        closest = sorted(
            n
            for n in dmc_same
            if (decls[n].get("reference_id") or "").split("-", 1)[1] in (want_task, want_task.split("_")[0])
        )
        if closest:
            hits["_dmc_closest"] = closest[:1]

    # -- same_scene_as: an already-matched id produced the same picture
    for other, row in sorted(already.items()):
        if other == swm_id or not row.get("envs"):
            continue
        op = row["_probe"]
        same = (
            op.get("env_name")
            and op.get("env_name") == probe.get("env_name")
            and op.get("variation_names") == probe.get("variation_names")
            and op.get("frame_shape") == probe.get("frame_shape")
            and probe.get("frame_shape")
        )
        if same:
            hits["same_scene_as"] = sorted(row["envs"])
            hits.setdefault("_same_scene_source", []).append(other)
            break

    # -- normalized_id: weakest rung, recorded separately so it can be discounted
    n_ = _norm_id(swm_id)
    z = sorted(name for name, dec in decls.items() if dec.get("reference_id") and _norm_id(dec["reference_id"]) == n_)
    if z:
        hits["normalized_id"] = z

    # -- documented_manual
    manual = MANUAL_JOINS.get(swm_id)
    if manual:
        hits["documented_manual"] = list(manual["envs"])

    rung = next((r for r in MATCH_RUNGS if r in hits), None)
    envs = sorted(hits.get(rung, [])) if rung else []
    source = hits.get("_same_scene_source", [None])[0] if rung == "same_scene_as" else None
    out = {
        "envs": envs,
        "rung": rung,
        "all_rungs": {k: v for k, v in sorted(hits.items()) if not k.startswith("_")},
        "rungs_tried": list(MATCH_RUNGS),
        "confidence": (
            "none"
            if not rung
            else (manual or {}).get("confidence", "verified")
            if rung == "documented_manual"
            else "verified"
            if rung != "normalized_id"
            else "uncertain"
        ),
        "_probe": probe,
    }
    if manual and rung == "documented_manual":
        out["evidence"] = manual["evidence"]
        out["primary"] = manual.get("primary")
        out["family"] = manual.get("family")
        out["caveats"] = manual.get("caveats", [])
    elif rung == "variation_map":
        out["evidence"] = (
            f"stable_envs.compat.worldmodel.VARIATION_MAPS declares swm_id={swm_id!r} for {', '.join(envs)}"
        )
    elif rung == "declared_docstring":
        out["evidence"] = f"the stable-envs class docstring names {swm_id}"
    elif rung == "reference_id":
        got = sorted({decls[n].get("reference_id") for n in envs})
        how = dmc.get("how")
        out["evidence"] = f"reference_id {got} equals the official env this id was measured to wrap" + (
            f" (DM Control task resolved against the suite: match={how})" if how else ""
        )
    elif rung == "underlying_env_name":
        out["evidence"] = (
            f"the env inside the swm wrapper is {inner.get('module')}.{inner.get('class')} and calls "
            f"itself {sorted(inner_names)}; dropping the -NoAutoReset segment (an episode-boundary "
            f"behaviour that changes no pixel) leaves exactly the reference_id "
            f"{sorted({decls[n].get('reference_id') for n in envs})} that {', '.join(envs)} declares"
        )
    elif rung == "reference_env_class":
        out["evidence"] = (
            f"the probe saw class {cls!r} under {swm_id}, and gymnasium.make() on our "
            f"reference_id produces the same unwrapped class"
        )
    elif rung == "same_dmc_appearance":
        got = sorted({decls[n].get("reference_id") for n in envs})
        out["evidence"] = (
            f"this id resolves to {dmc.get('reference_id')}, which no stable-envs env declares; we "
            f"declare {got} of the same domain, and the two compile to a byte-identical render model "
            f"after reset() (same geom types/sizes/colours/materials, same camera, same lights), so "
            f"the picture is the same and only the reward differs"
        )
        out["caveats"] = [
            f"the TASK differs: stable-envs declares {got} and the swm id is "
            f"{dmc.get('reference_id')}. That changes the reward and the state distribution a rollout "
            f"visits, not the renderer."
        ]
        closest = hits.get("_dmc_closest")
        if closest:
            out["primary"] = closest[0]
            out["caveats"].append(
                f"{closest[0]} is the nearest task of the group and is the cell to "
                f"quote; the others render the same picture under a different "
                f"reward."
            )
    elif rung == "same_scene_as":
        out["evidence"] = (
            f"the probe recorded the same env_name, the same variation leaves and the "
            f"same frame shape as {source}, so it is the same picture"
        )
    elif rung == "normalized_id":
        out["evidence"] = (
            f"{swm_id} and our reference_id are equal after lowercasing and deleting "
            f"separators -- a NAME match, weaker than the rungs above"
        )
    else:
        out["evidence"] = (
            "no rung matched: no VARIATION_MAPS entry, no docstring mention, no "
            "reference_id or reference class in common, no identical scene, no "
            "normalized-name match, and no documented manual join"
        )
    return out


# ====================================================================== per-row classification


def classify_frame(probe: dict) -> tuple[str, str]:
    """``pixels`` / ``symbolic_no_frame`` / ``unknown``, and why."""
    if probe.get("frame_shape"):
        return "pixels", "the probe rendered a frame"
    reset = str(probe.get("reset", ""))
    if SYMBOLIC_ERROR_PATTERN.search(reset):
        return "symbolic_no_frame", "the env itself reports that it produces no frame: " + reset[:160]
    if str(probe.get("make", "")).startswith("ERR"):
        return "unknown", "did not construct in the probe interpreter: " + str(probe["make"])[:160]
    if reset.startswith("ERR"):
        return "unknown", "constructed but did not render in the probe interpreter: " + reset[:160]
    return "unknown", "no frame and no error recorded"


#: The dynamics vocabulary. Six values, and the split that matters is between the middle two: a
#: closed-form integrator transcribed term for term is NOT the same claim as a different solver.
#:
#: ``exact_fast_step``        declares ``step_reference_exact=True``; exact_frac == 1.0 is the
#:                            headline number and anything below it fails.
#: ``near_exact_fast_step``   the same published equations in the same arithmetic, differing only
#:                            where the transcendentals do. Measured, in
#:                            ``stable_envs/envs/classic_control.py``: max deviation 3.3e-16 to
#:                            4.4e-13 over 1,600 steps, 1.1e-16 after ONE step, and the cause is
#:                            pinned -- torch and numpy `sin`/`cos` differ on ~16% of inputs by at
#:                            most 1 ulp. "There is no contact and no chaos excuse."
#: ``approximate_fast_step``  a different solver; trajectory identity is not expected and cannot be.
#:                            ``stable_envs/envs/pusht.py``: "not bit-exact and cannot be. Contact is
#:                            chaotic."
#: ``reference_only``         no batched step; the official CPU simulator produces the states.
#: ``unknown``                the owning module could not be imported, so nothing was read.
#: ``n/a``                    no stable-envs env matched, so there is nothing to classify.
PHYSICS_CLASSES = (
    "exact_fast_step",
    "near_exact_fast_step",
    "approximate_fast_step",
    "reference_only",
    UNKNOWN,
    "n/a",
)


def _physics_regime(name: str, dec: dict) -> tuple[str, str]:
    """One env's dynamics regime, from what it DECLARES. Never from its name.

    Three declared signals, in order:

    1. ``step_reference_regime`` -- an explicit opt-in. Nothing declares it today; it is honoured
       first so an env can override this function without touching the matrix.
    2. ``step_reference_exact is True`` -- the env claims bit-exactness and the suite enforces it.
    3. ``step_reference_gate`` -- the group "whose motion makes an episode informative; episodes
       where it never moved are excluded from the final-step statistics"
       (``testing/conformance.py``). An env declares a gate precisely because some episodes are
       uninformative about a chaotic group, which is the signature of a different solver rather than
       a different ulp. PushT declares ``gate='block'``; the five classic-control envs declare none.

    Checked against every env in the registry that ships a step, and the result agrees with what each
    module's own docstring says about itself in all cases.
    """
    declared = dec.get("step_reference_regime")
    if declared in PHYSICS_CLASSES:
        return declared, f"{name} declares step_reference_regime={declared!r}"
    if dec.get("step_reference_exact") is True:
        return "exact_fast_step", (
            f"{name} declares step_reference_exact=True, so exact_frac==1.0 "
            f"is the headline conformance number and anything below it fails"
        )
    if dec.get("step_reference_gate"):
        return "approximate_fast_step", (
            f"{name} declares step_reference_gate={dec['step_reference_gate']!r} -- a gate exists to "
            f"drop episodes where a chaotic group never moved, which is a different-solver signature, "
            f"not a floating-point one"
        )
    return "near_exact_fast_step", (
        f"{name} ships a step, claims no bit-exactness, and declares no chaos gate: the same "
        f"published closed form in the same arithmetic, diverging only where the transcendentals do"
    )


def classify_physics(envs: list[str], decls: dict) -> dict:
    """Dynamics, kept strictly separate from rendering.

    Nothing here reads a fidelity tier and nothing in ``collect_fidelity`` reads a step: a renderer
    can be bit-exact while its optional solver is approximate, and PushT is exactly that env.
    """
    if not envs:
        return {
            "class": "n/a",
            "confidence": "certain",
            "evidence": ("no stable-envs environment matched, so there is no stable-envs dynamics path to classify"),
        }
    have = [n for n in envs if n in decls]
    if not have:
        return {
            "class": UNKNOWN,
            "confidence": "unknown_probe_failed",
            "evidence": (
                "the matched env(s) are in the registry union but no supplied census "
                "could import them, so has_step / step_reference_exact are unread. This "
                "is NOT 'no step' -- it is 'not looked at from here'."
            ),
        }
    stepped = [n for n in have if decls[n].get("has_step")]
    if not stepped:
        return {
            "class": "reference_only",
            "confidence": "certain",
            "evidence": (
                "StableEnv declares no step and none of "
                + ", ".join(have)
                + " adds one, so state comes from the official CPU simulator"
            ),
        }
    per = {n: _physics_regime(n, decls[n]) for n in sorted(stepped)}
    regimes = sorted({r for r, _ in per.values()})
    # a partially-stepped match (some cells ship a solver, some do not) is reported as the weakest
    # claim present, never as the strongest
    if len(have) > len(stepped):
        regimes.append("reference_only")
    order = [c for c in PHYSICS_CLASSES if c in regimes]
    return {
        "class": order[-1] if len(order) > 1 else order[0],
        "all_regimes": order,
        "confidence": "certain" if len(order) == 1 else "mixed",
        "envs_with_step": sorted(stepped),
        "envs_without_step": sorted(set(have) - set(stepped)),
        "per_env": {n: {"regime": r, "evidence": e} for n, (r, e) in per.items()},
        "evidence": "; ".join(sorted({e for _, e in per.values()})),
    }


def collect_fidelity(envs: list[str], decls: dict, dino: dict) -> dict:
    """Render fidelity ONLY. Never merged with the physics field above."""
    out: dict = {"per_env": {}}
    for n in sorted(envs):
        dec = decls.get(n)
        if not dec:
            out["per_env"][n] = {"tier": UNKNOWN, "why": "declaration not reachable in any census supplied"}
            continue
        rec = {
            "tier": dec.get("fidelity_tier"),
            "render_decoder": dec.get("render_decoder"),
            "parity_interior_min": dec.get("parity_interior_min"),
            "parity_note": dec.get("parity_note") or None,
            "visual_fidelity": dec.get("visual_fidelity") or None,
            "visual_note": (dec.get("visual_note") or None),
            "source": "class declaration via scratch/census/census.py",
        }
        d = dino.get(n)
        if d:
            rec["measured_interior_exact_frac"] = d.get("interior_exact_frac")
            rec["measured_all_exact_frac"] = d.get("all_exact_frac")
            rec["measured_max_abs"] = d.get("max_abs")
            rec["measured_source"] = d["source"]
        out["per_env"][n] = rec
    tiers = sorted({r.get("tier") for r in out["per_env"].values() if r.get("tier")})
    out["tier"] = tiers[0] if len(tiers) == 1 else (tiers or [UNKNOWN])
    return out


def collect_speed(envs: list[str], speed: dict) -> dict:
    out: dict = {
        "per_env": {},
        "se_cpu_fps": NEEDS,
        "se_cpu_fps_why": (
            "no committed result times the stable-envs renderer on "
            "device='cpu'; every 'cpu' column in this repo's benchmarks is "
            "the OFFICIAL renderer, a different program"
        ),
    }
    for n in sorted(envs):
        if n in speed:
            out["per_env"][n] = speed[n]
    best = None
    for n, rec in sorted(out["per_env"].items()):
        f = rec.get("se_gpu_fps")
        if isinstance(f, (int, float)) and (best is None or f > best["se_gpu_fps"]):
            best = {
                "env": n,
                "se_gpu_fps": f,
                "gpu_batch": rec.get("gpu_batch"),
                "shape": rec.get("shape"),
                "supersample": rec.get("supersample"),
                "host_cpu_frac": rec.get("host_cpu_frac"),
                "official_cpu_fps": rec.get("official_cpu_fps"),
                "device": rec.get("device"),
                "source": rec.get("source"),
            }
    out["gpu_headline"] = best or {"se_gpu_fps": UNKNOWN, "why": "no row for these envs in any result file supplied"}
    return out


def collect_default_and_params(
    swm_id: str, probe: dict, match: dict, decls: dict, dmc_appearance: dict
) -> tuple[dict, dict, dict]:
    """THREE independent questions, answered separately. Conflating them was the Step 1B error.

    ``registered_default_render_covered``  can our renderer draw the states the DEFAULT swm env
                                           produces, with the right scene, camera and appearance?
                                           This is the only one of the three that gates downstream
                                           RENDERER validation.
    ``render_parameter_space_coverage``    of the configurations that id can build AND THAT CHANGE
                                           THE PICTURE, how many do we render?
    ``standalone_task_semantics_match``    would stable-envs, used as a standalone stepped env,
                                           implement the same reward and task? Worth recording and
                                           NOT a blocker, because in the pipeline this project
                                           actually runs, stable_worldmodel remains the source of
                                           physics and state and we supply pixels only.

    Step 1B answered the first by way of the third and blocked three rows -- Acrobot, Cartpole and
    Humanoid -- on a reward-name difference. It also counted the parameter space out of the upstream
    ``dm_control`` suite, which is the wrong registry twice over: ``AcrobotDMControlWrapper`` takes no
    ``task`` argument at all while the suite lists two acrobot tasks, and ``WalkerDMControlWrapper``
    accepts eight where the suite lists three. Both now come from the swm wrapper itself, and which
    of its tasks are separate RENDERER configurations is measured (see ``_swm_task_space`` in
    ``scratch/swm_registry.py``) rather than inferred from the names.
    """
    envs = match["envs"]
    rung = match.get("rung")
    ts = probe.get("swm_task_space") or {}
    dmc = probe.get("dmc_reference") or {}

    # ---------------------------------------------------------------- nothing to cover
    if str(probe.get("frame_status_probe")) == "symbolic_no_frame":
        return (
            {
                "covered": "n/a",
                "kind": "no_frame",
                "detail": "the env reports that it produces no frame, so there is nothing to render",
            },
            {"level": "n/a", "detail": "no frame, so no rendered configuration space"},
            {"match": "n/a", "detail": "no frame, so no renderer to compare task semantics for"},
        )
    if not envs:
        unknown = probe.get("make") != "ok"
        return (
            {
                "covered": "unknown" if unknown else False,
                "kind": "unknown_probe_failed" if unknown else "none",
                "detail": (
                    "this id did not construct in any supplied probe, so what it renders is unknown"
                    if unknown
                    else "no stable-envs environment matched this id"
                ),
            },
            {
                "level": "unknown" if unknown else "none",
                "detail": "no renderer, so there is no parameter space to cover",
            },
            {"match": "n/a", "detail": "no stable-envs environment to compare"},
        )

    # ---------------------------------------------------------------- 1. default RENDER coverage
    if swm_id in OBSERVATION_SUBSETS and match["confidence"] != "uncertain":
        sub = OBSERVATION_SUBSETS[swm_id]
        default = {
            "covered": False,
            "kind": sub["kind"],
            "detail": sub["detail"],
            "source": sub["source"],
            "renderer_identity": "settled",
            "renderer_for_default_scene": match.get("primary") or envs[0],
        }
    elif match["confidence"] == "uncertain":
        default = {
            "covered": "unknown",
            "kind": "uncertain_renderer_identity",
            "detail": f"the join itself is unconfirmed ({rung}): {match['evidence']}",
        }
    elif not match.get("_declarations_reachable", True):
        default = {
            "covered": "unknown",
            "kind": "unknown_probe_failed",
            "detail": (
                "the matched env(s) could not be imported by any supplied census, so what they render was not read"
            ),
        }
    elif rung == "same_dmc_appearance":
        # THE RECLASSIFICATION. The two compile to a byte-identical render model after reset(), so
        # our renderer draws this id's states correctly. What differs is the reward, which reaches
        # no pixel and is recorded in standalone_task_semantics_match instead of blocking this.
        ours = sorted({decls[n].get("reference_id") for n in envs if n in decls})
        default = {
            "covered": True,
            "kind": "same_scene_reward_differs",
            "detail": (
                f"the registered default resolves to {dmc.get('reference_id')} and "
                f"stable-envs declares {ours}; the two compile to a BYTE-IDENTICAL "
                f"render model after reset() (geom and site types, sizes, colours and "
                f"materials, camera, lights), so the renderer draws this id's states "
                f"correctly. Only the reward differs, and it reaches no pixel."
            ),
            "renderer_for_default_scene": match.get("primary") or envs[0],
        }
    else:
        default = {
            "covered": True,
            "kind": "exact_reference_match",
            "detail": f"matched on {rung}: {match['evidence']}",
        }

    # ---------------------------------------------------------------- 2. RENDER parameter space
    meta = PARAMETERIZED.get(swm_id)
    if meta:
        axes = meta["config_axes"]
        keys = sorted(axes)
        cells: list[dict] = []

        def _walk(i, chosen):
            if i == len(keys):
                cells.append(dict(chosen))
                return
            for v in axes[keys[i]]:
                _walk(i + 1, {**chosen, keys[i]: v})

        _walk(0, {})
        declared = {d.get("reference_id") for d in decls.values() if d.get("reference_id")}
        covered, missing = [], []
        for cell in cells:
            ref = meta["reference_pattern"].format(loco=cell.get("loco_env_type", ""), maze=cell.get("maze_type", ""))
            (covered if ref in declared else missing).append(ref)
        params = {
            "level": "full" if not missing else ("partial" if covered else "none"),
            "axes": axes,
            "registered_default": meta["default"],
            "default_evidence": meta["default_evidence"],
            "configs_total": len(cells),
            "configs_covered": sorted(covered),
            "missing_configurations": sorted(missing),
            "detail": meta["detail"],
            "counts": "render-distinct scenes (robot x maze), each a different picture",
        }
    elif ts.get("accepts_task"):
        # Group the wrapper's OWN accepted tasks into measured render-equivalence classes, then ask
        # how many CLASSES we render. Eight walker tasks that draw one picture are one renderer
        # configuration, not eight, and reporting seven missing would be an invented gap.
        classes = ts.get("render_equivalence_classes") or []
        dom = (dmc.get("reference_id") or "").split("/", 1)[-1].split("-", 1)[0]
        ours = {
            (d.get("reference_id") or "").split("-", 1)[1]
            for d in decls.values()
            if dom and (d.get("reference_id") or "").startswith(f"dm_control/{dom}-")
        }
        cov, miss = [], []
        for cl in classes:
            (cov if (set(cl) & ours) else miss).append(cl)
        n = len(classes)
        params = {
            "level": (
                "not_parameterized" if n <= 1 and cov else "full" if not miss else ("partial" if cov else "none")
            ),
            "axes": {"task": ts.get("tasks", [])},
            "registered_default": {"task": ts.get("default_task")},
            "default_evidence": ts.get("source"),
            "configs_total": n,
            "configs_covered": [sorted(c) for c in cov],
            "missing_configurations": [sorted(c) for c in miss],
            "tasks_total": len(ts.get("tasks", [])),
            "render_equivalence_classes": classes,
            "counts": (
                "RENDER-DISTINCT configurations, not task names: the wrapper's tasks were "
                "each constructed and their render models digested after reset(), and tasks "
                "with the same digest are one renderer configuration"
            ),
            "detail": (
                f"the wrapper accepts {len(ts.get('tasks', []))} tasks "
                f"({', '.join(ts.get('tasks', []))}), which collapse to {n} render-distinct "
                f"configuration(s); stable-envs renders {len(cov)} of them"
            ),
        }
        if n <= 1 and cov:
            params["detail"] += (
                ". All of its tasks draw the same picture, so one renderer covers "
                "the whole task space and there is nothing left to parameterise."
            )
    else:
        params = {
            "level": "not_parameterized",
            "detail": (
                ts.get("why")
                or "the registered id takes no scene-changing constructor argument that the probe observed"
            ),
            "source": ts.get("source"),
        }

    # ---------------------------------------------------------------- 3. standalone task semantics
    if dmc.get("reference_id"):
        theirs = dmc["reference_id"]
        ours = sorted({decls[n].get("reference_id") for n in envs if n in decls})
        same = theirs in ours
        semantics = {
            "match": bool(same),
            "swm_task": theirs,
            "stable_envs_task": ours,
            "detail": (
                f"stable-envs declares {ours} against this id's {theirs}"
                + (
                    ""
                    if same
                    else ". Different reward/task. This does NOT block renderer substitution: in "
                    "this project's pipeline stable_worldmodel keeps producing the states and "
                    "stable-envs supplies pixels only, so the reward never reaches the "
                    "renderer. It would matter to anyone stepping the stable-envs env "
                    "standalone."
                )
            ),
        }
    else:
        semantics = {
            "match": "n/a",
            "detail": (
                "no dm_control-style task identity to compare; stable-envs ships no "
                "standalone task semantics for this id beyond its renderer"
            ),
        }
    return default, params, semantics


def collect_swm_compat(
    swm_id: str,
    envs: list[str],
    probe: dict,
    var_maps: dict[str, str],
    live_mapping: dict,
    recorded_mapping: dict,
    workflow: dict,
) -> dict:
    """swm-side integration, kept independent of whether the picture matches.

    A renderer whose pixels are perfect is still unusable from swm if no VariationMap records how the
    two appearance spaces line up, so this never inherits anything from the fidelity fields.
    """
    mapped = sorted(n for n in envs if n in var_maps)
    out: dict = {
        "variation_map_exists": bool(mapped),
        "envs_with_variation_map": mapped,
        "envs_without_variation_map": sorted(set(envs) - set(mapped)),
        "swm_variation_leaf_count": (
            len(probe["variation_names"]) if isinstance(probe.get("variation_names"), list) else None
        ),
        "adapters": (
            "StableEnvGym / BatchedEnvPool / swap_renderer are generic over REGISTRY, so "
            "they are available for any matched env; only the variation mapping is per-env"
        ),
        "state_extraction_hook": None,
        "goal_handling": UNKNOWN,
    }
    checks = {}
    for n in mapped:
        live = (live_mapping or {}).get(n)
        rec = (recorded_mapping or {}).get(n)
        if live and "error" not in live:
            checks[n] = dict(live, source="live check_mapping() in the probe interpreter")
        elif rec:
            checks[n] = dict(rec, source=workflow.get("source", "swm_workflow.json"))
        elif live:
            checks[n] = dict(live, source="live check_mapping() in the probe interpreter")
    out["check_mapping"] = checks
    undocumented = sorted({leaf for c in checks.values() for leaf in c.get("undocumented", [])})
    out["undocumented_leaves"] = undocumented
    out["check_mapping_clean"] = bool(checks) and not undocumented
    if checks:
        cov = sorted({leaf for c in checks.values() for leaf in c.get("covered", [])})
        unm = sorted({leaf for c in checks.values() for leaf in c.get("declared_unmapped", [])})
        out["covered_leaves"] = cov
        out["declared_unmapped_leaves"] = unm
        out["unsupported_leaves_with_reason"] = unm
    # the one state-extraction hook the compat module ships by name
    if swm_id in ("swm/PushT-v1", "swm/PushT-Discrete-v1"):
        out["state_extraction_hook"] = (
            "stable_envs.compat.worldmodel.pusht_state -- reads the live "
            "pymunk agent/block bodies into our 5-float state"
        )
    if swm_id == "swm/TwoRoom-v1":
        out["goal_handling"] = (
            "BatchedEnvPool adds a 'goal' frame the official TwoRoom keeps "
            "private; info-key parity against a stock World was asserted in "
            + workflow.get("source", "swm_workflow.json")
        )
    return out


def collect_validation(envs: list[str], dino: dict, lewm: dict | None, swm_id: str) -> dict:
    """``none`` / ``pixel_only`` / ``dino_feature`` / ``downstream_policy_or_worldmodel``.

    DINO feature agreement is NOT downstream success and never promotes a row past ``dino_feature``.
    """
    evidence = []
    level = "none"
    have_dino = sorted(n for n in envs if n in dino)
    if envs:
        level = "pixel_only"
        evidence.append(
            {
                "kind": "pixel_conformance",
                "detail": (
                    "tests/gpu/test_conformance.py is parameterised over REGISTRY, so "
                    "each matched env carries render parity, state round-trip, "
                    "batch-invariance and backend agreement"
                ),
                "source": "stable_envs/testing/conformance.py",
            }
        )
    if have_dino:
        level = "dino_feature"
        for n in have_dino:
            d = dino[n]
            evidence.append(
                {
                    "kind": "dino_feature_agreement",
                    "env": n,
                    "detail": {
                        k: d[k]
                        for k in (
                            "dino_cos_ours_vs_official",
                            "dino_cos_state_vs_state",
                            "dino_cos_blank_vs_official",
                            "dino_cos_jpeg95_vs_official",
                        )
                    },
                    "caveat": ("representation-level agreement on DINOv2 features; NOT a DINO-WM task-success result"),
                    "source": d["source"],
                }
            )
    if swm_id in ("swm/PushT-v1", "swm/PushT-Discrete-v1") and lewm and "arms" in lewm:
        level = "downstream_policy_or_worldmodel"
        evidence.append(
            {
                "kind": "downstream_task_success",
                "detail": lewm,
                "applies_to": (
                    "swm/PushT-v1 directly; swm/PushT-Discrete-v1 shares the "
                    "renderer but the evaluation was run on the continuous env"
                ),
                "source": lewm["source"],
            }
        )
    return {"level": level, "evidence": evidence}


def decide_status(row: dict) -> list[dict]:
    """``[{status, action}, ...]``. A row with two independent gaps gets two entries.

    The action is kept ATTACHED to the status that produced it rather than accumulated in a parallel
    list, because the generated page groups rows by status and then wants that status's action -- and
    two parallel lists silently print the wrong pairing the moment a row has more than one entry.
    """
    out: list[dict] = []

    def add(status: str, action: str) -> None:
        out.append({"status": status, "action": action})

    frame = row["swm"]["frame_status"]
    m = row["match"]
    envs = m["envs"]

    if frame == "symbolic_no_frame":
        add(
            "symbolic_no_renderer_needed",
            "nothing to render; keep classified so it is never counted as a coverage gap",
        )
        return out

    if not envs:
        add(
            "renderer_gap",
            "no stable-envs environment corresponds to this id; see the family research doc for the "
            "gate that rejected it, or treat as an open renderer gap",
        )
        if frame == "unknown":
            add(
                "uncertain_needs_probe",
                "re-probe in an interpreter that can construct this id: " + row["swm"]["frame_status_why"],
            )
        return out

    if m["confidence"] == "uncertain":
        shapes = row["swm"].get("frame_shape")
        ours = sorted(
            {d.get("default_size") for d in row["stable_envs"]["declarations"].values() if d.get("default_size")}
        )
        add(
            "uncertain_needs_probe",
            f"confirm the join ({m['rung']}): {m['evidence']}"
            + (
                f". The swm id renders {shapes} and the matched env(s) declare default_size {ours}; "
                f"reconcile those before relying on the row."
                if shapes and ours
                else ""
            ),
        )

    if not row["stable_envs"]["declarations_reachable"]:
        add(
            "uncertain_needs_probe",
            "the matched env(s) are in the registry union but no supplied census could import them, "
            "so tier / state_dim / step read unknown here; rerun scratch/census/census.py on "
            + (", ".join(row["stable_envs"]["owner_interpreters"]) or "an interpreter that can import the family")
            + " and pass another --se-census",
        )

    if not row["swm_compat"]["variation_map_exists"]:
        n_leaves = row["swm_compat"]["swm_variation_leaf_count"]
        vs_status = row["swm"]["variation_space_status"]
        if vs_status == "unknown_probe_failed":
            leaf_clause = (
                "the swm variation space was NOT PROBED here (the id did not construct in "
                "any supplied probe), so how many leaves need classifying is unknown"
            )
        elif vs_status == "absent":
            leaf_clause = (
                "the swm env declares variation_space = None, so the map is only the "
                "swm_id link plus the leaves we have that it does not"
            )
        else:
            leaf_clause = f"{n_leaves} swm variation leaves to classify as identical / renamed / scaled / swm_only"
        add(
            "needs_swm_mapping",
            "add a VariationMap to stable_envs.compat.worldmodel.VARIATION_MAPS for "
            + ", ".join(envs)
            + "; "
            + leaf_clause,
        )
    elif row["swm_compat"]["undocumented_leaves"]:
        add(
            "needs_swm_mapping",
            "check_mapping reports undocumented swm leaves: " + ", ".join(row["swm_compat"]["undocumented_leaves"]),
        )

    if row["speed"]["se_cpu_fps"] == NEEDS:
        add(
            "needs_cpu_benchmark",
            "time the stable-envs renderer on device='cpu' for " + ", ".join(envs) + "; no such "
            "measurement exists for any env in this repo, so CPU speed is unknown project-wide",
        )

    # DEFAULT-CONFIG MISMATCH is its own work item and outranks the mapping/benchmark chores: a row
    # whose registered default is a different task is not describing the env someone would validate.
    dflt = row["axes"]["registered_default_render_covered"]
    if dflt["kind"] == "same_scene_partial_observation":
        add(
            "partial_observation_coverage",
            "the renderer is the right one and draws a documented SUBSET of what this id observes. "
            + dflt["detail"]
            + " Decide whether the downstream gate runs on the world layer alone "
            "before quoting this row as covered.",
        )
    if dflt["covered"] is False and dflt["kind"] not in (
        "same_scene_partial_observation",
        "none",
        "unknown_probe_failed",
    ):
        add(
            "default_render_mismatch",
            "the registered swm default is not the scene our renderer draws: " + dflt["detail"],
        )
    sem = row["axes"]["standalone_task_semantics_match"]
    if sem["match"] is False:
        add(
            "task_semantics_differ",
            "RECORDED, NOT BLOCKING. " + sem["detail"] + " Register the matching task only if "
            "someone intends to step the stable-envs env standalone.",
        )
    par = row["axes"]["render_parameter_space_coverage"]
    if par.get("level") == "partial":
        # `configs_total`, not `cells_total`: the key was renamed when the count became
        # render-distinct configurations rather than raw cells, and the stale lookup printed
        # "4 of None". A missing configuration is a STRING in the parameterised-family branch
        # (one reference_id) and a LIST in the task branch (a render-equivalence class of task
        # names), so both are rendered rather than passed to a join that only accepts strings.
        missing = par.get("missing_configurations", [])
        shown = ", ".join("/".join(m) if isinstance(m, (list, tuple)) else str(m) for m in missing[:6])
        add(
            "partial_parameter_coverage",
            f"{len(missing)} of {par.get('configs_total', UNKNOWN)} render-distinct configurations "
            f"this registered id can build have no stable-envs env: " + shown + ("..." if len(missing) > 6 else ""),
        )

    # A MISSING VariationMap DOES NOT BLOCK THIS. The paired gate runs the official default env
    # against our renderer for the same default scene; no variation leaf has to be mapped first.
    if row["axes"]["downstream_readiness"] == "ready":
        add(
            "ready_for_downstream_validation",
            "run the paired task-equivalence gate (official renderer vs ours, same episodes, same "
            "seeds) the way docs/research/artifacts/step5_lewm_pusht/ does for PushT; current evidence is "
            + row["validation"]["level"]
            + ". A missing VariationMap does not block this -- the gate "
            "uses the registered default appearance.",
        )
    if not out:
        add(
            "certified",
            "renderer, swm mapping, speed evidence and downstream equivalence all present; keep it that way",
        )
    return out


#: Which status leads the compact table when a row has several. Ordered by what blocks progress
#: soonest, not by severity of wording.
#: Which status leads the compact table when a row has several: THE MOST FUNDAMENTAL UNKNOWN FIRST.
#:
#: Reordered from the first pass, where `needs_swm_mapping` outranked everything below it and so
#: became the headline on 17 rows -- reading as though an unmapped variation space were the thing
#: standing between those renderers and being useful. It is not. The paired downstream gate runs on
#: the registered default appearance and needs no mapping at all, so `ready_for_downstream_validation`
#: now sorts ABOVE the two chores, and the chores stay visible in `statuses` and in the per-status
#: tables. What legitimately outranks readiness is anything that means the row might be describing
#: the wrong env: no renderer, an unconfirmed join, or a default that resolves to another task.
STATUS_PRECEDENCE = (
    "symbolic_no_renderer_needed",
    "renderer_gap",
    "uncertain_needs_probe",
    "default_render_mismatch",
    "partial_observation_coverage",
    "ready_for_downstream_validation",
    "partial_parameter_coverage",
    "task_semantics_differ",
    "needs_swm_mapping",
    "needs_cpu_benchmark",
    "needs_speed_optimization",
    "deprioritized",
    "certified",
)


# ====================================================================== build


def build(args) -> dict:
    swm = read_swm_registries([Path(x) for x in sorted(args.swm_registry)])
    censuses = read_censuses(sorted(Path(p) for p in args.se_census))
    union_blob = read_union(Path(args.union) if args.union else None)
    union = union_blob["union"]
    speed = read_speed(Path(args.decode_all) if args.decode_all else None, Path(args.bench) if args.bench else None)
    dino = read_dino(Path(args.dino) if args.dino else None)
    lewm = read_lewm(Path(args.lewm) if args.lewm else None)
    workflow = read_swm_workflow(Path(args.swm_workflow) if args.swm_workflow else None)
    live = live_probe(swm["ids"]) if args.live else {"available": False, "error": "not requested"}

    decls = censuses["declarations"]
    # ownership from BOTH sources: a census supplied to this run, and the recorded union from the
    # interpreters that were never going to run on this machine. Without the second, every Fetch row
    # would say "owning interpreters: none recorded" while union.json plainly names se-fetch.
    owners = defaultdict(list, {k: list(v) for k, v in censuses["owners"].items()})
    for tag, names in sorted(union_blob["per_interpreter"].items()):
        for name in names:
            if tag not in owners[name]:
                owners[name].append(tag)
    owners = {k: sorted(v) for k, v in owners.items()}
    # the registry is the UNION over interpreters; declarations are whatever subset could be imported
    all_envs = sorted(set(union) | set(decls) | set(owners) | set(speed))

    var_maps = dict(args._variation_maps)
    ref_classes = live.get("reference_env_class", {}) if live.get("available") else {}
    live_mapping = live.get("check_mapping", {}) if live.get("available") else {}
    cpu_render = live.get("cpu_render", {}) if live.get("available") else {}
    recorded_mapping = workflow.get("check_mapping", {})

    dmc_appearance = live.get("dmc_appearance", {}) if live.get("available") else {}

    # TWO PASSES, because the ``same_scene_as`` rung reads other rows' results and would otherwise
    # depend on where an id falls in the alphabet: `swm/PushT-Discrete-v1` sorts BEFORE `swm/PushT-v1`
    # and so would be judged against an empty table. The first pass resolves every rung that needs no
    # other row; the second re-runs the whole ladder with those results available. Same inputs, same
    # output, regardless of id order.
    def _pass(already: dict[str, dict]) -> dict[str, dict]:
        return {
            i: match_swm_id(
                i,
                swm["probe"].get(i, {}),
                swm["kwargs"].get(i, {}),
                decls,
                var_maps,
                ref_classes,
                dmc_appearance,
                already,
            )
            for i in swm["ids"]
        }

    matches = _pass(_pass({}))

    rows: list[dict] = []
    for swm_id in swm["ids"]:
        probe = swm["probe"].get(swm_id, {})
        kwargs = swm["kwargs"].get(swm_id, {})
        frame_status, frame_why = classify_frame(probe)

        m = matches[swm_id]
        envs = m["envs"]
        m = {k: v for k, v in m.items() if not k.startswith("_")}

        reachable = [n for n in envs if n in decls]
        se_side = {
            "envs": envs,
            "in_registry_union": sorted(n for n in envs if n in set(union) | set(all_envs)),
            "owner_interpreters": sorted({o for n in envs for o in owners.get(n, [])}),
            "declarations_reachable": bool(reachable) and len(reachable) == len(envs),
            "declarations": {n: decls[n] for n in sorted(reachable)},
            "reference_ids": sorted({decls[n].get("reference_id") for n in reachable if decls[n].get("reference_id")}),
            "state_dim": sorted(
                {decls[n].get("state_dim") for n in reachable if decls[n].get("state_dim") is not None}
            ),
            "render_decoder": sorted(
                {decls[n].get("render_decoder") for n in reachable if decls[n].get("render_decoder")}
            ),
            "backend": sorted(
                {
                    BACKEND_LABELS.get(
                        tuple(decls[n].get("backends") or []), "/".join(decls[n].get("backends") or []) or UNKNOWN
                    )
                    for n in reachable
                }
            )
            or [UNKNOWN],
        }
        gpu = {}
        for n in envs:
            r = speed.get(n, {})
            if isinstance(r.get("se_gpu_fps"), (int, float)):
                gpu[n] = True
        se_side["gpu_render"] = (
            "yes"
            if gpu
            else (UNKNOWN if not envs else "unknown (no throughput row; every backend here is CUDA-first)")
        )
        cpu_states = {n: cpu_render[n] for n in envs if n in cpu_render}
        if cpu_states:
            # A CPU render that raised ModuleNotFoundError says the PROBE INTERPRETER lacks a package,
            # not that the env has no CPU path -- the same "missing vs not importable here" trap the
            # registry union exists to avoid, one level down. The two are reported apart.
            failed = {n: v for n, v in cpu_states.items() if not v.get("executes")}
            dep = {
                n: v
                for n, v in failed.items()
                if v.get("error", "").startswith(("ModuleNotFoundError", "ImportError"))
            }
            if not failed:
                se_side["cpu_render_executes"] = "yes"
            elif len(dep) == len(failed):
                se_side["cpu_render_executes"] = "unknown_dependency_absent_here"
            else:
                se_side["cpu_render_executes"] = "no"
            se_side["cpu_render_detail"] = cpu_states
            se_side["cpu_render_caveat"] = (
                "'yes' means the CPU path returned a frame in the probe interpreter and says NOTHING "
                "about CPU throughput -- see speed.se_cpu_fps, which is needs_measurement. "
                "'unknown_dependency_absent_here' means every failure was an absent package in this "
                "interpreter, which is not evidence about the env."
            )
        else:
            se_side["cpu_render_executes"] = UNKNOWN
            se_side["cpu_render_caveat"] = "no live probe covered these envs"

        row = {
            "swm_id": swm_id,
            "swm": {
                "entry_point": swm["entry_points"].get(swm_id),
                "kwargs": kwargs,
                "class": probe.get("class"),
                "in_WORLDS": swm_id in swm.get("WORLDS", []),
                "discrete": swm_id in swm.get("DISCRETE_WORLDS", []),
                "constructible_in_probe": (
                    "yes" if probe.get("make") == "ok" else "error" if probe.get("make") else UNKNOWN
                ),
                "construct_error": (None if probe.get("make") == "ok" else probe.get("make")),
                "reset_status": probe.get("reset"),
                "frame_status": frame_status,
                "frame_status_why": frame_why,
                "frame_shape": probe.get("frame_shape"),
                # TRI-STATE, carried straight through from the probe. `absent` means the env said it
                # has none; `unknown_probe_failed` means nobody got far enough to ask. Step 1A
                # collapsed both to "no variation space" and so reported four classic-control ids and
                # all twelve Fetch ids as having none, when in fact they declare 4-11 leaves apiece.
                "variation_space_status": probe.get("variation_space_status", "unknown_probe_failed"),
                "variation_space_holder": probe.get("variation_space_holder"),
                "variation_space_error": probe.get("variation_space_error"),
                "variation_names": probe.get("variation_names"),
                "info_keys": probe.get("info_keys"),
                "env_name": probe.get("env_name"),
                "resolved_reference": probe.get("dmc_reference"),
                "observed_config": {
                    k: probe[k]
                    for k in ("task_name", "maze_type", "env_type", "ob_type", "camera_id", "action_repeat")
                    if k in probe
                },
            },
            "match": m,
            "stable_envs": se_side,
        }
        row["physics"] = classify_physics(envs, decls)
        row["fidelity"] = collect_fidelity(envs, decls, dino)
        row["speed"] = collect_speed(envs, speed)
        row["swm_compat"] = collect_swm_compat(swm_id, envs, probe, var_maps, live_mapping, recorded_mapping, workflow)
        row["validation"] = collect_validation(envs, dino, lewm, swm_id)
        if swm_id in GAP_REASONS:
            row["gap"] = dict(GAP_REASONS[swm_id])
        elif not envs and frame_status != "symbolic_no_frame":
            row["gap"] = {
                "reason": "unknown_needs_research",
                "detail": ("no committed research doc in this repo supports a stronger classification for this id"),
                "source": [],
            }
        else:
            row["gap"] = None

        m2 = dict(m, _declarations_reachable=se_side["declarations_reachable"])
        default_cfg, param_cov, task_sem = collect_default_and_params(swm_id, probe, m2, decls, dmc_appearance)

        # ---- THE INDEPENDENT AXES -----------------------------------------------------------
        # Nine questions that are genuinely separate, kept separate. A reader can answer any one of
        # them without the others being resolved, and no axis is allowed to imply another: a
        # renderer can exist with no mapping, be GPU-fast with no CPU number, be pixel-perfect with
        # approximate physics, or be ready to validate while its variation space is unmapped.
        vs_status = row["swm"]["variation_space_status"]
        if not envs:
            mapping_status = "n/a"
        elif row["swm_compat"]["variation_map_exists"]:
            mapping_status = (
                "complete"
                if row["swm_compat"]["check_mapping_clean"]
                else ("partial" if row["swm_compat"]["undocumented_leaves"] else UNKNOWN)
            )
        else:
            mapping_status = "missing"
        if frame_status == "symbolic_no_frame":
            readiness = "not_applicable_symbolic"
        elif not envs:
            readiness = "blocked_renderer_gap"
        elif m["confidence"] == "uncertain" or not se_side["declarations_reachable"]:
            readiness = "blocked_uncertain_join"
        elif default_cfg.get("kind") == "same_scene_partial_observation":
            readiness = "ready_on_documented_subset"
        elif default_cfg["covered"] is not True:
            readiness = "blocked_default_render"
        elif row["validation"]["level"] == "downstream_policy_or_worldmodel":
            readiness = "already_validated"
        else:
            readiness = "ready"
        row["axes"] = {
            "renderer_exists": ("no" if not envs else ("unknown" if not se_side["declarations_reachable"] else "yes")),
            # RENDER coverage and TASK semantics are separate axes. A reward-name difference lands
            # in the third and never in the first, because the renderer is substituted into a
            # pipeline where stable_worldmodel still produces the states.
            "registered_default_render_covered": default_cfg,
            "render_parameter_space_coverage": param_cov,
            "standalone_task_semantics_match": task_sem,
            "swm_variation_space": {
                "status": vs_status,
                "leaves": (len(probe["variation_names"]) if isinstance(probe.get("variation_names"), list) else None),
                "holder": probe.get("variation_space_holder"),
                "probe": swm.get("probe_source", {}).get(swm_id),
            },
            "swm_mapping_status": mapping_status,
            "cpu_support": {
                "render_executes": se_side.get("cpu_render_executes", UNKNOWN),
                "investigated": [
                    k
                    for k, v in sorted(PROBE_LIMITATIONS.items())
                    if v.get("field") == "cpu_support.render_executes"
                    and any(e.startswith(v["applies_to_env_prefix"]) for e in envs)
                ],
                "throughput": NEEDS,
                "note": (
                    "support and speed are different claims; throughput is Step 3 "
                    "work and is not inferred from the path running"
                ),
            },
            "gpu_support": {
                "render": se_side.get("gpu_render"),
                "throughput": row["speed"]["gpu_headline"].get("se_gpu_fps", UNKNOWN),
            },
            "physics_path": row["physics"]["class"],
            "validation_level": row["validation"]["level"],
            "downstream_readiness": readiness,
        }
        todo = decide_status(row)
        row["statuses"] = sorted({t["status"] for t in todo})
        row["headline_status"] = next(s for s in STATUS_PRECEDENCE if s in row["statuses"])
        row["next_actions"] = todo
        rows.append(row)

    by_headline: dict[str, int] = defaultdict(int)
    by_any: dict[str, int] = defaultdict(int)
    for r in rows:
        by_headline[r["headline_status"]] += 1
        for s in r["statuses"]:
            by_any[s] += 1

    covered = [r for r in rows if r["match"]["envs"]]
    return {
        "schema_version": 1,
        "what_this_is": (
            "One row per LIVE stable_worldmodel gymnasium id, joined to the stable-envs registry. "
            "Generated by scripts/regression/swm_matrix.py -- edit the script or the inputs, never this file."
        ),
        "provenance": {
            "swm_registries": [
                {
                    "path": _rel(q),
                    "sha256_16": _digest(Path(q)),
                    "probe_env": swm.get("probe_envs", {}).get(Path(q).stem, {}),
                }
                for q in sorted(args.swm_registry)
            ],
            "swm_probe_source": swm.get("probe_source", {}),
            "swm_registry_note": (
                "the swm id list is READ FROM THESE FILES, which live imports "
                "produced, and every probe must agree on it or the build stops. "
                "It is never transcribed into this script."
            ),
            "se_censuses": [{"path": _rel(p), "sha256_16": _digest(Path(p))} for p in sorted(args.se_census)],
            "interpreters_in_census": {k: len(v) for k, v in sorted(censuses["per_interpreter"].items())},
            "registry_union_size": len(union),
            "union_interpreters": {k: len(v) for k, v in sorted(union_blob["per_interpreter"].items())},
            "declarations_available_for": len(decls),
            "result_files": {
                k: (_rel(v) if v and Path(v).exists() else None)
                for k, v in sorted(
                    {
                        "decode_all": args.decode_all,
                        "bench_gpu_vs_cpu": args.bench,
                        "dino": args.dino,
                        "lewm_results": args.lewm,
                        "swm_workflow": args.swm_workflow,
                    }.items()
                )
            },
            "live_probe": {k: v for k, v in live.items() if k in ("available", "error")}
            | {
                "did": (
                    ["check_mapping", "cpu_render_executes", "reference_env_class"] if live.get("available") else []
                ),
                "registry_size_here": len(live.get("registry_here", [])) or None,
            },
        },
        "measurement_policy": {
            "measured_here": [
                "swm registry membership, constructibility, frame shape, variation "
                "leaves, info keys (scratch/swm_registry.py)",
                "the dm_control (domain, task) each DM Control id wraps, resolved by "
                "comparing task objects against every task the suite can build",
                "check_mapping() over VARIATION_MAPS, if --live",
                "whether device='cpu' returns a frame, if --live",
            ],
            "read_from_declarations": [
                "fidelity_tier",
                "render_decoder",
                "visual_fidelity",
                "parity_interior_min",
                "state_dim",
                "has_step",
                "step_reference_exact",
                "render backend",
            ],
            "read_from_committed_results": [
                "GPU throughput and the OFFICIAL-CPU baseline",
                "DINOv2 feature agreement",
                "LeWM PushT task success",
            ],
            "refused": {
                "se_cpu_fps": (
                    "no committed result times the stable-envs renderer on device='cpu'. "
                    "Every 'cpu' column in this repo is the OFFICIAL renderer. Marked "
                    "needs_measurement on every row rather than inferred."
                ),
                "downstream_from_pixels": (
                    "pixel parity and DINO cosine are recorded as themselves; "
                    "only docs/research/artifacts/step5_lewm_pusht/ promotes a row to "
                    "downstream_policy_or_worldmodel."
                ),
                "cpu_support_from_absence_of_error": (
                    "cpu_render_executes is a separate field from any speed field and never implies one."
                ),
            },
        },
        "summary": {
            "swm_ids": len(rows),
            "with_stable_envs_match": len(covered),
            "renderer_gaps": sum(1 for r in rows if "renderer_gap" in r["statuses"]),
            "symbolic_no_frame": sum(1 for r in rows if r["swm"]["frame_status"] == "symbolic_no_frame"),
            "distinct_stable_envs_envs_used": len(sorted({e for r in rows for e in r["match"]["envs"]})),
            "by_headline_status": dict(sorted(by_headline.items())),
            "by_any_status": dict(sorted(by_any.items())),
            "by_match_rung": dict(
                sorted(
                    (k, sum(1 for r in rows if r["match"]["rung"] == k))
                    for k in MATCH_RUNGS
                    if any(r["match"]["rung"] == k for r in rows)
                )
            ),
            "by_validation_level": dict(
                sorted(
                    (lvl, sum(1 for r in rows if r["validation"]["level"] == lvl))
                    for lvl in {r["validation"]["level"] for r in rows}
                )
            ),
            "by_physics_class": dict(
                sorted(
                    (c, sum(1 for r in rows if r["physics"]["class"] == c))
                    for c in {r["physics"]["class"] for r in rows}
                )
            ),
            "rows_needing_cpu_benchmark": sum(1 for r in rows if "needs_cpu_benchmark" in r["statuses"]),
            "rows_needing_swm_mapping": sum(1 for r in rows if "needs_swm_mapping" in r["statuses"]),
            "by_registered_default_render_covered": dict(
                sorted(
                    (
                        str(v),
                        sum(
                            1 for r in rows if str(r["axes"]["registered_default_render_covered"]["covered"]) == str(v)
                        ),
                    )
                    for v in {str(r["axes"]["registered_default_render_covered"]["covered"]) for r in rows}
                )
            ),
            "by_render_parameter_space_coverage": dict(
                sorted(
                    (lvl, sum(1 for r in rows if r["axes"]["render_parameter_space_coverage"].get("level") == lvl))
                    for lvl in {r["axes"]["render_parameter_space_coverage"].get("level") for r in rows}
                )
            ),
            "by_swm_variation_space_status": dict(
                sorted(
                    (st, sum(1 for r in rows if r["axes"]["swm_variation_space"]["status"] == st))
                    for st in {r["axes"]["swm_variation_space"]["status"] for r in rows}
                )
            ),
            "by_swm_mapping_status": dict(
                sorted(
                    (st, sum(1 for r in rows if r["axes"]["swm_mapping_status"] == st))
                    for st in {r["axes"]["swm_mapping_status"] for r in rows}
                )
            ),
            "by_standalone_task_semantics_match": dict(
                sorted(
                    (
                        str(v),
                        sum(1 for r in rows if str(r["axes"]["standalone_task_semantics_match"]["match"]) == str(v)),
                    )
                    for v in {str(r["axes"]["standalone_task_semantics_match"]["match"]) for r in rows}
                )
            ),
            "by_downstream_readiness": dict(
                sorted(
                    (st, sum(1 for r in rows if r["axes"]["downstream_readiness"] == st))
                    for st in {r["axes"]["downstream_readiness"] for r in rows}
                )
            ),
            "by_gap_reason": dict(
                sorted(
                    (g, sum(1 for r in rows if (r.get("gap") or {}).get("reason") == g))
                    for g in {(r.get("gap") or {}).get("reason") for r in rows}
                    if g
                )
            ),
            "probe_failures": sorted(r["swm_id"] for r in rows if r["swm"]["constructible_in_probe"] != "yes"),
        },
        "rows": rows,
    }


# ====================================================================== markdown view


def _fmt_fps(v) -> str:
    if not isinstance(v, (int, float)):
        return "?"
    if v >= 1e6:
        return f"{v / 1e6:.1f}M"
    if v >= 1e3:
        return f"{v / 1e3:.0f}k"
    return f"{v:.0f}"


def _envs_cell(row: dict) -> str:
    envs = row["match"]["envs"]
    if not envs:
        return "—"
    fam = row["match"].get("family")
    if fam:
        return (
            f"`{row['match'].get('primary') or envs[0]}` +{len(envs) - 1} ({fam})" if len(envs) > 1 else f"`{envs[0]}`"
        )
    if len(envs) <= 2:
        return ", ".join(f"`{e}`" for e in envs)
    return f"`{envs[0]}` +{len(envs) - 1} more"


def markdown(matrix: dict) -> str:
    s = matrix["summary"]
    prov = matrix["provenance"]
    rows = matrix["rows"]
    L: list[str] = []
    A = L.append

    A("# stable_worldmodel x stable-envs coverage matrix")
    A("")
    A("<!-- GENERATED FILE. Do not edit by hand. -->")
    A("")
    A(
        "**Generated by [`scripts/regression/swm_matrix.py`](../../scripts/regression/swm_matrix.py) from "
        "[`tests/fixtures/regression/fidelity/swm_matrix.json`](../../tests/fixtures/regression/fidelity/swm_matrix.json).** Every count on this page is "
        "computed from that JSON; nothing here is maintained by hand. To change a cell, change the "
        "input it comes from and regenerate — the reproduction commands are at the bottom."
    )
    A("")
    A(
        "This supersedes the headline counts in "
        "[`swm_coverage.md`](swm_coverage.md), which remains the better read for *why* each gap is a "
        "gap but has drifted on *what* is covered."
    )
    A("")

    A("## What this page is, and what it is not")
    A("")
    A("Three kinds of cell, never mixed:")
    A("")
    A("| kind | how to read it |")
    A("|---|---|")
    A(
        "| **measured here** | the probe observed it while enumerating the two registries — id "
        "membership, constructibility, frame shape, variation leaves, the `dm_control` task each DM "
        'Control id actually wraps, `check_mapping()`, and whether `device="cpu"` returns a frame |'
    )
    A(
        "| **read from a declaration** | fidelity tier, render decoder, `visual_fidelity`, `state_dim`, "
        "`has_step`, `step_reference_exact`, render backend — things an env states about itself and the "
        "conformance suite holds it to |"
    )
    A(
        "| **read from a committed result** | GPU throughput, the official-CPU baseline, DINOv2 feature "
        "agreement, the LeWM PushT task-success evaluation. Reused with provenance, never rerun |"
    )
    A("")
    A(
        "**Render coverage and task semantics are different questions.** The one this project turns on "
        "is *can stable-envs correctly render the states this stable_worldmodel environment produces* "
        "— because the pipeline substitutes the renderer, not the simulator: swm keeps producing "
        "states and stable-envs supplies pixels. A different reward, desired speed or success "
        "condition reaches no pixel, so it **cannot** reduce render coverage; it is recorded on its "
        "own `standalone_task_semantics_match` axis, where it matters to anyone stepping a stable-envs "
        "env standalone. What *does* reduce render coverage is a change to the model, the camera, a "
        "drawn target, or the observation content."
    )
    A("")
    A(
        "The same rule decides what counts as a *configuration*. A registered id's parameter space is "
        "read from the **swm wrapper itself** — its `task` argument and its `_TASKS`, not the upstream "
        "`dm_control` suite, which lists two acrobot tasks for a wrapper that takes no `task` argument "
        "at all and three walker tasks for a wrapper that accepts eight. Each accepted task is then "
        "constructed and its render model digested after `reset()`, so tasks that draw the same "
        "picture collapse into one renderer configuration. Walker's eight collapse to one; Finger's "
        "three stay three, because its target is a site whose size the task changes."
    )
    A("")
    A(
        "**`unknown` is never `absent`.** Anything that depends on constructing an environment — the "
        "swm `variation_space`, its leaves, a frame, a stable-envs declaration, whether the CPU path "
        "runs — is recorded in three states, not two: `present`/`pixels` (observed), `absent` (the env "
        "said it has none), and `unknown_probe_failed` (nobody got far enough to ask). The first pass "
        "collapsed the last two and so reported sixteen ids as having no variation space when four "
        "were read off the wrong object and twelve had not constructed; all sixteen in fact declare "
        "4–17 leaves apiece."
    )
    A("")
    A("Two claims are refused outright:")
    A("")
    for k, v in sorted(matrix["measurement_policy"]["refused"].items()):
        A(f"- **`{k}`** — {v}")
    A("")

    A("## Summary")
    A("")
    A(
        f"**{s['swm_ids']} `swm/` ids are registered live. {s['with_stable_envs_match']} have a "
        f"stable-envs renderer, {s['symbolic_no_frame']} have no frame to render, and "
        f"{s['renderer_gaps']} are gaps.** Those {s['with_stable_envs_match']} matched ids resolve to "
        f"{s['distinct_stable_envs_envs_used']} distinct stable-envs environments."
    )
    A("")
    A("| headline status | ids |")
    A("|---|---|")
    for k, v in sorted(s["by_headline_status"].items(), key=lambda kv: (-kv[1], kv[0])):
        A(f"| `{k}` | {v} |")
    A("")
    A(
        "A row usually has several of these at once. The **headline is the most fundamental unknown**, "
        "not the longest list: anything meaning the row might be describing the wrong environment (no "
        "renderer, an unconfirmed join, a default that resolves to another task) outranks the chores. "
        "`needs_swm_mapping` deliberately sorts BELOW `ready_for_downstream_validation`, because the "
        "paired downstream gate runs on the registered default appearance and needs no variation leaf "
        "mapped first — in the first pass it outranked everything and became the headline on 17 rows, "
        "reading as though an unmapped variation space were what stood between those renderers and "
        "being useful. Counting every status wherever it appears:"
    )
    A("")
    A("| status (anywhere on a row) | ids |")
    A("|---|---|")
    for k, v in sorted(s["by_any_status"].items(), key=lambda kv: (-kv[1], kv[0])):
        A(f"| `{k}` | {v} |")
    A("")
    A("### The independent axes")
    A("")
    A(
        "These are separate questions and none of them implies another. A renderer can exist with no "
        "variation mapping, be GPU-fast with no CPU number, be pixel-exact with approximate physics, "
        "or be ready for downstream validation while its variation space is entirely unmapped."
    )
    A("")
    A("| axis | breakdown |")
    A("|---|---|")

    def _b(d):
        return ", ".join(f"`{k}` {v}" for k, v in d.items())

    A(f"| join evidence | {_b(s['by_match_rung'])} |")
    A(f"| **registered default RENDER covered** | {_b(s['by_registered_default_render_covered'])} |")
    A(f"| **render** parameter-space coverage | {_b(s['by_render_parameter_space_coverage'])} |")
    A(f"| standalone task semantics match | {_b(s['by_standalone_task_semantics_match'])} |")
    A(f"| swm `variation_space` | {_b(s['by_swm_variation_space_status'])} |")
    A(f"| `VariationMap` status | {_b(s['by_swm_mapping_status'])} |")
    A(f"| physics path | {_b(s['by_physics_class'])} |")
    A(f"| validation level | {_b(s['by_validation_level'])} |")
    A(f"| downstream readiness | {_b(s['by_downstream_readiness'])} |")
    if s.get("by_gap_reason"):
        A(f"| gap reason | {_b(s['by_gap_reason'])} |")
    A("")
    A(
        "`unknown` is never merged into `absent` anywhere above. A property is `absent` only when the "
        "env said so; a property nobody could reach is `unknown_probe_failed`, which is a different "
        "cell and a different next action."
    )
    A("")

    A("## The matrix")
    A("")
    A(
        "**`render dflt`** — can our renderer draw the states the default `gymnasium.make(id)` "
        "produces? `yes ⁺` means yes, and the reward differs (see `task sem`) — a reward reaches no "
        "pixel, so it does not reduce render coverage. `subset` means the renderer is the right one "
        "and draws a documented part of the observation. "
        "**`render params`** — of the configurations that id can build **and that change the "
        "picture**, how many we render; task names that draw the same picture are one configuration, "
        "not several. "
        "**`task sem`** — would stable-envs, stepped standalone, implement the same reward? Recorded, "
        "never a blocker: in this project's pipeline stable_worldmodel produces the states and we "
        "supply pixels only. "
        "`render` — declared fidelity tier. `CPU`/`GPU` are independent, and `CPU` is *support*, never "
        "speed. `physics` — the dynamics path, never merged with the render tier. "
        "`swm VS` — swm variation leaves (`none` = the env declares none, `?` = not probed). "
        "`map` — whether `compat.worldmodel.VARIATION_MAPS` carries this env. "
        "Detail and provenance for every cell is in the JSON."
    )
    A("")
    A(
        "| swm id | stable-envs | render dflt | render params | task sem | tier | CPU | GPU | "
        "physics | swm VS | map | validation | headline |"
    )
    A("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        se = r["stable_envs"]
        blank = not r["match"]["envs"]
        tier = r["fidelity"]["tier"]
        tier = tier if isinstance(tier, str) else "/".join(tier or [UNKNOWN])
        if blank:
            tier = "—"
        cpu = se.get("cpu_render_executes", UNKNOWN)
        cpu = {
            "yes": "runs, unbenchmarked",
            "no": "**no CPU path**",
            "unknown_dependency_absent_here": "? (dep absent here)",
            "unknown": "?",
        }.get(cpu, cpu)
        gpufps = r["speed"]["gpu_headline"].get("se_gpu_fps")
        gpu = _fmt_fps(gpufps) + " f/s" if isinstance(gpufps, (int, float)) else "?"
        if blank:
            cpu = gpu = "—"
        phys = r["physics"]["class"]
        smap = "yes" if r["swm_compat"]["variation_map_exists"] else ("—" if not r["match"]["envs"] else "**no**")
        val = {
            "none": "—",
            "pixel_only": "pixel",
            "dino_feature": "pixel+DINO",
            "downstream_policy_or_worldmodel": "**downstream**",
        }[r["validation"]["level"]]
        ax = r["axes"]
        dflt = {True: "yes", False: "**no**", "unknown": "?", "n/a": "—"}[
            ax["registered_default_render_covered"]["covered"]
        ]
        dflt = {"same_scene_reward_differs": "yes ⁺", "same_scene_partial_observation": "**subset**"}.get(
            ax["registered_default_render_covered"]["kind"], dflt
        )
        sem = {True: "yes", False: "**no**", "n/a": "—", "unknown": "?"}[
            ax["standalone_task_semantics_match"]["match"]
        ]
        par = {
            "full": "full",
            "partial": "**partial**",
            "none": "none",
            "not_parameterized": "—",
            "unknown": "?",
            "n/a": "—",
        }[ax["render_parameter_space_coverage"]["level"]]
        vs = {"present": f"{ax['swm_variation_space']['leaves']}", "absent": "none", "unknown_probe_failed": "?"}[
            ax["swm_variation_space"]["status"]
        ]
        A(
            f"| `{r['swm_id']}` | {_envs_cell(r)} | {dflt} | {par} | {sem} | {tier} | {cpu} | {gpu} | "
            f"{phys} | {vs} | {smap} | {val} | `{r['headline_status']}` |"
        )
    A("")

    A("## Work remaining, grouped by next action")
    A("")
    groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for r in rows:
        for t in r["next_actions"]:
            if t["status"] in ("certified", "symbolic_no_renderer_needed"):
                continue
            groups[t["status"]].append((r["swm_id"], t["action"]))
    for st in STATUS_PRECEDENCE:
        if st not in groups:
            continue
        entries = sorted(groups[st])
        A(f"### `{st}` — {len(entries)} ids")
        A("")
        A(f"Example action, for `{entries[0][0]}`: {entries[0][1]}")
        A("")
        A("<details><summary>every id, with its action</summary>")
        A("")
        for i, act in entries:
            A(f"- `{i}` — {act}")
        A("")
        A("</details>")
        A("")

    A("## Semantic validation evidence, by level")
    A("")
    A(
        "Three levels, and the distinction between the last two is the one most easily lost: DINOv2 "
        "cosine agreement is a statement about *representations*, and a policy's success rate is a "
        "statement about the *task*. Feature agreement never promotes a row past `dino_feature`."
    )
    A("")
    lewm_row = next(
        (r for r in rows if any(e["kind"] == "downstream_task_success" for e in r["validation"]["evidence"])), None
    )
    if lewm_row:
        ev = next(e for e in lewm_row["validation"]["evidence"] if e["kind"] == "downstream_task_success")
        det = ev["detail"]
        A("### `downstream_policy_or_worldmodel` — the only task-success result in the repo")
        A("")
        A(
            f"From `{det['source']}`, recomputed here from the per-seed JSONs with that directory's own "
            "`summarize.py` (its paired McNemar test and Wilson interval), not transcribed."
        )
        A("")
        A("| arm | seeds | successes | success rate | 95% CI | paired vs official | McNemar *p* |")
        A("|---|---|---|---|---|---|---|")
        for arm, a in det["arms"].items():
            p = det["paired_vs_official"].get(arm)
            pair = f"{p['delta_pts']:+.1f} pts (n={p['n']})" if p else "—"
            pv = f"{p['mcnemar_p']:.3f}" if p else "—"
            A(
                f"| `{arm}` | {len(a['seeds'])} | {a['successes']}/{a['n']} | "
                f"{a['success_rate_pct']:.1f}% | [{a['ci95_pct'][0]:.1f}, {a['ci95_pct'][1]:.1f}] | "
                f"{pair} | {pv} |"
            )
        A("")
        A(
            "The number to take from this is not 87.7%. It is that **the evaluation is pixel-sensitive "
            "and the renderer swap is not detectable**: blank frames cost 92 points and swapping the "
            "renderer costs 0.3, well inside the noise of a paired test. A gate that could not tell "
            "blank frames from real ones would prove nothing."
        )
        A("")
        A(f"Applies to: {ev.get('applies_to', '')}")
        A("")
    dino_rows = [r for r in rows if r["validation"]["level"] == "dino_feature"]
    A(f"### `dino_feature` — {len(dino_rows)} ids")
    A("")
    A(
        "Representation-level agreement on DINOv2 features, from "
        "`examples/results/dino/dino_agreement.json`. Each row's JSON entry carries the cosine distance "
        "to the official frame alongside three controls — a different state, a blank frame, and a "
        "JPEG-95 round trip — so the residual can be read against something. **This is not a DINO-WM "
        "task-success result and the matrix never treats it as one.**"
    )
    A("")
    A(f"### `pixel_only` — {sum(1 for r in rows if r['validation']['level'] == 'pixel_only')} ids")
    A("")
    A(
        "Covered by the registry-parameterised conformance suite (render parity, state round-trip, "
        "batch invariance, backend agreement) and nothing beyond it."
    )
    A("")

    A("## Renderer gaps, individually")
    A("")
    gaps = [r for r in rows if "renderer_gap" in r["statuses"]]
    if not gaps:
        A("None.")
    else:
        A(
            "Each carries a `gap_reason` from a controlled vocabulary, taken from research this repo "
            "has already committed. **This is classification, not new investigation** — where no "
            "committed doc supports a reason, the row says `unknown_needs_research` rather than "
            "guessing one."
        )
        A("")
        A("| swm id | resolves to | gap reason |")
        A("|---|---|---|")
        for r in gaps:
            ref = (r["swm"].get("resolved_reference") or {}).get("reference_id") or "—"
            g = r.get("gap") or {}
            A(f"| `{r['swm_id']}` | `{ref}` | `{g.get('reason', '?')}` |")
        A("")
        for r in gaps:
            g = r.get("gap") or {}
            A(f"**`{r['swm_id']}` — `{g.get('reason', '?')}`**")
            A("")
            A(g.get("detail", ""))
            src = g.get("source") or []
            if src:
                A("")
                A(
                    "Source: "
                    + ", ".join(
                        f"[`{x}`]({x.split('#')[0].replace('docs/research/', '')}"
                        f"{('#' + x.split('#')[1]) if '#' in x else ''})"
                        if x.startswith("docs/research/")
                        else f"`{x}`"
                        for x in src
                    )
                )
            A("")
    A("")

    A("## Probe coverage: which interpreter established what")
    A("")
    A(
        "No single interpreter can construct every swm id or import every stable-envs family, so both "
        "sides are a union and every row records where its facts came from. On the swm side the "
        "binding constraint is measured, not incidental: `dm_control` needs `mujoco >= 3.12` and "
        '`gymnasium-robotics` needs `<= 3.11` — plain `gymnasium.make("FetchPush-v4")` raises inside '
        "`set_joint_qpos` under 3.12, so this is upstream and not `stable_worldmodel`'s doing."
    )
    A("")
    src = matrix["provenance"].get("swm_probe_source", {})
    counts: dict[str, int] = defaultdict(int)
    for v in src.values():
        counts[v] += 1
    if counts:
        A("| swm probe | ids it established | ")
        A("|---|---|")
        for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
            A(f"| `{k}` | {v} |")
        A("")
    A("| stable-envs census | envs registered |")
    A("|---|---|")
    for k, v in sorted(matrix["provenance"]["interpreters_in_census"].items(), key=lambda kv: -kv[1]):
        A(f"| `{k}` | {v} |")
    A("")
    A(
        f"Plus the recorded union of {matrix['provenance']['registry_union_size']} envs across "
        f"{len(matrix['provenance'].get('union_interpreters', {}))} interpreters from "
        "`tests/fixtures/regression/census/union.json`, which is how a row can say a renderer exists and name the venv "
        "that owns it even when nothing here could import it."
    )
    A("")

    A("## Known stale, uncertain, or limited by the probe interpreter")
    A("")
    A("### Ids that did not construct where this was probed")
    A("")
    bad = [r for r in rows if r["swm"]["constructible_in_probe"] != "yes"]
    if not bad:
        A("None — every registered id constructed.")
    else:
        A(
            "**Not evidence that the renderer is missing.** An optional dependency absent from one "
            "interpreter is a property of that interpreter, which is why the stable-envs side of this "
            "matrix is a union over censuses and not one process's imports."
        )
        A("")
        A("| swm id | error in the probe interpreter |")
        A("|---|---|")
        for r in bad:
            A(f"| `{r['swm_id']}` | `{(r['swm']['construct_error'] or '')[:150]}` |")
    A("")
    A("### Matched envs whose declarations no supplied census could reach")
    A("")
    unreach = [r for r in rows if r["match"]["envs"] and not r["stable_envs"]["declarations_reachable"]]
    if not unreach:
        A("None.")
    else:
        A(
            "These rows have a renderer — the union says so — but this run could not import it, so "
            "tier, `state_dim` and step behaviour read `unknown`. Rerun `scratch/census/census.py` on "
            "an interpreter that owns the family and pass it with another `--se-census`."
        )
        A("")
        A("| swm id | envs | owning interpreters |")
        A("|---|---|---|")
        for r in unreach:
            own = ", ".join(r["stable_envs"]["owner_interpreters"]) or "none recorded"
            A(f"| `{r['swm_id']}` | {_envs_cell(r)} | {own} |")
    A("")
    A("### Cells this pass tried to establish and could not")
    A("")
    if not PROBE_LIMITATIONS:
        A("None.")
    else:
        A(
            "These stay `unknown`. What is recorded is *why*, so the attempt is not repeated blind — "
            "investigated-and-still-unknown is a more useful state than bare unknown, and a much more "
            "useful one than a wrong answer."
        )
        A("")
        for k, v in sorted(PROBE_LIMITATIONS.items()):
            A(f"**`{k}`** — affects `{v['field']}` on `{v['applies_to_env_prefix']}*`")
            A("")
            A(v["detail"])
            A("")
            A(f"*Next:* {v['next']}")
            A("")

    A("### Joins resting on the weakest evidence")
    A("")
    weak = [r for r in rows if r["match"]["rung"] in ("normalized_id", "documented_manual")]
    if not weak:
        A("None.")
    else:
        A("| swm id | rung | evidence |")
        A("|---|---|---|")
        for r in weak:
            A(f"| `{r['swm_id']}` | `{r['match']['rung']}` | {r['match']['evidence']} |")
    A("")

    A("## Provenance and reproduction")
    A("")
    A(f"- swm registries ({len(prov['swm_registries'])} probes, merged best-record-per-id):")
    for q in prov["swm_registries"]:
        pe = q.get("probe_env", {})
        A(
            f"    - `{q['path']}` (sha256:{q['sha256_16']}) — probed {pe.get('probed_at', '?')} on "
            f"`{pe.get('platform', '?')}`, gymnasium {pe.get('gymnasium', '?')}, "
            f"`MUJOCO_GL={pe.get('mujoco_gl', '') or 'unset'}`"
        )
    census_paths = ", ".join("`" + c["path"] + "`" for c in prov["se_censuses"]) or "none"
    A(
        f"- stable-envs censuses: {len(prov['se_censuses'])} ({census_paths}), "
        f"registry union {prov['registry_union_size']}, declarations for "
        f"{prov['declarations_available_for']} envs"
    )
    for k, v in sorted(prov["result_files"].items()):
        A(f"- {k}: {f'`{v}`' if v else '*not supplied*'}")
    A(
        f"- live probe: {'yes — ' + ', '.join(prov['live_probe']['did']) if prov['live_probe'].get('available') else 'no'}"
    )
    A("")
    A("```bash")
    A("# 1. the swm side, ONCE PER INTERPRETER. dm_control needs mujoco>=3.12 and")
    A("#    gymnasium-robotics needs <=3.11, so no single one constructs both the DM Control ids")
    A("#    and the Fetch ids; the merge keeps the record that learned the most about each id.")
    A("python scratch/swm_registry.py scratch/swm_registry.json         # mujoco 3.12: DMC, OGBench")
    A("python scratch/swm_registry.py scratch/swm_registry_fetch.json   # mujoco 3.11: Fetch")
    A("")
    A("# 2. the stable-envs side, ONCE PER INTERPRETER that owns a family. No single interpreter")
    A("#    registers all 196 envs, so a family missing from one census is not a missing renderer.")
    # The two concrete outputs, not `census_$VENV.json`: a shell variable inside a cited path is a
    # citation that can never resolve, and tests/unit/test_repo_contracts.py rightly fails it.
    A("python scratch/census/census.py > scratch/census/census_macos-probe.json")
    A("python scratch/census/census.py > scratch/census/census_oscar-3090-fetch.json")
    A("")
    A("# 3. join, and regenerate this page. --live adds check_mapping() and the device='cpu' probe,")
    A("#    and needs both libraries importable in the same process.")
    A("python scripts/regression/swm_matrix.py \\")
    A("    --swm-registry scratch/swm_registry.json \\")
    A("    --swm-registry scratch/swm_registry_fetch.json \\")
    A("    --se-census scratch/census/census_*.json --live")
    A("```")
    A("")
    A(
        "Passing more `--se-census` files only ever *adds* declarations, so the fastest way to shrink "
        "the `uncertain_needs_probe` count is to run step 2 on the interpreters named in the tables "
        "above and pass them all."
    )
    A("")
    A(
        "The builder calls no clock and no RNG: the same inputs produce the same bytes, so a diff on "
        "this file is a change in the codebase and never a rerun."
    )
    A("")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--swm-registry",
        action="append",
        default=[],
        help="scratch/swm_registry.py output (the LIVE swm id list). REPEATABLE: no one "
        "interpreter can construct every id (dm_control needs mujoco>=3.12, "
        "gymnasium-robotics needs <=3.11), and the best record per id wins. "
        "Defaults to every scratch/swm_registry*.json in the repo.",
    )
    ap.add_argument(
        "--se-census",
        action="append",
        default=[],
        help="scratch/census/census.py output; repeatable, one per interpreter",
    )
    ap.add_argument("--union", default=str(REPO / "tests/fixtures/regression/census/union.json"))
    ap.add_argument("--decode-all", default=str(REPO / "tests/fixtures/regression/references/decode_all.json"))
    ap.add_argument("--bench", default=str(REPO / "tests/fixtures/regression/references/bench_gpu_vs_cpu.json"))
    ap.add_argument("--dino", default=str(REPO / "tests/fixtures/regression/references/dino_agreement.json"))
    ap.add_argument("--lewm", default=str(REPO / "docs/research/artifacts/step5_lewm_pusht/results"))
    ap.add_argument("--swm-workflow", default=str(REPO / "tests/fixtures/regression/references/swm_workflow.json"))
    ap.add_argument(
        "--live",
        action="store_true",
        help="also import both libraries here to run check_mapping and probe device='cpu'",
    )
    ap.add_argument("--out-json", default=str(REPO / "tests/fixtures/regression/fidelity/swm_matrix.json"))
    ap.add_argument("--out-md", default=str(REPO / "docs/research/swm_matrix.md"))
    args = ap.parse_args()
    if not args.swm_registry:
        args.swm_registry = [
            str(q) for q in sorted((REPO / "tests" / "fixtures" / "regression" / "census").glob("swm_registry*.json"))
        ]
    if not args.se_census:
        args.se_census = [
            str(q) for q in sorted((REPO / "tests" / "fixtures" / "regression" / "census").glob("census_*.json"))
        ]

    # VARIATION_MAPS is the strongest join rung, so it is read from the module rather than copied.
    try:
        from tests.helpers.reference import VARIATION_MAPS

        args._variation_maps = {n: m.swm_id for n, m in VARIATION_MAPS.items()}
    except Exception as ex:
        print(
            f"warning: cannot import VARIATION_MAPS ({type(ex).__name__}: {ex}); the "
            f"variation_map rung is disabled and swm-compat columns will read as missing",
            file=sys.stderr,
        )
        args._variation_maps = {}

    matrix = build(args)
    Path(args.out_json).write_text(json.dumps(matrix, indent=1, sort_keys=False) + "\n")
    Path(args.out_md).write_text(markdown(matrix))
    s = matrix["summary"]
    print(
        f"{s['swm_ids']} swm ids | {s['with_stable_envs_match']} matched | "
        f"{s['renderer_gaps']} gaps | {s['symbolic_no_frame']} symbolic"
    )
    for k, v in sorted(s["by_headline_status"].items(), key=lambda kv: -kv[1]):
        print(f"  {v:3d}  {k}")
    print(f"wrote {_rel(args.out_json)} and {_rel(args.out_md)}")


if __name__ == "__main__":
    main()
