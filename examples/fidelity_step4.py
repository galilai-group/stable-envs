"""Step 4: freeze what the current renderers DRAW, before Step 6 is allowed to change any of them.

Step 3 measured how fast the renderers are. This measures what they produce, on states that are
written down, so that a Step 6 rewrite can be replayed against the identical inputs and the question
"was this broken before?" has an answer instead of an opinion.

Nothing here is a new correctness framework. Every check is a function in
``stable_envs.testing.conformance``, called with the arguments ``tests/gpu/test_conformance.py`` calls
it with; this script decides WHICH checks apply to which configuration, feeds the ones that accept
states from a frozen corpus, and records the numbers. The thresholds are not applied here at all --
``scratch/step4/certify.py`` reads them out of the frozen Step 2 spec and out of each env's own
declaration, so a bar cannot be moved in the same file that reports the measurement.

    raw shard  ->  examples/results/step4/conformance/<shard>.json
    corpus     ->  examples/results/step4/states/<config>.npz     (shared by every shard and rerun)

THREE THINGS THIS SCRIPT IS CAREFUL ABOUT, each of which has already cost somebody a wrong number:

**A dependency failure is not a fidelity failure.** An official renderer that is not installed in this
interpreter makes a check ``blocked_environment``. It never makes it ``fail``. The distinction is the
reason Step 3's shards could be merged at all, and section 12 of the Step 4 brief restates it.

**A check that draws its own states still has to be reproducible.** Most of the conformance checks
call ``env.sample_states(env.B)`` internally, which consumes the env's own generator -- so the states
a check sees depend on how many checks ran before it, and under a different check order the same
commit measures something else. ``env.rng`` is therefore re-seeded before every check. That changes no
check's semantics: each one still draws from the env's own reset distribution, it just draws the same
thing every time. The same reset puts the APPEARANCE back to the official default, which is not
cosmetic: ``check_render_is_pure`` legitimately leaves a randomised per-row style on the env, and a
parity number taken after it would compare our randomised frame against the official default one and
read as a broken renderer. See ``reseed``.

**The corpus is written once and then read.** If ``--states-dir`` already holds a config's ``.npz``,
it is LOADED, never regenerated -- across shards, across hosts and across the Step 6 rerun. Freezing
the states is the entire point of the step, and a corpus that is silently re-drawn because it was
convenient is not frozen.

    python examples/fidelity_step4.py --shard dmc --out examples/results/step4/conformance/dmc.json
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scratch" / "step4"))
import universe  # noqa: E402  (the shared, stdlib-only definition of what Step 4 validates)

import stable_envs as se  # noqa: E402
from stable_envs.testing import conformance as C  # noqa: E402

#: The draw the frozen corpora come from. One constant for the whole step, and deliberately NOT
#: ``test_conformance.PARITY_STATE_SEED`` or ``render_worker.STATE_SEED``: both of those are protocol
#: constants owned by something else, and coupling to either would mean a change over there silently
#: re-freezes this baseline. Changing this number invalidates every corpus in ``states/``.
STATE_SEED = 20260909

#: Re-seeded into ``env.rng`` before each check, so a check that draws its own states draws the same
#: ones regardless of what ran before it. See the module docstring.
CHECK_SEED = 4041

#: The corpus: ``POOL`` states drawn from the env's own reset+rollout distribution, of which ``N``
#: are kept by ``select()``. Drawing wide and selecting is what puts the extremes in: the env's own
#: distribution supplies validity, and the selection supplies coverage.
POOL, N_STATES = 256, 64

#: ``check_render_is_pure`` and friends are cheap; ``check_variations_do_something`` re-bakes a MuJoCo
#: plate per leaf on the OGBench mazes. A check that blows this budget is recorded as blocked WITH the
#: elapsed time, never dropped and never marked not_applicable.
DEFAULT_CHECK_BUDGET_S = 900.0


# ------------------------------------------------------------------ the frozen state corpus

def select(pool: np.ndarray, n: int) -> tuple[np.ndarray, dict]:
    """``n`` rows of ``pool``: the most typical state, then every per-dimension extreme, then spread.

    The brief asks for reset/typical states, extreme-but-VALID positions, orientations, targets and
    contact-relevant states. Synthesising those per family would be a second state generator with a
    per-env notion of "extreme", and the states it invented would not be states the env reaches --
    a parity number on an unreachable pose describes nothing. So every row here comes out of the env's
    own ``sample_states``, which for DM Control is ``initialize_episode`` plus up to 40 real actions,
    and the SELECTION supplies the coverage:

      1. the row nearest the pool mean in z-space -- the typical state, present by construction;
      2. up to HALF the budget on per-dimension extremes, widest dimension first: the pool's argmin
         and argmax on each state dimension, which are the extremes of the joints, targets and object
         positions the env actually produced;
      3. the remaining slots by farthest-point traversal in z-space, which fills the interior with
         states that are far from everything already chosen rather than with another near-duplicate.

    Deterministic given ``pool``: no generator is consulted, ties break on the lower index.
    """
    x = np.asarray(pool, np.float64)
    sd = x.std(0)
    z = (x - x.mean(0)) / np.where(sd > 0, sd, 1.0)
    picks: list[int] = [int(np.abs(z).sum(1).argmin())]
    # Half the budget for extremes, and the WIDEST dimensions first. Both halves are load-bearing.
    # Craftax Classic's state is 127-dimensional, so taking two extremes per dimension in index order
    # spent all 64 slots before dimension 32 and left no room for the spread at all -- a corpus in
    # which every state is extreme in some coordinate and none is ordinary. Ordering by standard
    # deviation puts the dimensions that MOVE first, which for a scene is the ones a renderer draws.
    budget = max(1, n // 2)
    for d in sorted(range(x.shape[1]), key=lambda k: (-sd[k], k)):
        if len(picks) >= budget + 1:
            break
        for i in (int(x[:, d].argmin()), int(x[:, d].argmax())):
            if i not in picks and len(picks) < budget + 1:
                picks.append(i)
    n_extreme = len(picks)
    if len(picks) < n:
        d2 = ((z[:, None, :] - z[None, picks, :]) ** 2).sum(-1).min(1)
        while len(picks) < n and len(picks) < len(x):
            d2[picks] = -1.0
            i = int(d2.argmax())
            picks.append(i)
            d2 = np.minimum(d2, ((z - z[i]) ** 2).sum(-1))
    idx = np.array(picks[:n], int)
    return pool[idx], {"pool": int(len(pool)), "kept": int(len(idx)),
                       "n_typical": 1, "n_extreme": int(n_extreme - 1),
                       "n_spread": int(len(idx) - n_extreme), "indices": idx.tolist()}


def corpus_for(env, config_id: str, states_dir: Path, n: int, pool: int) -> tuple[np.ndarray, dict]:
    """Load ``config_id``'s frozen corpus, or draw and freeze it. Never redraws an existing one."""
    p = states_dir / f"{config_id}.npz"
    if p.exists():
        z = np.load(p, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        meta["loaded_from"] = str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)
        return z["states"], meta
    raw = env.sample_states(int(pool), rng=np.random.default_rng(STATE_SEED))
    states, how = select(np.asarray(raw, np.float32), int(n))
    states = np.ascontiguousarray(states, np.float32)
    meta = {"seed": STATE_SEED, "selection": "typical + per-dim extremes + farthest-point spread",
            "state_dim": int(states.shape[1]), "state_names": list(env.state_names or []),
            "env": config_id, "sha256": C.frames_digest(states), **how}
    states_dir.mkdir(parents=True, exist_ok=True)
    np.savez(p, states=states, meta=np.array(json.dumps(meta, sort_keys=True)))
    meta["written_to"] = str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)
    return states, meta


# ------------------------------------------------------------------ running one check

#: An exception that says the interpreter could not REACH the renderer, as opposed to one that says
#: the renderer was reached and was wrong. Only the second kind may become ``fail``.
BLOCKING = (ImportError, ModuleNotFoundError, MemoryError, torch.cuda.OutOfMemoryError)


def run(fn, *a, budget: float = DEFAULT_CHECK_BUDGET_S, **kw) -> dict:
    """Call one conformance check and classify the outcome. Never raises, never omits."""
    t0 = time.perf_counter()
    try:
        r = fn(*a, **kw)
    except BLOCKING as e:
        return {"status": "blocked_environment", "why": f"{type(e).__name__}: {str(e)[:300]}",
                "seconds": round(time.perf_counter() - t0, 2)}
    except NotImplementedError as e:
        return {"status": "not_applicable", "why": f"{type(e).__name__}: {str(e)[:300]}",
                "seconds": round(time.perf_counter() - t0, 2)}
    except BaseException as e:                                              # noqa: BLE001
        return {"status": "error", "why": f"{type(e).__name__}: {str(e)[:600]}",
                "traceback": traceback.format_exc()[-1200:],
                "seconds": round(time.perf_counter() - t0, 2)}
    dt = round(time.perf_counter() - t0, 2)
    if isinstance(r, dict) and "skipped" in r:
        return {"status": "not_applicable", "why": r["skipped"], "seconds": dt}
    out = {"status": "measured", "evidence": _jsonable(r), "seconds": dt}
    if dt > budget:
        out["over_budget_s"] = budget
    return out


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def reseed(env) -> None:
    """Put the env back to a known start: seeded generator, OFFICIAL appearance.

    The generator is the reproducibility half (module docstring). The style is a CORRECTNESS half, and
    it is not optional: ``check_render_is_pure`` legitimately calls ``set_state(states, style=...)``,
    which leaves a 64-row randomised appearance ON the env, and every later call that renders without
    naming a style then draws THAT appearance. The official frames are drawn at the default, so a
    parity number taken in that state compares two pictures of different scenes and comes out looking
    like a broken renderer. ``set_style(None)`` rebuilds from the variation space's own defaults, which
    is what ``default_style()`` packs and what parity is defined against.
    """
    env.rng = np.random.default_rng(CHECK_SEED)
    if env.style_dim:
        env.set_style(None)


# ------------------------------------------------------------------ the check battery

def mandatory(env, states, style) -> dict:
    """Section 4.1's mandatory list, in a fixed order. Every certified env, both tiers, no exceptions."""
    out = {}
    for name, call in (
        ("check_state_roundtrip", lambda: C.check_state_roundtrip(env)),
        ("check_state_dims_matter", lambda: C.check_state_dims_matter(env)),
        ("check_batch_invariance", lambda: C.check_batch_invariance(env)),
        ("check_batch_size_invariance", lambda: C.check_batch_size_invariance(env, states, style)),
        ("check_order_independence", lambda: C.check_order_independence(env, states, style)),
        ("check_determinism", lambda: C.check_determinism(env)),
        ("check_render_is_pure", lambda: C.check_render_is_pure(env, states, style)),
        ("check_chunked_render_matches", lambda: C.check_chunked_render_matches(env, states, style)),
        ("check_input_form_invariance", lambda: C.check_input_form_invariance(env, states, style)),
        ("check_set_state_copies", lambda: C.check_set_state_copies(env)),
        ("check_style_is_state", lambda: C.check_style_is_state(env)),
        ("check_default_style_is_official", lambda: C.check_default_style_is_official(env)),
        ("check_per_env_variation", lambda: C.check_per_env_variation(env)),
        ("check_fidelity_declaration", lambda: _fidelity_declaration(env)),
    ):
        reseed(env)
        out[name] = run(call)
    return out


def _fidelity_declaration(env) -> dict:
    """``register``'s own enforcement, re-run on the live class. Raises if the tier is half-declared."""
    from stable_envs.core.env import check_fidelity_declaration, reference_versions_of
    check_fidelity_declaration(type(env))
    d = env.describe()
    return {"tier": env.fidelity_tier, "describe_reports_tier": d.get("fidelity_tier"),
            "agrees": d.get("fidelity_tier") == env.fidelity_tier,
            "parity_interior_min": _jsonable(getattr(env, "parity_interior_min", None)),
            "parity_all_exact_min": _jsonable(getattr(env, "parity_all_exact_min", None)),
            "visual_fidelity": _jsonable(getattr(env, "visual_fidelity", None) or None),
            # Recorded separately and never defaulted from visual_fidelity: they are different
            # declarations with different numbers, and only a LEARNED decoder carries this one.
            "variation_fidelity": _jsonable(getattr(env, "variation_fidelity", None) or None),
            # The reference versions the declaration names, for either tier. The certifier compares
            # them with this run's ``reference_packages`` and blocks a mismatch rather than failing it.
            "reference_versions": _jsonable(reference_versions_of(type(env)) or None),
            "render_decoder": getattr(env, "render_decoder", "renderer")}


def family_specific(env, states, style) -> dict:
    """Section 4.1's family-specific list: mandatory where DECLARED, ``not_applicable`` otherwise.

    The applicability test is written out per check rather than inferred from whether the call happened
    to succeed, because "it raised, so it must not apply" is how a real failure becomes a skip.
    """
    cls, tier = type(env), env.fidelity_tier
    has_ref = bool(getattr(cls, "reference_id", None)) and hasattr(cls, "reference_env")
    out = {}

    def na(name, why):
        out[name] = {"status": "not_applicable", "why": why}

    if tier == "parity" and has_ref:
        reseed(env)
        out["render_parity_with_official_renderer"] = run(_parity, env, states, style)
    else:
        na("render_parity_with_official_renderer",
           "declares the VISUAL tier; asserted against its own numbers instead" if tier == "visual"
           else "no official reference renderer is declared")

    if tier == "visual" and has_ref:
        reseed(env)
        out["check_visual_fidelity"] = run(C.check_visual_fidelity, env, n=len(states))
        reseed(env)
        out["check_visual_fidelity_under_variation"] = run(
            C.check_visual_fidelity_under_variation, env, n=len(states), seed=0)
    else:
        why = ("declares the PARITY tier; held to interior-pixel exactness instead" if tier == "parity"
               else "no official reference renderer is declared")
        na("check_visual_fidelity", why)
        na("check_visual_fidelity_under_variation", why)

    reseed(env)
    out["check_backend_agreement"] = run(C.check_backend_agreement, env)

    if env.variation_space.names():
        reseed(env)
        out["check_variations_do_something"] = run(C.check_variations_do_something, env)
        reseed(env)
        out["check_colors_apply_exactly"] = run(C.check_colors_apply_exactly, env)
    else:
        na("check_variations_do_something", "the env declares an empty variation space")
        na("check_colors_apply_exactly", "the env declares an empty variation space")

    # No style, deliberately. The floor this is judged against -- DOWNSAMPLE_EXACT_FRAC in
    # tests/gpu/test_reproducibility.py -- was recorded at the DEFAULT appearance, because the number
    # is a property of the geometry; measuring under a sampled style would produce a different
    # quantity and compare it to that floor anyway.
    reseed(env)
    out["check_resolution_consistency"] = _resolution(env, states)
    return out


def _resolution(env, states) -> dict:
    """The 2x-render-and-downsample check, with the suite's own applicability rule applied.

    An env that renders at one size only raises ``ValueError`` from ``render(size=2S)``, and that is
    the env ANSWERING the applicability question in section 4.1 -- "applies when the env can render at
    more than one size". ``test_the_downsampled_2x_render_lands_where_the_1x_render_does`` skips on
    exactly that exception; classifying it as an error here would turn six envs that are behaving as
    designed into Gate A failures.

    The JUDGED number is measured on the bound's own protocol (``C.downsample_protocol_states``: the
    8 states the floor was recorded on), not on the frozen corpus. The floor is a property of the
    state set as much as of the renderer -- the same renderer measures 0.6188 on walker's 64-state
    corpus and 0.6290, the recorded floor to six decimals, on the protocol's 8 (both mujoco
    versions; ``examples/results/step6/texture-fidelity/resprobe31{1,2}.json``) -- so a floor applied
    to another draw with a 0.005 margin failed the draw, not the renderer. The corpus number is still
    measured and recorded beside it as ``frozen_corpus``, because Step 6 replays the corpus and a
    regression that moves a shape shows up in either.
    """
    # The draw first and on its own: the ``except ValueError`` below is the env saying it renders at
    # one size only, and a ValueError out of ``sample_states`` is not that -- it must be an error,
    # not a skip.
    try:
        proto = C.downsample_protocol_states(env)
    except BLOCKING as e:
        return {"status": "blocked_environment", "why": f"{type(e).__name__}: {str(e)[:300]}"}
    except BaseException as e:                                              # noqa: BLE001
        return {"status": "error", "why": f"protocol draw: {type(e).__name__}: {str(e)[:400]}",
                "traceback": traceback.format_exc()[-1200:]}
    try:
        r = C.check_resolution_consistency(env, proto)
        r["protocol"] = C.downsample_protocol(proto)
        r["frozen_corpus"] = C.check_resolution_consistency(env, states)
        r["frozen_corpus"]["n_states"] = int(len(states))
        r["frozen_corpus"]["states_sha256"] = C.frames_digest(np.ascontiguousarray(states, np.float32))
    except ValueError as e:
        return {"status": "not_applicable", "why": f"{type(e).__name__}: {str(e)[:300]}"}
    except BLOCKING as e:
        return {"status": "blocked_environment", "why": f"{type(e).__name__}: {str(e)[:300]}"}
    except BaseException as e:                                              # noqa: BLE001
        return {"status": "error", "why": f"{type(e).__name__}: {str(e)[:400]}",
                "traceback": traceback.format_exc()[-1200:]}
    if isinstance(r, dict) and "skipped" in r:
        return {"status": "not_applicable", "why": r["skipped"]}
    return {"status": "measured", "evidence": _jsonable(r)}


def _parity(env, states, style) -> dict:
    """Our frame against the OFFICIAL one, on the frozen corpus. The measurement, not the verdict.

    ``test_render_parity_with_official_renderer`` draws its own 64 states from its own seed; this uses
    the frozen corpus instead, because a baseline Step 6 must be able to replay has to be replayable.
    The numbers are the same numbers and the floor is applied later, by the certifier, out of the env's
    own ``parity_interior_min``.
    """
    cls = type(env)
    ref = C.reference_env(cls, env.size)
    official = cls.reference_render(ref, states)
    ours = env.render(states).cpu().numpy()
    if ours.shape != official.shape:
        raise AssertionError(f"shape {ours.shape} vs official {official.shape}")
    out = C.pixel_agreement(ours, official)
    out["ssim"] = C.ssim(ours, official)
    out["n_states"] = int(len(states))
    out["reference_id"] = cls.reference_id
    out["digest_ours"] = C.frames_digest(ours)
    out["digest_official"] = C.frames_digest(official)
    return out


def physics(env) -> dict:
    """Gate E, kept on its own axis. A ``reference_only`` env is not a renderer failure -- section 3."""
    cls = type(env)
    if "step" not in cls.__dict__ or not getattr(cls, "step_reference_checkable", False):
        return {"status": "not_applicable",
                "why": "no batched simulator declared checkable against a reference "
                       "(step_reference_checkable is unset), so this env's physics is reference_only"}
    reseed(env)
    return run(C.check_step_matches_reference, env)


# ------------------------------------------------------------------ the regression oracle

def oracle(env, config: dict, states, style) -> dict:
    """What Step 6 replays against: our frames, the official frames, and digests of both.

    Digests rather than pixels. A frame stack for one config at 480 px is 42 MB and the repo's whole
    thesis is that frames are cheap to regenerate; 32 bytes per side is what a later run needs to
    prove it produced the same thing, and the metrics beside them are what it needs when it did not.
    """
    cls = type(env)
    out = {"size": int(env.size), "supersample": int(env.ss), "batch": int(env.B),
           "state_dim": int(env.state_dim), "style_dim": int(env.style_dim),
           "device": str(env.device), "n_states": int(len(states))}
    ours = env.render(states).cpu().numpy()
    out["frames_shape"] = list(ours.shape)
    out["digest_ours"] = C.frames_digest(ours)
    out["digest_ours_default_style"] = C.frames_digest(
        env.render(states, style=np.tile(np.asarray(env.default_style(), np.float32), (len(states), 1)))
    ) if env.style_dim else out["digest_ours"]
    out["mean_pixel"] = float(ours.mean())
    out["distinct_colors"] = int(len(np.unique(ours.reshape(-1, 3), axis=0)))
    if style is not None:
        out["digest_ours_corpus_style"] = C.frames_digest(env.render(states, style=style))
        out["style_digest"] = C.frames_digest(np.ascontiguousarray(style, np.float32))
    try:
        ref = C.reference_env(cls, env.size)
        official = cls.reference_render(ref, states)
        out["digest_official"] = C.frames_digest(official)
        out["reference_id"] = cls.reference_id
    except BLOCKING as e:
        out["reference"] = {"status": "blocked_environment", "why": f"{type(e).__name__}: {str(e)[:200]}"}
    except BaseException as e:                                              # noqa: BLE001
        out["reference"] = {"status": "error", "why": f"{type(e).__name__}: {str(e)[:300]}"}
    # Its own wrapper, deliberately after the digest. This is section 4.1's DIAGNOSTIC -- a property of
    # the node's rasterizer, never of our renderer -- and it builds its own env, so letting it share the
    # try above would let a diagnostic failure delete the official digest that had already been taken.
    out["reference_stable"] = run(C.check_reference_renderer_is_stable, cls, env.size)
    return out


def signature(env) -> dict:
    """The render-model signature Step 4 certifies against, in Step 1's own terms where it has them.

    For a DM Control cell this is Step 1's digest verbatim -- the same ``FIELDS`` over the compiled
    model taken AFTER ``reset()``, which is what caught ``reacher`` defaulting to ``hard`` -- so a
    Step 4 result can be tied to the row Step 1 matched rather than to a name that happens to agree.
    Everything else digests the env's own declared configuration, which is the whole of what the frame
    is a function of for a renderer that has no MuJoCo model behind it.
    """
    import hashlib
    d = env.describe()
    keys = ("name", "state_dim", "state_names", "size", "style_dim", "style_names", "reference_id",
            "fidelity_tier", "variations", "parity_interior_min")
    payload = json.dumps({k: _jsonable(d.get(k)) for k in keys}, sort_keys=True)
    out = {"describe_digest": hashlib.sha256(payload.encode()).hexdigest()[:16],
           "describe": _jsonable({k: d.get(k) for k in keys})}
    dom, task = getattr(type(env), "DOMAIN", ""), getattr(type(env), "TASK", "")
    if dom and task:
        out["dm_control"] = {"domain": dom, "task": task,
                             "model_digest": _dmc_model_digest(dom, task)}
    return out


_DMC_DIGESTS: dict = {}


def _dmc_model_digest(domain: str, task: str) -> str | None:
    """Step 1's ``_dmc_appearance_groups`` digest for one cell, computed by Step 1's own function."""
    if not _DMC_DIGESTS:
        try:
            sys.path.insert(0, str(ROOT))
            from scratch.swm_matrix import _dmc_appearance_groups
            _DMC_DIGESTS.update(_dmc_appearance_groups() or {"__none__": {}})
        except BaseException:                                               # noqa: BLE001
            _DMC_DIGESTS["__none__"] = {}
    return (_DMC_DIGESTS.get(domain) or {}).get(task)


# ------------------------------------------------------------------ shard driver

def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return ""


def provenance(device: str) -> dict:
    cpu = platform.processor()
    if platform.system() == "Linux":
        cpu = _run(["bash", "-c", "lscpu | grep 'Model name' | head -1"]).split(":", 1)[-1].strip() or cpu
    elif platform.system() == "Darwin":
        cpu = _run(["sysctl", "-n", "machdep.cpu.brand_string"]) or cpu
    # The measured copy on the cluster is an rsync without .git, which is how Step 4's shards came to
    # record git_commit: null. The launcher stamps the SHA into STABLE_ENVS_GIT_COMMIT or a GIT_COMMIT
    # file at the root; a real checkout still wins, and the source is recorded beside the value.
    commit, source = _run(["git", "-C", str(ROOT), "rev-parse", "HEAD"]) or None, "git"
    dirty = bool(_run(["git", "-C", str(ROOT), "status", "--porcelain"])) if commit else None
    if not commit and os.environ.get("STABLE_ENVS_GIT_COMMIT", "").strip():
        commit, source = os.environ["STABLE_ENVS_GIT_COMMIT"].strip(), "env:STABLE_ENVS_GIT_COMMIT"
    elif not commit and (ROOT / "GIT_COMMIT").is_file():
        commit, source = (ROOT / "GIT_COMMIT").read_text().strip() or None, "file:GIT_COMMIT"
    p = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
         "git_commit": commit, "git_commit_source": source if commit else None,
         "git_dirty": dirty,
         "host": platform.node(), "os": f"{platform.system()} {platform.release()}",
         "cpu": cpu, "python": platform.python_version(), "torch": torch.__version__,
         "device": device, "state_seed": STATE_SEED, "check_seed": CHECK_SEED,
         "n_states": N_STATES, "pool": POOL,
         "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
         "slurm_nodelist": os.environ.get("SLURM_JOB_NODELIST"),
         "mujoco_gl": os.environ.get("MUJOCO_GL"),
         "registry_here": sorted(se.REGISTRY), "import_errors": _jsonable(se.envs.import_errors())}
    if device == "cuda" and torch.cuda.is_available():
        p["gpu"] = torch.cuda.get_device_name(0)
        p["driver"] = _run(["bash", "-c",
                            "nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1"])
    import importlib.metadata as md
    for m in ("mujoco", "mujoco_warp", "dm_control", "ogbench", "gym_pusht", "gymnasium",
              "gymnasium_robotics", "craftax", "stable_worldmodel", "warp"):
        # The distribution's version first: ``dm_control`` has no ``__version__`` and was recorded as
        # "present", which the certifier cannot compare with a declared reference version.
        try:
            p.setdefault("reference_packages", {})[m] = md.version(m)
            continue
        except BaseException:                                               # noqa: BLE001
            pass
        try:
            p.setdefault("reference_packages", {})[m] = getattr(__import__(m), "__version__", "present")
        except BaseException:                                               # noqa: BLE001
            p.setdefault("reference_packages", {})[m] = None
    return p


def measure(config: dict, states_dir: Path, n: int, pool: int, device: str, images: Path | None) -> dict:
    """One configuration, end to end. Any failure to even build it is recorded, never raised."""
    name = config["config_id"]
    row = {"config_id": name, "env": config["env"], "swm_ids": config["swm_ids"],
           "role": config["role"], "siblings": config["siblings"],
           "registered_default_coverage": config["registered_default_coverage"],
           "declared_tier": config["fidelity_tier"], "backends": config["backends"]}
    t0 = time.perf_counter()
    if name not in se.REGISTRY:
        row["status"] = "blocked_environment"
        row["why"] = (f"{name} is not registered in this interpreter; env-module import errors: "
                      f"{se.envs.import_errors()}")
        return row
    try:
        env = se.make(name, batch=n, device=device)
    except BaseException as e:                                              # noqa: BLE001
        row["status"] = ("blocked_environment" if isinstance(e, BLOCKING) else "error")
        row["why"] = f"{type(e).__name__}: {str(e)[:400]}"
        row["traceback"] = traceback.format_exc()[-1200:]
        return row

    row["status"] = "measured"
    states, row["corpus"] = corpus_for(env, name, states_dir, n, pool)
    style = C.uniform_style(env, len(states), groups=2, seed=0)
    row["signature"] = signature(env)
    row["mandatory"] = mandatory(env, states, style)
    row["family_specific"] = family_specific(env, states, style)
    row["physics"] = physics(env)
    reseed(env)
    row["oracle"] = oracle(env, config, states, style)
    reseed(env)
    row["sibling_digest_matches"] = siblings(env, config, states)
    if images is not None:
        row["images"] = contact_sheet(env, name, states, images)
    row["seconds"] = round(time.perf_counter() - t0, 1)
    return row


def siblings(env, config: dict, states) -> dict:
    """Re-measure the collapse: does each sibling draw THIS config's picture on these exact states?

    Step 1 measured that OGBench's dataset suffixes over one ``robot x maze`` cell compile to one
    scene, and Step 4 validates one env per cell on the strength of it. That is evidence propagation,
    and section 6 says it has to be explicit rather than assumed -- so the siblings are rendered once
    each here and their digests compared. A mismatch does not fail the config; it says the collapse is
    wrong and the sibling needs validating in its own right.
    """
    out: dict = {}
    if not config["siblings"]:
        return out
    mine = C.frames_digest(env.render(states))
    for s in config["siblings"]:
        if s not in se.REGISTRY:
            out[s] = {"status": "blocked_environment", "why": "not registered in this interpreter"}
            continue
        try:
            other = se.make(s, batch=len(states), size=env.size, device=str(env.device))
            if other.state_dim != env.state_dim:
                out[s] = {"status": "not_comparable",
                          "why": f"state_dim {other.state_dim} vs {env.state_dim}: not the same corpus"}
                continue
            d = C.frames_digest(other.render(states))
            out[s] = {"status": "measured", "identical": d == mine, "digest": d}
        except BaseException as e:                                          # noqa: BLE001
            out[s] = {"status": ("blocked_environment" if isinstance(e, BLOCKING) else "error"),
                      "why": f"{type(e).__name__}: {str(e)[:200]}"}
    out["_digest_this_config"] = mine
    return out


def contact_sheet(env, name: str, states, outdir: Path, rows: int = 4) -> dict:
    """``official | ours | 4x |difference|`` for a few states. For a human, not for a threshold.

    Section 11 wants approximations made visible before Step 6 changes them, and explicitly not an
    eyeball pass/fail. The difference panel is amplified 4x and clipped, because a visual-tier residual
    of 5/255 is invisible at 1x and the point of the sheet is to see WHERE the renderers disagree.
    """
    from PIL import Image
    cls = type(env)
    idx = np.linspace(0, len(states) - 1, rows).astype(int)
    sub = states[idx]
    # The default appearance, named rather than inherited: the env carries whatever the last check
    # left on it, and this sheet is the record of what the DEFAULT render looks like.
    sty = np.tile(np.asarray(env.default_style(), np.float32), (len(sub), 1)) if env.style_dim else None
    ours = env.render(sub, style=sty).cpu().numpy()
    try:
        official = cls.reference_render(C.reference_env(cls, env.size), sub)
    except BaseException as e:                                              # noqa: BLE001
        official = None
        why = f"{type(e).__name__}: {str(e)[:200]}"
    outdir.mkdir(parents=True, exist_ok=True)
    if official is None:
        panels = ours
        meta = {"panels": ["stable-envs"], "reference": why}
    else:
        d = np.abs(ours.astype(np.int16) - official.astype(np.int16))
        amp = np.clip(d * 4, 0, 255).astype(np.uint8)
        panels = np.concatenate([official, ours, amp], axis=2)
        meta = {"panels": ["official", "stable-envs", "|difference| x4"],
                "max_abs": int(d.max()), "mean_abs": float(d.mean())}
    sheet = np.concatenate(list(panels), axis=0)
    p = outdir / f"{name}.png"
    Image.fromarray(sheet).save(p)
    meta.update({"path": str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p),
                 "state_indices": idx.tolist(), "rows": int(rows),
                 "sha256": C.frames_digest(sheet)})
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="", help="dmc | fetch | ogb | raster | swm | '' for all here")
    ap.add_argument("--configs", default="", help="comma-separated config ids, overrides --shard")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True)
    ap.add_argument("--states-dir", default=str(ROOT / "examples" / "results" / "step4" / "states"))
    ap.add_argument("--images", default="", help="write contact sheets here; off when empty")
    ap.add_argument("--image-configs", default="",
                    help="comma-separated config ids to sheet; empty means every config in the shard. "
                         "Section 11 asks for a REPRESENTATIVE set, and a sheet per configuration "
                         "would put ~40 MB of PNG in a repository whose thesis is that frames are "
                         "cheap to regenerate.")
    ap.add_argument("--n-states", type=int, default=N_STATES)
    ap.add_argument("--pool", type=int, default=POOL)
    ap.add_argument("--resume", action="store_true", help="keep configs already in --out")
    a = ap.parse_args()

    all_cfg = universe.configurations()
    if a.configs:
        want = [c for c in all_cfg if c["config_id"] in a.configs.split(",")]
    elif a.shard:
        want = [c for c in all_cfg if universe.shard_of(c) == a.shard]
    else:
        want = [c for c in all_cfg if c["config_id"] in se.REGISTRY]

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = {"schema_version": 1, "step": 4, "shard": a.shard or "adhoc",
           "what_this_is": "raw Step 4 conformance/fidelity measurements for one interpreter. No "
                           "threshold is applied here; scratch/step4/certify.py does that from the "
                           "frozen Step 2 spec.",
           "provenance": provenance(a.device), "configs": {}}
    if a.resume and out.exists():
        prev = json.loads(out.read_text())
        doc["configs"] = {k: v for k, v in prev.get("configs", {}).items()
                          if v.get("status") == "measured"}
        print(f"resuming: {len(doc['configs'])} config(s) already measured", flush=True)

    states_dir = Path(a.states_dir)
    images = Path(a.images) if a.images else None
    print(f"{len(want)} configuration(s) on {a.device}; corpus {a.n_states} of {a.pool} "
          f"@ seed {STATE_SEED}", flush=True)
    for c in want:
        if c["config_id"] in doc["configs"]:
            continue
        want_img = images if (images and (not a.image_configs
                                          or c["config_id"] in a.image_configs.split(","))) else None
        r = measure(c, states_dir, a.n_states, a.pool, a.device, want_img)
        doc["configs"][c["config_id"]] = r
        n_bad = sum(1 for g in ("mandatory", "family_specific")
                    for v in (r.get(g) or {}).values() if v.get("status") == "error")
        print(f"[{c['config_id']:26s}] {r['status']:20s} {r.get('seconds', 0)}s"
              + (f"  {n_bad} check error(s)" if n_bad else "")
              + (f"  {r.get('why', '')[:120]}" if r["status"] != "measured" else ""), flush=True)
        out.write_text(json.dumps(doc, indent=1, sort_keys=True))
    out.write_text(json.dumps(doc, indent=1, sort_keys=True))
    print(f"\nwrote {out}  ({len(doc['configs'])} configs)")


if __name__ == "__main__":
    main()
