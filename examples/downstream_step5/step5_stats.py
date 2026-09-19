"""The Step 2 downstream gate, computed rather than restated.

Every bound this module applies is read from ``scratch/certification.json`` at import time -- the
same file Step 2 froze and Step 4's certifier reads. Nothing here declares a threshold of its own,
so a number in a Step 5 report cannot drift from the policy that authorised it: change the frozen
spec and these verdicts change with it, edit this file and the tests that compare it against the
spec fail.

The gate is PAIRED. ``b`` is the count of episodes the official renderer won and ours lost, ``c``
the reverse; concordant episodes carry no information about a renderer difference and are excluded
from the test by construction. A non-significant McNemar p is NOT the claim -- an underpowered
experiment produces one for free. The claim is that the 95% confidence interval on the paired
difference lies entirely inside the equivalence margin, which is a statement an underpowered
experiment cannot make.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

#: The frozen Step 2 downstream policy. Read, never edited: Step 5 is a measurement against the
#: rule Step 2 wrote down before any downstream number existed.
SPEC = json.loads((ROOT / "scratch" / "certification.json").read_text())["downstream"]


def wilson(k: int, n: int, z: float | None = None) -> tuple[float, float]:
    """Wilson score interval, in percentage points.

    Wilson rather than the normal approximation because the arms sit near 90% success, where the
    textbook interval runs past 100 and is asymmetric in the wrong direction.
    """
    z = SPEC["ci_z"] if z is None else z
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * max(0.0, centre - half), 100 * min(1.0, centre + half))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar: the sign test on the ``b + c`` discordant pairs.

    Exact rather than the chi-square approximation because the control arm has ``c = 0`` and the
    approximation is not trustworthy in that corner -- which is precisely the corner the control is
    supposed to land in.
    """
    n = b + c
    if n == 0:
        return 1.0
    lo = min(b, c)
    tail = sum(math.comb(n, i) for i in range(lo + 1)) * (0.5**n)
    return min(1.0, 2 * tail)


def paired(off: list[int], ours: list[int]) -> dict:
    """The paired difference and its interval, by the frozen formula.

    ``ci_formula`` in the spec is ``delta +/- 1.96 * sqrt(b + c) / n``; it is reproduced here rather
    than reimplemented from a textbook so that the interval this returns is the interval Step 2
    described.
    """
    if len(off) != len(ours):
        raise ValueError(f"arms are not paired: {len(off)} vs {len(ours)} episodes")
    n = len(off)
    b = sum(1 for a, o in zip(off, ours) if a == 1 and o == 0)
    c = sum(1 for a, o in zip(off, ours) if a == 0 and o == 1)
    z = SPEC["ci_z"]
    delta = 100.0 * (c - b) / n if n else 0.0
    half = 100.0 * z * math.sqrt(b + c) / n if n else float("inf")
    disc = (b + c) / n if n else 0.0
    margin = SPEC["equivalence_margin_points"]
    return {
        "n": n,
        "b": b,
        "c": c,
        "delta_points": delta,
        "ci": [delta - half, delta + half],
        "half_width_points": half,
        "mcnemar_p": mcnemar_exact(b, c),
        "discordance": disc,
        "inside_margin": bool(delta - half >= -margin and delta + half <= margin),
        "required_n_for_margin": math.ceil(3.8416 * disc / (margin / 100.0) ** 2) if disc else 0,
    }


def arm(successes: list[int]) -> dict:
    k, n = int(sum(successes)), len(successes)
    lo, hi = wilson(k, n)
    return {"successes": k, "n": n, "success_rate": 100.0 * k / n if n else 0.0, "wilson_ci": [lo, hi]}


def gate_h(
    official: list[int],
    ours: list[int],
    control: list[int] | None,
    seeds: list[int] | None = None,
    per_seed_n: list[int] | None = None,
) -> dict:
    """The whole verdict, including the parts that make a null result mean something.

    Three ways to not be a PASS, kept distinct because they call for different work:
    ``FAIL`` (the renderers differ by more than the margin), ``INCONCLUSIVE`` (the evaluation was
    never shown to depend on pixels, or is too noisy at this ``n`` to resolve the margin), and
    ``DESIGN_SHORTFALL`` (the run is smaller than the design the spec requires).
    """
    eq = paired(official, ours)
    out = {
        "official": arm(official),
        "stable_envs": arm(ours),
        "equivalence": eq,
        "margin_points": SPEC["equivalence_margin_points"],
        "design": {
            "min_paired_episodes": SPEC["min_paired_episodes"],
            "min_seeds": SPEC["min_seeds"],
            "min_episodes_per_seed": SPEC["min_episodes_per_seed"],
            "paired_episodes": eq["n"],
            "seeds": sorted(set(seeds)) if seeds else [],
            "n_seeds": len(set(seeds)) if seeds else 0,
            "episodes_per_seed": sorted(set(per_seed_n)) if per_seed_n else [],
        },
    }
    out["design"]["met"] = bool(
        eq["n"] >= SPEC["min_paired_episodes"]
        and (not seeds or len(set(seeds)) >= SPEC["min_seeds"])
        and (not per_seed_n or min(per_seed_n) >= SPEC["min_episodes_per_seed"])
    )

    if control is None:
        out["control"] = {"present": False, "degraded": False, "note": "no destructive control arm was run"}
    else:
        cp = paired(official[: len(control)], control)
        out["control"] = {
            "present": True,
            "arm": arm(control),
            "paired": cp,
            "max_delta_points": SPEC["control_max_delta_points"],
            "max_p": SPEC["control_max_p"],
            "degraded": bool(
                cp["delta_points"] <= SPEC["control_max_delta_points"] and cp["mcnemar_p"] < SPEC["control_max_p"]
            ),
        }

    # The guard is about whether THIS n can resolve the margin at THIS discordance, which is what
    # `required_n_formula` computes. `max_discordance_at_n300` is, by its own name, the ceiling for
    # a 300-episode run -- 3.8416 * 0.19 / 0.05^2 = 292, i.e. it IS that formula evaluated at 300.
    # Testing discordance against it regardless of n turns an adequately powered FAIL at n=600 into
    # an INCONCLUSIVE, which is the comfortable answer rather than the true one.
    required = eq["required_n_for_margin"]
    underpowered = eq["n"] < required
    out["power_guard"] = {
        "max_discordance_at_n300": SPEC["power_guard"]["max_discordance_at_n300"],
        "observed_discordance": eq["discordance"],
        "underpowered": bool(underpowered),
        "required_n": required,
        "n": eq["n"],
    }

    if not out["control"]["degraded"]:
        verdict, why = (
            "INCONCLUSIVE",
            SPEC["outcome_without_control_degradation"] + ": the evaluation was not shown to depend on the pixels",
        )
    elif not out["design"]["met"]:
        verdict, why = (
            "DESIGN_SHORTFALL",
            (f"{eq['n']} paired episodes against a required {SPEC['min_paired_episodes']}"),
        )
    elif eq["inside_margin"]:
        verdict, why = (
            "PASS",
            (
                f"95% CI [{eq['ci'][0]:.2f}, {eq['ci'][1]:.2f}] lies inside "
                f"+/-{SPEC['equivalence_margin_points']} points"
            ),
        )
    elif underpowered:
        verdict, why = (
            "INCONCLUSIVE",
            (
                f"discordance {eq['discordance']:.3f} needs {required} paired episodes to resolve a "
                f"+/-{SPEC['equivalence_margin_points']} point margin and {eq['n']} were run"
            ),
        )
    else:
        verdict, why = (
            "FAIL",
            (
                f"95% CI [{eq['ci'][0]:.2f}, {eq['ci'][1]:.2f}] is not inside "
                f"+/-{SPEC['equivalence_margin_points']} points, at {eq['n']} paired episodes against "
                f"the {required} this discordance requires"
            ),
        )
    out["verdict"], out["verdict_reason"] = verdict, why
    return out
