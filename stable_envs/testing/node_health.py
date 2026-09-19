"""Is this MACHINE fit to be measured on? Asked once, first, before any test blames the code.

One of the four EGL incidents in this repo's history was not an incident at all. Twenty-five OGBench
cells failed with 69-90/255 pixel differences on identical states, the interpreter then aborted, and
the obvious reading — the compact state does not determine the frame — is the single most damaging
conclusion this repo could reach by accident. The actual cause was that the node had been condemned by
the cluster's own health checker **41 minutes before the run started** (``sinfo`` reason
``NHC: auto-reboot``). The suite was reporting, correctly, that the OFFICIAL renderer on that host was
broken. Nobody read it that way, because nothing in the output named the host.

So this module asks two questions in increasing order of cost, and stops at the first bad answer:

  1. **What does the scheduler think of this node?** Free, no GPU work, and it is the question that
     would have ended the incident above in one line. A node in ``drain``/``down``/``fail``/``maint``
     with a reason string is not a machine to measure a renderer on, whatever the reason says.
  2. **Does the official renderer return the same pixels twice?** Two frames through ONE reference env
     via ``conformance.check_reference_renderer_is_stable``. Before any of our code is involved. If
     this fails the run is worthless and every parity number it would produce is noise.

The probe deliberately uses an env the run needs anyway, at the size the run will use, so the reference
env it builds lands in the process-wide cache and the parity tests that follow pay nothing for it.

Run standalone:  python -m stable_envs.testing.node_health
"""
from __future__ import annotations

import os
import platform
import subprocess

#: Scheduler states that mean "this host is not available for honest measurement". SLURM appends ``*``
#: (unreachable), ``~`` (powered down), ``#`` (powering up) and ``$`` (reservation) to the state, so
#: matching is on the alphabetic stem.
#: ``resv`` is deliberately NOT here: a node inside a reservation is a perfectly healthy node that your
#: job is entitled to be on, and refusing it would be a gate that fires on a good machine — which gets
#: the gate disabled, which is worse than not having one.
CONDEMNED_STATES = frozenset({"down", "drain", "drng", "draining", "drained", "fail", "failing",
                              "maint", "err", "error", "unk", "unknown", "boot", "reboot"})


def hostname() -> str:
    return os.environ.get("SLURMD_NODENAME") or platform.node()


def slurm_node_status(host: str | None = None, timeout: float = 20.0) -> dict:
    """``{state, reason, source}`` for this node, or ``{}`` when SLURM is not answering.

    ``sinfo -h -n <host> -o '%t|%E'`` is one call and is available on the compute nodes (verified).
    The reason field is the payload: it is where ``NHC: auto-reboot`` lives, along with a timestamp for
    when the node was flagged, which is what turns "the tests are broken" into "the host was condemned
    before the job started".
    """
    host = host or hostname()
    try:
        p = subprocess.run(["sinfo", "-h", "-n", host, "-o", "%t|%E"],
                           capture_output=True, text=True, timeout=timeout)
    except Exception as e:                                                   # noqa: BLE001
        return {"state": "", "reason": "", "source": f"sinfo unavailable ({type(e).__name__}: {e})"}
    if p.returncode != 0 or not p.stdout.strip():
        return {"state": "", "reason": "", "source": f"sinfo exit {p.returncode}"}
    # A node in several partitions is listed once per partition; the state is the same in each.
    line = p.stdout.strip().splitlines()[0]
    state, _, reason = line.partition("|")
    return {"state": state.strip(), "reason": reason.strip(), "source": "sinfo"}


def state_is_condemned(state: str) -> bool:
    stem = "".join(ch for ch in state.lower() if ch.isalpha())
    return stem in CONDEMNED_STATES


def _egl_probe_class(candidates=None):
    """The cheapest registered class whose reference renderer goes through EGL, or ``None``.

    Chosen by mechanism rather than by name: the EGL-backed families are the ones that bake a static
    plate (``PLATE_PATHS``) or declare a mujoco-backed ``reference_id``. Among those, the smallest maze
    is preferred — a medium pointmaze builds in a couple of seconds where a giant humanoid maze does
    not, and the question being asked ("does this rasterizer repeat itself") does not care which scene
    it is asked about.
    """
    import stable_envs as se
    names = list(candidates) if candidates else sorted(se.registry.REGISTRY)
    scored = []
    for n in names:
        cls = se.registry.REGISTRY.get(n)
        if cls is None or not hasattr(cls, "reference_env") or getattr(cls, "reference_id", None) is None:
            continue
        egl = hasattr(cls, "PLATE_PATHS") or "ogbench" in cls.__module__ or "mujoco" in str(
            getattr(cls, "reference_id", ""))
        if not egl:
            continue
        # smaller mazes first, then a stable alphabetical tiebreak
        rank = {"medium": 0, "teleport": 1, "large": 2, "giant": 3}.get(
            str(getattr(cls, "MAZE", "")).lower(), 1)
        loco = {"point": 0, "ant": 1, "humanoid": 2}.get(str(getattr(cls, "LOCO", "")).lower(), 1)
        scored.append((rank, loco, n, cls))
    if not scored:
        return None
    scored.sort(key=lambda t: (t[0], t[1], t[2]))
    return scored[0][3]


def reference_renderer_probe(candidates=None) -> dict:
    """Two frames through one reference env. ``{}`` when there is no EGL-backed env to ask about."""
    cls = _egl_probe_class(candidates)
    if cls is None:
        return {"skipped": "no EGL-backed reference env in this selection"}
    from . import conformance as C
    size = int(getattr(cls, "default_size", 224))
    try:
        r = C.check_reference_renderer_is_stable(cls, size)
    except ImportError as e:
        return {"skipped": f"{cls.__name__}: reference env not installed ({e})"}
    return {"env": cls.__name__, "size": size, "max_abs_diff": int(r["max_abs_diff"])}


def gate(candidates=None, check_scheduler: bool = True, check_renderer: bool = True) -> dict:
    """The whole precondition. ``ok`` False means: do not measure anything on this host.

    Returns a dict rather than raising so the caller decides the consequence — the pytest plugin ends
    the session with it, a benchmark script might only print it.
    """
    host = hostname()
    out = {"host": host, "ok": True, "why": "", "scheduler": {}, "renderer": {}}
    if check_scheduler:
        st = slurm_node_status(host)
        out["scheduler"] = st
        if state_is_condemned(st.get("state", "")):
            out["ok"] = False
            out["why"] = (
                f"the scheduler has condemned this node: state={st['state']!r} "
                f"reason={st.get('reason') or '(none given)'!r}. A host the cluster's own health "
                f"checker has flagged is not a host to measure a renderer on — this exact situation "
                f"once produced 25 parity failures and a SIGABRT that were read as a repo bug for "
                f"hours. Resubmit elsewhere: --exclude={host}")
            return out
    if check_renderer:
        r = reference_renderer_probe(candidates)
        out["renderer"] = r
        if r.get("max_abs_diff", 0) != 0:
            out["ok"] = False
            out["why"] = (
                f"the OFFICIAL {r.get('env')} renderer produced two different frames for the SAME "
                f"states on {host} (max {r['max_abs_diff']}/255), before any stable-envs code was "
                f"involved. This is a node fault, not a repo fault. Every parity number measurable "
                f"here is meaningless; rerun with --exclude={host}"
                + (f". Scheduler says state={out['scheduler'].get('state')!r} "
                   f"reason={out['scheduler'].get('reason')!r}" if out["scheduler"] else ""))
    return out


def main(argv=None) -> int:
    import json
    g = gate()
    print(json.dumps(g, indent=2))
    return 0 if g["ok"] else 78


if __name__ == "__main__":
    raise SystemExit(main())
