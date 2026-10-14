"""Cluster node health check run before measuring.

Reads the node's state from the cluster scheduler (``sinfo``), when one is available, and renders the
same states twice through an official reference renderer to confirm the host draws deterministically.
"""

from __future__ import annotations

import os
import platform
import subprocess

# Compare alphabetic state stems, ignoring scheduler suffixes.
# Reservations are valid allocations and are not condemned states.
CONDEMNED_STATES = frozenset(
    {
        "down",
        "drain",
        "drng",
        "draining",
        "drained",
        "fail",
        "failing",
        "maint",
        "err",
        "error",
        "unk",
        "unknown",
        "boot",
        "reboot",
    }
)


def hostname() -> str:
    return os.environ.get("SLURMD_NODENAME") or platform.node()


def slurm_node_status(host: str | None = None, timeout: float = 20.0) -> dict:
    """Return scheduler state, reason, and query status; unavailable queries have empty state."""
    host = host or hostname()
    try:
        p = subprocess.run(["sinfo", "-h", "-n", host, "-o", "%t|%E"], capture_output=True, text=True, timeout=timeout)
    except Exception as e:  # noqa: BLE001
        return {"state": "", "reason": "", "source": f"sinfo unavailable ({type(e).__name__}: {e})"}
    if p.returncode != 0 or not p.stdout.strip():
        return {"state": "", "reason": "", "source": f"sinfo exit {p.returncode}"}
    # Multiple partitions can repeat a node; use the first reported state.
    line = p.stdout.strip().splitlines()[0]
    state, _, reason = line.partition("|")
    return {"state": state.strip(), "reason": reason.strip(), "source": "sinfo"}


def state_is_condemned(state: str) -> bool:
    stem = "".join(ch for ch in state.lower() if ch.isalpha())
    return stem in CONDEMNED_STATES


def _egl_probe_class(candidates=None):
    """Select a likely EGL-backed reference environment, preferring small mazes."""
    import stable_envs as se

    names = list(candidates) if candidates else sorted(se.registry.REGISTRY)
    scored = []
    for n in names:
        cls = se.registry.REGISTRY.get(n)
        if cls is None or not hasattr(cls, "reference_env") or getattr(cls, "reference_id", None) is None:
            continue
        egl = (
            hasattr(cls, "PLATE_PATHS")
            or "ogbench" in cls.__module__
            or "mujoco" in str(getattr(cls, "reference_id", ""))
        )
        if not egl:
            continue
        rank = {"medium": 0, "teleport": 1, "large": 2, "giant": 3}.get(str(getattr(cls, "MAZE", "")).lower(), 1)
        loco = {"point": 0, "ant": 1, "humanoid": 2}.get(str(getattr(cls, "LOCO", "")).lower(), 1)
        scored.append((rank, loco, n, cls))
    if not scored:
        return None
    scored.sort(key=lambda t: (t[0], t[1], t[2]))
    return scored[0][3]


def reference_renderer_probe(candidates=None) -> dict:
    """Compare two reference renders; report a skip when no suitable environment is available."""
    cls = _egl_probe_class(candidates)
    if cls is None:
        return {"skipped": "no EGL-backed reference env in this selection"}
    from stable_envs.testing import conformance as C

    size = int(getattr(cls, "default_size", 224))
    try:
        r = C.check_reference_renderer_is_stable(cls, size)
    except ImportError as e:
        return {"skipped": f"{cls.__name__}: reference env not installed ({e})"}
    return {"env": cls.__name__, "size": size, "max_abs_diff": int(r["max_abs_diff"])}


def gate(candidates=None, check_scheduler: bool = True, check_renderer: bool = True) -> dict:
    """Report scheduler and renderer preconditions without raising.

    A false ok blocks measurement. A true ok can include unavailable or skipped
    checks, so inspect their details before treating it as evidence of node health.
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
                f"reason={st.get('reason') or '(none given)'!r}. Do not measure a renderer on a host "
                f"the cluster's health checker has flagged. Resubmit elsewhere: --exclude={host}"
            )
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
                + (
                    f". Scheduler says state={out['scheduler'].get('state')!r} "
                    f"reason={out['scheduler'].get('reason')!r}"
                    if out["scheduler"]
                    else ""
                )
            )
    return out


def main(argv=None) -> int:
    import json

    g = gate()
    print(json.dumps(g, indent=2))
    return 0 if g["ok"] else 78


if __name__ == "__main__":
    raise SystemExit(main())
