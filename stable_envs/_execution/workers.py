"""Setup and replies shared by spawned simulator workers."""

import os
import traceback

import torch


def pin_worker(cpus):
    """Restrict this process to ``cpus`` where supported and run Torch on one thread."""
    if cpus is not None and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, cpus)
    torch.set_num_threads(1)


def send_ready(conn):
    """Report this worker's pid and CPU affinity to the parent."""
    conn.send(
        (
            True,
            dict(
                pid=os.getpid(),
                affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
            ),
        )
    )


def send_failure(conn):
    """Send the exception being handled to the parent, ignoring a closed pipe."""
    try:
        conn.send((False, traceback.format_exc()))
    except (BrokenPipeError, EOFError, OSError):
        pass
