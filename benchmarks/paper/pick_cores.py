"""Print N idle physical cores (one hardware thread each, comma-separated) for taskset.

On a shared host every container sees the host's CPU numbers, so a fixed set (0-7) can be the same cores another
container uses. This samples /proc/stat for 3 s, keeps the physical cores whose threads were all below 10% busy,
prefers the NUMA node of the GPU, and picks N of them at random so that two pods on one host rarely pick the same.
Falls back to the N idlest cores; with PICK_STRICT=1 it exits non-zero instead.
"""

import os
import random
import subprocess
import sys
import time
from pathlib import Path


def usage(seconds: float) -> dict[int, float]:
    def snap():
        out = {}
        for line in open("/proc/stat"):
            if line.startswith("cpu") and line[3].isdigit():
                w = line.split()
                v = list(map(int, w[1:]))
                out[int(w[0][3:])] = (sum(v), v[3] + v[4])
        return out

    a = snap()
    time.sleep(seconds)
    b = snap()
    return {c: 1 - (b[c][1] - a[c][1]) / max(1, b[c][0] - a[c][0]) for c in a if c in b}


def gpu_node() -> int | None:
    """NUMA node of the job's GPU, or None."""
    try:
        gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
        cmd = ["nvidia-smi", "--query-gpu=pci.bus_id", "--format=csv,noheader", "-i", gpu]
        bus = subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip().lower()
        node = int(Path(f"/sys/bus/pci/devices/{bus[-12:]}/numa_node").read_text())
        return node if node >= 0 else None
    except Exception:  # noqa: BLE001
        return None


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    allowed = sorted(os.sched_getaffinity(0))
    busy = usage(3.0)
    cores: dict[str, list[int]] = {}
    for c in allowed:
        sib = Path(f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list").read_text().strip()
        cores.setdefault(sib, []).append(c)
    node = gpu_node()

    def numa(c):
        return int(next(Path(f"/sys/devices/system/cpu/cpu{c}").glob("node*")).name[4:])

    # PICK_EXCLUDE: cores (comma-separated) that other jobs on this host hold; their physical cores are not picked.
    held = {int(c) for c in os.environ.get("PICK_EXCLUDE", "").replace(" ", ",").split(",") if c.strip()}
    cores = {k: v for k, v in cores.items() if not held & set(v)}
    score = {k: max(busy.get(c, 1.0) for c in v) for k, v in cores.items()}
    idle = [k for k in cores if score[k] < 0.1]
    near = [k for k in idle if node is None or numa(cores[k][0]) == node]
    pool = near if len(near) >= n else idle
    random.seed(f"{os.uname().nodename}-{os.getpid()}-{time.time()}")
    if len(pool) < n and os.environ.get("PICK_STRICT") == "1":
        sys.exit(f"{len(pool)} idle physical cores, {n} wanted")
    pick = random.sample(pool, n) if len(pool) >= n else sorted(cores, key=score.get)[:n]
    # PICK_MAX_BUSY: fail when the picked cores were busier than this on average (other work on a shared host).
    mean = sum(score[k] for k in pick) / len(pick)
    if os.environ.get("PICK_MAX_BUSY") and mean > float(os.environ["PICK_MAX_BUSY"]):
        sys.exit(f"the {n} least busy physical cores are {mean:.2f} busy")
    print(f"busy {mean:.3f}", file=sys.stderr)
    print(",".join(str(cores[k][0]) for k in sorted(pick, key=lambda k: cores[k][0])))


if __name__ == "__main__":
    main()
