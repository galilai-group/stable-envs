"""``stable-envs doctor``: what works on this machine, per family, and what to install or set for the rest.

GL backends are probed in subprocesses with a timeout because context creation can hang or crash.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field

from ._core import mujoco_gl
from ._core.warp_cuda import CERTIFIED_WARP, warp_cuda_problem, warp_runtime

OK, WARN, FAIL, SKIP = "ok", "warn", "fail", "skip"
_RANK = {OK: 0, SKIP: 0, WARN: 1, FAIL: 2}
GL_PROBE_TIMEOUT = 60.0
SUPPORTED_PYTHON = ((3, 10), (3, 12))


@dataclass
class Check:
    """One check result: a status, what was found, and the fix for a warning or failure."""

    name: str
    status: str
    detail: str = ""
    fix: str = ""


@dataclass(frozen=True)
class Family:
    """A family's install extra, distribution, modules and the capabilities its CPU and GPU paths need."""

    name: str
    extra: str | None
    dist: str | None
    modules: tuple[str, ...]
    cpu: tuple[str, ...] = ()
    gpu: tuple[str, ...] = ("cuda",)
    #: The extra that installs Warp for this family's GPU path.
    gpu_extra: str | None = None
    #: Cells that need Warp on CUDA when the rest of the family does not.
    warp_cells: str = ""


# Every MuJoCo render path, CPU or CUDA, bakes its static background with mujoco.Renderer.
FAMILIES = (
    Family("classic_control", None, None, ("classic_control",)),
    Family("navigation", "navigation", "matplotlib", ("navigation",)),
    Family("pusht", "pusht", "pymunk", ("pusht", "pusht_semantics")),
    Family(
        "dmcontrol",
        "dmcontrol",
        "dm-control",
        ("dmcontrol",),
        cpu=("gl",),
        gpu=("gl", "cuda"),
        gpu_extra="dmcontrol-gpu",
        warp_cells="TrackCom domains (cheetah, hopper, humanoid, quadruped, swimmer, walker)",
    ),
    Family(
        "fetch",
        "robotics",
        "gymnasium-robotics",
        ("fetch",),
        cpu=("gl",),
        gpu=("gl", "cuda", "warp"),
        gpu_extra="robotics,mjwarp",
    ),
    Family(
        "ogbench",
        "ogbench",
        "ogbench",
        ("ogbench",),
        cpu=("gl",),
        gpu=("gl", "cuda"),
        gpu_extra="ogbench-manipulation-gpu",
        warp_cells="Cube and Scene",
    ),
    Family(
        "rocket",
        "rocket",
        "PyFlyt",
        ("rocket_landing", "rocket_render", "rocket_semantics"),
        gpu=("cuda", "triton", "nvdiffrast"),
    ),
    Family("craftax", "craftax-semantics", "craftax", ("craftax", "craftax_full", "craftax_semantics")),
)


@dataclass
class FamilyReport:
    """Overall, CPU-path and GPU-path status of one family, with the fixes that apply."""

    name: str
    status: str
    cpu: str = SKIP
    gpu: str = SKIP
    detail: str = ""
    fixes: list[str] = field(default_factory=list)


@dataclass
class Report:
    """System checks, GL checks and per-family results; ``ok`` is False when any family fails."""

    system: list[Check] = field(default_factory=list)
    gl: list[Check] = field(default_factory=list)
    families: list[FamilyReport] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(f.status != FAIL for f in self.families)

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "system": [asdict(c) for c in self.system],
            "gl": [asdict(c) for c in self.gl],
            "families": [asdict(f) for f in self.families],
        }


def _version(pair) -> str:
    return ".".join(str(int(v)) for v in pair)


def cuda_driver_version() -> tuple[int, int] | None:
    """The CUDA version the installed NVIDIA driver supports, from libcuda, or None without one."""
    names = {"linux": ("libcuda.so.1", "libcuda.so"), "win32": ("nvcuda.dll",)}.get(sys.platform, ())
    for name in names:
        try:
            lib = ctypes.CDLL(name)
        except OSError:
            continue
        v = ctypes.c_int()
        if lib.cuDriverGetVersion(ctypes.byref(v)) == 0 and v.value:
            return v.value // 1000, v.value % 1000 // 10
    return None


def nvidia_driver_release() -> str | None:
    """Return the NVIDIA kernel module release on Linux, or None."""
    try:
        with open("/proc/driver/nvidia/version") as fh:
            m = re.search(r"Kernel Module(?: for \S+)?\s+([\d.]+)", fh.read())
    except OSError:
        return None
    return m.group(1) if m else None


def _dist_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def check_python() -> Check:
    v = sys.version_info[:2]
    detail = f"{platform.python_version()} on {platform.platform()}"
    lo, hi = SUPPORTED_PYTHON
    if lo <= v <= hi:
        return Check("python", OK, detail)
    return Check("python", WARN, detail, f"supported: Python {_version(lo)} to {_version(hi)}")


def check_driver(driver: tuple[int, int] | None) -> Check:
    if driver is None:
        what = "macOS has no CUDA" if sys.platform == "darwin" else "no NVIDIA driver found"
        return Check("nvidia driver", SKIP, f"{what}; GPU render paths are unavailable")
    release = nvidia_driver_release()
    return Check("nvidia driver", OK, f"{release + ' ' if release else ''}(supports CUDA {_version(driver)})")


def check_torch(driver: tuple[int, int] | None) -> Check:
    import torch

    built = torch.version.cuda
    detail = f"{torch.__version__}, " + (f"built for CUDA {built}" if built else "CPU-only build")
    pick = "pick the build for your driver at https://pytorch.org/get-started/locally/"
    if driver is None:
        return Check("torch", OK, detail)
    if built is None:
        return Check(
            "torch", FAIL, detail + " on a machine with an NVIDIA driver", f"install a CUDA build of torch: {pick}"
        )
    if not torch.cuda.is_available():
        major = int(built.split(".")[0])
        if major > driver[0]:
            fix = f"the driver supports CUDA {_version(driver)}: upgrade it, or {pick}"
        else:
            fix = "check CUDA_VISIBLE_DEVICES and that nvidia-smi lists a GPU"
        return Check("torch", FAIL, detail + ", but torch.cuda.is_available() is False", fix)
    devices, missing = [], []
    arches = set(torch.cuda.get_arch_list())
    for i in range(torch.cuda.device_count()):
        major, minor = torch.cuda.get_device_capability(i)
        sm = f"sm_{major}{minor}"
        devices.append(f"{torch.cuda.get_device_name(i)} ({sm})")
        if arches and sm not in arches and not any(a.startswith("compute_") for a in arches):
            missing.append(sm)
    detail += "; " + ", ".join(devices)
    if missing:
        return Check("torch", WARN, detail, f"this torch build has no kernels for {', '.join(missing)}: {pick}")
    return Check("torch", OK, detail)


def check_warp(driver: tuple[int, int] | None) -> Check:
    if importlib.util.find_spec("warp") is None:
        return Check("warp", SKIP, "not installed", 'GPU MuJoCo rendering needs it: pip install "stable-envs[mjwarp]"')
    try:
        import warp as wp

        if hasattr(wp, "LOG_WARNING"):
            wp.config.log_level = wp.LOG_WARNING
        else:
            wp.config.quiet = True
        problem = warp_cuda_problem()
        runtime = warp_runtime()
    except Exception as e:  # noqa: BLE001 — reported
        return Check("warp", FAIL, f"warp does not initialise: {type(e).__name__}: {e}")
    toolkit = getattr(runtime, "toolkit_version", None)
    detail = f"{wp.config.version}" + (f", CUDA toolkit {_version(toolkit)}" if toolkit else "")
    mjw = _dist_version("mujoco-warp") or _dist_version("mujoco_warp")
    detail += f", mujoco-warp {mjw}" if mjw else ", mujoco-warp not installed"
    if problem is not None:
        if driver is None:
            return Check("warp", SKIP, f"{detail}; {problem}")
        return Check(
            "warp", FAIL, f"{detail}; {problem}", f"pip install warp-lang=={CERTIFIED_WARP} or upgrade the driver"
        )
    if mjw is None:
        return Check("warp", FAIL, detail, 'pip install "stable-envs[mjwarp]"')
    return Check("warp", OK, detail + f"; sees {', '.join(str(d) for d in wp.get_cuda_devices())}")


def _module_check(name: str, module: str, fix: str) -> Check:
    if importlib.util.find_spec(module) is None:
        return Check(name, FAIL, "not installed", fix)
    return Check(name, OK, _dist_version(module) or "installed")


def gl_env_checks(environ, platform_name: str = sys.platform) -> list[Check]:
    """Consistency of MUJOCO_GL, PYOPENGL_PLATFORM, DISPLAY and MUJOCO_EGL_DEVICE_ID, without creating a context."""
    raw = environ.get("MUJOCO_GL")
    backend = mujoco_gl.selected_backend(environ, platform_name)
    shown = f"MUJOCO_GL={raw}" if raw is not None else "MUJOCO_GL unset"
    out = []
    valid = {"cgl", "glfw"} if platform_name == "darwin" else {"egl", "osmesa", "glfw"}
    if backend not in valid:
        out.append(Check("MUJOCO_GL", FAIL, f"{shown}: not a rendering backend", f"use one of {sorted(valid)}"))
    else:
        out.append(Check("MUJOCO_GL", OK, f"{shown}; MuJoCo binds {backend}"))
    pyopengl = environ.get("PYOPENGL_PLATFORM")
    if pyopengl and backend in ("egl", "osmesa") and pyopengl.lower() != backend:
        out.append(
            Check(
                "PYOPENGL_PLATFORM",
                FAIL,
                f"PYOPENGL_PLATFORM={pyopengl} with MUJOCO_GL={backend}: MuJoCo refuses the mismatch",
                f"set PYOPENGL_PLATFORM={backend} or unset it",
            )
        )
    elif pyopengl and backend == "glfw" and pyopengl.lower() in ("egl", "osmesa"):
        out.append(
            Check(
                "PYOPENGL_PLATFORM",
                WARN,
                f"PYOPENGL_PLATFORM={pyopengl} with MUJOCO_GL=glfw",
                "unset PYOPENGL_PLATFORM (or set it to glx) for glfw",
            )
        )
    if (
        backend == "glfw"
        and platform_name != "darwin"
        and not (environ.get("DISPLAY") or environ.get("WAYLAND_DISPLAY"))
    ):
        out.append(
            Check(
                "DISPLAY",
                FAIL,
                "MUJOCO_GL=glfw needs a display and DISPLAY is unset",
                "on a headless machine use MUJOCO_GL=egl (GPU) or MUJOCO_GL=osmesa (no GPU), or run under xvfb-run",
            )
        )
    device_id, visible = environ.get("MUJOCO_EGL_DEVICE_ID"), environ.get("CUDA_VISIBLE_DEVICES")
    if backend == "egl" and device_id and visible and device_id not in visible.split(","):
        out.append(
            Check(
                "MUJOCO_EGL_DEVICE_ID",
                WARN,
                f"MUJOCO_EGL_DEVICE_ID={device_id} is not in CUDA_VISIBLE_DEVICES={visible}; "
                "EGL counts every GPU on the machine, so this renders on a GPU CUDA does not use",
                "unset MUJOCO_EGL_DEVICE_ID, or set it to the physical index of the GPU CUDA uses",
            )
        )
    return out


def gl_candidates(platform_name: str = sys.platform) -> tuple[str, ...]:
    """Return the MUJOCO_GL backends to probe on ``platform_name``."""
    return ("cgl", "glfw") if platform_name == "darwin" else ("egl", "osmesa", "glfw")


_GL_PROBE = """
import json, runpy, sys
gl = runpy.run_path(sys.argv[1])
import mujoco
try:
    gl["mujoco_renderer"](mujoco.MjModel.from_xml_string("<mujoco/>"), height=8, width=8).render()
except RuntimeError as e:
    cause = e.__cause__
    print(json.dumps({"error": f"{type(cause).__name__}: {cause}" if cause else str(e)}))
    sys.exit(1)
try:
    from OpenGL import GL
    name = GL.glGetString(GL.GL_RENDERER).decode()
except Exception:
    name = "context created"
print(json.dumps({"renderer": name}))
"""


def probe_gl(backend: str, environ) -> tuple[str, str]:
    """Create a MuJoCo renderer under ``backend`` in a fresh interpreter; return (status, detail)."""
    env = dict(environ)
    env["MUJOCO_GL"] = backend
    if backend in ("egl", "osmesa"):
        env["PYOPENGL_PLATFORM"] = backend
    elif env.get("PYOPENGL_PLATFORM", "").lower() in ("egl", "osmesa"):
        env.pop("PYOPENGL_PLATFORM")
    # Load the helper by path: importing stable_envs would discover every family in the probe.
    cmd = [sys.executable, "-c", _GL_PROBE, mujoco_gl.__file__]
    try:
        r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=GL_PROBE_TIMEOUT)
    except subprocess.TimeoutExpired:
        return FAIL, f"timed out after {GL_PROBE_TIMEOUT:.0f} s creating a context"
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith("{")]
    if lines:
        out = json.loads(lines[-1])
        if "renderer" in out:
            return OK, out["renderer"]
        return FAIL, out["error"]
    if r.returncode < 0:
        return FAIL, f"crashed with signal {-r.returncode}"
    tail = [ln for ln in r.stderr.splitlines() if ln.strip()]
    return FAIL, tail[-1] if tail else f"exit code {r.returncode}"


def _gl_fix(working: list[str], environ, driver) -> str:
    if working:
        alt = working[0]
        extra = f" PYOPENGL_PLATFORM={alt}" if alt in ("egl", "osmesa") else ""
        return f"export MUJOCO_GL={alt}{extra} (works on this machine)"
    if sys.platform == "darwin":
        return "MuJoCo's cgl backend should work on macOS; reinstall mujoco for this Python's architecture"
    hints = [mujoco_gl.NO_RENDERER_FIX]
    in_container = os.path.exists("/.dockerenv") or os.path.exists("/run/.containerenv")
    caps = environ.get("NVIDIA_DRIVER_CAPABILITIES")
    if driver is not None and in_container and caps is not None and not {"all", "graphics"} & set(caps.split(",")):
        hints.append(
            f"This container has NVIDIA_DRIVER_CAPABILITIES={caps}; EGL on the GPU needs "
            "NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics (or all)."
        )
    return " ".join(hints)


_GL_LIBRARIES = {"egl": "EGL", "osmesa": "OSMesa"}


def gl_checks(environ, driver) -> tuple[list[Check], Check]:
    """Environment and per-backend probe checks, and the check that decides the ``gl`` capability."""
    if importlib.util.find_spec("mujoco") is None:
        skipped = Check("gl", SKIP, "mujoco not installed; only the MuJoCo families need OpenGL")
        return [skipped], skipped
    checks = gl_env_checks(environ)
    selected = mujoco_gl.selected_backend(environ)
    headless = sys.platform != "darwin" and not (environ.get("DISPLAY") or environ.get("WAYLAND_DISPLAY"))
    backends = [b for b in gl_candidates() if not (b == "glfw" and headless)]
    with ThreadPoolExecutor(len(backends)) as pool:
        probes = dict(zip(backends, pool.map(lambda b: probe_gl(b, environ), backends)))
    if headless:
        probes["glfw"] = (SKIP, "no DISPLAY")
    for backend, (status, detail) in probes.items():
        if backend == selected:
            checks.append(Check(f"gl {backend} (selected)", status, detail))
        elif (
            status == FAIL and _GL_LIBRARIES.get(backend) and ctypes.util.find_library(_GL_LIBRARIES[backend]) is None
        ):
            checks.append(Check(f"gl {backend}", SKIP, f"lib{_GL_LIBRARIES[backend]} not found"))
        else:
            # Only the selected backend decides whether the families render.
            checks.append(Check(f"gl {backend}", WARN if status == FAIL else status, detail))
    working = [b for b, (s, _) in probes.items() if s == OK and b != selected]
    env_fail = next((c for c in checks if c.status == FAIL and not c.name.startswith("gl ")), None)
    status, detail = probes.get(selected, (FAIL, f"MUJOCO_GL={selected} is not a backend"))
    if env_fail is not None:
        decision = Check("gl", FAIL, env_fail.detail, _gl_fix(working, environ, driver) if working else env_fail.fix)
    elif status == OK:
        decision = Check("gl", OK, f"{selected}: {detail}")
    else:
        decision = Check("gl", FAIL, f"{selected}: {detail}", _gl_fix(working, environ, driver))
    for c in checks:
        if c.name.startswith(f"gl {selected}") and c.status == FAIL:
            c.fix = decision.fix
    return checks, decision


def _worst(statuses) -> str:
    return max(statuses, key=lambda s: _RANK[s], default=OK)


def family_report(fam: Family, errors: dict[str, str], caps: dict[str, Check], has_gpu: bool) -> FamilyReport:
    """Combine a family's import errors with the capability checks its CPU and GPU paths need."""
    install = f'pip install "stable-envs[{fam.extra}]"' if fam.extra else ""
    if fam.dist is not None and _dist_version(fam.dist) is None:
        return FamilyReport(fam.name, SKIP, detail="not installed", fixes=[install])
    failed = {m: errors[m] for m in fam.modules if m in errors}
    if len(failed) == len(fam.modules):
        first = next(iter(failed.values()))
        return FamilyReport(fam.name, FAIL, FAIL, FAIL, f"does not import: {first}", [install] if install else [])
    notes, fixes = [], []
    if failed:
        notes.append("not imported: " + "; ".join(f"{m} ({e})" for m, e in failed.items()))

    def broken(cap: str) -> bool:
        if caps[cap].status in (OK, SKIP):
            return False
        fix = caps[cap].fix
        if cap == "warp" and caps[cap].detail == "not installed" and fam.gpu_extra:
            fix = f'pip install "stable-envs[{fam.gpu_extra}]"'
        if fix and fix not in fixes:
            fixes.append(fix)
        return True

    def path(required) -> str:
        return _worst([caps[c].status for c in required if broken(c)])

    cpu = path(fam.cpu)
    gpu = path(fam.gpu) if has_gpu else SKIP
    if has_gpu and fam.warp_cells and broken("warp"):
        notes.append(f"{fam.warp_cells} need Warp on CUDA: {caps['warp'].detail}")
        gpu = _worst([gpu, WARN])
    status = _worst([cpu, gpu, WARN if failed else OK])
    return FamilyReport(fam.name, status, cpu, gpu, "; ".join(notes), fixes)


def run(environ=None) -> Report:
    """Collect the full report. ``environ`` defaults to the GL environment this process started with."""
    from . import registry

    if environ is None:
        environ = dict(os.environ)
        for k, v in mujoco_gl.INITIAL_GL_ENV.items():
            if v is None:
                environ.pop(k, None)
            else:
                environ[k] = v
    driver = cuda_driver_version()
    report = Report()
    torch_check, warp_check = check_torch(driver), check_warp(driver)
    report.system = [check_python(), check_driver(driver), torch_check, warp_check]
    report.gl, gl_cap = gl_checks(environ, driver)
    caps = {
        "gl": gl_cap,
        "cuda": torch_check,
        "warp": Check("warp", FAIL, "not installed", warp_check.fix)
        if warp_check.detail == "not installed"
        else warp_check,
        "triton": _module_check("triton", "triton", "pip install triton (Linux CUDA torch wheels include it)"),
        "nvdiffrast": _module_check(
            "nvdiffrast", "nvdiffrast", "build nvdiffrast from source: see docs/install.md#nvdiffrast"
        ),
    }
    errors = registry.discover()
    has_gpu = driver is not None
    report.families = [family_report(f, errors, caps, has_gpu) for f in FAMILIES]
    return report


_COLOURS = {OK: "32", WARN: "33", FAIL: "31", SKIP: "2"}


def _label(status: str, colour: bool) -> str:
    text = f"{status:<5}"
    return f"\033[{_COLOURS[status]}m{text}\033[0m" if colour else text


def format_report(report: Report, colour: bool = False) -> str:
    """Return the report as text, with ANSI status colours when ``colour``."""
    from .__about__ import __version__

    lines = [f"stable-envs {__version__} doctor", ""]
    for title, checks in (("System", report.system), ("MuJoCo OpenGL", report.gl)):
        lines.append(title)
        for c in checks:
            lines.append(f"  {_label(c.status, colour)} {c.name:<22} {c.detail}")
            if c.fix and c.status in (WARN, FAIL):
                lines.append(f"  {'':<5} {'':<22} fix: {c.fix}")
        lines.append("")
    lines.append(f"Families  {'':<14} cpu   gpu")
    for f in report.families:
        if f.cpu == SKIP and f.gpu == SKIP and f.status == SKIP:
            row = f"{f.detail}"
        else:
            row = f"{_label(f.cpu, colour)} {_label(f.gpu, colour)} {f.detail}"
        lines.append(f"  {_label(f.status, colour)} {f.name:<17} {row}".rstrip())
        if f.status != OK:
            for fix in f.fixes:
                lines.append(f"  {'':<5} {'':<17} {'fix' if f.status != SKIP else 'install'}: {fix}")
    lines.append("")
    lines.append(
        "All installed families can render here."
        if report.ok
        else "Some installed families cannot render here; see the fixes above."
    )
    return "\n".join(lines)


def main(json_output: bool = False) -> int:
    """Print the report as text or JSON and return 0 when every installed family works, else 1."""
    report = run()
    if json_output:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        colour = sys.stdout.isatty() and "NO_COLOR" not in os.environ
        print(format_report(report, colour))
    return 0 if report.ok else 1
