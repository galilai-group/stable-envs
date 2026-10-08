"""Matched-frame metrics, numeric identities and public-render timing utilities."""

from __future__ import annotations

import hashlib
import importlib.metadata as md
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict

ROOT = Path(__file__).resolve().parents[3]
PIN = "176d54532b3c0a251a4f509b74f2fdacc143a63d"
METRICS = {
    "mse_normalized": "mean((candidate.astype(float64)/255 - oracle.astype(float64)/255)**2) over H*W*3",
    "mse_uint8": "mean((candidate.astype(float64) - oracle.astype(float64))**2) over H*W*3",
    "mae_rgb": "mean(abs(candidate.astype(float64) - oracle.astype(float64))) over H*W*3",
    "mean_max_channel_abs": "mean(max(abs(candidate-oracle), axis=RGB)) over H*W",
    "exact_fraction": "fraction of pixels with all three channels equal",
    "within8_fraction": "fraction of pixels with maximum channel difference <= 8",
    "ssim": "float64 luminance SSIM, 7x7 uniform window, nearest boundary, C1=(.01*255)^2, C2=(.03*255)^2",
    "max_channel_abs": "maximum absolute channel difference over H*W*3",
}


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=lambda x: array(x).tolist()) + "\n")


def array(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def array_hash(value):
    a = np.ascontiguousarray(array(value))
    h = hashlib.sha256()
    h.update(json.dumps([a.dtype.str, list(a.shape)], separators=(",", ":")).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def state_arrays(state):
    return {"/".join(k if isinstance(k, tuple) else (k,)): array(v) for k, v in state.items(True, True)}


def state_hash(state):
    values = state_arrays(state) if isinstance(state, TensorDict) else state
    return hashlib.sha256(
        json.dumps({k: array_hash(v) for k, v in sorted(values.items())}, sort_keys=True).encode()
    ).hexdigest()


def equal_state(a, b):
    aa, bb = state_arrays(a), state_arrays(b)
    assert aa.keys() == bb.keys(), (aa.keys() - bb.keys(), bb.keys() - aa.keys())
    for key in aa:
        assert (
            aa[key].dtype == bb[key].dtype and aa[key].shape == bb[key].shape and np.array_equal(aa[key], bb[key])
        ), key
    return dict(
        kind="EXACT_NUMERIC_STATE",
        candidate_state_hash=state_hash(a),
        oracle_state_hash=state_hash(b),
        compared_leaves=len(aa),
        equal=True,
    )


def save_state(path, state):
    np.savez_compressed(
        path, **{json.dumps(list(k if isinstance(k, tuple) else (k,))): array(v) for k, v in state.items(True, True)}
    )


def load_state(path, device="cpu", template=None):
    with np.load(path, allow_pickle=False) as a:
        paths = None
        if template is not None:
            paths = {"/".join(k if isinstance(k, tuple) else (k,)): k for k in template.keys(True, True)}
            assert len(paths) == len(template.keys(True, True)), "ambiguous legacy State archive keys"

        def key(k):
            if k.startswith("["):
                return tuple(json.loads(k))
            if paths is not None:
                return paths[k]
            return tuple(k.split("/"))

        return TensorDict({key(k): torch.from_numpy(a[k].copy()).to(device) for k in a.files}, [1], device=device)


def rgb(value):
    a = array(value)
    if a.ndim == 4 and a.shape[0] == 1:
        a = a[0]
    assert a.ndim == 3 and a.shape[2] in (3, 4), a.shape
    assert a.dtype == np.uint8, a.dtype
    return np.ascontiguousarray(a[..., :3])


def quality(candidate, oracle):
    from scipy.ndimage import uniform_filter

    a, b = rgb(candidate), rgb(oracle)
    assert a.shape == b.shape, (a.shape, b.shape)
    af, bf = a.astype(np.float64), b.astype(np.float64)
    diff = af - bf
    absolute = np.abs(diff)
    maximum = absolute.max(axis=-1)
    x, y = af @ [0.299, 0.587, 0.114], bf @ [0.299, 0.587, 0.114]
    f = lambda v: uniform_filter(v, size=7, mode="nearest")
    ux, uy = f(x), f(y)
    vx, vy, vxy = f(x * x) - ux * ux, f(y * y) - uy * uy, f(x * y) - ux * uy
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    ssim = ((2 * ux * uy + c1) * (2 * vxy + c2) / ((ux * ux + uy * uy + c1) * (vx + vy + c2))).mean()
    return dict(
        mse_normalized=float(np.mean((af / 255.0 - bf / 255.0) ** 2)),
        mse_uint8=float(np.mean(diff * diff)),
        mae_rgb=float(absolute.mean()),
        mean_max_channel_abs=float(maximum.mean()),
        exact_fraction=float((maximum == 0).mean()),
        within8_fraction=float((maximum <= 8).mean()),
        ssim=float(ssim),
        max_channel_abs=float(absolute.max()),
        candidate_sha256=hashlib.sha256(a.tobytes()).hexdigest(),
        oracle_sha256=hashlib.sha256(b.tobytes()).hexdigest(),
        shape=list(a.shape),
    )


def distribution(samples):
    a = np.array(samples, dtype=np.float64)
    assert len(a) and np.isfinite(a).all()
    return dict(
        count=len(a),
        mean=float(a.mean()),
        median=float(np.median(a)),
        p95=float(np.percentile(a, 95)),
        maximum=float(a.max()),
        minimum=float(a.min()),
        std=float(a.std()),
    )


def public_render(env, state, **kwargs):
    if hasattr(env, "to_render_state"):
        return env.render(env.to_render_state(state), **kwargs)
    return env.render(state, **kwargs)


def timed(fn, *, gpu=False, samples=6, warmups=10, to_host=False):
    def sync():
        if gpu:
            torch.cuda.synchronize()

    def call():
        value = fn()
        if to_host and isinstance(value, torch.Tensor):
            value = value.cpu()
        if hasattr(value, "block_until_ready"):
            value.block_until_ready()
        return value

    sync()
    begin = time.perf_counter_ns()
    result = call()
    sync()
    cold = (time.perf_counter_ns() - begin) / 1e6
    for _ in range(warmups):
        result = call()
    sync()
    raw = []
    for _ in range(samples):
        sync()
        begin = time.perf_counter_ns()
        result = call()
        sync()
        raw.append((time.perf_counter_ns() - begin) / 1e6)
    return dict(
        samples_ms=raw,
        statistics_ms=distribution(raw),
        first_call_ms=cold,
        warmups=warmups,
        synchronization="torch.cuda.synchronize before and after every timed call"
        if gpu
        else "eager materialized CPU array/Tensor (JAX block_until_ready if applicable)",
        output_location="cpu" if to_host or not gpu else "cuda",
        includes_reset_step_get_set_state=False,
    ), result


def profile():
    names = [
        "torch",
        "numpy",
        "tensordict",
        "gymnasium",
        "mujoco",
        "dm-control",
        "gymnasium-robotics",
        "ogbench",
        "craftax",
        "jax",
        "jaxlib",
        "flax",
        "pymunk",
        "pybullet",
        "PyFlyt",
        "nvdiffrast",
        "pygame",
    ]
    packages = {}
    for name in names:
        try:
            d = md.distribution(name)
            packages[name] = dict(version=d.version, path=str(d.locate_file("")))
        except md.PackageNotFoundError:
            pass
    cpu = platform.processor()
    if Path("/proc/cpuinfo").exists():
        cpu = next(
            x.split(":", 1)[1].strip()
            for x in Path("/proc/cpuinfo").read_text().splitlines()
            if x.startswith("model name")
        )
    return dict(
        host=platform.node(),
        cpu=cpu,
        gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        python=platform.python_version(),
        packages=packages,
        torch_threads=torch.get_num_threads(),
        environment={
            k: os.environ.get(k)
            for k in [
                "SLURM_JOB_ID",
                "MUJOCO_GL",
                "PYOPENGL_PLATFORM",
                "LIBGL_ALWAYS_SOFTWARE",
                "GALLIUM_DRIVER",
                "DISPLAY",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "LP_NUM_THREADS",
                "JAX_PLATFORMS",
                "CRAFTAX_JAX_BACKEND",
                "CUDA_VISIBLE_DEVICES",
            ]
        },
    )


def unavailable(reason, status="UNSUPPORTED"):
    assert status in {"UNSUPPORTED", "NONCOMPARABLE", "NOT IMPLEMENTED"}
    return dict(status=status, reason=reason)
