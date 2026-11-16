"""Measure LeWM training speed and GPU use for three frame inputs: stored, source and ours.

    python examples/lewm/throughput.py --env pusht --input ours --data data/pusht_state.h5 --out runs/tp
    python examples/lewm/throughput.py --env pusht --input stored --data data/pusht_expert_train.h5 --out runs/tp
    python examples/lewm/throughput.py --env pusht --input source --data data/pusht_state.h5 --out runs/tp

Inputs:
  stored  the DataLoader reads and decodes the ``pixels`` column of the dataset.
  source  DataLoader workers write each state row into a stable-worldmodel (swm) env and draw it on the CPU.
  ours    the DataLoader reads only state rows; ``Env.render`` draws the frames on the training GPU.

The model is the full size of stable-worldmodel ``lewm.yaml`` (``config/pusht.yaml``); ``--model-overrides`` changes
it. The script trains ``--steps`` steps and leaves the first ``--warmup-steps`` out of the steady-state numbers. It
writes ``summary.json`` and one CSV row to ``--out``:

  steps_per_s         training steps per second in the steady state.
  train_busy_frac     GPU time of the training step alone (CUDA events around the step, without frame drawing)
                      divided by the steady-state wall time.
  render_busy_frac    GPU time of the frame drawing of ``ours`` divided by the steady-state wall time.
  render_ms           mean GPU time of the frame drawing of ``ours`` per batch.
  data_wait_frac      wall time between the end of one step and the start of the next (DataLoader wait and the
                      copy of the batch to the GPU) divided by the steady-state wall time.
  gpu_util_mean       the mean of NVML's GPU-utilisation samples in the steady state (the whole GPU, frame drawing
                      included).
  proc_sm_train / proc_sm_workers / proc_sm_other
                      the mean per-process NVML SM utilisation of the training process, of its DataLoader workers,
                      and of other processes.

On a GPU it also writes three traces for the whole run: ``nvml-*.csv`` (NVML polled at ``--nvml-hz``),
``proc-*.csv`` (per-process SM samples) and ``steps-*.csv`` (the times of each step).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

#: env -> (stable-envs id, state column, State leaves, zeroed leaves, swm id, swm kwargs)
ENVS = {
    "pusht": (
        "PushT",
        "state",
        [
            ("current.body.agent.position", 0, 2),
            ("current.body.block.position", 2, 4),
            ("current.body.block.angle", 4, 5),
        ],
        ["current.render_segments", "current.render_segment_count"],
        "swm/PushT-v1",
        {},
    ),
    "tworoom": ("TwoRoom", "pos_agent", [("agent.position", 0, 2)], [], "swm/TwoRoom-v1", {}),
    "reacher": ("DMCReacher", "qpos", [("qpos", 0, 2)], [], "swm/ReacherDMControl-v0", {}),
    "cube": (
        "OGBCube",
        "qpos",
        [("physics.qpos", 0, 21)],
        [],
        "swm/OGBCube-v0",
        {
            "env_type": "single",
            "ob_type": "states",
            "multiview": False,
            "width": 224,
            "height": 224,
            "visualize_info": False,
        },
    ),
}


class SourceRender:
    """DataLoader transform: draw each state row of ``col`` with a per-worker swm env; adds ``pixels`` uint8."""

    def __init__(self, env: str, col: str):
        self.key, self.col, self._env = env, col, None

    def _make(self):
        import gymnasium as gym
        import stable_worldmodel  # noqa: F401  registers the swm/ ids

        _, _, _, _, swm_id, kwargs = ENVS[self.key]
        self._env = gym.make(swm_id, render_mode="rgb_array", **kwargs).unwrapped
        self._env.reset(seed=0)
        self._pid = os.getpid()

    def _draw(self, s):
        import numpy as np

        env = self._env
        s = np.asarray(s, np.float64)
        if self.key == "pusht":
            env._set_state(s)
        elif self.key == "tworoom":
            env._set_state(s[:2])
        elif self.key == "reacher":
            env.set_state(s, np.zeros_like(s))
        else:
            env.set_state(s, np.zeros(env._model.nv))
        return np.asarray(env.render())[..., :3]

    def __call__(self, steps):
        import numpy as np
        import torch

        if self._env is None or self._pid != os.getpid():
            self._make()
        s = steps[self.col]
        s = s.numpy() if torch.is_tensor(s) else np.asarray(s)
        steps["pixels"] = torch.from_numpy(np.stack([self._draw(x) for x in s])).permute(0, 3, 1, 2).contiguous()
        return steps


class Monitor(threading.Thread):
    """Poll NVML for the GPU of ``device``: the GPU utilisation, the driver's sample buffer and per-process SM use."""

    def __init__(self, device, hz: float = 10.0):
        super().__init__(daemon=True)
        self.device, self.period = device, 1.0 / hz
        self.polls, self.util, self.proc, self.error = [], {}, {}, None
        self._halt = threading.Event()

    def _handle(self, pynvml):
        import torch

        # NVML numbers every GPU of the node, so CUDA_VISIBLE_DEVICES does not apply; find the GPU by its UUID.
        uuid = str(torch.cuda.get_device_properties(self.device).uuid)
        return pynvml.nvmlDeviceGetHandleByUUID(uuid if uuid.startswith("GPU-") else f"GPU-{uuid}")

    def run(self):
        import psutil

        try:
            import pynvml

            pynvml.nvmlInit()
            h = self._handle(pynvml)
        except Exception as e:  # noqa: BLE001
            self.error = repr(e)
            return
        me, last_ts, last_pts = psutil.Process(), 0, 0
        while not self._halt.is_set():
            t = time.time()
            try:
                rates = pynvml.nvmlDeviceGetUtilizationRates(h)
                self.polls.append((t, rates.gpu, rates.memory))
            except pynvml.NVMLError:
                pass
            try:
                _, samples = pynvml.nvmlDeviceGetSamples(h, pynvml.NVML_GPU_UTILIZATION_SAMPLES, last_ts)
                for x in samples:
                    self.util[x.timeStamp] = x.sampleValue.uiVal
                    last_ts = max(last_ts, x.timeStamp)
            except pynvml.NVMLError:
                pass
            try:
                pu = pynvml.nvmlDeviceGetProcessUtilization(h, last_pts)
            except pynvml.NVMLError:
                pu = []
            children = {c.pid for c in me.children(recursive=True)}
            for u in pu:
                role = "train" if u.pid == me.pid else "worker" if u.pid in children else "other"
                self.proc[(u.pid, u.timeStamp)] = (role, u.smUtil, u.memUtil)
                last_pts = max(last_pts, u.timeStamp)
            self._halt.wait(max(0.0, self.period - (time.time() - t)))

    def stop(self):
        self._halt.set()
        self.join(5)

    def summary(self, t0: float, t1: float) -> dict:
        """Means over ``[t0, t1]``. A process reports no sample for a period with zero use, so each role's SM
        mean is the sum of its samples divided by the number of driver sample periods in the window."""
        import numpy as np

        util = [v for ts, v in self.util.items() if t0 <= ts / 1e6 <= t1]
        polls = [g for t, g, _ in self.polls if t0 <= t <= t1]
        out = {
            "gpu_util_mean": float(np.mean(util)) if util else None,
            "gpu_util_poll_mean": float(np.mean(polls)) if polls else None,
            "nvml_samples": len(util),
            "nvml_polls": len(polls),
            "nvml_error": self.error,
        }
        for role in ("train", "worker", "other"):
            sm = sum(v[1] for (_, ts), v in self.proc.items() if v[0] == role and t0 <= ts / 1e6 <= t1)
            out[f"proc_sm_{'workers' if role == 'worker' else role}"] = sm / len(util) if util else None
        return out

    def write(self, out: Path, tag: str) -> None:
        with open(out / f"nvml-{tag}.csv", "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["t", "source", "gpu_util", "mem_util"])
            wr.writerows((t, "poll", g, m) for t, g, m in self.polls)
            wr.writerows((ts / 1e6, "buffer", v, "") for ts, v in sorted(self.util.items()))
        with open(out / f"proc-{tag}.csv", "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["t", "pid", "role", "sm_util", "mem_util"])
            wr.writerows((ts / 1e6, pid, *v) for (pid, ts), v in sorted(self.proc.items(), key=lambda kv: kv[0][1]))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", choices=sorted(ENVS), default="pusht")
    p.add_argument("--input", choices=["stored", "source", "ours"], required=True)
    p.add_argument("--data", required=True, help="HDF5 dataset: pixels for stored, state rows for source and ours")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--warmup-steps", type=int, default=200, help="first steps left out (compile, worker start)")
    p.add_argument(
        "--model-overrides",
        nargs="*",
        default=[],
        help="OmegaConf dotlist of the config, e.g. model.predictor.depth=3",
    )
    p.add_argument("--accelerator", default="gpu", help="Lightning accelerator; cpu reports no GPU numbers")
    p.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True, help="torch.compile each module")
    p.add_argument("--nvml-hz", type=float, default=10.0, help="NVML poll rate")
    p.add_argument("--out", default="runs/throughput")
    a = p.parse_args()
    if a.input == "ours":
        import stable_envs  # noqa: F401  before torch, as a user's script does

    import hydra
    import lightning as pl
    import numpy as np
    import stable_pretraining as spt
    import stable_worldmodel as swm
    import torch
    from lightning.pytorch.callbacks import Callback
    from omegaconf import OmegaConf
    from stable_pretraining import data as dt
    from stable_worldmodel.data import column_normalizer
    from stable_worldmodel.wm.loss import SIGReg
    from train import CONFIG, Pixels, lewm_forward

    env_id, col, leaves, zero, swm_id, _ = ENVS[a.env]
    cfg = OmegaConf.load(CONFIG)
    cfg.env = OmegaConf.create(
        {
            "id": env_id,
            "kwargs": {},
            "state_column": col,
            "zero": zero,
            "leaves": [{"leaf": k, "columns": [s, e]} for k, s, e in leaves],
        }
    )
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(a.model_overrides))
    if cfg.loader.num_workers == 0:
        cfg.loader.persistent_workers = False
        cfg.loader.prefetch_factor = None
    gpu = a.accelerator == "gpu"
    stats = dt.dataset_stats.ImageNet
    keys = ["pixels", "action"] if a.input == "stored" else ["action", col]
    dataset = swm.data.load_dataset(
        str(Path(a.data).resolve()),
        num_steps=cfg.wm.history_size + cfg.wm.num_preds,
        frameskip=cfg.data.frameskip,
        keys_to_load=keys,
        keys_to_cache=[k for k in keys if k != "pixels"],
        transform=None,
    )
    transforms = []
    if a.input == "source":
        transforms.append(SourceRender(a.env, col))
    if a.input != "ours":
        transforms.append(
            dt.transforms.Compose(
                dt.transforms.ToImage(**stats, source="pixels", target="pixels"),
                dt.transforms.Resize(cfg.img_size, source="pixels", target="pixels"),
            )
        )
    transforms.append(column_normalizer(dataset, "action", "action"))
    dataset.transform = spt.data.transforms.Compose(*transforms)
    cfg.model.action_encoder.input_dim = cfg.data.frameskip * dataset.get_dim("action")
    gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=gen)
    loader = OmegaConf.to_container(cfg.loader)
    train = torch.utils.data.DataLoader(train_set, **loader, generator=gen)
    val = torch.utils.data.DataLoader(val_set, **{**loader, "shuffle": False, "drop_last": False})

    model = hydra.utils.instantiate(cfg.model)
    if a.compile:
        for module in model.children():
            module.compile()
    optim = {
        "model_opt": {
            "modules": "model",
            "optimizer": OmegaConf.to_container(cfg.optimizer),
            "scheduler": {
                "type": "LinearWarmupCosineAnnealingLR",
                "warmup_steps": max(1, a.steps // 100),
                "max_steps": a.steps,
            },
            "interval": "epoch",
        }
    }
    pixels = None
    render_ms: list[float] = []
    if a.input == "ours":
        pixels = Pixels(OmegaConf.to_container(cfg.env), cfg.img_size, stats["mean"], stats["std"])
        draw = pixels.__call__

        def timed_draw(rows):
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            x = draw(rows)
            e1.record()
            e1.synchronize()
            render_ms.append(e0.elapsed_time(e1))
            return x

        if gpu:
            pixels = timed_draw
    module = spt.Module(
        model=model,
        sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
        forward=lambda self, batch, stage: lewm_forward(self, batch, stage, cfg=cfg, pixels=pixels),
        optim=optim,
    )

    class Timing(Callback):
        def __init__(self):
            self.starts, self.ends, self.gpu_ms, self.render_ms = [], [], [], []

        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            self.starts.append(time.time())
            self.r0 = len(render_ms)
            if gpu:
                self.e0 = torch.cuda.Event(enable_timing=True)
                self.e1 = torch.cuda.Event(enable_timing=True)
                self.e0.record()

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            if gpu:
                self.e1.record()
                torch.cuda.synchronize()
                self.gpu_ms.append(self.e0.elapsed_time(self.e1))
            self.render_ms.append(sum(render_ms[self.r0 :]))
            self.ends.append(time.time())

    timing = Timing()
    monitor = Monitor(torch.device("cuda", torch.cuda.current_device()), a.nvml_hz) if gpu else None
    trainer = pl.Trainer(
        **{**OmegaConf.to_container(cfg.trainer), "max_epochs": 1},
        accelerator=a.accelerator,
        devices=1,
        max_steps=a.steps,
        limit_val_batches=1,
        callbacks=[timing],
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
        default_root_dir=a.out,
    )
    if monitor:
        monitor.start()
    spt.Manager(trainer=trainer, module=module, data=spt.data.DataModule(train=train, val=val), seed=cfg.seed)()
    if monitor:
        monitor.stop()

    w = a.warmup_steps
    starts, ends = np.array(timing.starts), np.array(timing.ends)
    gpu_ms, step_render_ms = np.array(timing.gpu_ms), np.array(timing.render_ms)
    t0, t1 = starts[w], ends[-1]
    wall = t1 - t0
    steady = len(starts) - w
    render_sum = step_render_ms[w:].sum()
    nvml = monitor.summary(t0, t1) if monitor else {}

    summary = {
        "env": a.env,
        "input": a.input,
        "gpu": torch.cuda.get_device_name(0) if gpu else "",
        "steps": len(starts),
        "steady_steps": steady,
        "steps_per_s": steady / wall,
        "samples_per_s": steady * int(cfg.loader.batch_size) / wall,
        "train_busy_frac": (gpu_ms[w:].sum() - render_sum) / 1000 / wall if gpu else None,
        "render_busy_frac": render_sum / 1000 / wall if render_ms else None,
        "render_ms": render_sum / steady if render_ms else None,
        "data_wait_frac": float((starts[w + 1 :] - ends[w:-1]).sum() / wall),
        **nvml,
        "num_workers": int(cfg.loader.num_workers),
        "model_params": sum(x.numel() for x in model.parameters()),
        "model_overrides": a.model_overrides,
        "torch": torch.__version__,
    }
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{a.env}-{a.input}"
    (out / f"summary-{tag}.json").write_text(json.dumps(summary, indent=1))
    if monitor:
        monitor.write(out, tag)
    with open(out / f"steps-{tag}.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["step", "start", "end", "step_gpu_ms", "render_ms"])
        wr.writerows(
            zip(range(1, len(starts) + 1), starts, ends, gpu_ms if gpu else [""] * len(starts), step_render_ms)
        )
    csv_path = out / "throughput.csv"
    new = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        wr = csv.DictWriter(f, list(summary))
        if new:
            wr.writeheader()
        wr.writerow({k: json.dumps(v) if isinstance(v, list) else v for k, v in summary.items()})
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
