"""Train LeWM on a stable-envs dataset, with frames drawn on the training device or read from the file.

``--render online`` (the default) reads only states and actions from the dataset. After a batch moves to
the training device, ``Env.render`` draws its frames there from the state rows. ``--render offline`` reads
the ``pixels`` column, for example a file from ``make_dataset.py``. The model, loss, optimizer and data
split are those of stable-worldmodel ``scripts/train/lewm.py``.

    python examples/lewm/train.py --data pusht_state.h5 --half --compile
    python examples/lewm/train.py --data pusht_snv.h5 --render offline --half --compile

Arguments after the options are OmegaConf overrides of the config, for example ``trainer.max_epochs=10``.
"""

from __future__ import annotations

import argparse
import json
import time
from functools import partial
from pathlib import Path

import torch

CONFIG = Path(__file__).resolve().parent / "config" / "pusht.yaml"


class Pixels:
    """Draw ``(B, T, 3, H, W)`` normalised frames from ``(B, T, D)`` state rows on the rows' device."""

    def __init__(self, env_cfg, size, mean, std):
        self.env_cfg = env_cfg
        self.size = size
        self.mean = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
        self.renderer = None

    @torch.no_grad()
    def __call__(self, rows: torch.Tensor) -> torch.Tensor:
        if self.renderer is None:
            from state_render import from_config

            self.renderer = from_config(self.env_cfg, device=str(rows.device), size=self.size)
            self.mean, self.std = self.mean.to(rows.device), self.std.to(rows.device)
        b, n = rows.shape[:2]
        x = self.renderer(rows.float()).reshape(b * n, self.size, self.size, 3)
        x = x.permute(0, 3, 1, 2).float().div_(255.0)
        return ((x - self.mean) / self.std).reshape(b, n, 3, self.size, self.size)


def lewm_forward(self, batch, stage, cfg, pixels=None):
    """Encode the frames, predict the next embeddings and return the LeWM loss (as in lewm.py)."""
    if pixels is not None:
        batch["pixels"] = pixels(batch[cfg.env.state_column])
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    output = self.model.encode(batch)
    emb, act_emb = output["emb"], output["act_emb"]
    ctx = cfg.wm.history_size
    pred = self.model.predict(emb[:, :ctx], act_emb[:, :ctx])
    output["pred_loss"] = (pred - emb[:, cfg.wm.num_preds :]).pow(2).mean()
    output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))
    output["loss"] = output["pred_loss"] + cfg.loss.sigreg.weight * output["sigreg_loss"]
    self.log_dict({f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}, on_step=True, sync_dist=True)
    return output


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--data", required=True, help="HDF5 dataset (state rows for online, pixels for offline)")
    p.add_argument("--config", default=str(CONFIG), help="default: config/pusht.yaml")
    p.add_argument("--render", choices=["online", "offline"], default="online")
    p.add_argument("--out", default="runs", help="output folder; weights go to OUT/checkpoints/NAME/")
    p.add_argument("--name", default="lewm-pusht", help="run name")
    p.add_argument("--half", action="store_true", help="half-size model (the config's `half` overrides)")
    p.add_argument("--precision", default=None, help="Lightning precision, e.g. bf16-mixed (default) or 32")
    p.add_argument("--compile", action="store_true", help="torch.compile each LeWM submodule")
    p.add_argument("--accelerator", default="auto", help="Lightning accelerator: auto, gpu or cpu")
    p.add_argument("--max-steps", type=int, default=-1, help="stop after this many optimizer steps")
    p.add_argument("--wandb", default=None, metavar="PROJECT", help="log to this Weights & Biases project")
    p.add_argument("overrides", nargs="*", help="OmegaConf dotlist overrides, e.g. loader.num_workers=4")
    a = p.parse_args()

    import hydra
    import lightning as pl
    import stable_pretraining as spt
    import stable_worldmodel as swm
    from lightning.pytorch.callbacks import Callback
    from omegaconf import OmegaConf
    from stable_pretraining import data as dt
    from stable_worldmodel.data import column_normalizer
    from stable_worldmodel.wm.loss import SIGReg
    from stable_worldmodel.wm.utils import save_pretrained

    cfg = OmegaConf.load(a.config)
    if a.half:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(cfg.half)))
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(a.overrides))
    if a.precision is not None:
        cfg.trainer.precision = a.precision
    if cfg.loader.num_workers == 0:
        cfg.loader.persistent_workers = False
        cfg.loader.prefetch_factor = None
    online = a.render == "online"
    state_col = cfg.env.state_column

    keys = ["action", state_col] if online else ["pixels", "action"]
    # stable-worldmodel resolves a relative path under its cache folder, not the working directory.
    dataset = swm.data.load_dataset(
        str(Path(a.data).resolve()),
        num_steps=cfg.wm.history_size + cfg.wm.num_preds,
        frameskip=cfg.data.frameskip,
        keys_to_load=keys,
        keys_to_cache=[k for k in keys if k != "pixels"],
        transform=None,
    )
    stats = dt.dataset_stats.ImageNet
    transforms = [column_normalizer(dataset, "action", "action")]
    if not online:
        transforms.insert(
            0,
            dt.transforms.Compose(
                dt.transforms.ToImage(**stats, source="pixels", target="pixels"),
                dt.transforms.Resize(cfg.img_size, source="pixels", target="pixels"),
            ),
        )
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
    total_steps = cfg.trainer.max_epochs * len(train) if a.max_steps < 0 else a.max_steps
    optim = {
        "model_opt": {
            "modules": "model",
            "optimizer": OmegaConf.to_container(cfg.optimizer),
            "scheduler": {
                "type": "LinearWarmupCosineAnnealingLR",
                "warmup_steps": max(1, int(0.01 * total_steps)),
                "max_steps": total_steps,
            },
            "interval": "epoch",
        },
    }
    pixels = Pixels(OmegaConf.to_container(cfg.env), cfg.img_size, stats["mean"], stats["std"]) if online else None
    module = spt.Module(
        model=model,
        sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lewm_forward, cfg=cfg, pixels=pixels),
        optim=optim,
    )

    out = Path(a.out)

    class Report(Callback):
        """Print the epoch time and save the weights after each epoch."""

        def on_train_epoch_start(self, trainer, pl_module):
            self.t0, self.samples = time.perf_counter(), 0

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            self.samples += int(batch["action"].shape[0])

        def on_train_epoch_end(self, trainer, pl_module):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            seconds = time.perf_counter() - self.t0
            row = {
                "epoch": trainer.current_epoch + 1,
                "steps": int(trainer.global_step),
                "epoch_seconds": round(seconds, 2),
                "samples_per_s": round(self.samples / max(seconds, 1e-9), 1),
                **{k: round(float(v), 5) for k, v in trainer.callback_metrics.items() if k.startswith("train/")},
            }
            print(json.dumps(row), flush=True)
            if trainer.is_global_zero:
                save_pretrained(
                    pl_module.model,
                    run_name=a.name,
                    config=cfg.model,
                    filename=f"weights_epoch_{trainer.current_epoch + 1}.pt",
                    cache_dir=str(out),
                )

    logger = False
    if a.wandb:
        from lightning.pytorch.loggers import WandbLogger

        logger = WandbLogger(project=a.wandb, name=a.name)
        logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
    trainer = pl.Trainer(
        **OmegaConf.to_container(cfg.trainer),
        accelerator=a.accelerator,
        devices=1,
        max_steps=a.max_steps,
        callbacks=[Report()],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=False,
        default_root_dir=str(out),
    )
    (out / "checkpoints" / a.name).mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out / "checkpoints" / a.name / "train_config.yaml")
    data = spt.data.DataModule(train=train, val=val)
    spt.Manager(trainer=trainer, module=module, data=data, seed=cfg.seed)()


if __name__ == "__main__":
    main()
