"""Measure per-frame MSE against the source for each class and write tests/fixtures/source_mse.json."""

import argparse
import json

import torch

import stable_envs as se
from stable_envs.env import reference_versions_of
from stable_envs.testing import conformance as C
from tests.helpers import source_mse as SM
from tests.helpers.reference.certification import has_reference_renderer, install_on_classes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--update",
        nargs="*",
        default=None,
        metavar="CLASS",
        help="replace the records of these classes (every measured class when empty) in an existing file",
    )
    parser.add_argument(
        "--envs", nargs="*", default=None, help="class names to measure; default every registered class"
    )
    args = parser.parse_args()
    install_on_classes()
    if SM.PATH.exists() and args.update is None:
        raise SystemExit(f"{SM.PATH} exists; pass --update [CLASS ...] to replace records")
    names = args.envs or args.update or sorted(se.registry.REGISTRY)
    records = SM.load()
    gpu = torch.cuda.get_device_name()
    for name in names:
        cls = se.registry.REGISTRY[name]
        if not has_reference_renderer(cls):
            print(f"SKIP {name}: no reference renderer", flush=True)
            continue
        try:
            env = se.make(name, num_envs=16, device="cuda")
        except ModuleNotFoundError as e:
            print(f"SKIP {name}: {e}", flush=True)
            continue
        try:
            got = C.check_source_mse(env, n=SM.N, seed=SM.SEED)
        except ImportError as e:
            print(f"SKIP {name}: reference env not installed ({e})", flush=True)
            continue
        finally:
            env.close()
        records[name] = {
            "seed": got["seed"],
            "n": got["n"],
            "size": got["size"],
            **{k: round(got[k], 4) for k in SM.KEYS},
            "reference_versions": reference_versions_of(cls),
            "gpu": gpu,
            "torch": torch.__version__,
        }
        print(f"RECORDED {name} " + " ".join(f"{k}={got[k]:.4f}" for k in SM.KEYS), flush=True)
    SM.PATH.write_text(json.dumps({"classes": dict(sorted(records.items()))}, indent=1) + "\n")
    print(f"wrote {len(records)} records to {SM.PATH}")


if __name__ == "__main__":
    main()
