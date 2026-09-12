"""Run ``stable_worldmodel``'s own PreJEPA training, with the one callback it names but nobody ships.

``scripts/train/prejepa.py`` builds its trainer with ``spt.callbacks.CPUOffloadCallback()``. That
attribute does not exist in ``stable-pretraining==0.1.8`` -- the version stable-worldmodel's own
``uv.lock`` pins -- nor in the 0.1.8 installed here; ``stable_pretraining.callbacks`` exports
thirty-odd callbacks and none of them offloads anything. So the committed DINO-WM training script
raises ``AttributeError`` before the first batch, against its own locked dependencies.

Rather than edit their repository, this shim installs a no-op ``lightning.pytorch.Callback`` under
that name and then calls the script's own ``run()``. A no-op is the safe direction: whatever the
callback was meant to move to the CPU stays on the GPU, which costs memory and changes no result.
Everything else -- the config, the data pipeline, the model, the optimiser, the loss, the
checkpoint writer -- is theirs, untouched.

    python scratch/step5/train_prejepa_shim.py --swm /path/to/stable-worldmodel -- \
        --config-dir ... --config-name prejepa_reacher trainer.max_epochs=3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--swm", type=Path, required=True, help="stable-worldmodel checkout")
    ap.add_argument("rest", nargs=argparse.REMAINDER, help="hydra arguments, after --")
    args = ap.parse_args()

    train_dir = args.swm / "scripts" / "train"
    if not (train_dir / "prejepa.py").is_file():
        raise SystemExit(f"no prejepa.py under {train_dir}")

    import lightning
    import stable_pretraining as spt

    if not hasattr(spt.callbacks, "CPUOffloadCallback"):
        class CPUOffloadCallback(lightning.pytorch.Callback):
            """No-op stand-in. Named in the training script, absent from the pinned library."""

        spt.callbacks.CPUOffloadCallback = CPUOffloadCallback
        print("[shim] installed a no-op spt.callbacks.CPUOffloadCallback", flush=True)

    passthrough = [a for a in args.rest if a != "--"]
    sys.argv = [str(train_dir / "prejepa.py"), *passthrough]
    sys.path.insert(0, str(train_dir))

    # runpy rather than `import prejepa; prejepa.run()`. @hydra.main resolves its relative
    # `config_path='./config'` against the DECORATED FUNCTION'S MODULE: run as a script the module
    # is __main__ and hydra uses the file's directory, but imported it is the module `prejepa`,
    # which has no package, and hydra builds the primary config path as '..config' and dies with
    # MissingConfigException. Executing the file under run_name='__main__' reproduces the exact
    # execution mode the script was written for -- the shim changes what is in spt.callbacks and
    # nothing else.
    import runpy

    runpy.run_path(str(train_dir / "prejepa.py"), run_name="__main__")


if __name__ == "__main__":
    main()
