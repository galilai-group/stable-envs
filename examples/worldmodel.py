"""stable-worldmodel integration: SWM ids, the gym adapter, and ``swap_renderer``.

This is the supported path. ``gymnasium`` / ``stable_worldmodel`` are optional; missing extras
print the reason and exit 0 on the arms that need them.

    python examples/worldmodel.py
    python examples/worldmodel.py --mode ids
    python examples/worldmodel.py --mode gym --device cpu
    python examples/worldmodel.py --mode swap --device cpu
    python examples/worldmodel.py --mode all --device cpu
"""

from __future__ import annotations

import argparse
import sys


def resolve_device(requested: str) -> str:
    import torch

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(f"{requested} was requested but torch.cuda.is_available() is False. Pass --device cpu.")
    return requested


def cmd_ids() -> int:
    import stable_envs as snv
    from stable_envs import interop

    print(f"stable-envs {snv.__version__}")
    print(f"SWM_IDS {len(interop.SWM_IDS)}  (stable_envs.interop.SWM_IDS; 40 pixel + 2 symbolic)")
    print(f"info('swm/PushT-v1')['class_name'] = {snv.info('swm/PushT-v1')['class_name']}")
    print(f"info('PushT')['swm_ids'] = {snv.info('PushT')['swm_ids']}")
    try:
        env = snv.make("swm/PushT-v1", num_envs=2, device="cpu")
    except Exception as e:
        print(f"make('swm/PushT-v1') refused: {type(e).__name__}: {e}")
        return 0
    print(f"make('swm/PushT-v1') -> {type(env).__name__} num_envs={env.num_envs}")
    env.close()
    try:
        snv.make("swm/CraftaxSymbolic-v1")
    except KeyError as e:
        print(f"symbolic id refused: {e}")
    return 0


def cmd_gym(device: str) -> int:
    try:
        import gymnasium  # noqa: F401
    except ImportError:
        print("gymnasium is a core dependency; this interpreter cannot import it. Skipping gym arm.")
        return 0
    try:
        from stable_envs import interop
    except ImportError as e:
        print(f"stable_envs.interop failed to import: {e}")
        return 0

    try:
        gym_env = interop.StableEnvGym("PushT", device=device)
    except Exception as e:
        print(f"StableEnvGym('PushT', device={device!r}) failed: {type(e).__name__}: {e}")
        return 0
    try:
        obs, info = gym_env.reset(seed=0)
        print(f"StableEnvGym reset obs {getattr(obs, 'shape', type(obs))} info_keys={sorted(info)}")
        frame = gym_env.render()
        print(f"render {getattr(frame, 'shape', type(frame))}")
    finally:
        gym_env.close()
    print("A swm.World is constructed from a gym id string (or this adapter), not from a native snv.Env.")
    return 0


def cmd_swap(device: str) -> int:
    try:
        import gymnasium as gym
        import numpy as np

        from stable_envs import interop
    except ImportError as e:
        print(f"swap arm needs gymnasium (+ numpy): {e}")
        return 0
    try:
        official = gym.make("gym_pusht/PushT-v0", render_mode="rgb_array")
    except Exception as e:
        print(f"gym_pusht is optional (pip install -e '.[pusht]'): {type(e).__name__}: {e}")
        return 0
    try:
        wrapped = interop.swap_renderer(official, "PushT", read_state=interop.pusht_state, device=device)
        wrapped.reset(seed=0)
        frame = wrapped.render()
        print(f"swap_renderer frame {np.asarray(frame).shape} — official dynamics, our pixels")
    finally:
        official.close()
    return 0


def cmd_limitation() -> None:
    print(
        "Native World(env=snv.PushT(...)) is not shipped. Use the adapter above "
        "(SWM ids, StableEnvGym, swap_renderer). stable-envs owns reset/step/transition/unroll/render; "
        "stable-worldmodel owns policy rollouts, MPC, planning, dataset collection, evaluation."
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Supported stable-worldmodel integration: ids, gym adapter, swap_renderer."
    )
    p.add_argument("--mode", choices=("ids", "gym", "swap", "all"), default="ids")
    p.add_argument("--device", default="auto")
    args = p.parse_args(argv)

    try:
        import stable_envs as snv  # noqa: F401
        import stable_envs.interop  # noqa: F401
    except ImportError as e:
        print(f"core dependency missing: {e}", file=sys.stderr)
        return 1

    device = resolve_device(args.device)
    if args.mode in ("ids", "all"):
        cmd_ids()
    if args.mode in ("gym", "all"):
        cmd_gym(device)
    if args.mode in ("swap", "all"):
        cmd_swap(device)
    cmd_limitation()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
