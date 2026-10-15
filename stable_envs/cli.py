"""The ``stable-envs`` command."""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    """Run the ``stable-envs`` command with ``argv`` and return its exit code."""
    parser = argparse.ArgumentParser(prog="stable-envs")
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser(
        "doctor", help="check which environment families work on this machine and what to fix for the rest"
    )
    doctor.add_argument("--json", action="store_true", help="print the report as JSON")
    tune = commands.add_parser("tune", help="measure the fastest tuning knobs of one environment's render path")
    tune.add_argument("env", nargs="?", help="environment class name, such as DMCCheetahRun")
    tune.add_argument("--batch", type=int, default=64, help="number of environments (default 64)")
    tune.add_argument("--size", type=int, help="frame size in pixels (default: the environment's)")
    tune.add_argument("--device", help="cpu, cuda or cuda:N (default: the configured default device)")
    tune.add_argument("--knob", action="append", help="sweep only this knob; repeat for more")
    tune.add_argument("--repeats", type=int, default=5, help="timed calls per value (default 5)")
    tune.add_argument("--show", action="store_true", help="list the cached values of this machine")
    tune.add_argument("--clear", action="store_true", help="delete the cached values of this machine")
    tune.add_argument("--all", action="store_true", help="tune one representative environment per render path")
    tune.add_argument("--yes", action="store_true", help="with --all, start without asking")
    args = parser.parse_args(argv)
    if args.command == "tune":
        from . import _tune

        return _tune.main(args)
    from . import doctor as _doctor

    return _doctor.main(json_output=args.json)
