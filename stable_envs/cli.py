"""The ``stable-envs`` command."""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="stable-envs")
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser(
        "doctor", help="check which environment families work on this machine and what to fix for the rest"
    )
    doctor.add_argument("--json", action="store_true", help="print the report as JSON")
    args = parser.parse_args(argv)
    from . import doctor as _doctor

    return _doctor.main(json_output=args.json)
