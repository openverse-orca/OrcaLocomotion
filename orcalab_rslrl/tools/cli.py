"""Unified public ``orca`` command."""

from __future__ import annotations

import argparse
import sys

from .. import __version__


def _run(module_main, command: str, args: list[str]) -> None:
    sys.argv = [f"orca {command}", *args]
    module_main()


def main() -> None:
    command = sys.argv[1] if len(sys.argv) > 1 else None
    if command in {"train", "play", "inspect", "benchmark"}:
        if command == "train":
            from .train import main as command_main
        elif command == "play":
            from .play import main as command_main
        elif command == "inspect":
            from .inspect_env import main as command_main
        else:
            from .benchmark import main as command_main
        _run(command_main, command, sys.argv[2:])
        return

    parser = argparse.ArgumentParser(
        prog="orca",
        description="Train, inspect, and play robot-learning tasks with Orca.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("command", nargs="?", choices=("list", "train", "play", "inspect", "benchmark"))
    args, remainder = parser.parse_known_args()

    if args.command is None:
        parser.print_help()
        return
    if args.command == "list":
        if remainder:
            parser.error("orca list takes no arguments")
        from ..orca import list_tasks

        for task in list_tasks():
            print(f"{task.name}\t{task.description}")
        return
    parser.error(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
