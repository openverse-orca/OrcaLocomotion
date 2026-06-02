from __future__ import annotations

from orca_rl.registry import list_tasks


def main() -> None:
    for spec in list_tasks():
        suffix = f" - {spec.description}" if spec.description else ""
        print(f"{spec.name}{suffix}")


if __name__ == "__main__":
    main()
