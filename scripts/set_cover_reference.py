#!/usr/bin/env python3
"""Thin compatibility launcher for the Rust set-cover reference."""

import argparse
import json
import os


def result(status: str, solver: str, message: str) -> dict:
    return {
        "status": status,
        "solver": solver,
        "selectedSetIndices": [],
        "selectedSets": [],
        "objective": None,
        "coveredElements": [],
        "message": message,
    }


def local_rust_binary_is_current(repo_root: str, binary_path: str) -> bool:
    if not os.path.exists(binary_path):
        return False
    binary_mtime = os.path.getmtime(binary_path)
    source_paths = [
        os.path.join(repo_root, "src", "bin", "set_cover_reference.rs"),
        os.path.join(repo_root, "src", "des", "general", "external_set_cover_reference.rs"),
        os.path.join(repo_root, "src", "des", "general", "set_cover.rs"),
    ]
    return all(
        not os.path.exists(source_path) or os.path.getmtime(source_path) <= binary_mtime
        for source_path in source_paths
    )


def rust_reference_command(solver: str) -> list[str]:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    binary_name = "set_cover_reference"
    explicit = os.environ.get("SET_COVER_REFERENCE_RUST_BIN")
    if explicit:
        return [explicit, "--solver", solver]
    local_binary = os.path.join(repo_root, "target", "debug", binary_name)
    if local_rust_binary_is_current(repo_root, local_binary):
        return [local_binary, "--solver", solver]
    os.chdir(repo_root)
    return ["cargo", "run", "--quiet", "--bin", binary_name, "--", "--solver", solver]


def exec_rust_reference(solver: str) -> None:
    command = rust_reference_command(solver)
    os.execvp(command[0], command)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--solver",
        default="auto",
        metavar="SOLVER",
        help="solver alias to pass through to the Rust set_cover_reference binary",
    )
    args = parser.parse_args()

    try:
        exec_rust_reference(args.solver)
    except Exception as exc:
        print(
            json.dumps(
                result(
                    "error",
                    "rust:set-cover-reference",
                    str(exc),
                )
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
