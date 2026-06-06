#!/usr/bin/env python3
"""Compatibility launcher for the Rust scheduling reference solver.

The deterministic Rust exact solvers and the Rust-owned OR-Tools CP-SAT bridge
live in the crate. This script exists only so older tooling that invokes
``scripts/scheduling_reference.py`` keeps working without vendoring solver
executables or duplicating model logic in Python.
"""

from __future__ import annotations

import argparse
import json
import os


def local_rust_binary_is_current(repo_root: str, binary_path: str) -> bool:
    if not os.path.exists(binary_path):
        return False
    binary_mtime = os.path.getmtime(binary_path)
    source_paths = [
        os.path.join(repo_root, "src", "bin", "scheduling_reference.rs"),
        os.path.join(repo_root, "src", "des", "general", "external_scheduling_reference.rs"),
        os.path.join(repo_root, "src", "des", "general", "classical_optimization_models.rs"),
    ]
    return all(
        not os.path.exists(source_path) or os.path.getmtime(source_path) <= binary_mtime
        for source_path in source_paths
    )


def rust_reference_command() -> list[str]:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    binary_name = "scheduling_reference"
    explicit = os.environ.get("SCHEDULING_REFERENCE_RUST_BIN")
    if explicit:
        return [explicit]
    local_binary = os.path.join(repo_root, "target", "debug", binary_name)
    if local_rust_binary_is_current(repo_root, local_binary):
        return [local_binary]
    return ["cargo", "run", "--quiet", "--bin", binary_name, "--"]


def exec_rust_reference(solver: str, kind: str) -> None:
    command = rust_reference_command() + ["--solver", solver, "--kind", kind]
    if command[0] == "cargo":
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(os.path.dirname(script_dir))
    os.execvp(command[0], command)


def error_result(message: str) -> dict:
    return {
        "status": "error",
        "solver": "rust:scheduling-reference",
        "schedule": [],
        "sequence": [],
        "makespan": None,
        "totalFlowTime": None,
        "message": message,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--solver",
        default="auto",
        metavar="SOLVER",
        help="solver alias to pass through to the Rust scheduling_reference binary",
    )
    parser.add_argument(
        "--kind",
        default="auto",
        metavar="KIND",
        help="problem kind alias to pass through to the Rust scheduling_reference binary",
    )
    args = parser.parse_args()
    try:
        exec_rust_reference(args.solver, args.kind)
    except Exception as exc:
        print(json.dumps(error_result(f"failed to exec Rust scheduling reference: {exc}")))
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
