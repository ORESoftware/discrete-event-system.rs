#!/usr/bin/env python3
"""Compatibility launcher for the Rust stochastic LP reference solver.

The Rust crate owns the monolithic SLP reference and the explicit SciPy/HiGHS
adapter. This script exists only so older tooling that invokes
``scripts/stochastic_lp_reference.py`` keeps working.
"""

from __future__ import annotations

import argparse
import json
import os


def result(status: str, solver: str, message: str) -> dict[str, object]:
    return {
        "status": status,
        "solver": solver,
        "x": [],
        "objective": None,
        "cFirstX": None,
        "expectedQ": None,
        "yByScenario": [],
        "scenarioValues": [],
        "iterations": None,
        "message": message,
    }


def local_rust_binary_is_current(repo_root: str, binary_path: str) -> bool:
    if not os.path.exists(binary_path):
        return False
    binary_mtime = os.path.getmtime(binary_path)
    source_paths = [
        os.path.join(repo_root, "src", "bin", "stochastic_lp_reference.rs"),
        os.path.join(repo_root, "src", "des", "general", "external_stochastic_lp_reference.rs"),
        os.path.join(repo_root, "src", "des", "general", "stochastic_lp.rs"),
    ]
    return all(
        not os.path.exists(source_path) or os.path.getmtime(source_path) <= binary_mtime
        for source_path in source_paths
    )


def rust_reference_command() -> list[str]:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    binary_name = "stochastic_lp_reference"
    explicit = os.environ.get("STOCHASTIC_LP_REFERENCE_RUST_BIN")
    if explicit:
        return [explicit]
    local_binary = os.path.join(repo_root, "target", "debug", binary_name)
    if local_rust_binary_is_current(repo_root, local_binary):
        return [local_binary]
    return ["cargo", "run", "--quiet", "--bin", binary_name, "--"]


def exec_rust_reference(solver: str) -> None:
    command = rust_reference_command()
    if command[0] == "cargo":
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(os.path.dirname(script_dir))
    os.execvp(command[0], [*command, "--solver", solver])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--solver",
        choices=[
            "auto",
            "rust",
            "rust-monolithic",
            "monolithic",
            "scipy",
            "scipy-highs",
            "highs",
            "fallback",
            "rust-fallback",
        ],
        default="auto",
    )
    args = parser.parse_args()
    try:
        exec_rust_reference(args.solver)
    except Exception as exc:
        print(
            json.dumps(
                result(
                    "numerical-error",
                    "rust:stochastic-lp-reference",
                    f"failed to exec Rust stochastic LP reference: {exc}",
                )
            )
        )
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
