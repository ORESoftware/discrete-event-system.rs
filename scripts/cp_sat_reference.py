#!/usr/bin/env python3
"""Thin compatibility launcher for the Rust CP-SAT reference."""

from __future__ import annotations

import argparse
import json
import os
import sys


def result(
    status: str,
    solver: str,
    message: str,
) -> dict[str, object]:
    return {
        "status": status,
        "assignment": [],
        "objective": None,
        "nodes": 0,
        "solver": solver,
        "message": message,
    }


def local_rust_binary_is_current(repo_root: str, binary_path: str) -> bool:
    if not os.path.exists(binary_path):
        return False
    binary_mtime = os.path.getmtime(binary_path)
    source_paths = [
        os.path.join(repo_root, "src", "bin", "cp_sat_reference.rs"),
        os.path.join(repo_root, "src", "des", "general", "external_cp_sat_reference.rs"),
        os.path.join(repo_root, "src", "des", "general", "cp_sat.rs"),
    ]
    return all(
        not os.path.exists(source_path) or os.path.getmtime(source_path) <= binary_mtime
        for source_path in source_paths
    )


def rust_reference_command() -> list[str]:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    binary_name = "cp_sat_reference"
    explicit = os.environ.get("CP_SAT_REFERENCE_RUST_BIN")
    if explicit:
        return [explicit]
    local_binary = os.path.join(repo_root, "target", "debug", binary_name)
    if local_rust_binary_is_current(repo_root, local_binary):
        return [local_binary]
    return ["cargo", "run", "--quiet", "--bin", binary_name, "--"]


def exec_rust_reference(args: argparse.Namespace) -> None:
    command = rust_reference_command()
    command_args = ["--solver", args.solver]
    if args.enumerate_solutions is not None:
        command_args.extend(["--enumerate-solutions", str(args.enumerate_solutions)])
    if args.assumption_core:
        command_args.append("--assumption-core")
    if command[0] == "cargo":
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(os.path.dirname(script_dir))
    os.execvp(command[0], [*command, *command_args])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", default="auto")
    parser.add_argument("--enumerate-solutions", type=int)
    parser.add_argument("--assumption-core", action="store_true")
    args = parser.parse_args()
    try:
        exec_rust_reference(args)
    except Exception as exc:
        print(
            json.dumps(
                result(
                    "error",
                    "rust:cp-sat-reference",
                    str(exc),
                )
            )
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
