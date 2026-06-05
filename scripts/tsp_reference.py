#!/usr/bin/env python3
"""Thin launcher for the Rust TSP reference bridge."""

from __future__ import annotations

import argparse
import json
import os


def result(status: str, solver: str, message: str) -> dict:
    return {
        "status": status,
        "solver": solver,
        "tour": [],
        "objective": None,
        "message": message,
    }


def local_rust_binary_is_current(repo_root: str, binary_path: str) -> bool:
    if not os.path.exists(binary_path):
        return False
    binary_mtime = os.path.getmtime(binary_path)
    source_paths = [
        os.path.join(repo_root, "src", "bin", "tsp_reference.rs"),
        os.path.join(repo_root, "src", "des", "general", "external_tsp_reference.rs"),
    ]
    return all(
        not os.path.exists(source_path) or os.path.getmtime(source_path) <= binary_mtime
        for source_path in source_paths
    )


def rust_reference_command(solver: str) -> tuple[str, list[str], str]:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    binary_name = "tsp_reference"
    explicit = os.environ.get("TSP_REFERENCE_RUST_BIN")
    if explicit:
        return explicit, [explicit, "--solver", solver], repo_root
    local_binary = os.path.join(repo_root, "target", "debug", binary_name)
    if local_rust_binary_is_current(repo_root, local_binary):
        return local_binary, [local_binary, "--solver", solver], repo_root
    return (
        "cargo",
        ["cargo", "run", "--quiet", "--bin", binary_name, "--", "--solver", solver],
        repo_root,
    )


def exec_rust_reference(solver: str) -> None:
    executable, argv, cwd = rust_reference_command(solver)
    os.chdir(cwd)
    os.execvp(executable, argv)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--solver",
        choices=["auto", "ortools", "fallback", "rust-held-karp", "rust-exact"],
        default="auto",
    )
    args = parser.parse_args()
    try:
        exec_rust_reference(args.solver)
    except Exception as exc:
        print(json.dumps(result("error", "rust:tsp-reference", str(exc))))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
