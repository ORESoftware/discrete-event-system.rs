#!/usr/bin/env python3
"""Thin reference bridge for simulation validation payloads.

The reusable simulation validators live in the Rust crate. This script keeps
the existing Python CLI shape for adapters that still invoke
``scripts/simulation_validation_reference.py --engine ...``.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any


def result(
    status: str,
    verdict: str,
    simulator: str,
    message: str = "",
    metrics: dict[str, float] | None = None,
    checks: list[dict[str, Any]] | None = None,
    trace: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "verdict": verdict,
        "simulator": simulator,
        "message": message,
        "metrics": metrics or {},
        "checks": checks or [],
        "trace": trace or [],
    }


def rust_reference_command() -> list[str]:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    binary_name = "simulation_validation_reference"
    explicit = os.environ.get("SIMULATION_VALIDATION_REFERENCE_RUST_BIN")
    if explicit:
        return [explicit]
    local_binary = os.path.join(repo_root, "target", "debug", binary_name)
    if local_rust_binary_is_current(repo_root, local_binary):
        return [local_binary]
    return ["cargo", "run", "--quiet", "--bin", binary_name, "--"]


def local_rust_binary_is_current(repo_root: str, binary_path: str) -> bool:
    if not os.path.exists(binary_path):
        return False
    binary_mtime = os.path.getmtime(binary_path)
    source_paths = [
        os.path.join(repo_root, "src", "bin", "simulation_validation_reference.rs"),
        os.path.join(repo_root, "src", "des", "general", "external_validation_tools.rs"),
    ]
    return all(
        not os.path.exists(source_path) or os.path.getmtime(source_path) <= binary_mtime
        for source_path in source_paths
    )


def exec_rust_reference(engine: str | None) -> None:
    command = rust_reference_command()
    args = []
    if engine:
        args.extend(["--engine", engine])
    if command[0] == "cargo":
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(os.path.dirname(script_dir))
    os.execvp(command[0], [*command, *args])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine")
    args = parser.parse_args()
    try:
        exec_rust_reference(args.engine)
    except Exception as exc:
        print(json.dumps(result("failed", "failure", args.engine or "simulation", str(exc))))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
