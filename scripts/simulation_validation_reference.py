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
import subprocess
import sys
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


def rust_reference(payload: dict[str, Any], engine: str | None) -> dict[str, Any]:
    command = rust_reference_command()
    args = []
    if engine:
        args.extend(["--engine", engine])
    cwd = None
    if command[0] == "cargo":
        script_dir = os.path.dirname(os.path.abspath(__file__))
        cwd = os.path.dirname(script_dir)
    completed = subprocess.run(
        [*command, *args],
        input=json.dumps(payload),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=cwd,
        check=False,
    )
    try:
        parsed = json.loads(completed.stdout)
    except Exception as exc:
        return result(
            "failed",
            "failure",
            "rust:simulation-validation-reference",
            f"failed to parse Rust simulation validation output: {exc}; stderr={completed.stderr.strip()}",
        )
    if completed.returncode != 0 and not parsed.get("message"):
        parsed["message"] = completed.stderr.strip()
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine")
    args = parser.parse_args()
    try:
        payload = json.load(sys.stdin)
        print(json.dumps(rust_reference(payload, args.engine)))
    except Exception as exc:
        print(json.dumps(result("failed", "failure", args.engine or "simulation", str(exc))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
