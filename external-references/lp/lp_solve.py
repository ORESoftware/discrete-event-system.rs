#!/usr/bin/env python3
"""Compatibility launcher for the Rust LP reference solver.

The Rust crate owns the LP reference implementation. This script remains as the
stable legacy path for callers that still invoke
``external-references/lp/lp_solve.py`` directly.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any


def _truthy_python_override(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
        "python",
        "py",
        "scipy",
        "legacy-python",
    }


def _force_python_reference() -> bool:
    return any(
        _truthy_python_override(os.environ.get(name))
        for name in (
            "LP_SOLVE_REFERENCE_FORCE_PYTHON",
            "LP_EXTERNAL_REFERENCE_FORCE_PYTHON",
            "LP_EXTERNAL_BRIDGE",
            "ORES_LP_EXTERNAL_BRIDGE",
        )
    )


def _result(status: str, message: str) -> dict[str, Any]:
    return {
        "status": status,
        "x": [],
        "objective": None,
        "iters": None,
        "solver": "rust:lp-reference",
        "message": message,
    }


def _repo_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(here, "..", ".."))
    if os.path.exists(os.path.join(repo_root, "Cargo.toml")):
        return repo_root
    return os.getcwd()


def _local_rust_binary_is_current(repo_root: str, binary_path: str) -> bool:
    if not os.path.exists(binary_path):
        return False
    binary_mtime = os.path.getmtime(binary_path)
    source_paths = [
        os.path.join(repo_root, "src", "bin", "lp_solve_reference.rs"),
        os.path.join(repo_root, "src", "des", "general", "lp.rs"),
    ]
    return all(
        not os.path.exists(source_path) or os.path.getmtime(source_path) <= binary_mtime
        for source_path in source_paths
    )


def _rust_reference_command() -> tuple[list[str], str | None]:
    repo_root = _repo_root()
    explicit = os.environ.get("LP_SOLVE_REFERENCE_RUST_BIN")
    if explicit:
        return [explicit], None
    local_binary = os.path.join(repo_root, "target", "debug", "lp_solve_reference")
    if _local_rust_binary_is_current(repo_root, local_binary):
        return [local_binary], None
    return ["cargo", "run", "--quiet", "--bin", "lp_solve_reference", "--"], repo_root


def _exec_rust_reference(method: str) -> None:
    command, cwd = _rust_reference_command()
    if cwd is not None:
        os.chdir(cwd)
    os.execvp(command[0], [*command, "--method", method])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="highs")
    args = parser.parse_args()

    if _force_python_reference():
        print(
            json.dumps(
                _result(
                    "unavailable",
                    "the legacy Python SciPy LP bridge has been retired; use the Rust lp_solve_reference binary",
                )
            )
        )
        return 2

    try:
        _exec_rust_reference(args.method)
    except Exception as exc:
        print(json.dumps(_result("numerical-error", f"failed to exec Rust LP reference: {exc}")))
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
