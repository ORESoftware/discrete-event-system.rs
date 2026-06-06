#!/usr/bin/env python3
"""Compatibility launcher for the Rust linear CLI reference solver.

The Rust crate owns the local LP/MIP CLI model writing, process execution, and
solution parsing paths. This script remains only for older callers that still
invoke ``scripts/linear_cli_reference.py`` directly.
"""

from __future__ import annotations

import json
import os
import sys


def truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on", "python", "legacy-python"}


def result(status: str, message: str) -> dict[str, object]:
    return {
        "status": status,
        "solver": "rust:linear-cli-reference",
        "x": [],
        "objective": None,
        "elapsedMs": 0.0,
        "message": message,
    }


def local_rust_binary_is_current(repo_root: str, binary_path: str) -> bool:
    if not os.path.exists(binary_path):
        return False
    binary_mtime = os.path.getmtime(binary_path)
    source_paths = [
        os.path.join(repo_root, "src", "bin", "linear_cli_reference.rs"),
        os.path.join(repo_root, "src", "des", "general", "external_linear_cli.rs"),
    ]
    return all(
        not os.path.exists(source_path) or os.path.getmtime(source_path) <= binary_mtime
        for source_path in source_paths
    )


def rust_reference_command() -> tuple[list[str], str | None]:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    binary_name = "linear_cli_reference"
    explicit = os.environ.get("LINEAR_CLI_REFERENCE_RUST_BIN")
    if explicit:
        return [explicit], None
    local_binary = os.path.join(repo_root, "target", "debug", binary_name)
    if local_rust_binary_is_current(repo_root, local_binary):
        return [local_binary], None
    return ["cargo", "run", "--quiet", "--bin", binary_name, "--"], repo_root


def exec_rust_reference() -> None:
    if truthy(os.environ.get("LINEAR_CLI_REFERENCE_FORCE_PYTHON")):
        print(
            json.dumps(
                result(
                    "unavailable",
                    "the legacy Python linear CLI implementation has been retired; use the Rust linear_cli_reference binary",
                )
            )
        )
        raise SystemExit(2)
    if truthy(os.environ.get("LINEAR_CLI_REFERENCE_FROM_RUST")):
        print(
            json.dumps(
                result(
                    "unavailable",
                    "Rust linear CLI bridge reached retired Python fallback; use a native Rust-supported option set",
                )
            )
        )
        raise SystemExit(2)

    command, cwd = rust_reference_command()
    if cwd is not None:
        os.chdir(cwd)
    os.execvp(command[0], [*command, *sys.argv[1:]])


def main() -> int:
    try:
        exec_rust_reference()
    except Exception as exc:
        print(json.dumps(result("numerical-error", f"failed to exec Rust linear CLI reference: {exc}")))
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
