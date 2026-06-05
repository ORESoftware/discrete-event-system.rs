#!/usr/bin/env python3
"""Thin compatibility launcher for the Rust model validation reference."""

from __future__ import annotations

import argparse
import json
import os
import sys


def result(
    status: str,
    verdict: str,
    validator: str,
    message: str,
) -> dict[str, object]:
    return {
        "status": status,
        "verdict": verdict,
        "validator": validator,
        "message": message,
        "stdout": "",
        "stderr": "",
    }


def local_rust_binary_is_current(repo_root: str, binary_path: str) -> bool:
    if not os.path.exists(binary_path):
        return False
    binary_mtime = os.path.getmtime(binary_path)
    source_paths = [
        os.path.join(repo_root, "src", "bin", "model_validation_reference.rs"),
        os.path.join(repo_root, "src", "des", "general", "external_validation_tools.rs"),
    ]
    return all(
        not os.path.exists(source_path) or os.path.getmtime(source_path) <= binary_mtime
        for source_path in source_paths
    )


def rust_reference_command() -> list[str]:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    binary_name = "model_validation_reference"
    explicit = os.environ.get("MODEL_VALIDATION_REFERENCE_RUST_BIN")
    if explicit:
        return [explicit]
    local_binary = os.path.join(repo_root, "target", "debug", binary_name)
    if local_rust_binary_is_current(repo_root, local_binary):
        return [local_binary]
    return ["cargo", "run", "--quiet", "--bin", binary_name, "--"]


def exec_rust_reference(tool: str | None) -> None:
    command = rust_reference_command()
    args: list[str] = []
    if tool:
        args.extend(["--tool", tool])
    if command[0] == "cargo":
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(os.path.dirname(script_dir))
    os.execvp(command[0], [*command, *args])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool", default=None)
    args = parser.parse_args()
    try:
        exec_rust_reference(args.tool)
    except Exception as exc:
        print(
            json.dumps(
                result(
                    "failed",
                    "failure",
                    "rust:model-validation-reference",
                    str(exc),
                )
            )
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
