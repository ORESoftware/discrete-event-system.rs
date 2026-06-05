#!/usr/bin/env python3
"""Thin proof-checker reference bridge.

The reusable DRAT/LRAT/FRAT and pseudo-Boolean proof checks live in Rust. This
script only preserves the previous Python CLI contract.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any


def result(tool: str, message: str) -> dict[str, Any]:
    return {
        "kind": "proof-validation-result",
        "tool": tool,
        "validator": "rust:proof-validation-reference",
        "status": "ok",
        "verdict": "invalid",
        "message": message,
    }


def rust_reference_command() -> list[str]:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    binary_name = "proof_validation_reference"
    explicit = os.environ.get("PROOF_VALIDATION_REFERENCE_RUST_BIN")
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
        os.path.join(repo_root, "src", "bin", "proof_validation_reference.rs"),
        os.path.join(repo_root, "src", "des", "general", "external_validation_tools.rs"),
    ]
    return all(
        not os.path.exists(source_path) or os.path.getmtime(source_path) <= binary_mtime
        for source_path in source_paths
    )


def exec_rust_reference(tool: str) -> None:
    command = rust_reference_command()
    if command[0] == "cargo":
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(os.path.dirname(script_dir))
    os.execvp(command[0], [*command, "--tool", tool])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool", default="drat")
    args = parser.parse_args()
    tool = args.tool.lower().replace("_", "-")
    try:
        exec_rust_reference(tool)
        return 0
    except Exception as exc:
        print(json.dumps(result(tool, str(exc)), sort_keys=True))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
