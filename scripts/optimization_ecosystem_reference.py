#!/usr/bin/env python3
"""Thin JSON bridge for non-LP/MIP optimization ecosystem references.

The reusable ecosystem smoke solvers live in Rust. This script preserves the
existing Python-facing CLI contract for adapter invocations that still call
``scripts/optimization_ecosystem_reference.py --tool ...``.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any


def result(tool: str, message: str) -> dict[str, Any]:
    return {
        "kind": "optimization-ecosystem-reference-result",
        "tool": tool,
        "family": "unknown",
        "status": "invalid",
        "objective": None,
        "x": None,
        "message": message,
        "backend": "builtin-rust:optimization-ecosystem-reference",
    }


def arg_tool(args_tool: str | None) -> str:
    raw = args_tool or os.environ.get("ORES_EXTERNAL_OPTIMIZATION_TOOL") or "auto"
    return raw.lower().replace("_", "-")


def rust_reference_command() -> list[str]:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    binary_name = "optimization_ecosystem_reference"
    explicit = os.environ.get("OPTIMIZATION_ECOSYSTEM_REFERENCE_RUST_BIN")
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
        os.path.join(repo_root, "src", "bin", "optimization_ecosystem_reference.rs"),
        os.path.join(repo_root, "src", "des", "general", "external_optimization_tools.rs"),
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
    parser.add_argument("--tool", default=None)
    args = parser.parse_args()
    tool = arg_tool(args.tool)
    try:
        exec_rust_reference(tool)
        return 0
    except Exception as exc:
        print(json.dumps(result(tool, str(exc)), sort_keys=True))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
