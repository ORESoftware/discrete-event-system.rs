#!/usr/bin/env python3
"""Thin reference bridge for output/data validators.

Reusable validation logic lives in the Rust crate. This script only preserves
the existing Python-facing CLI shape for callers that expect
``scripts/output_validation_reference.py --tool ...``.
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
    validator: str,
    message: str = "",
    errors: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "verdict": verdict,
        "validator": validator,
        "message": message,
        "errors": errors or [],
    }


def rust_reference_command() -> list[str]:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    binary_name = "output_validation_reference"
    explicit = os.environ.get("OUTPUT_VALIDATION_REFERENCE_RUST_BIN")
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
        os.path.join(repo_root, "src", "bin", "output_validation_reference.rs"),
        os.path.join(repo_root, "src", "des", "general", "external_validation_tools.rs"),
    ]
    return all(
        not os.path.exists(source_path) or os.path.getmtime(source_path) <= binary_mtime
        for source_path in source_paths
    )


def rust_reference(payload: dict[str, Any], tool: str) -> dict[str, Any]:
    command = rust_reference_command()
    cwd = None
    if command[0] == "cargo":
        script_dir = os.path.dirname(os.path.abspath(__file__))
        cwd = os.path.dirname(script_dir)
    completed = subprocess.run(
        [*command, "--tool", tool],
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
            "rust:output-validation-reference",
            f"failed to parse Rust output validation output: {exc}; stderr={completed.stderr.strip()}",
        )
    if completed.returncode != 0 and not parsed.get("message"):
        parsed["message"] = completed.stderr.strip()
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool", default="json-schema")
    args = parser.parse_args()
    try:
        payload = json.load(sys.stdin)
        print(json.dumps(rust_reference(payload, args.tool)))
    except Exception as exc:
        print(json.dumps(result("failed", "failure", args.tool, str(exc))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
