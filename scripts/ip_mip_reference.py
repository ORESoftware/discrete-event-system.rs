#!/usr/bin/env python3
"""Thin compatibility launcher for the Rust IP/MIP reference."""

from __future__ import annotations

import argparse
import json
import os


def payload(status: str, solver: str, message: str = "") -> dict[str, object]:
    return {
        "result": {
            "status": status,
            "solver": solver,
            "x": None,
            "objective": None,
            "message": message,
            "enumerated": 0,
        }
    }


def local_rust_binary_is_current(repo_root: str, binary_path: str) -> bool:
    if not os.path.exists(binary_path):
        return False
    binary_mtime = os.path.getmtime(binary_path)
    source_paths = [
        os.path.join(repo_root, "src", "bin", "ip_mip_reference.rs"),
        os.path.join(repo_root, "src", "des", "general", "lp.rs"),
    ]
    return all(
        not os.path.exists(source_path) or os.path.getmtime(source_path) <= binary_mtime
        for source_path in source_paths
    )


def rust_reference_command() -> list[str]:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    binary_name = "ip_mip_reference"
    explicit = os.environ.get("IP_MIP_REFERENCE_RUST_BIN")
    if explicit:
        return [explicit]
    local_binary = os.path.join(repo_root, "target", "debug", binary_name)
    if local_rust_binary_is_current(repo_root, local_binary):
        return [local_binary]
    return ["cargo", "run", "--quiet", "--bin", binary_name, "--"]


def exec_rust_reference(args: argparse.Namespace) -> None:
    command = rust_reference_command()
    command_args = [
        "--problem",
        args.problem,
        "--out",
        args.out,
        "--solver",
        args.solver,
        "--max-enumerations",
        str(args.max_enumerations),
    ]
    if args.pool_size is not None:
        command_args.extend(["--pool-size", str(args.pool_size)])
    if command[0] == "cargo":
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(os.path.dirname(script_dir))
    os.execvp(command[0], [*command, *command_args])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--solver", default="auto")
    parser.add_argument("--max-enumerations", type=int, default=1_000_000)
    parser.add_argument("--pool-size", type=int)
    args = parser.parse_args()
    try:
        exec_rust_reference(args)
    except Exception as exc:
        print(
            json.dumps(
                payload("unavailable", "rust:ip-mip-reference", str(exc)),
                allow_nan=True,
            )
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
