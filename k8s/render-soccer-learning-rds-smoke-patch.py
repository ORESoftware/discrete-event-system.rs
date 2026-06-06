#!/usr/bin/env python3
"""Compatibility shim for the Rust smoke-patch renderer."""

import os
import sys


def local_rust_binary_is_current(repo_root: str, binary_path: str) -> bool:
    if not os.path.exists(binary_path):
        return False
    binary_mtime = os.path.getmtime(binary_path)
    source_paths = [
        os.path.join(repo_root, "src", "bin", "render_soccer_learning_rds_smoke_patch.rs"),
    ]
    return all(
        not os.path.exists(source_path) or os.path.getmtime(source_path) <= binary_mtime
        for source_path in source_paths
    )


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    os.chdir(repo_root)

    explicit = os.environ.get("SOCCER_RDS_SMOKE_PATCH_RENDERER_BIN")
    if explicit:
        os.execv(explicit, [explicit, *sys.argv[1:]])

    binary = "render_soccer_learning_rds_smoke_patch"
    local_binary = os.path.join(repo_root, "target", "debug", binary)
    if local_rust_binary_is_current(repo_root, local_binary):
        os.execv(local_binary, [local_binary, *sys.argv[1:]])

    os.execvp("cargo", ["cargo", "run", "--quiet", "--bin", binary, "--", *sys.argv[1:]])


if __name__ == "__main__":
    main()
