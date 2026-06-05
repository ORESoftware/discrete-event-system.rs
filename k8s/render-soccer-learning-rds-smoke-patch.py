#!/usr/bin/env python3
"""Compatibility shim for the Rust smoke-patch renderer."""

import os
import sys


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    os.chdir(repo_root)

    explicit = os.environ.get("SOCCER_RDS_SMOKE_PATCH_RENDERER_BIN")
    if explicit:
        os.execv(explicit, [explicit, *sys.argv[1:]])

    binary = "render_soccer_learning_rds_smoke_patch"
    local_binary = os.path.join(repo_root, "target", "debug", binary)
    if os.path.exists(local_binary):
        os.execv(local_binary, [local_binary, *sys.argv[1:]])

    os.execvp("cargo", ["cargo", "run", "--quiet", "--bin", binary, "--", *sys.argv[1:]])


if __name__ == "__main__":
    main()
