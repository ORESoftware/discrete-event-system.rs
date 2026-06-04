#!/usr/bin/env python3
"""Compatibility shim for the Rust migration_status CLI.

The TS->RS coverage implementation now lives in `src/bin/migration_status.rs`.
This wrapper keeps older invocations working while keeping the migration logic
in Rust.
"""

import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    binary = os.environ.get("MIGRATION_STATUS_BIN")
    if binary:
        cmd = [binary, *sys.argv[1:]]
    else:
        cmd = [
            os.environ.get("CARGO", "cargo"),
            "run",
            "--quiet",
            "--bin",
            "migration_status",
            "--",
            *sys.argv[1:],
        ]
    return subprocess.call(cmd, cwd=repo)


if __name__ == "__main__":
    raise SystemExit(main())
