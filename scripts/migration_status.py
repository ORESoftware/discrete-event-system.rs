#!/usr/bin/env python3
"""Compatibility shim for the Rust migration_status CLI.

The TS->RS coverage implementation now lives in `src/bin/migration_status.rs`.
This wrapper keeps older invocations working while keeping the migration logic
in Rust.
"""

import os
import sys
from pathlib import Path


def local_rust_binary_is_current(repo: Path, binary: Path) -> bool:
    if not binary.exists():
        return False
    binary_mtime = binary.stat().st_mtime
    sources = [
        repo / "src" / "bin" / "migration_status.rs",
    ]
    return all(not source.exists() or source.stat().st_mtime <= binary_mtime for source in sources)


def exec_rust_reference() -> None:
    repo = Path(__file__).resolve().parents[1]
    binary = os.environ.get("MIGRATION_STATUS_BIN")
    if binary:
        os.execvp(binary, [binary, *sys.argv[1:]])
    local_binary = repo / "target" / "debug" / "migration_status"
    if local_rust_binary_is_current(repo, local_binary):
        os.execv(str(local_binary), [str(local_binary), *sys.argv[1:]])
    cargo = os.environ.get("CARGO", "cargo")
    os.chdir(repo)
    os.execvp(
        cargo,
        [
            cargo,
            "run",
            "--quiet",
            "--bin",
            "migration_status",
            "--",
            *sys.argv[1:],
        ],
    )


def main() -> int:
    try:
        exec_rust_reference()
    except Exception as exc:
        print(f"migration_status.py: failed to exec Rust migration_status: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
