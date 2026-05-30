#!/usr/bin/env python3
"""Report TS->RS migration coverage by mapping each .ts file to its expected
.rs counterpart (hyphens->underscores, .ts->.rs, index.ts->mod.rs)."""
import os
import sys

TS_ROOT = os.path.expanduser("~/codes/ores/des-engine/src")
RS_ROOT = os.path.expanduser("~/codes/ores/discrete-event-system.rs/src")


def ts_files():
    out = []
    for dirpath, _dirs, files in os.walk(TS_ROOT):
        for f in files:
            if f.endswith(".ts") and not f.endswith(".d.ts"):
                out.append(os.path.relpath(os.path.join(dirpath, f), TS_ROOT))
    return sorted(out)


def expected_rs(rel):
    parts = rel.split(os.sep)
    fname = parts[-1]
    dirs = [p.replace("-", "_") for p in parts[:-1]]
    if fname == "index.ts":
        base = "mod.rs"
    else:
        base = fname[:-3].replace("-", "_") + ".rs"
    return os.path.join(*dirs, base) if dirs else base


def main():
    ts = ts_files()
    matched, missing = [], []
    for rel in ts:
        rs_rel = expected_rs(rel)
        if os.path.exists(os.path.join(RS_ROOT, rs_rel)):
            matched.append((rel, rs_rel))
        else:
            missing.append((rel, rs_rel))
    total = len(ts)
    print(f"TOTAL TS: {total}")
    print(f"MATCHED:  {len(matched)}")
    print(f"MISSING:  {len(missing)}")
    print(f"COVERAGE: {len(matched)*100//total}%  ({len(matched)}/{total})")
    if len(sys.argv) > 1 and sys.argv[1] == "--missing":
        # group missing by top-level dir under src/des
        from collections import defaultdict
        groups = defaultdict(list)
        for rel, _ in missing:
            parts = rel.split(os.sep)
            # group key: first 2-3 path components
            key = os.sep.join(parts[:3]) if len(parts) > 3 else os.sep.join(parts[:-1])
            groups[key].append(rel)
        for key in sorted(groups, key=lambda k: -len(groups[k])):
            print(f"  [{len(groups[key]):3d}] {key}/")
    if len(sys.argv) > 1 and sys.argv[1] == "--list-missing":
        for rel, rs_rel in missing:
            print(f"{rel}  ->  {rs_rel}")


if __name__ == "__main__":
    main()
