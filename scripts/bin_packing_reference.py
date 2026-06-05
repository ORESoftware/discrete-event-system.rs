#!/usr/bin/env python3
"""Reference bridge for small one-dimensional bin-packing instances.

The deterministic branch-and-bound oracle lives in Rust. This Python bridge
remains as thin adapter glue for explicit OR-Tools CP-SAT checks.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Optional


EPS = 1e-9
SCALES = (1, 10, 100, 1000, 10000, 100000, 1000000)


def exec_rust_reference(solver: str) -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    binary_name = "bin_packing_reference"
    explicit = os.environ.get("BIN_PACKING_REFERENCE_RUST_BIN")
    if explicit:
        os.execv(explicit, [explicit, "--solver", solver])
    local_binary = os.path.join(repo_root, "target", "debug", binary_name)
    if local_rust_binary_is_current(repo_root, local_binary):
        os.execv(local_binary, [local_binary, "--solver", solver])
    os.chdir(repo_root)
    os.execvp(
        "cargo",
        ["cargo", "run", "--quiet", "--bin", binary_name, "--", "--solver", solver],
    )


def local_rust_binary_is_current(repo_root: str, binary_path: str) -> bool:
    if not os.path.exists(binary_path):
        return False
    binary_mtime = os.path.getmtime(binary_path)
    source_paths = [
        os.path.join(repo_root, "src", "bin", "bin_packing_reference.rs"),
        os.path.join(repo_root, "src", "des", "general", "external_bin_packing_reference.rs"),
        os.path.join(repo_root, "src", "des", "general", "bin_packing.rs"),
    ]
    return all(
        not os.path.exists(source_path) or os.path.getmtime(source_path) <= binary_mtime
        for source_path in source_paths
    )


def normalize(raw: dict) -> dict:
    capacity = float(raw.get("capacity", 0.0))
    if not math.isfinite(capacity) or capacity <= 0.0:
        raise ValueError("capacity must be finite and > 0")

    raw_items = raw.get("items")
    if raw_items is None:
        raw_weights = raw.get("weights") or []
        raw_items = [
            {"id": f"I{index + 1}", "weight": weight}
            for index, weight in enumerate(raw_weights)
        ]
    if not raw_items:
        raise ValueError("items must be non-empty")

    items = []
    seen = set()
    for index, raw_item in enumerate(raw_items):
        item_id = str(raw_item.get("id", f"I{index + 1}"))
        if not item_id.strip():
            raise ValueError(f"items[{index}].id must be non-empty")
        if item_id in seen:
            raise ValueError(f"duplicate item id {item_id!r}")
        seen.add(item_id)
        weight = float(raw_item.get("weight", 0.0))
        if not math.isfinite(weight) or weight <= 0.0:
            raise ValueError(f"items[{index}].weight must be finite and > 0")
        if weight > capacity + EPS:
            raise ValueError(f"items[{index}].weight exceeds capacity")
        items.append({"id": item_id, "weight": weight, "index": index})
    return {"capacity": capacity, "items": items}


def total_weight(problem: dict) -> float:
    return float(sum(item["weight"] for item in problem["items"]))


def lower_bound_bins(problem: dict) -> int:
    return int(math.ceil(total_weight(problem) / problem["capacity"]))


def result(
    status: str,
    solver: str,
    problem: dict,
    bins: Optional[list[dict]] = None,
    message: str = "",
) -> dict:
    packed_bins = [] if bins is None else bins
    return {
        "status": status,
        "solver": solver,
        "bins": [
            {
                "items": list(bin_["items"]),
                "load": float(bin_["load"]),
            }
            for bin_ in packed_bins
        ],
        "objective": None if bins is None else len(packed_bins),
        "totalWeight": total_weight(problem),
        "lowerBoundBins": lower_bound_bins(problem),
        "message": message,
    }


def normalize_bins(problem: dict, bins: list[dict]) -> list[dict]:
    by_index = {item["index"]: item for item in problem["items"]}
    normalized = []
    for bin_ in bins:
        indices = sorted(int(index) for index in bin_["indices"])
        load = sum(by_index[index]["weight"] for index in indices)
        normalized.append(
            {
                "items": [by_index[index]["id"] for index in indices],
                "indices": indices,
                "load": float(load),
            }
        )
    return normalized


def choose_scale(values: list[float]) -> Optional[int]:
    for scale in SCALES:
        if all(abs(round(value * scale) - value * scale) <= 1e-6 for value in values):
            return scale
    return None


def ortools_bin_packing(problem: dict) -> dict:
    try:
        from ortools.sat.python import cp_model  # type: ignore
    except Exception as exc:
        return result("unavailable", "ortools:cp-sat-bin-packing", problem, None, str(exc))

    items = problem["items"]
    scale = choose_scale([problem["capacity"]] + [item["weight"] for item in items])
    if scale is None:
        return result(
            "unsupported",
            "ortools:cp-sat-bin-packing",
            problem,
            None,
            "OR-Tools CP-SAT bridge requires integer-scalable weights/capacity",
        )

    n = len(items)
    capacity = int(round(problem["capacity"] * scale))
    weights = [int(round(item["weight"] * scale)) for item in items]
    model = cp_model.CpModel()
    x = {
        (item, bin_): model.NewBoolVar(f"x_i{item}_b{bin_}")
        for item in range(n)
        for bin_ in range(n)
    }
    y = {bin_: model.NewBoolVar(f"y_b{bin_}") for bin_ in range(n)}
    for item in range(n):
        model.AddExactlyOne(x[(item, bin_)] for bin_ in range(n))
    for bin_ in range(n):
        model.Add(sum(weights[item] * x[(item, bin_)] for item in range(n)) <= capacity * y[bin_])
    for bin_ in range(n - 1):
        model.Add(y[bin_] >= y[bin_ + 1])
    model.Minimize(sum(y.values()))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10.0
    solver.parameters.num_search_workers = 1
    status_code = solver.Solve(model)
    status_name = solver.StatusName(status_code).lower()
    if status_code not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return result(
            "infeasible" if status_name == "infeasible" else status_name,
            "ortools:cp-sat-bin-packing",
            problem,
            None,
            f"OR-Tools CP-SAT status {status_name}",
        )

    bins = []
    for bin_ in range(n):
        indices = [item for item in range(n) if solver.BooleanValue(x[(item, bin_)])]
        if not indices:
            continue
        bins.append(
            {
                "indices": indices,
                "load": sum(items[item]["weight"] for item in indices),
            }
        )
    output = result(
        "optimal" if status_code == cp_model.OPTIMAL else "feasible",
        "ortools:cp-sat-bin-packing",
        problem,
        normalize_bins(problem, bins),
        f"OR-Tools CP-SAT status {status_name}",
    )
    output["objectiveBound"] = solver.BestObjectiveBound()
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--solver",
        choices=["auto", "ortools", "fallback", "rust-exact"],
        default="auto",
    )
    args = parser.parse_args()
    if args.solver in ("auto", "fallback", "rust-exact"):
        exec_rust_reference(args.solver)

    try:
        problem = normalize(json.load(sys.stdin))
        output = ortools_bin_packing(problem)
        print(json.dumps(output))
        return 0 if output["status"] in ("optimal", "feasible", "unavailable", "unsupported") else 1
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "solver": "python:bin-packing-reference",
                    "bins": [],
                    "objective": None,
                    "totalWeight": None,
                    "lowerBoundBins": None,
                    "message": str(exc),
                }
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
