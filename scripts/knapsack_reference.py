#!/usr/bin/env python3
"""Reference bridge for small 0/1 knapsack instances.

The deterministic branch-and-bound oracle lives in Rust. This Python bridge
remains as thin adapter glue for explicit OR-Tools CP-SAT checks.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
from typing import Optional


SCALES = (1, 10, 100, 1000, 10000, 100000, 1000000)
RUST_REFERENCE_SOLVERS = ("auto", "fallback", "rust-branch-and-bound", "rust-exact")


def exec_rust_reference(solver: str) -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    binary_name = "knapsack_reference"
    explicit = os.environ.get("KNAPSACK_REFERENCE_RUST_BIN")
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
        os.path.join(repo_root, "src", "bin", "knapsack_reference.rs"),
        os.path.join(repo_root, "src", "des", "general", "external_knapsack_reference.rs"),
        os.path.join(repo_root, "src", "des", "general", "knapsack.rs"),
    ]
    return all(
        not os.path.exists(source_path) or os.path.getmtime(source_path) <= binary_mtime
        for source_path in source_paths
    )


def package_available(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except Exception:
        return False


def external_rust_fallback_enabled() -> bool:
    value = os.environ.get("KNAPSACK_REFERENCE_EXTERNAL_FALLBACK", "")
    return value.strip().lower() in ("1", "true", "yes", "on", "rust")


def normalize(raw: dict) -> dict:
    capacity = float(raw.get("capacity", 0.0))
    if not math.isfinite(capacity) or capacity <= 0.0:
        raise ValueError("capacity must be finite and > 0")

    raw_items = raw.get("items")
    if raw_items is None:
        raw_weights = raw.get("weights") or []
        raw_values = raw.get("values") or []
        if len(raw_weights) != len(raw_values):
            raise ValueError("weights and values must have the same length")
        raw_items = [
            {"id": f"I{index + 1}", "weight": weight, "value": value}
            for index, (weight, value) in enumerate(zip(raw_weights, raw_values))
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
        value = float(raw_item.get("value", 0.0))
        if not math.isfinite(weight) or weight <= 0.0:
            raise ValueError(f"items[{index}].weight must be finite and > 0")
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"items[{index}].value must be finite and non-negative")
        items.append({"id": item_id, "weight": weight, "value": value, "index": index})
    return {"capacity": capacity, "items": items}


def solution(
    status: str,
    solver: str,
    problem: dict,
    selected_indices: Optional[list[int]] = None,
    upper_bound: Optional[float] = None,
    message: str = "",
) -> dict:
    indices = [] if selected_indices is None else sorted(int(index) for index in selected_indices)
    items_by_index = {int(item["index"]): item for item in problem["items"]}
    selected_ids = [items_by_index[index]["id"] for index in indices]
    total_weight = sum(float(items_by_index[index]["weight"]) for index in indices)
    total_value = sum(float(items_by_index[index]["value"]) for index in indices)
    return {
        "status": status,
        "solver": solver,
        "selectedItemIndices": indices,
        "selectedItemIds": selected_ids,
        "totalWeight": total_weight,
        "totalValue": total_value,
        "objective": total_value,
        "upperBound": upper_bound,
        "message": message,
    }


def choose_scale(values: list[float]) -> Optional[int]:
    for scale in SCALES:
        if all(abs(round(value * scale) - value * scale) <= 1e-6 for value in values):
            return scale
    return None


def ortools_knapsack(problem: dict) -> dict:
    try:
        from ortools.sat.python import cp_model  # type: ignore
    except Exception as exc:
        return solution("unavailable", "ortools:cp-sat-knapsack", problem, None, None, str(exc))

    items = problem["items"]
    weight_scale = choose_scale([problem["capacity"]] + [item["weight"] for item in items])
    value_scale = choose_scale([item["value"] for item in items])
    if weight_scale is None or value_scale is None:
        return solution(
            "unsupported",
            "ortools:cp-sat-knapsack",
            problem,
            None,
            None,
            "OR-Tools CP-SAT bridge requires integer-scalable weights/capacity and values",
        )

    capacity = int(round(problem["capacity"] * weight_scale))
    weights = [int(round(item["weight"] * weight_scale)) for item in items]
    values = [int(round(item["value"] * value_scale)) for item in items]

    model = cp_model.CpModel()
    x = [model.NewBoolVar(f"x_{item['id']}") for item in items]
    model.Add(sum(weights[index] * x[index] for index in range(len(items))) <= capacity)
    model.Maximize(sum(values[index] * x[index] for index in range(len(items))))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10.0
    solver.parameters.num_search_workers = 1
    status_code = solver.Solve(model)
    status_name = solver.StatusName(status_code).lower()
    if status_code not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return solution(
            "infeasible" if status_name == "infeasible" else status_name,
            "ortools:cp-sat-knapsack",
            problem,
            None,
            None,
            f"OR-Tools CP-SAT status {status_name}",
        )

    selected = [index for index, var in enumerate(x) if solver.Value(var)]
    result = solution(
        "optimal" if status_code == cp_model.OPTIMAL else "feasible",
        "ortools:cp-sat-knapsack",
        problem,
        selected,
        solver.BestObjectiveBound() / value_scale,
        f"OR-Tools CP-SAT status {status_name}",
    )
    result["objectiveBound"] = solver.BestObjectiveBound() / value_scale
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--solver",
        choices=["auto", "ortools", "fallback", "rust-branch-and-bound", "rust-exact"],
        default="auto",
    )
    args = parser.parse_args()
    if args.solver in RUST_REFERENCE_SOLVERS:
        exec_rust_reference(args.solver)
    if (
        external_rust_fallback_enabled()
        and args.solver == "ortools"
        and not package_available("ortools")
    ):
        exec_rust_reference("rust-branch-and-bound")

    try:
        problem = normalize(json.load(sys.stdin))
        output = ortools_knapsack(problem)
        print(json.dumps(output))
        return 0 if output["status"] in ("optimal", "feasible", "unavailable", "unsupported") else 1
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "solver": "python:knapsack-reference",
                    "selectedItemIndices": [],
                    "selectedItemIds": [],
                    "totalWeight": 0.0,
                    "totalValue": 0.0,
                    "objective": None,
                    "upperBound": None,
                    "message": str(exc),
                }
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
