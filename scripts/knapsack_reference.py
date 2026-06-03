#!/usr/bin/env python3
"""Reference bridge for small 0/1 knapsack instances.

The deterministic oracle uses branch-and-bound with a fractional knapsack upper
bound. When OR-Tools is installed and the input can be safely integer-scaled,
the same model is solved with CP-SAT.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from typing import Optional


EPS = 1e-9
MAX_EXACT_ITEMS = 64
SCALES = (1, 10, 100, 1000, 10000, 100000, 1000000)


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


def sorted_items(problem: dict) -> list[dict]:
    return sorted(
        problem["items"],
        key=lambda item: (
            -float(item["value"]) / float(item["weight"]),
            -float(item["value"]),
            float(item["weight"]),
            int(item["index"]),
        ),
    )


def fractional_upper_bound(
    capacity: float,
    order: list[dict],
    pos: int,
    current_weight: float,
    current_value: float,
) -> float:
    if current_weight > capacity + EPS:
        return -math.inf
    bound = current_value
    remaining = capacity - current_weight
    for item in order[pos:]:
        weight = float(item["weight"])
        value = float(item["value"])
        if weight <= remaining + EPS:
            bound += value
            remaining -= weight
        elif remaining > EPS:
            bound += value * (remaining / weight)
            break
        else:
            break
    return bound


def candidate_better(
    value: float,
    weight: float,
    indices: list[int],
    best_value: float,
    best_weight: float,
    best_indices: list[int],
) -> bool:
    if value > best_value + EPS:
        return True
    if abs(value - best_value) <= EPS and weight < best_weight - EPS:
        return True
    if abs(value - best_value) <= EPS and abs(weight - best_weight) <= EPS:
        return sorted(indices) < sorted(best_indices)
    return False


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


def greedy_density(problem: dict) -> dict:
    selected = []
    total_weight = 0.0
    for item in sorted_items(problem):
        if total_weight + item["weight"] <= problem["capacity"] + EPS:
            selected.append(int(item["index"]))
            total_weight += float(item["weight"])
    return solution(
        "feasible",
        "python:greedy-density-knapsack",
        problem,
        selected,
        None,
        "greedy value-density heuristic",
    )


def exact_knapsack(problem: dict) -> dict:
    items = problem["items"]
    if len(items) > MAX_EXACT_ITEMS:
        return solution(
            "unsupported",
            "python:exact-knapsack",
            problem,
            [],
            None,
            f"exact knapsack only practical for <= {MAX_EXACT_ITEMS} items, got {len(items)}",
        )

    order = sorted_items(problem)
    root_bound = fractional_upper_bound(problem["capacity"], order, 0, 0.0, 0.0)
    incumbent = greedy_density(problem)
    best_indices = list(incumbent["selectedItemIndices"])
    best_weight = float(incumbent["totalWeight"])
    best_value = float(incumbent["totalValue"])
    current: list[int] = []

    def search(pos: int, weight: float, value: float) -> None:
        nonlocal best_indices, best_weight, best_value
        if weight > problem["capacity"] + EPS:
            return
        if pos == len(order):
            if candidate_better(value, weight, current, best_value, best_weight, best_indices):
                best_indices = list(current)
                best_weight = weight
                best_value = value
            return
        bound = fractional_upper_bound(problem["capacity"], order, pos, weight, value)
        if bound + EPS < best_value:
            return

        item = order[pos]
        current.append(int(item["index"]))
        search(pos + 1, weight + float(item["weight"]), value + float(item["value"]))
        current.pop()
        search(pos + 1, weight, value)

    search(0, 0.0, 0.0)
    return solution(
        "optimal",
        "python:exact-knapsack",
        problem,
        best_indices,
        root_bound,
        "exact branch-and-bound with fractional-relaxation bound",
    )


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
    parser.add_argument("--solver", choices=["auto", "ortools", "fallback"], default="auto")
    args = parser.parse_args()

    try:
        problem = normalize(json.load(sys.stdin))
        exact = exact_knapsack(problem)
        if args.solver == "fallback":
            print(json.dumps(exact))
            return 0 if exact["status"] in ("optimal", "feasible", "unsupported") else 1

        ortools = ortools_knapsack(problem)
        if args.solver == "ortools":
            print(json.dumps(ortools))
            return 0 if ortools["status"] in ("optimal", "feasible", "unavailable", "unsupported") else 1

        result = dict(exact)
        result["solver"] = (
            "ortools:cp-sat-knapsack+python:exact-knapsack"
            if ortools.get("status") != "unavailable"
            else "python:exact-knapsack"
        )
        result["ortoolsStatus"] = ortools.get("status")
        result["ortoolsSelectedItemIndices"] = ortools.get("selectedItemIndices", [])
        result["ortoolsSelectedItemIds"] = ortools.get("selectedItemIds", [])
        result["ortoolsTotalWeight"] = ortools.get("totalWeight")
        result["ortoolsTotalValue"] = ortools.get("totalValue")
        result["ortoolsObjective"] = ortools.get("objective")
        result["ortoolsObjectiveBound"] = ortools.get("objectiveBound")
        result["ortoolsMessage"] = ortools.get("message")
        print(json.dumps(result))
        return 0 if result["status"] in ("optimal", "feasible", "unsupported") else 1
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
