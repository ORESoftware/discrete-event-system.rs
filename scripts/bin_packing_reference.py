#!/usr/bin/env python3
"""Reference bridge for small one-dimensional bin-packing instances.

The deterministic oracle is an exact branch-and-bound over bin assignments.
When OR-Tools is installed and the inputs can be safely integer-scaled, the
same instance is solved with CP-SAT using item-bin assignment Booleans.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from typing import Optional


EPS = 1e-9
MAX_EXACT_ITEMS = 24
SCALES = (1, 10, 100, 1000, 10000, 100000, 1000000)


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


def sorted_items(problem: dict) -> list[dict]:
    return sorted(
        problem["items"],
        key=lambda item: (-float(item["weight"]), int(item["index"])),
    )


def first_fit_decreasing(problem: dict) -> dict:
    bins: list[dict] = []
    for item in sorted_items(problem):
        placed = False
        for bin_ in bins:
            if bin_["load"] + item["weight"] <= problem["capacity"] + EPS:
                bin_["items"].append(item["id"])
                bin_["indices"].append(item["index"])
                bin_["load"] += item["weight"]
                placed = True
                break
        if not placed:
            bins.append(
                {
                    "items": [item["id"]],
                    "indices": [item["index"]],
                    "load": item["weight"],
                }
            )
    return result(
        "feasible",
        "python:first-fit-decreasing",
        problem,
        normalize_bins(problem, bins),
        "first-fit-decreasing heuristic",
    )


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


def exact_bin_packing(problem: dict) -> dict:
    if len(problem["items"]) > MAX_EXACT_ITEMS:
        return result(
            "unsupported",
            "python:exact-bin-packing",
            problem,
            None,
            f"exact bin-packing only practical for <= {MAX_EXACT_ITEMS} items, got {len(problem['items'])}",
        )

    ffd = first_fit_decreasing(problem)
    best_bins = [
        {
            "indices": [
                next(item["index"] for item in problem["items"] if item["id"] == item_id)
                for item_id in bin_["items"]
            ],
            "load": float(bin_["load"]),
        }
        for bin_ in ffd["bins"]
    ]
    best_count = len(best_bins)
    if best_count == lower_bound_bins(problem):
        return result(
            "optimal",
            "python:exact-bin-packing",
            problem,
            normalize_bins(problem, best_bins),
            "exact branch-and-bound certified by volume lower bound",
        )

    order = sorted_items(problem)
    suffix = [0.0 for _ in range(len(order) + 1)]
    for index in range(len(order) - 1, -1, -1):
        suffix[index] = suffix[index + 1] + order[index]["weight"]
    current: list[dict] = []

    def search(pos: int) -> None:
        nonlocal best_bins, best_count
        if len(current) >= best_count:
            return
        if pos == len(order):
            best_count = len(current)
            best_bins = [
                {"indices": list(bin_["indices"]), "load": float(bin_["load"])}
                for bin_ in current
            ]
            return

        free_capacity = sum(max(0.0, problem["capacity"] - bin_["load"]) for bin_ in current)
        extra_weight = max(0.0, suffix[pos] - free_capacity)
        extra_bins = int(math.ceil(extra_weight / problem["capacity"]))
        if len(current) + extra_bins >= best_count:
            return

        item = order[pos]
        tried_loads: list[float] = []
        for bin_ in current:
            load = float(bin_["load"])
            if load + item["weight"] > problem["capacity"] + EPS:
                continue
            if any(abs(previous - load) <= EPS for previous in tried_loads):
                continue
            tried_loads.append(load)
            bin_["load"] = load + item["weight"]
            bin_["indices"].append(item["index"])
            search(pos + 1)
            bin_["indices"].pop()
            bin_["load"] = load

        if len(current) + 1 < best_count:
            current.append({"indices": [item["index"]], "load": item["weight"]})
            search(pos + 1)
            current.pop()

    search(0)
    return result(
        "optimal",
        "python:exact-bin-packing",
        problem,
        normalize_bins(problem, best_bins),
        "exact branch-and-bound",
    )


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
    parser.add_argument("--solver", choices=["auto", "ortools", "fallback"], default="auto")
    args = parser.parse_args()

    try:
        problem = normalize(json.load(sys.stdin))
        exact = exact_bin_packing(problem)
        if args.solver == "fallback":
            print(json.dumps(exact))
            return 0 if exact["status"] in ("optimal", "feasible", "unsupported") else 1

        ortools = ortools_bin_packing(problem)
        if args.solver == "ortools":
            print(json.dumps(ortools))
            return 0 if ortools["status"] in ("optimal", "feasible", "unavailable", "unsupported") else 1

        output = dict(exact)
        output["solver"] = (
            "ortools:cp-sat-bin-packing+python:exact-bin-packing"
            if ortools.get("status") != "unavailable"
            else "python:exact-bin-packing"
        )
        output["ortoolsStatus"] = ortools.get("status")
        output["ortoolsBins"] = ortools.get("bins", [])
        output["ortoolsObjective"] = ortools.get("objective")
        output["ortoolsMessage"] = ortools.get("message")
        output["ortoolsObjectiveBound"] = ortools.get("objectiveBound")
        print(json.dumps(output))
        return 0 if output["status"] in ("optimal", "feasible", "unsupported") else 1
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
