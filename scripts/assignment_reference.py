#!/usr/bin/env python3
"""Reference bridge for small linear-sum assignment instances.

The deterministic oracle is an exact dynamic program over row assignments.
When OR-Tools is installed and the costs can be safely integer-scaled, the
same input is also sent to OR-Tools SimpleLinearSumAssignment. SciPy's
linear_sum_assignment is used as another open-source reference when present.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from functools import lru_cache
from typing import Optional


SCALES = (1, 10, 100, 1000, 10000, 100000, 1000000)


def normalize(raw: dict) -> list[list[float]]:
    cost = raw.get("cost")
    if not cost:
        raise ValueError("cost matrix must be non-empty")
    rows = [[float(v) for v in row] for row in cost]
    cols = len(rows[0])
    if cols == 0:
        raise ValueError("cost matrix rows must be non-empty")
    if len(rows) > cols:
        raise ValueError("assignment bridge requires rows <= columns")
    for i, row in enumerate(rows):
        if len(row) != cols:
            raise ValueError(f"cost row {i} length {len(row)} != {cols}")
        if any(not math.isfinite(v) for v in row):
            raise ValueError(f"cost row {i} contains a non-finite value")
    return rows


def result(
    status: str,
    solver: str,
    assignment: Optional[list[int]] = None,
    objective: Optional[float] = None,
    message: str = "",
) -> dict:
    return {
        "status": status,
        "solver": solver,
        "assignment": [] if assignment is None else [int(v) for v in assignment],
        "objective": None if objective is None else float(objective),
        "message": message,
    }


def exact_assignment(cost: list[list[float]]) -> dict:
    rows = len(cost)
    cols = len(cost[0])

    @lru_cache(maxsize=None)
    def solve(row: int, used_mask: int) -> tuple[float, tuple[int, ...]]:
        if row == rows:
            return 0.0, ()
        best_cost = math.inf
        best_assignment: tuple[int, ...] = ()
        for col in range(cols):
            if used_mask & (1 << col):
                continue
            tail_cost, tail_assignment = solve(row + 1, used_mask | (1 << col))
            candidate = cost[row][col] + tail_cost
            candidate_assignment = (col,) + tail_assignment
            if candidate < best_cost - 1e-12 or (
                abs(candidate - best_cost) <= 1e-12 and candidate_assignment < best_assignment
            ):
                best_cost = candidate
                best_assignment = candidate_assignment
        return best_cost, best_assignment

    objective, assignment_tuple = solve(0, 0)
    if not math.isfinite(objective):
        return result("infeasible", "python:assignment-dp", message="no assignment")
    return result(
        "optimal",
        "python:assignment-dp",
        assignment=list(assignment_tuple),
        objective=objective,
        message="exact assignment dynamic program",
    )


def choose_scale(values: list[float]) -> Optional[int]:
    for scale in SCALES:
        if all(abs(round(value * scale) - value * scale) <= 1e-6 for value in values):
            return scale
    return None


def status_name(status: object) -> str:
    return str(status).split(".")[-1].lower()


def ortools_assignment(cost: list[list[float]]) -> dict:
    try:
        from ortools.graph.python import linear_sum_assignment  # type: ignore
    except Exception as exc:
        return result("unavailable", "ortools:simple-linear-sum-assignment", message=str(exc))

    flat = [v for row in cost for v in row]
    scale = choose_scale(flat)
    if scale is None:
        return result(
            "unsupported",
            "ortools:simple-linear-sum-assignment",
            message="OR-Tools SimpleLinearSumAssignment requires integer-scalable costs",
        )

    solver = linear_sum_assignment.SimpleLinearSumAssignment()
    for row, values in enumerate(cost):
        for col, value in enumerate(values):
            solver.add_arc_with_cost(row, col, int(round(value * scale)))
    status = solver.solve()
    if status != solver.OPTIMAL:
        mapped = status_name(status)
        return result(
            "infeasible" if mapped == "infeasible" else mapped,
            "ortools:simple-linear-sum-assignment",
            message=f"OR-Tools SimpleLinearSumAssignment status {mapped}",
        )
    assignment = [int(solver.right_mate(row)) for row in range(len(cost))]
    objective = solver.optimal_cost() / scale
    return result(
        "optimal",
        "ortools:simple-linear-sum-assignment",
        assignment=assignment,
        objective=objective,
        message="OR-Tools SimpleLinearSumAssignment",
    )


def scipy_assignment(cost: list[list[float]]) -> dict:
    try:
        from scipy.optimize import linear_sum_assignment  # type: ignore
    except Exception as exc:
        return result("unavailable", "scipy:linear_sum_assignment", message=str(exc))
    row_ind, col_ind = linear_sum_assignment(cost)
    assignment = [-1 for _ in cost]
    objective = 0.0
    for row, col in zip(row_ind, col_ind):
        assignment[int(row)] = int(col)
        objective += cost[int(row)][int(col)]
    if any(col < 0 for col in assignment):
        return result("infeasible", "scipy:linear_sum_assignment", message="not all rows assigned")
    return result(
        "optimal",
        "scipy:linear_sum_assignment",
        assignment=assignment,
        objective=objective,
        message="SciPy linear_sum_assignment",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", choices=["auto", "ortools", "scipy", "fallback"], default="auto")
    args = parser.parse_args()

    try:
        cost = normalize(json.load(sys.stdin))
        exact = exact_assignment(cost)
        if args.solver == "fallback":
            print(json.dumps(exact))
            return 0 if exact["status"] in ("optimal", "infeasible") else 1

        ortools = ortools_assignment(cost)
        scipy = scipy_assignment(cost)
        if args.solver == "ortools":
            output = dict(ortools)
            output["referenceStatus"] = exact.get("status")
            output["referenceObjective"] = exact.get("objective")
            print(json.dumps(output))
            return 0 if output["status"] in ("optimal", "infeasible", "unsupported", "unavailable") else 1
        if args.solver == "scipy":
            output = dict(scipy)
            output["referenceStatus"] = exact.get("status")
            output["referenceObjective"] = exact.get("objective")
            print(json.dumps(output))
            return 0 if output["status"] in ("optimal", "infeasible", "unavailable") else 1

        output = dict(exact)
        output["solver"] = (
            "ortools:simple-linear-sum-assignment+python:assignment-dp"
            if ortools.get("status") != "unavailable"
            else "python:assignment-dp"
        )
        output["ortoolsStatus"] = ortools.get("status")
        output["ortoolsAssignment"] = ortools.get("assignment", [])
        output["ortoolsObjective"] = ortools.get("objective")
        output["ortoolsMessage"] = ortools.get("message", "")
        output["scipyStatus"] = scipy.get("status")
        output["scipyAssignment"] = scipy.get("assignment", [])
        output["scipyObjective"] = scipy.get("objective")
        output["scipyMessage"] = scipy.get("message", "")
        print(json.dumps(output))
        return 0 if output["status"] in ("optimal", "infeasible") else 1
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "solver": "assignment-reference",
                    "assignment": [],
                    "objective": None,
                    "message": str(exc),
                }
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
