#!/usr/bin/env python3
"""Reference bridge for small linear-sum assignment instances.

The deterministic exact DP oracle lives in Rust. This Python bridge remains as
adapter glue for explicit OR-Tools SimpleLinearSumAssignment and SciPy
linear_sum_assignment checks when those packages are installed.
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
RUST_REFERENCE_SOLVERS = ("auto", "fallback", "rust-dp", "rust-exact")
EXTERNAL_REFERENCE_SOLVERS = ("ortools", "scipy")


def exec_rust_reference(solver: str) -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    binary_name = "assignment_reference"
    explicit = os.environ.get("ASSIGNMENT_REFERENCE_RUST_BIN")
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
        os.path.join(repo_root, "src", "bin", "assignment_reference.rs"),
        os.path.join(repo_root, "src", "des", "general", "external_assignment_reference.rs"),
        os.path.join(repo_root, "src", "des", "general", "assignment.rs"),
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
    value = os.environ.get("ASSIGNMENT_REFERENCE_EXTERNAL_FALLBACK", "")
    return value.strip().lower() in ("1", "true", "yes", "on", "rust")


def external_solver_package_available(solver: str) -> bool:
    if solver == "ortools":
        return package_available("ortools")
    if solver == "scipy":
        return package_available("scipy")
    return False


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
    parser.add_argument(
        "--solver",
        choices=["auto", "ortools", "scipy", "fallback", "rust-dp", "rust-exact"],
        default="auto",
    )
    args = parser.parse_args()
    if args.solver in RUST_REFERENCE_SOLVERS:
        exec_rust_reference(args.solver)
    if (
        external_rust_fallback_enabled()
        and args.solver in EXTERNAL_REFERENCE_SOLVERS
        and not external_solver_package_available(args.solver)
    ):
        exec_rust_reference("rust-dp")

    try:
        cost = normalize(json.load(sys.stdin))
        if args.solver == "ortools":
            output = ortools_assignment(cost)
            print(json.dumps(output))
            return 0 if output["status"] in ("optimal", "infeasible", "unsupported", "unavailable") else 1
        if args.solver == "scipy":
            output = scipy_assignment(cost)
            print(json.dumps(output))
            return 0 if output["status"] in ("optimal", "infeasible", "unavailable") else 1

        raise ValueError(f"unsupported solver {args.solver}")
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
