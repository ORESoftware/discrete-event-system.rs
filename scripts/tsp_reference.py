#!/usr/bin/env python3
"""Reference bridge for small traveling-salesman instances.

The deterministic oracle is Held-Karp dynamic programming over a dense distance
matrix. When OR-Tools is installed, the same matrix is also sent to OR-Tools
Routing as a one-vehicle TSP so Rust validation can cross-check the local
external routing engine without vendoring solver executables.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Optional


DISTANCE_SCALE = 1_000_000


def exec_rust_reference(solver: str) -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    binary_name = "tsp_reference"
    explicit = os.environ.get("TSP_REFERENCE_RUST_BIN")
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
        os.path.join(repo_root, "src", "bin", "tsp_reference.rs"),
        os.path.join(repo_root, "src", "des", "general", "external_tsp_reference.rs"),
        os.path.join(repo_root, "src", "des", "general", "tsp.rs"),
    ]
    return all(
        not os.path.exists(source_path) or os.path.getmtime(source_path) <= binary_mtime
        for source_path in source_paths
    )


def euclidean_distance(a: dict, b: dict) -> float:
    return math.hypot(float(a["x"]) - float(b["x"]), float(a["y"]) - float(b["y"]))


def parse_point(raw: object, index: int) -> dict:
    if isinstance(raw, dict):
        return {
            "id": str(raw.get("id", index)),
            "x": float(raw["x"]),
            "y": float(raw["y"]),
        }
    if isinstance(raw, (list, tuple)) and len(raw) >= 2:
        return {"id": str(index), "x": float(raw[0]), "y": float(raw[1])}
    raise ValueError(f"point {index} must be an object with x/y or a length-2 array")


def build_distance_matrix(points: list[dict]) -> list[list[float]]:
    return [[euclidean_distance(a, b) for b in points] for a in points]


def normalize(raw: dict) -> dict:
    matrix_raw = raw.get("distanceMatrix", raw.get("distance_matrix"))
    points_raw = raw.get("points", raw.get("cities"))
    points: list[dict] = []
    if points_raw is not None:
        points = [parse_point(point, i) for i, point in enumerate(points_raw)]

    if matrix_raw is None:
        if not points:
            raise ValueError("points or distanceMatrix is required")
        matrix = build_distance_matrix(points)
    else:
        matrix = [[float(v) for v in row] for row in matrix_raw]

    n = len(matrix)
    if n < 2:
        raise ValueError("TSP requires at least two cities")
    for i, row in enumerate(matrix):
        if len(row) != n:
            raise ValueError(f"distance row {i} length {len(row)} != {n}")
        for j, value in enumerate(row):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"distance[{i}][{j}] must be finite and non-negative")
        if abs(row[i]) > 1e-9:
            raise ValueError(f"distance[{i}][{i}] must be zero")

    if not points:
        points = [{"id": str(i), "x": float(i), "y": 0.0} for i in range(n)]
    if len(points) != n:
        raise ValueError(f"points length {len(points)} != distance matrix size {n}")

    return {"points": points, "distanceMatrix": matrix}


def tour_length(matrix: list[list[float]], tour: list[int]) -> float:
    if not tour:
        return 0.0
    total = 0.0
    for a, b in zip(tour, tour[1:]):
        total += matrix[a][b]
    total += matrix[tour[-1]][tour[0]]
    return total


def result(
    status: str,
    solver: str,
    tour: Optional[list[int]] = None,
    objective: Optional[float] = None,
    message: str = "",
) -> dict:
    return {
        "status": status,
        "solver": solver,
        "tour": [] if tour is None else [int(v) for v in tour],
        "objective": None if objective is None else float(objective),
        "message": message,
    }


def ortools_tsp(problem: dict) -> dict:
    try:
        from ortools.constraint_solver import pywrapcp, routing_enums_pb2  # type: ignore
    except Exception as exc:
        return result("unavailable", "ortools:routing-tsp", message=str(exc))

    matrix = problem["distanceMatrix"]
    n = len(matrix)
    manager = pywrapcp.RoutingIndexManager(n, 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index: int, to_index: int) -> int:
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(round(matrix[from_node][to_node] * DISTANCE_SCALE))

    transit = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.FromSeconds(5)

    solution = routing.SolveWithParameters(params)
    if solution is None:
        return result("infeasible", "ortools:routing-tsp", message="OR-Tools Routing found no tour")

    tour: list[int] = []
    index = routing.Start(0)
    while not routing.IsEnd(index):
        tour.append(manager.IndexToNode(index))
        index = solution.Value(routing.NextVar(index))
    return result(
        "optimal",
        "ortools:routing-tsp",
        tour=tour,
        objective=tour_length(matrix, tour),
        message="OR-Tools Routing one-vehicle TSP",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--solver",
        choices=["auto", "ortools", "fallback", "rust-held-karp", "rust-exact"],
        default="auto",
    )
    args = parser.parse_args()
    if args.solver in ("auto", "fallback", "rust-held-karp", "rust-exact"):
        exec_rust_reference(args.solver)

    try:
        problem = normalize(json.load(sys.stdin))
        routing = ortools_tsp(problem)
        print(json.dumps(routing))
        return 0 if routing["status"] in ("optimal", "infeasible", "unavailable") else 1
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "solver": "tsp-reference",
                    "tour": [],
                    "objective": None,
                    "message": str(exc),
                }
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
