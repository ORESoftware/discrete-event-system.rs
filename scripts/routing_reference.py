#!/usr/bin/env python3
"""Reference bridge for small capacitated vehicle-routing instances.

The deterministic exact CVRP oracle lives in Rust. This Python bridge remains
as thin adapter glue for explicit OR-Tools Routing checks without vendoring
external solver executables.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys


def exec_rust_reference(solver: str) -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    binary_name = "routing_reference"
    explicit = os.environ.get("ROUTING_REFERENCE_RUST_BIN")
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
        os.path.join(repo_root, "src", "bin", "routing_reference.rs"),
        os.path.join(repo_root, "src", "des", "general", "external_routing_reference.rs"),
        os.path.join(repo_root, "src", "des", "general", "routing.rs"),
    ]
    return all(
        not os.path.exists(source_path) or os.path.getmtime(source_path) <= binary_mtime
        for source_path in source_paths
    )


def distance(a: dict, b: dict) -> float:
    return math.hypot(float(a["x"]) - float(b["x"]), float(a["y"]) - float(b["y"]))


def route_distance(depot: dict, route: list[dict]) -> float:
    if not route:
        return 0.0
    total = distance(depot, route[0])
    for a, b in zip(route, route[1:]):
        total += distance(a, b)
    total += distance(route[-1], depot)
    return total


def normalize(raw: dict) -> dict:
    depot = raw.get("depot") or {"x": 0.0, "y": 0.0}
    customers = raw.get("customers") or []
    capacity = raw.get("vehicle_capacity", raw.get("capacity"))
    if capacity is None:
        raise ValueError("vehicle_capacity is required")
    return {
        "depot": {"x": float(depot["x"]), "y": float(depot["y"])},
        "customers": [
            {
                "id": str(customer.get("id", f"c{i}")),
                "x": float(customer["x"]),
                "y": float(customer["y"]),
                "demand": float(customer.get("demand", 0.0)),
            }
            for i, customer in enumerate(customers)
        ],
        "capacity": float(capacity),
    }


def ortools_routing(problem: dict) -> dict:
    try:
        from ortools.constraint_solver import pywrapcp, routing_enums_pb2  # type: ignore
    except Exception as exc:
        return {"status": "unavailable", "message": f"OR-Tools Routing unavailable: {exc}"}

    depot = problem["depot"]
    customers = problem["customers"]
    capacity = problem["capacity"]
    n = len(customers)
    if n == 0:
        return {"status": "optimal", "objective": 0.0, "routes": [], "message": "empty instance"}

    points = [depot] + customers
    distance_scale = 1_000_000
    demand_scale = 1_000
    manager = pywrapcp.RoutingIndexManager(n + 1, n, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index: int, to_index: int) -> int:
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(round(distance(points[from_node], points[to_node]) * distance_scale))

    transit = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit)

    scaled_demands = [0] + [int(round(customer["demand"] * demand_scale)) for customer in customers]
    scaled_capacity = int(round(capacity * demand_scale))

    def demand_callback(index: int) -> int:
        return scaled_demands[manager.IndexToNode(index)]

    demand = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand,
        0,
        [scaled_capacity for _ in range(n)],
        True,
        "Capacity",
    )

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.FromSeconds(5)

    solution = routing.SolveWithParameters(params)
    if solution is None:
        return {"status": "infeasible", "message": "OR-Tools Routing found no solution"}

    routes = []
    for vehicle in range(n):
        index = routing.Start(vehicle)
        ids = []
        route_customers = []
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            if node != 0:
                customer = customers[node - 1]
                ids.append(customer["id"])
                route_customers.append(customer)
            index = solution.Value(routing.NextVar(index))
        if ids:
            routes.append(
                {
                    "customers": ids,
                    "load": sum(customer["demand"] for customer in route_customers),
                    "distance": route_distance(depot, route_customers),
                }
            )
    routes.sort(key=lambda route: route["customers"])
    return {
        "status": "optimal",
        "objective": sum(route["distance"] for route in routes),
        "routes": routes,
        "message": "OR-Tools Routing local-search solution",
    }


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
        routing = ortools_routing(problem)
        print(json.dumps(routing))
        return 0 if routing["status"] in ("optimal", "infeasible", "unavailable") else 1
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "solver": "routing-reference",
                    "routes": [],
                    "objective": None,
                    "message": str(exc),
                }
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
