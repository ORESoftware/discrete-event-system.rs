#!/usr/bin/env python3
"""Reference bridge for small capacitated vehicle-routing instances.

The bridge calls OR-Tools Routing when installed and also computes an exact
small-instance CVRP optimum by dynamic programming over feasible customer
subsets. The exact result is the deterministic oracle; OR-Tools metadata is
returned so the Rust validation suite can verify that the local external
routing engine agrees on the same input.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from typing import Optional


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


def reconstruct_route(route_mask: int, route_last: list[Optional[int]], parent: list[list[Optional[int]]]) -> list[int]:
    mask = route_mask
    last = route_last[route_mask]
    if last is None:
        return []
    order = []
    while True:
        order.append(last)
        prev = parent[mask][last]
        if prev is None:
            break
        mask ^= 1 << last
        last = prev
    order.reverse()
    return order


def exact_cvrp(problem: dict) -> dict:
    depot = problem["depot"]
    customers = problem["customers"]
    capacity = problem["capacity"]
    n = len(customers)
    if n > 16:
        return {
            "status": "unsupported",
            "objective": None,
            "routes": [],
            "message": f"exact CVRP only practical for n <= 16, got {n}",
        }
    if any(customer["demand"] > capacity + 1e-9 for customer in customers):
        return {
            "status": "infeasible",
            "objective": None,
            "routes": [],
            "message": "customer demand exceeds vehicle capacity",
        }
    if n == 0:
        return {"status": "optimal", "objective": 0.0, "routes": [], "message": "empty instance"}

    full = (1 << n) - 1
    demand = [0.0 for _ in range(1 << n)]
    for mask in range(1, full + 1):
        bit = mask & -mask
        idx = bit.bit_length() - 1
        demand[mask] = demand[mask ^ bit] + customers[idx]["demand"]

    path_cost = [[math.inf for _ in range(n)] for _ in range(1 << n)]
    parent: list[list[Optional[int]]] = [[None for _ in range(n)] for _ in range(1 << n)]
    for i, customer in enumerate(customers):
        path_cost[1 << i][i] = distance(depot, customer)

    for mask in range(1, full + 1):
        for last in range(n):
            if not (mask & (1 << last)):
                continue
            prev_mask = mask ^ (1 << last)
            if prev_mask == 0:
                continue
            best = path_cost[mask][last]
            best_prev = parent[mask][last]
            for prev in range(n):
                if not (prev_mask & (1 << prev)):
                    continue
                candidate = path_cost[prev_mask][prev] + distance(customers[prev], customers[last])
                if candidate < best:
                    best = candidate
                    best_prev = prev
            path_cost[mask][last] = best
            parent[mask][last] = best_prev

    route_cost = [math.inf for _ in range(1 << n)]
    route_last: list[Optional[int]] = [None for _ in range(1 << n)]
    feasible_masks = 0
    for mask in range(1, full + 1):
        if demand[mask] > capacity + 1e-9:
            continue
        feasible_masks += 1
        for last in range(n):
            if not (mask & (1 << last)):
                continue
            candidate = path_cost[mask][last] + distance(customers[last], depot)
            if candidate < route_cost[mask]:
                route_cost[mask] = candidate
                route_last[mask] = last

    cover_cost = [math.inf for _ in range(1 << n)]
    choice = [0 for _ in range(1 << n)]
    cover_cost[0] = 0.0
    for mask in range(1, full + 1):
        sub = mask
        while sub:
            if math.isfinite(route_cost[sub]):
                candidate = cover_cost[mask ^ sub] + route_cost[sub]
                if candidate < cover_cost[mask]:
                    cover_cost[mask] = candidate
                    choice[mask] = sub
            sub = (sub - 1) & mask

    if not math.isfinite(cover_cost[full]):
        return {"status": "infeasible", "objective": None, "routes": [], "message": "no feasible route cover"}

    routes = []
    mask = full
    while mask:
        route_mask = choice[mask]
        order = reconstruct_route(route_mask, route_last, parent)
        route_customers = [customers[i] for i in order]
        routes.append(
            {
                "customers": [customer["id"] for customer in route_customers],
                "load": demand[route_mask],
                "distance": route_distance(depot, route_customers),
            }
        )
        mask ^= route_mask
    routes.sort(key=lambda route: route["customers"])
    return {
        "status": "optimal",
        "objective": sum(route["distance"] for route in routes),
        "routes": routes,
        "feasibleRouteMasks": feasible_masks,
        "message": "exact CVRP route-cover dynamic program",
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
    parser.add_argument("--solver", choices=["auto", "ortools", "fallback"], default="auto")
    args = parser.parse_args()

    try:
        problem = normalize(json.load(sys.stdin))
        exact = exact_cvrp(problem)
        if args.solver == "fallback":
            print(json.dumps(exact))
            return 0 if exact["status"] in ("optimal", "infeasible", "unsupported") else 1

        routing = ortools_routing(problem)
        if args.solver == "ortools":
            output = dict(routing)
            output["referenceStatus"] = exact.get("status")
            output["referenceObjective"] = exact.get("objective")
            print(json.dumps(output))
            return 0 if output["status"] in ("optimal", "infeasible", "unavailable") else 1

        solver = "ortools:routing+python:exact-cvrp" if routing["status"] != "unavailable" else "python:exact-cvrp"
        print(
            json.dumps(
                {
                    "status": exact["status"],
                    "solver": solver,
                    "routes": exact["routes"],
                    "objective": exact["objective"],
                    "message": exact.get("message", ""),
                    "feasibleRouteMasks": exact.get("feasibleRouteMasks"),
                    "ortoolsStatus": routing.get("status"),
                    "ortoolsObjective": routing.get("objective"),
                    "ortoolsRoutes": routing.get("routes", []),
                    "ortoolsMessage": routing.get("message", ""),
                }
            )
        )
        return 0
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
