#!/usr/bin/env python3
"""Reference bridge for small minimum-cost-flow instances.

The bridge always computes a deterministic successive-shortest-path reference
with lower-bound normalization. When OR-Tools is installed and the numeric data
can be safely integer-scaled, it also calls OR-Tools SimpleMinCostFlow on the
same normalized network so the Rust validator can compare against a specialized
external network optimizer.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from typing import Optional


EPS = 1e-9
SCALES = (1, 10, 100, 1000, 10000, 100000, 1000000)


def normalize(raw: dict) -> dict:
    num_nodes = int(raw.get("num_nodes", raw.get("numNodes", 0)))
    supplies = [float(v) for v in raw.get("supplies", [])]
    arcs = raw.get("arcs") or []
    if num_nodes <= 0:
        raise ValueError("num_nodes must be positive")
    if len(supplies) != num_nodes:
        raise ValueError(f"supplies length {len(supplies)} != num_nodes {num_nodes}")
    if abs(sum(supplies)) > 1e-7:
        raise ValueError(f"supplies must sum to zero, got {sum(supplies):.3e}")
    if not arcs:
        raise ValueError("arcs must be non-empty")

    normalized_arcs = []
    for index, raw_arc in enumerate(arcs):
        from_node = int(raw_arc["from"])
        to_node = int(raw_arc["to"])
        lower_bound = float(raw_arc.get("lower_bound", raw_arc.get("lowerBound", 0.0)))
        capacity = float(raw_arc["capacity"])
        cost = float(raw_arc["cost"])
        if from_node < 0 or from_node >= num_nodes or to_node < 0 or to_node >= num_nodes:
            raise ValueError(f"arc {index} endpoint out of range")
        if from_node == to_node:
            raise ValueError(f"arc {index} is a self-loop")
        if any(not math.isfinite(v) for v in (lower_bound, capacity, cost)):
            raise ValueError(f"arc {index} fields must be finite")
        if lower_bound < -EPS:
            raise ValueError(f"arc {index} lower_bound must be non-negative")
        if capacity + EPS < lower_bound:
            raise ValueError(f"arc {index} capacity {capacity} < lower_bound {lower_bound}")
        normalized_arcs.append(
            {
                "from": from_node,
                "to": to_node,
                "lowerBound": lower_bound,
                "capacity": capacity,
                "cost": cost,
                "name": raw_arc.get("name"),
            }
        )
    return {"numNodes": num_nodes, "supplies": supplies, "arcs": normalized_arcs}


def arc_payload(problem: dict, flows: list[float]) -> list[dict]:
    return [
        {
            "from": arc["from"],
            "to": arc["to"],
            "lowerBound": arc["lowerBound"],
            "capacity": arc["capacity"],
            "cost": arc["cost"],
            "flow": float(flow),
            "name": arc.get("name"),
        }
        for arc, flow in zip(problem["arcs"], flows)
    ]


def node_balance(problem: dict, flows: list[float]) -> list[float]:
    balance = [0.0 for _ in range(problem["numNodes"])]
    for arc, flow in zip(problem["arcs"], flows):
        balance[arc["from"]] += flow
        balance[arc["to"]] -= flow
    return balance


def add_residual_arc(
    residual: list[list[dict]],
    from_node: int,
    to_node: int,
    capacity: float,
    cost: float,
    original: Optional[int],
) -> None:
    forward_index = len(residual[from_node])
    reverse_index = len(residual[to_node])
    residual[from_node].append(
        {
            "to": to_node,
            "rev": reverse_index,
            "cap": capacity,
            "cost": cost,
            "original": original,
            "direction": 1.0,
        }
    )
    residual[to_node].append(
        {
            "to": from_node,
            "rev": forward_index,
            "cap": 0.0,
            "cost": -cost,
            "original": original,
            "direction": -1.0,
        }
    )


def shortest_path(residual: list[list[dict]], source: int, sink: int) -> Optional[tuple[list[int], list[int], float]]:
    n = len(residual)
    dist = [math.inf for _ in range(n)]
    prev_node = [-1 for _ in range(n)]
    prev_edge = [-1 for _ in range(n)]
    dist[source] = 0.0
    for _ in range(max(0, n - 1)):
        changed = False
        for node, edges in enumerate(residual):
            if not math.isfinite(dist[node]):
                continue
            for edge_index, edge in enumerate(edges):
                if edge["cap"] <= EPS:
                    continue
                candidate = dist[node] + edge["cost"]
                if candidate < dist[edge["to"]] - EPS:
                    dist[edge["to"]] = candidate
                    prev_node[edge["to"]] = node
                    prev_edge[edge["to"]] = edge_index
                    changed = True
        if not changed:
            break
    if not math.isfinite(dist[sink]):
        return None
    return prev_node, prev_edge, dist[sink]


def path_nodes(prev_node: list[int], source: int, sink: int) -> list[int]:
    out = [sink]
    node = sink
    while node != source:
        node = prev_node[node]
        out.append(node)
    out.reverse()
    return out


def exact_min_cost_flow(problem: dict) -> dict:
    num_nodes = problem["numNodes"]
    source = num_nodes
    sink = num_nodes + 1
    residual: list[list[dict]] = [[] for _ in range(num_nodes + 2)]
    adjusted_supply = list(problem["supplies"])
    flows = [arc["lowerBound"] for arc in problem["arcs"]]
    total_cost = 0.0

    for index, arc in enumerate(problem["arcs"]):
        adjusted_supply[arc["from"]] -= arc["lowerBound"]
        adjusted_supply[arc["to"]] += arc["lowerBound"]
        total_cost += arc["lowerBound"] * arc["cost"]
        add_residual_arc(
            residual,
            arc["from"],
            arc["to"],
            arc["capacity"] - arc["lowerBound"],
            arc["cost"],
            index,
        )

    required = 0.0
    for node, supply in enumerate(adjusted_supply):
        if supply > EPS:
            add_residual_arc(residual, source, node, supply, 0.0, None)
            required += supply
        elif supply < -EPS:
            add_residual_arc(residual, node, sink, -supply, 0.0, None)

    sent = 0.0
    trace = []
    while sent < required - EPS:
        path = shortest_path(residual, source, sink)
        if path is None:
            return {
                "status": "infeasible",
                "solver": "python:ssp-min-cost-flow",
                "objective": None,
                "flows": arc_payload(problem, flows),
                "nodeBalance": node_balance(problem, flows),
                "iterations": len(trace),
                "message": "not enough residual capacity to satisfy demands",
            }
        prev_node, prev_edge, unit_cost = path
        bottleneck = required - sent
        node = sink
        while node != source:
            parent = prev_node[node]
            edge_index = prev_edge[node]
            bottleneck = min(bottleneck, residual[parent][edge_index]["cap"])
            node = parent

        node = sink
        while node != source:
            parent = prev_node[node]
            edge_index = prev_edge[node]
            edge = residual[parent][edge_index]
            to_node = edge["to"]
            reverse_index = edge["rev"]
            edge["cap"] -= bottleneck
            residual[to_node][reverse_index]["cap"] += bottleneck
            if edge["original"] is not None:
                flows[edge["original"]] += edge["direction"] * bottleneck
            node = parent

        sent += bottleneck
        total_cost += bottleneck * unit_cost
        trace.append(
            {
                "iter": len(trace),
                "path": path_nodes(prev_node, source, sink),
                "bottleneck": float(bottleneck),
                "unitCost": float(unit_cost),
                "totalCost": float(total_cost),
            }
        )

    return {
        "status": "optimal",
        "solver": "python:ssp-min-cost-flow",
        "objective": float(total_cost),
        "flows": arc_payload(problem, flows),
        "nodeBalance": node_balance(problem, flows),
        "iterations": len(trace),
        "message": "successive shortest augmenting path reference",
    }


def choose_scale(values: list[float]) -> Optional[int]:
    for scale in SCALES:
        if all(abs(round(value * scale) - value * scale) <= 1e-6 for value in values):
            return scale
    return None


def status_name(status: object) -> str:
    text = str(status)
    return text.split(".")[-1].lower()


def ortools_min_cost_flow(problem: dict) -> dict:
    try:
        from ortools.graph.python import min_cost_flow  # type: ignore
    except Exception as exc:
        return {"status": "unavailable", "message": f"OR-Tools SimpleMinCostFlow unavailable: {exc}"}

    flow_values = list(problem["supplies"])
    for arc in problem["arcs"]:
        flow_values.extend([arc["lowerBound"], arc["capacity"]])
    cost_values = [arc["cost"] for arc in problem["arcs"]]
    flow_scale = choose_scale(flow_values)
    cost_scale = choose_scale(cost_values)
    if flow_scale is None or cost_scale is None:
        return {
            "status": "unsupported",
            "message": "OR-Tools SimpleMinCostFlow requires integer-scalable supplies/capacities/costs",
        }

    solver = min_cost_flow.SimpleMinCostFlow()
    adjusted_supply = list(problem["supplies"])
    base_cost = 0.0
    for arc in problem["arcs"]:
        adjusted_supply[arc["from"]] -= arc["lowerBound"]
        adjusted_supply[arc["to"]] += arc["lowerBound"]
        base_cost += arc["lowerBound"] * arc["cost"]
        capacity = int(round((arc["capacity"] - arc["lowerBound"]) * flow_scale))
        cost = int(round(arc["cost"] * cost_scale))
        solver.add_arc_with_capacity_and_unit_cost(arc["from"], arc["to"], capacity, cost)
    for node, supply in enumerate(adjusted_supply):
        solver.set_node_supply(node, int(round(supply * flow_scale)))

    status = solver.solve()
    mapped = status_name(status)
    if status != solver.OPTIMAL:
        return {"status": mapped, "message": f"OR-Tools SimpleMinCostFlow status {mapped}"}

    flows = []
    for index, arc in enumerate(problem["arcs"]):
        flows.append(arc["lowerBound"] + solver.flow(index) / flow_scale)
    objective = base_cost + solver.optimal_cost() / (flow_scale * cost_scale)
    return {
        "status": "optimal",
        "solver": "ortools:simple-min-cost-flow",
        "objective": float(objective),
        "flows": arc_payload(problem, flows),
        "nodeBalance": node_balance(problem, flows),
        "message": "OR-Tools SimpleMinCostFlow",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", choices=["auto", "ortools", "fallback"], default="auto")
    args = parser.parse_args()

    try:
        problem = normalize(json.load(sys.stdin))
        reference = exact_min_cost_flow(problem)
        if args.solver == "fallback":
            print(json.dumps(reference))
            return 0 if reference["status"] in ("optimal", "infeasible") else 1

        ortools = ortools_min_cost_flow(problem)
        if args.solver == "ortools":
            output = dict(ortools)
            output.setdefault("solver", "ortools:simple-min-cost-flow")
            output.setdefault("objective", None)
            output.setdefault("flows", [])
            output.setdefault("nodeBalance", [])
            output["referenceStatus"] = reference.get("status")
            output["referenceObjective"] = reference.get("objective")
            print(json.dumps(output))
            return 0 if output["status"] in ("optimal", "infeasible", "unavailable", "unsupported") else 1

        output = dict(reference)
        output["solver"] = (
            "ortools:simple-min-cost-flow+python:ssp"
            if ortools.get("status") != "unavailable"
            else "python:ssp-min-cost-flow"
        )
        output["ortoolsStatus"] = ortools.get("status")
        output["ortoolsObjective"] = ortools.get("objective")
        output["ortoolsFlows"] = ortools.get("flows", [])
        output["ortoolsNodeBalance"] = ortools.get("nodeBalance", [])
        output["ortoolsMessage"] = ortools.get("message", "")
        print(json.dumps(output))
        return 0 if output["status"] in ("optimal", "infeasible") else 1
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "solver": "min-cost-flow-reference",
                    "objective": None,
                    "flows": [],
                    "nodeBalance": [],
                    "message": str(exc),
                }
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
