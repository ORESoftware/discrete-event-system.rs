#!/usr/bin/env python3
"""Reference bridge for directed maximum-flow instances.

The deterministic Edmonds-Karp oracle lives in Rust. This Python bridge remains
as thin adapter glue for explicit OR-Tools SimpleMaxFlow checks when installed
and when capacities can be safely integer-scaled.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Optional


SCALES = (1, 10, 100, 1000, 10000, 100000, 1000000)


def exec_rust_reference(solver: str) -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    binary_name = "max_flow_reference"
    explicit = os.environ.get("MAX_FLOW_REFERENCE_RUST_BIN")
    if explicit:
        os.execv(explicit, [explicit, "--solver", solver])
    local_binary = os.path.join(repo_root, "target", "debug", binary_name)
    if os.path.exists(local_binary):
        os.execv(local_binary, [local_binary, "--solver", solver])
    os.chdir(repo_root)
    os.execvp(
        "cargo",
        ["cargo", "run", "--quiet", "--bin", binary_name, "--", "--solver", solver],
    )


def normalize(raw: dict) -> dict:
    num_nodes = int(raw.get("numNodes", raw.get("num_nodes", 0)))
    source = int(raw.get("source", -1))
    sink = int(raw.get("sink", -1))
    edges_raw = raw.get("edges") or []
    if num_nodes < 2:
        raise ValueError("numNodes must be at least 2")
    if source < 0 or source >= num_nodes:
        raise ValueError("source is outside node range")
    if sink < 0 or sink >= num_nodes:
        raise ValueError("sink is outside node range")
    if source == sink:
        raise ValueError("source and sink must differ")
    if not edges_raw:
        raise ValueError("edges must be non-empty")
    edges = []
    for i, edge in enumerate(edges_raw):
        from_node = int(edge["from"])
        to_node = int(edge["to"])
        capacity = float(edge["capacity"])
        if from_node < 0 or from_node >= num_nodes or to_node < 0 or to_node >= num_nodes:
            raise ValueError(f"edge {i} endpoint is outside node range")
        if not math.isfinite(capacity) or capacity < 0.0:
            raise ValueError(f"edge {i} capacity must be finite and non-negative")
        edges.append(
            {
                "from": from_node,
                "to": to_node,
                "capacity": capacity,
                "name": edge.get("name"),
            }
        )
    return {"numNodes": num_nodes, "source": source, "sink": sink, "edges": edges}


def result(
    status: str,
    solver: str,
    max_flow: Optional[float] = None,
    edge_flows: Optional[list[dict]] = None,
    min_cut: Optional[dict] = None,
    node_balance: Optional[list[float]] = None,
    iterations: Optional[int] = None,
    trace: Optional[list[dict]] = None,
    message: str = "",
) -> dict:
    return {
        "status": status,
        "solver": solver,
        "maxFlow": None if max_flow is None else float(max_flow),
        "edgeFlows": [] if edge_flows is None else edge_flows,
        "minCut": {} if min_cut is None else min_cut,
        "nodeBalance": [] if node_balance is None else [float(v) for v in node_balance],
        "iterations": iterations,
        "trace": [] if trace is None else trace,
        "message": message,
    }


def edge_flow_payload(edge: dict, flow: float) -> dict:
    return {
        "from": int(edge["from"]),
        "to": int(edge["to"]),
        "capacity": float(edge["capacity"]),
        "name": edge.get("name"),
        "flow": float(flow),
    }


def node_balance(num_nodes: int, edge_flows: list[dict]) -> list[float]:
    balance = [0.0 for _ in range(num_nodes)]
    for edge in edge_flows:
        balance[int(edge["from"])] -= float(edge["flow"])
        balance[int(edge["to"])] += float(edge["flow"])
    return balance


def cut_payload(problem: dict, source_side: list[int], edge_flows: list[dict]) -> dict:
    source_set = set(source_side)
    sink_side = [i for i in range(problem["numNodes"]) if i not in source_set]
    cut_edges = [
        edge
        for edge in edge_flows
        if int(edge["from"]) in source_set and int(edge["to"]) not in source_set
    ]
    return {
        "sourceSide": [int(v) for v in source_side],
        "sinkSide": sink_side,
        "cutEdges": cut_edges,
        "capacity": float(sum(float(edge["capacity"]) for edge in cut_edges)),
    }


def choose_scale(values: list[float]) -> Optional[int]:
    for scale in SCALES:
        if all(abs(round(value * scale) - value * scale) <= 1e-6 for value in values):
            return scale
    return None


def status_name(status: object) -> str:
    return str(status).split(".")[-1].lower()


def ortools_max_flow(problem: dict) -> dict:
    try:
        from ortools.graph.python import max_flow  # type: ignore
    except Exception as exc:
        return result("unavailable", "ortools:simple-max-flow", message=str(exc))

    capacities = [edge["capacity"] for edge in problem["edges"]]
    scale = choose_scale(capacities)
    if scale is None:
        return result(
            "unsupported",
            "ortools:simple-max-flow",
            message="OR-Tools SimpleMaxFlow requires integer-scalable capacities",
        )

    solver = max_flow.SimpleMaxFlow()
    for edge in problem["edges"]:
        solver.add_arc_with_capacity(
            int(edge["from"]),
            int(edge["to"]),
            int(round(edge["capacity"] * scale)),
        )
    status = solver.solve(problem["source"], problem["sink"])
    if status != solver.OPTIMAL:
        mapped = status_name(status)
        return result(
            "infeasible" if mapped == "bad_input" else mapped,
            "ortools:simple-max-flow",
            message=f"OR-Tools SimpleMaxFlow status {mapped}",
        )

    edge_flows = [
        edge_flow_payload(edge, solver.flow(i) / scale) for i, edge in enumerate(problem["edges"])
    ]
    source_side = [int(v) for v in solver.get_source_side_min_cut()]
    balances = node_balance(problem["numNodes"], edge_flows)
    return result(
        "optimal",
        "ortools:simple-max-flow",
        max_flow=solver.optimal_flow() / scale,
        edge_flows=edge_flows,
        min_cut=cut_payload(problem, source_side, edge_flows),
        node_balance=balances,
        message="OR-Tools SimpleMaxFlow",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--solver",
        choices=["auto", "ortools", "fallback", "rust-edmonds-karp", "rust-exact"],
        default="auto",
    )
    args = parser.parse_args()
    if args.solver in ("auto", "fallback", "rust-edmonds-karp", "rust-exact"):
        exec_rust_reference(args.solver)

    try:
        problem = normalize(json.load(sys.stdin))
        output = ortools_max_flow(problem)
        print(json.dumps(output))
        return 0 if output["status"] in ("optimal", "infeasible", "unsupported", "unavailable") else 1
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "solver": "max-flow-reference",
                    "maxFlow": None,
                    "edgeFlows": [],
                    "minCut": {},
                    "nodeBalance": [],
                    "message": str(exc),
                }
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
