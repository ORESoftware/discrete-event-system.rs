#!/usr/bin/env python3
"""Reference bridge for small minimum-cost-flow instances.

The deterministic successive-shortest-path reference lives in Rust. This
Python bridge remains as thin adapter glue for explicit OR-Tools
SimpleMinCostFlow checks when installed and when numeric data can be safely
integer-scaled.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
from typing import Optional


EPS = 1e-9
SCALES = (1, 10, 100, 1000, 10000, 100000, 1000000)
RUST_REFERENCE_SOLVERS = ("auto", "fallback", "rust-ssp", "rust-exact")


def exec_rust_reference(solver: str) -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    binary_name = "min_cost_flow_reference"
    explicit = os.environ.get("MIN_COST_FLOW_REFERENCE_RUST_BIN")
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
        os.path.join(repo_root, "src", "bin", "min_cost_flow_reference.rs"),
        os.path.join(repo_root, "src", "des", "general", "external_min_cost_flow_reference.rs"),
        os.path.join(repo_root, "src", "des", "general", "min_cost_flow.rs"),
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
    value = os.environ.get("MIN_COST_FLOW_REFERENCE_EXTERNAL_FALLBACK", "")
    return value.strip().lower() in ("1", "true", "yes", "on", "rust")


def external_rust_first_enabled() -> bool:
    values = (
        os.environ.get("MIN_COST_FLOW_REFERENCE_RUST_FIRST", ""),
        os.environ.get("ORES_EXTERNAL_REFERENCE_RUST_FIRST", ""),
    )
    return any(value.strip().lower() in ("1", "true", "yes", "on", "rust") for value in values)


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
    parser.add_argument(
        "--solver",
        choices=["auto", "ortools", "fallback", "rust-ssp", "rust-exact"],
        default="auto",
    )
    args = parser.parse_args()
    if args.solver in RUST_REFERENCE_SOLVERS:
        exec_rust_reference(args.solver)
    if external_rust_first_enabled() and args.solver == "ortools":
        os.environ["MIN_COST_FLOW_REFERENCE_EXTERNAL_FALLBACK"] = "rust"
        exec_rust_reference(args.solver)
    if (
        external_rust_fallback_enabled()
        and args.solver == "ortools"
        and not package_available("ortools")
    ):
        exec_rust_reference("rust-exact")

    try:
        problem = normalize(json.load(sys.stdin))
        output = dict(ortools_min_cost_flow(problem))
        output.setdefault("solver", "ortools:simple-min-cost-flow")
        output.setdefault("objective", None)
        output.setdefault("flows", [])
        output.setdefault("nodeBalance", [])
        print(json.dumps(output))
        return 0 if output["status"] in ("optimal", "infeasible", "unavailable", "unsupported") else 1
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
