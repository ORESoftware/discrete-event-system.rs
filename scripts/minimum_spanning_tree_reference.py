#!/usr/bin/env python3
"""Reference bridge for small minimum-spanning-tree instances.

The deterministic Kruskal oracle lives in Rust. This Python bridge remains as
thin adapter glue for explicit OR-Tools CP-SAT checks using binary
selected-edge variables and integer flow variables that force root
connectivity.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Optional


OBJECTIVE_SCALE = 1_000_000


def exec_rust_reference(solver: str) -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    binary_name = "minimum_spanning_tree_reference"
    explicit = os.environ.get("MINIMUM_SPANNING_TREE_REFERENCE_RUST_BIN")
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
        os.path.join(repo_root, "src", "bin", "minimum_spanning_tree_reference.rs"),
        os.path.join(
            repo_root,
            "src",
            "des",
            "general",
            "external_minimum_spanning_tree_reference.rs",
        ),
        os.path.join(repo_root, "src", "des", "general", "minimum_spanning_tree.rs"),
    ]
    return all(
        not os.path.exists(source_path) or os.path.getmtime(source_path) <= binary_mtime
        for source_path in source_paths
    )


def normalize(raw: dict) -> dict:
    vertices = [str(value) for value in (raw.get("vertices") or [])]
    if not vertices:
        raise ValueError("vertices must be non-empty")
    if any(not vertex.strip() for vertex in vertices):
        raise ValueError("vertices must be non-empty strings")
    if len(set(vertices)) != len(vertices):
        raise ValueError("vertices must be unique")
    index = {vertex: i for i, vertex in enumerate(vertices)}
    seen_ids = set()
    seen_edges = set()
    edges = []
    for edge_index, raw_edge in enumerate(raw.get("edges") or []):
        edge_id = str(raw_edge.get("id") or f"E{edge_index + 1}")
        if not edge_id.strip():
            raise ValueError(f"edges[{edge_index}].id must be non-empty")
        if edge_id in seen_ids:
            raise ValueError(f"duplicate edge id {edge_id!r}")
        seen_ids.add(edge_id)
        a = str(raw_edge.get("from"))
        b = str(raw_edge.get("to"))
        if a not in index or b not in index:
            raise ValueError(f"edges[{edge_index}] endpoints must belong to vertices")
        ai = index[a]
        bi = index[b]
        if ai == bi:
            raise ValueError(f"edges[{edge_index}] must not be a self-loop")
        key = (ai, bi) if ai < bi else (bi, ai)
        if key in seen_edges:
            raise ValueError(f"duplicate undirected edge {a!r}-{b!r}")
        seen_edges.add(key)
        weight = float(raw_edge.get("weight"))
        if not math.isfinite(weight):
            raise ValueError(f"edges[{edge_index}].weight must be finite")
        edges.append({"id": edge_id, "from": ai, "to": bi, "weight": weight})
    return {"vertices": vertices, "edges": edges}


def output(
    status: str,
    solver: str,
    problem: dict,
    selected: Optional[list[int]] = None,
    message: str = "",
) -> dict:
    if selected is None:
        total = None
        ids: list[str] = []
    else:
        selected = sorted(selected)
        total = sum(problem["edges"][idx]["weight"] for idx in selected)
        ids = [problem["edges"][idx]["id"] for idx in selected]
    return {
        "status": status,
        "solver": solver,
        "selectedEdgeIndices": [] if selected is None else selected,
        "selectedEdgeIds": ids,
        "objective": total,
        "totalWeight": total,
        "message": message,
    }


def ortools_mst(problem: dict) -> dict:
    try:
        from ortools.sat.python import cp_model  # type: ignore
    except Exception as exc:
        return output("unavailable", "ortools:cp-sat-mst", problem, None, str(exc))

    n = len(problem["vertices"])
    if n == 1:
        return output("optimal", "ortools:cp-sat-mst", problem, [], "single-vertex MST")
    if not problem["edges"]:
        return output("infeasible", "ortools:cp-sat-mst", problem, None, "graph is disconnected")

    model = cp_model.CpModel()
    selected = [model.NewBoolVar(f"select_{edge['id']}") for edge in problem["edges"]]
    forward = []
    reverse = []
    max_flow = n - 1
    for edge_idx, edge in enumerate(problem["edges"]):
        fwd = model.NewIntVar(0, max_flow, f"flow_{edge['from']}_{edge['to']}_{edge_idx}")
        rev = model.NewIntVar(0, max_flow, f"flow_{edge['to']}_{edge['from']}_{edge_idx}")
        model.Add(fwd + rev <= max_flow * selected[edge_idx])
        forward.append(fwd)
        reverse.append(rev)

    model.Add(sum(selected) == n - 1)
    for vertex in range(n):
        inflow = []
        outflow = []
        for edge_idx, edge in enumerate(problem["edges"]):
            if edge["to"] == vertex:
                inflow.append(forward[edge_idx])
                outflow.append(reverse[edge_idx])
            elif edge["from"] == vertex:
                inflow.append(reverse[edge_idx])
                outflow.append(forward[edge_idx])
        if vertex == 0:
            model.Add(sum(outflow) - sum(inflow) == n - 1)
        else:
            model.Add(sum(inflow) - sum(outflow) == 1)

    objective = sum(
        int(round(edge["weight"] * OBJECTIVE_SCALE)) * selected[idx]
        for idx, edge in enumerate(problem["edges"])
    )
    model.Minimize(objective)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10.0
    solver.parameters.num_search_workers = 1
    status_code = solver.Solve(model)
    status_name = solver.StatusName(status_code).lower()
    if status_code not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return output(
            "infeasible" if status_name == "infeasible" else status_name,
            "ortools:cp-sat-mst",
            problem,
            None,
            f"OR-Tools CP-SAT status {status_name}",
        )
    selected_indices = [idx for idx, var in enumerate(selected) if solver.Value(var)]
    result = output(
        "optimal" if status_code == cp_model.OPTIMAL else "feasible",
        "ortools:cp-sat-mst",
        problem,
        selected_indices,
        f"OR-Tools CP-SAT status {status_name}",
    )
    result["objectiveBound"] = solver.BestObjectiveBound() / OBJECTIVE_SCALE
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--solver",
        choices=["auto", "ortools", "fallback", "rust-kruskal", "rust-exact"],
        default="auto",
    )
    args = parser.parse_args()
    if args.solver in ("auto", "fallback", "rust-kruskal", "rust-exact"):
        exec_rust_reference(args.solver)

    try:
        problem = normalize(json.load(sys.stdin))
        output = ortools_mst(problem)
        print(json.dumps(output))
        return 0 if output["status"] in ("optimal", "feasible", "infeasible", "unavailable") else 1
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "solver": "python:minimum-spanning-tree-reference",
                    "selectedEdgeIndices": [],
                    "selectedEdgeIds": [],
                    "objective": None,
                    "totalWeight": None,
                    "message": str(exc),
                }
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
