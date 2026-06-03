#!/usr/bin/env python3
"""Reference bridge for small weighted independent-set instances.

The deterministic oracle uses branch-and-bound with a remaining-weight upper
bound. When OR-Tools is installed and weights can be safely integer-scaled, the
same conflict graph is solved with CP-SAT.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from typing import Optional


EPS = 1e-9
MAX_EXACT_VERTICES = 64
SCALES = (1, 10, 100, 1000, 10000, 100000, 1000000)


def normalize(raw: dict) -> dict:
    raw_vertices = raw.get("vertices") or []
    if not raw_vertices:
        raise ValueError("vertices must be non-empty")
    vertices = []
    seen = set()
    for index, raw_vertex in enumerate(raw_vertices):
        if isinstance(raw_vertex, dict):
            vertex_id = str(raw_vertex.get("id", f"V{index + 1}"))
            weight = float(raw_vertex.get("weight", 0.0))
        else:
            vertex_id = str(raw_vertex)
            weight = 1.0
        if not vertex_id.strip():
            raise ValueError(f"vertices[{index}].id must be non-empty")
        if vertex_id in seen:
            raise ValueError(f"duplicate vertex id {vertex_id!r}")
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError(f"vertices[{index}].weight must be finite and non-negative")
        seen.add(vertex_id)
        vertices.append({"id": vertex_id, "weight": weight, "index": index})

    index_by_id = {vertex["id"]: vertex["index"] for vertex in vertices}
    edges = []
    edge_seen = set()
    for edge_index, raw_edge in enumerate(raw.get("edges") or []):
        if not isinstance(raw_edge, list) or len(raw_edge) != 2:
            raise ValueError(f"edges[{edge_index}] must be a two-item list")
        a = str(raw_edge[0])
        b = str(raw_edge[1])
        if a not in index_by_id or b not in index_by_id:
            raise ValueError(f"edges[{edge_index}] endpoints must belong to vertices")
        ai = index_by_id[a]
        bi = index_by_id[b]
        if ai == bi:
            raise ValueError(f"edges[{edge_index}] must not be a self-loop")
        key = (ai, bi) if ai < bi else (bi, ai)
        if key in edge_seen:
            raise ValueError(f"duplicate undirected edge {a!r}-{b!r}")
        edge_seen.add(key)
        edges.append([ai, bi])
    return {"vertices": vertices, "edges": edges}


def adjacency(problem: dict) -> list[list[bool]]:
    n = len(problem["vertices"])
    matrix = [[False for _ in range(n)] for _ in range(n)]
    for ai, bi in problem["edges"]:
        matrix[ai][bi] = True
        matrix[bi][ai] = True
    return matrix


def sorted_vertices(problem: dict) -> list[dict]:
    return sorted(
        problem["vertices"],
        key=lambda vertex: (-float(vertex["weight"]), str(vertex["id"])),
    )


def compatible(adj: list[list[bool]], vertex: int, selected: list[int]) -> bool:
    return all(not adj[vertex][other] for other in selected)


def candidate_better(
    problem: dict,
    weight: float,
    indices: list[int],
    best_weight: float,
    best_indices: list[int],
) -> bool:
    if weight > best_weight + EPS:
        return True
    if abs(weight - best_weight) <= EPS and len(indices) < len(best_indices):
        return True
    if abs(weight - best_weight) <= EPS and len(indices) == len(best_indices):
        by_index = {vertex["index"]: vertex for vertex in problem["vertices"]}
        lhs = sorted(by_index[index]["id"] for index in indices)
        rhs = sorted(by_index[index]["id"] for index in best_indices)
        return lhs < rhs
    return False


def output(
    status: str,
    solver: str,
    problem: dict,
    selected_indices: Optional[list[int]] = None,
    upper_bound: Optional[float] = None,
    message: str = "",
) -> dict:
    indices = [] if selected_indices is None else sorted(int(index) for index in selected_indices)
    by_index = {vertex["index"]: vertex for vertex in problem["vertices"]}
    selected_ids = [by_index[index]["id"] for index in indices]
    total_weight = float(sum(by_index[index]["weight"] for index in indices))
    return {
        "status": status,
        "solver": solver,
        "selectedVertexIndices": indices,
        "selectedVertexIds": selected_ids,
        "totalWeight": total_weight,
        "objective": total_weight,
        "upperBound": upper_bound,
        "message": message,
    }


def greedy_independent_set(problem: dict) -> dict:
    adj = adjacency(problem)
    selected = []
    for vertex in sorted_vertices(problem):
        index = int(vertex["index"])
        if compatible(adj, index, selected):
            selected.append(index)
    return output(
        "feasible",
        "python:greedy-weighted-independent-set",
        problem,
        selected,
        None,
        "greedy descending-weight independent set",
    )


def exact_independent_set(problem: dict) -> dict:
    n = len(problem["vertices"])
    if n > MAX_EXACT_VERTICES:
        return output(
            "unsupported",
            "python:exact-weighted-independent-set",
            problem,
            [],
            None,
            f"exact weighted independent set only practical for <= {MAX_EXACT_VERTICES} vertices, got {n}",
        )

    adj = adjacency(problem)
    order = sorted_vertices(problem)
    suffix_weight = [0.0 for _ in range(len(order) + 1)]
    for index in range(len(order) - 1, -1, -1):
        suffix_weight[index] = suffix_weight[index + 1] + float(order[index]["weight"])

    incumbent = greedy_independent_set(problem)
    best_indices = list(incumbent["selectedVertexIndices"])
    best_weight = float(incumbent["totalWeight"])
    current: list[int] = []

    def search(pos: int, current_weight: float) -> None:
        nonlocal best_indices, best_weight
        if pos == len(order):
            if candidate_better(problem, current_weight, current, best_weight, best_indices):
                best_indices = list(current)
                best_weight = current_weight
            return
        if current_weight + suffix_weight[pos] + EPS < best_weight:
            return

        vertex = order[pos]
        vertex_index = int(vertex["index"])
        if compatible(adj, vertex_index, current):
            current.append(vertex_index)
            search(pos + 1, current_weight + float(vertex["weight"]))
            current.pop()
        search(pos + 1, current_weight)

    search(0, 0.0)
    return output(
        "optimal",
        "python:exact-weighted-independent-set",
        problem,
        best_indices,
        suffix_weight[0],
        "exact branch-and-bound weighted independent set",
    )


def choose_scale(values: list[float]) -> Optional[int]:
    for scale in SCALES:
        if all(abs(round(value * scale) - value * scale) <= 1e-6 for value in values):
            return scale
    return None


def ortools_independent_set(problem: dict) -> dict:
    try:
        from ortools.sat.python import cp_model  # type: ignore
    except Exception as exc:
        return output("unavailable", "ortools:cp-sat-weighted-independent-set", problem, None, None, str(exc))

    vertices = problem["vertices"]
    value_scale = choose_scale([vertex["weight"] for vertex in vertices])
    if value_scale is None:
        return output(
            "unsupported",
            "ortools:cp-sat-weighted-independent-set",
            problem,
            None,
            None,
            "OR-Tools CP-SAT bridge requires integer-scalable vertex weights",
        )

    weights = [int(round(vertex["weight"] * value_scale)) for vertex in vertices]
    model = cp_model.CpModel()
    x = [model.NewBoolVar(f"x_{vertex['id']}") for vertex in vertices]
    for ai, bi in problem["edges"]:
        model.Add(x[ai] + x[bi] <= 1)
    model.Maximize(sum(weights[index] * x[index] for index in range(len(vertices))))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10.0
    solver.parameters.num_search_workers = 1
    status_code = solver.Solve(model)
    status_name = solver.StatusName(status_code).lower()
    if status_code not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return output(
            "infeasible" if status_name == "infeasible" else status_name,
            "ortools:cp-sat-weighted-independent-set",
            problem,
            None,
            None,
            f"OR-Tools CP-SAT status {status_name}",
        )
    selected = [index for index, var in enumerate(x) if solver.Value(var)]
    result = output(
        "optimal" if status_code == cp_model.OPTIMAL else "feasible",
        "ortools:cp-sat-weighted-independent-set",
        problem,
        selected,
        solver.BestObjectiveBound() / value_scale,
        f"OR-Tools CP-SAT status {status_name}",
    )
    result["objectiveBound"] = solver.BestObjectiveBound() / value_scale
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", choices=["auto", "ortools", "fallback"], default="auto")
    args = parser.parse_args()

    try:
        problem = normalize(json.load(sys.stdin))
        exact = exact_independent_set(problem)
        if args.solver == "fallback":
            print(json.dumps(exact))
            return 0 if exact["status"] in ("optimal", "feasible", "unsupported") else 1

        ortools = ortools_independent_set(problem)
        if args.solver == "ortools":
            print(json.dumps(ortools))
            return 0 if ortools["status"] in ("optimal", "feasible", "unavailable", "unsupported") else 1

        result = dict(exact)
        result["solver"] = (
            "ortools:cp-sat-weighted-independent-set+python:exact-weighted-independent-set"
            if ortools.get("status") != "unavailable"
            else "python:exact-weighted-independent-set"
        )
        result["ortoolsStatus"] = ortools.get("status")
        result["ortoolsSelectedVertexIndices"] = ortools.get("selectedVertexIndices", [])
        result["ortoolsSelectedVertexIds"] = ortools.get("selectedVertexIds", [])
        result["ortoolsTotalWeight"] = ortools.get("totalWeight")
        result["ortoolsObjective"] = ortools.get("objective")
        result["ortoolsObjectiveBound"] = ortools.get("objectiveBound")
        result["ortoolsMessage"] = ortools.get("message")
        print(json.dumps(result))
        return 0 if result["status"] in ("optimal", "feasible", "unsupported") else 1
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "solver": "python:weighted-independent-set-reference",
                    "selectedVertexIndices": [],
                    "selectedVertexIds": [],
                    "totalWeight": 0.0,
                    "objective": None,
                    "upperBound": None,
                    "message": str(exc),
                }
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
