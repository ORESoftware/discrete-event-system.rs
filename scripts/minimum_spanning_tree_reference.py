#!/usr/bin/env python3
"""Reference bridge for small minimum-spanning-tree instances.

The deterministic oracle is Kruskal's algorithm. When OR-Tools is installed,
the same undirected graph is also sent to CP-SAT using binary selected-edge
variables and integer flow variables that force root connectivity.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from typing import Optional


OBJECTIVE_SCALE = 1_000_000


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


class DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0 for _ in range(size)]

    def find(self, value: int) -> int:
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, a: int, b: int) -> bool:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


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


def exact_mst(problem: dict) -> dict:
    n = len(problem["vertices"])
    if n == 1:
        return output("optimal", "python:kruskal-mst", problem, [], "single-vertex MST")
    dsu = DisjointSet(n)
    selected = []
    order = list(range(len(problem["edges"])))
    order.sort(key=lambda idx: (problem["edges"][idx]["weight"], problem["edges"][idx]["id"]))
    for edge_idx in order:
        edge = problem["edges"][edge_idx]
        if dsu.union(edge["from"], edge["to"]):
            selected.append(edge_idx)
            if len(selected) + 1 == n:
                break
    if len(selected) + 1 != n:
        return output("infeasible", "python:kruskal-mst", problem, None, "graph is disconnected")
    return output("optimal", "python:kruskal-mst", problem, selected, "Kruskal minimum spanning tree")


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
    parser.add_argument("--solver", choices=["auto", "ortools", "fallback"], default="auto")
    args = parser.parse_args()

    try:
        problem = normalize(json.load(sys.stdin))
        exact = exact_mst(problem)
        if args.solver == "fallback":
            print(json.dumps(exact))
            return 0 if exact["status"] in ("optimal", "infeasible") else 1

        ortools = ortools_mst(problem)
        if args.solver == "ortools":
            print(json.dumps(ortools))
            return 0 if ortools["status"] in ("optimal", "feasible", "infeasible", "unavailable") else 1

        result = dict(exact)
        result["solver"] = (
            "ortools:cp-sat-mst+python:kruskal-mst"
            if ortools.get("status") != "unavailable"
            else "python:kruskal-mst"
        )
        result["ortoolsStatus"] = ortools.get("status")
        result["ortoolsSelectedEdgeIndices"] = ortools.get("selectedEdgeIndices", [])
        result["ortoolsSelectedEdgeIds"] = ortools.get("selectedEdgeIds", [])
        result["ortoolsObjective"] = ortools.get("objective")
        result["ortoolsTotalWeight"] = ortools.get("totalWeight")
        result["ortoolsMessage"] = ortools.get("message")
        result["ortoolsObjectiveBound"] = ortools.get("objectiveBound")
        print(json.dumps(result))
        return 0 if result["status"] in ("optimal", "infeasible") else 1
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
