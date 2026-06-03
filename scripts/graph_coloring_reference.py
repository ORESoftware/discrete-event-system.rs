#!/usr/bin/env python3
"""Reference bridge for small graph-coloring instances.

The deterministic oracle is a DSATUR-style chromatic-number search. When
OR-Tools is installed, the same graph is also sent to CP-SAT with one integer
color variable per vertex and an objective minimizing max(color) + 1.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional


UNCOLORED = -1
MAX_EXACT_VERTICES = 40


def normalize(raw: dict) -> dict:
    vertices = [str(value) for value in (raw.get("vertices") or [])]
    if not vertices:
        raise ValueError("vertices must be non-empty")
    if any(not vertex.strip() for vertex in vertices):
        raise ValueError("vertices must be non-empty strings")
    if len(set(vertices)) != len(vertices):
        raise ValueError("vertices must be unique")
    index = {vertex: i for i, vertex in enumerate(vertices)}
    edges = []
    seen = set()
    for edge_index, raw_edge in enumerate(raw.get("edges") or []):
        if not isinstance(raw_edge, list) or len(raw_edge) != 2:
            raise ValueError(f"edges[{edge_index}] must be a two-item list")
        a = str(raw_edge[0])
        b = str(raw_edge[1])
        if a not in index or b not in index:
            raise ValueError(f"edges[{edge_index}] endpoints must belong to vertices")
        ai = index[a]
        bi = index[b]
        if ai == bi:
            raise ValueError(f"edges[{edge_index}] must not be a self-loop")
        key = (ai, bi) if ai < bi else (bi, ai)
        if key in seen:
            raise ValueError(f"duplicate undirected edge {a!r}-{b!r}")
        seen.add(key)
        edges.append([ai, bi])
    return {"vertices": vertices, "edges": edges}


def adjacency(problem: dict) -> list[list[int]]:
    adj = [[] for _ in problem["vertices"]]
    for ai, bi in problem["edges"]:
        adj[ai].append(bi)
        adj[bi].append(ai)
    return [sorted(set(row)) for row in adj]


def color_names(count: int) -> list[str]:
    return [f"C{index + 1}" for index in range(count)]


def output(
    status: str,
    solver: str,
    problem: dict,
    colors: Optional[list[int]] = None,
    message: str = "",
) -> dict:
    if colors is None:
        used = None
        names: list[str] = []
        objective = None
    else:
        used = max(colors) + 1 if colors else 0
        names = color_names(used)
        objective = float(used)
    return {
        "status": status,
        "solver": solver,
        "colorIndices": [] if colors is None else colors,
        "colorNames": names,
        "usedColorCount": used,
        "objective": objective,
        "message": message,
    }


def greedy_coloring(problem: dict) -> dict:
    adj = adjacency(problem)
    order = list(range(len(problem["vertices"])))
    order.sort(key=lambda vertex: (-len(adj[vertex]), problem["vertices"][vertex]))
    colors = [UNCOLORED for _ in problem["vertices"]]
    for vertex in order:
        unavailable = {colors[neighbor] for neighbor in adj[vertex] if colors[neighbor] != UNCOLORED}
        color = 0
        while color in unavailable:
            color += 1
        colors[vertex] = color
    return output("feasible", "python:greedy-graph-coloring", problem, colors, "Welsh-Powell greedy graph coloring")


def select_dsatur_vertex(adj: list[list[int]], colors: list[int]) -> Optional[int]:
    best = None
    for vertex, color in enumerate(colors):
        if color != UNCOLORED:
            continue
        sat = len({colors[neighbor] for neighbor in adj[vertex] if colors[neighbor] != UNCOLORED})
        degree = len(adj[vertex])
        if best is None or sat > best[1] or (sat == best[1] and degree > best[2]):
            best = (vertex, sat, degree)
    return None if best is None else best[0]


def can_use_color(adj: list[list[int]], colors: list[int], vertex: int, color: int) -> bool:
    return all(colors[neighbor] != color for neighbor in adj[vertex])


def dsatur_color(adj: list[list[int]], max_colors: int, colors: list[int], used_colors: int) -> bool:
    vertex = select_dsatur_vertex(adj, colors)
    if vertex is None:
        return True
    for color in range(min(used_colors + 1, max_colors)):
        if not can_use_color(adj, colors, vertex, color):
            continue
        colors[vertex] = color
        if dsatur_color(adj, max_colors, colors, max(used_colors, color + 1)):
            return True
        colors[vertex] = UNCOLORED
    return False


def exact_graph_coloring(problem: dict) -> dict:
    n = len(problem["vertices"])
    if n > MAX_EXACT_VERTICES:
        return output(
            "unsupported",
            "python:exact-graph-coloring",
            problem,
            None,
            f"exact graph-coloring only practical for <= {MAX_EXACT_VERTICES} vertices, got {n}",
        )
    adj = adjacency(problem)
    greedy = greedy_coloring(problem)
    upper = int(greedy["usedColorCount"] or max(1, n))
    lower = 1 if not problem["edges"] else 2
    for k in range(lower, upper + 1):
        colors = [UNCOLORED for _ in problem["vertices"]]
        if dsatur_color(adj, k, colors, 0):
            return output(
                "optimal",
                "python:exact-graph-coloring",
                problem,
                colors,
                "exact DSATUR-style chromatic search",
            )
    return output("infeasible", "python:exact-graph-coloring", problem, None, "no coloring found")


def ortools_graph_coloring(problem: dict) -> dict:
    try:
        from ortools.sat.python import cp_model  # type: ignore
    except Exception as exc:
        return output("unavailable", "ortools:cp-sat-graph-coloring", problem, None, str(exc))

    n = len(problem["vertices"])
    model = cp_model.CpModel()
    colors = [model.NewIntVar(0, max(0, n - 1), f"color_v{index}") for index in range(n)]
    for ai, bi in problem["edges"]:
        model.Add(colors[ai] != colors[bi])
    max_color = model.NewIntVar(0, max(0, n - 1), "max_color")
    model.AddMaxEquality(max_color, colors)
    model.Minimize(max_color)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10.0
    solver.parameters.num_search_workers = 1
    status_code = solver.Solve(model)
    status_name = solver.StatusName(status_code).lower()
    if status_code not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return output(
            "infeasible" if status_name == "infeasible" else status_name,
            "ortools:cp-sat-graph-coloring",
            problem,
            None,
            f"OR-Tools CP-SAT status {status_name}",
        )
    assignment = [int(solver.Value(var)) for var in colors]
    result = output(
        "optimal" if status_code == cp_model.OPTIMAL else "feasible",
        "ortools:cp-sat-graph-coloring",
        problem,
        assignment,
        f"OR-Tools CP-SAT status {status_name}",
    )
    result["objectiveBound"] = solver.BestObjectiveBound() + 1.0
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", choices=["auto", "ortools", "fallback"], default="auto")
    args = parser.parse_args()

    try:
        problem = normalize(json.load(sys.stdin))
        exact = exact_graph_coloring(problem)
        if args.solver == "fallback":
            print(json.dumps(exact))
            return 0 if exact["status"] in ("optimal", "feasible", "infeasible", "unsupported") else 1

        ortools = ortools_graph_coloring(problem)
        if args.solver == "ortools":
            print(json.dumps(ortools))
            return 0 if ortools["status"] in ("optimal", "feasible", "infeasible", "unavailable", "unsupported") else 1

        result = dict(exact)
        result["solver"] = (
            "ortools:cp-sat-graph-coloring+python:exact-graph-coloring"
            if ortools.get("status") != "unavailable"
            else "python:exact-graph-coloring"
        )
        result["ortoolsStatus"] = ortools.get("status")
        result["ortoolsColorIndices"] = ortools.get("colorIndices", [])
        result["ortoolsColorNames"] = ortools.get("colorNames", [])
        result["ortoolsUsedColorCount"] = ortools.get("usedColorCount")
        result["ortoolsObjective"] = ortools.get("objective")
        result["ortoolsMessage"] = ortools.get("message")
        result["ortoolsObjectiveBound"] = ortools.get("objectiveBound")
        print(json.dumps(result))
        return 0 if result["status"] in ("optimal", "feasible", "infeasible", "unsupported") else 1
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "solver": "python:graph-coloring-reference",
                    "colorIndices": [],
                    "colorNames": [],
                    "usedColorCount": None,
                    "objective": None,
                    "message": str(exc),
                }
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
