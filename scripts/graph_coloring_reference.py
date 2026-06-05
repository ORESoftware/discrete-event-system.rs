#!/usr/bin/env python3
"""Reference bridge for small graph-coloring instances.

The deterministic DSATUR-style chromatic-number search lives in Rust. This
Python bridge remains as thin adapter glue for explicit OR-Tools CP-SAT checks.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from typing import Optional


RUST_REFERENCE_SOLVERS = ("auto", "fallback", "rust-dsatur", "rust-exact")


def exec_rust_reference(solver: str) -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    binary_name = "graph_coloring_reference"
    explicit = os.environ.get("GRAPH_COLORING_REFERENCE_RUST_BIN")
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
        os.path.join(repo_root, "src", "bin", "graph_coloring_reference.rs"),
        os.path.join(repo_root, "src", "des", "general", "external_graph_coloring_reference.rs"),
        os.path.join(repo_root, "src", "des", "general", "graph_coloring.rs"),
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
    value = os.environ.get("GRAPH_COLORING_REFERENCE_EXTERNAL_FALLBACK", "")
    return value.strip().lower() in ("1", "true", "yes", "on", "rust")


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
    parser.add_argument(
        "--solver",
        choices=["auto", "ortools", "fallback", "rust-dsatur", "rust-exact"],
        default="auto",
    )
    args = parser.parse_args()
    if args.solver in RUST_REFERENCE_SOLVERS:
        exec_rust_reference(args.solver)
    if (
        external_rust_fallback_enabled()
        and args.solver == "ortools"
        and not package_available("ortools")
    ):
        exec_rust_reference("rust-exact")

    try:
        problem = normalize(json.load(sys.stdin))
        output = ortools_graph_coloring(problem)
        print(json.dumps(output))
        return 0 if output["status"] in ("optimal", "feasible", "infeasible", "unavailable", "unsupported") else 1
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
