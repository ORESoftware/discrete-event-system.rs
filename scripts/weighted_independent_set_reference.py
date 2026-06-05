#!/usr/bin/env python3
"""Reference bridge for small weighted independent-set instances.

The deterministic oracle uses branch-and-bound with a remaining-weight upper
bound. When OR-Tools is installed and weights can be safely integer-scaled, the
same conflict graph is solved with CP-SAT.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
from typing import Optional


SCALES = (1, 10, 100, 1000, 10000, 100000, 1000000)
RUST_REFERENCE_SOLVERS = ("auto", "fallback", "rust-branch-and-bound", "rust-exact")


def exec_rust_reference(solver: str) -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    binary_name = "weighted_independent_set_reference"
    explicit = os.environ.get("WEIGHTED_INDEPENDENT_SET_REFERENCE_RUST_BIN")
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
        os.path.join(repo_root, "src", "bin", "weighted_independent_set_reference.rs"),
        os.path.join(
            repo_root,
            "src",
            "des",
            "general",
            "external_weighted_independent_set_reference.rs",
        ),
        os.path.join(repo_root, "src", "des", "general", "weighted_independent_set.rs"),
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
    value = os.environ.get("WEIGHTED_INDEPENDENT_SET_REFERENCE_EXTERNAL_FALLBACK", "")
    return value.strip().lower() in ("1", "true", "yes", "on", "rust")


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
    parser.add_argument(
        "--solver",
        choices=["auto", "ortools", "fallback", "rust-branch-and-bound", "rust-exact"],
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
        ortools = ortools_independent_set(problem)
        print(json.dumps(ortools))
        return 0 if ortools["status"] in ("optimal", "feasible", "unavailable", "unsupported") else 1
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
