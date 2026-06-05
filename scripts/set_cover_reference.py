#!/usr/bin/env python3
"""Reference bridge for small weighted set-cover instances.

The deterministic oracle is a branch-and-bound set-cover search. When OR-Tools
is installed and costs can be safely integer-scaled, the same model is also sent
to CP-SAT with one Boolean variable per candidate set.
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
    binary_name = "set_cover_reference"
    explicit = os.environ.get("SET_COVER_REFERENCE_RUST_BIN")
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
    universe = [str(element) for element in (raw.get("universe") or [])]
    if not universe:
        raise ValueError("universe must be non-empty")
    if any(not element.strip() for element in universe):
        raise ValueError("universe elements must be non-empty")
    if len(set(universe)) != len(universe):
        raise ValueError("universe elements must be unique")

    raw_sets = raw.get("sets") or []
    if not raw_sets:
        raise ValueError("sets must be non-empty")
    universe_set = set(universe)
    sets = []
    seen_ids = set()
    for index, raw_set in enumerate(raw_sets):
        set_id = str(raw_set.get("id", f"S{index + 1}"))
        if not set_id.strip():
            raise ValueError(f"sets[{index}].id must be non-empty")
        if set_id in seen_ids:
            raise ValueError(f"duplicate set id {set_id!r}")
        seen_ids.add(set_id)
        cost = float(raw_set.get("cost", 0.0))
        if not math.isfinite(cost) or cost < 0.0:
            raise ValueError(f"sets[{index}].cost must be finite and >= 0")
        elements = [str(element) for element in (raw_set.get("elements") or [])]
        if not elements:
            raise ValueError(f"sets[{index}].elements must be non-empty")
        if len(set(elements)) != len(elements):
            raise ValueError(f"sets[{index}].elements must be unique")
        missing = [element for element in elements if element not in universe_set]
        if missing:
            raise ValueError(f"sets[{index}].elements not in universe: {missing}")
        sets.append(
            {
                "id": set_id,
                "cost": cost,
                "elements": elements,
                "index": index,
            }
        )
    return {"universe": universe, "sets": sets}


def result(
    status: str,
    solver: str,
    problem: dict,
    selected_indices: Optional[list[int]] = None,
    message: str = "",
) -> dict:
    if selected_indices is None:
        selected_ids: list[str] = []
        objective = None
        covered_elements: list[str] = []
    else:
        selected_indices = sorted(set(int(index) for index in selected_indices))
        selected_ids = [problem["sets"][index]["id"] for index in selected_indices]
        objective = float(sum(problem["sets"][index]["cost"] for index in selected_indices))
        covered = {
            element
            for index in selected_indices
            for element in problem["sets"][index]["elements"]
        }
        covered_elements = [element for element in problem["universe"] if element in covered]
    return {
        "status": status,
        "solver": solver,
        "selectedSetIndices": [] if selected_indices is None else selected_indices,
        "selectedSets": selected_ids,
        "objective": objective,
        "coveredElements": covered_elements,
        "message": message,
    }


def choose_scale(values: list[float]) -> Optional[int]:
    for scale in SCALES:
        if all(abs(round(value * scale) - value * scale) <= 1e-6 for value in values):
            return scale
    return None


def ortools_set_cover(problem: dict) -> dict:
    try:
        from ortools.sat.python import cp_model  # type: ignore
    except Exception as exc:
        return result("unavailable", "ortools:cp-sat-set-cover", problem, None, str(exc))

    scale = choose_scale([set_["cost"] for set_ in problem["sets"]])
    if scale is None:
        return result(
            "unsupported",
            "ortools:cp-sat-set-cover",
            problem,
            None,
            "OR-Tools CP-SAT bridge requires integer-scalable costs",
        )

    model = cp_model.CpModel()
    x = [model.NewBoolVar(f"x_s{index}") for index in range(len(problem["sets"]))]
    for element in problem["universe"]:
        covering = [
            x[index]
            for index, set_ in enumerate(problem["sets"])
            if element in set_["elements"]
        ]
        if not covering:
            return result(
                "infeasible",
                "ortools:cp-sat-set-cover",
                problem,
                None,
                f"element {element!r} is uncovered by all sets",
            )
        model.Add(sum(covering) >= 1)
    model.Minimize(
        sum(int(round(set_["cost"] * scale)) * x[index] for index, set_ in enumerate(problem["sets"]))
    )

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10.0
    solver.parameters.num_search_workers = 1
    status_code = solver.Solve(model)
    status_name = solver.StatusName(status_code).lower()
    if status_code not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return result(
            "infeasible" if status_name == "infeasible" else status_name,
            "ortools:cp-sat-set-cover",
            problem,
            None,
            f"OR-Tools CP-SAT status {status_name}",
        )

    selected = [index for index, var in enumerate(x) if solver.BooleanValue(var)]
    output = result(
        "optimal" if status_code == cp_model.OPTIMAL else "feasible",
        "ortools:cp-sat-set-cover",
        problem,
        selected,
        f"OR-Tools CP-SAT status {status_name}",
    )
    output["objectiveBound"] = solver.BestObjectiveBound() / scale
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--solver",
        choices=["auto", "ortools", "fallback", "rust-exact"],
        default="auto",
    )
    args = parser.parse_args()
    if args.solver in ("auto", "fallback", "rust-exact"):
        exec_rust_reference(args.solver)

    try:
        problem = normalize(json.load(sys.stdin))
        ortools = ortools_set_cover(problem)
        print(json.dumps(ortools))
        return 0 if ortools["status"] in ("optimal", "feasible", "infeasible", "unavailable", "unsupported") else 1
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "solver": "python:set-cover-reference",
                    "selectedSetIndices": [],
                    "selectedSets": [],
                    "objective": None,
                    "coveredElements": [],
                    "message": str(exc),
                }
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
