#!/usr/bin/env python3
"""Reference bridge for small weighted partial Max-SAT instances.

The deterministic oracle enumerates all assignments for validation-scale
models. When OR-Tools is installed, the same weighted Boolean model is also
sent to CP-SAT with hard clauses enforced and soft clauses contributing to a
maximized weighted objective.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
from typing import Optional


OBJECTIVE_SCALE = 1_000_000
RUST_REFERENCE_SOLVERS = ("auto", "fallback", "rust-enumeration", "rust-exact")


def exec_rust_reference(solver: str) -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    binary_name = "weighted_max_sat_reference"
    explicit = os.environ.get("WEIGHTED_MAX_SAT_REFERENCE_RUST_BIN")
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
        os.path.join(repo_root, "src", "bin", "weighted_max_sat_reference.rs"),
        os.path.join(repo_root, "src", "des", "general", "external_weighted_max_sat_reference.rs"),
        os.path.join(repo_root, "src", "des", "general", "weighted_max_sat.rs"),
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
    value = os.environ.get("WEIGHTED_MAX_SAT_REFERENCE_EXTERNAL_FALLBACK", "")
    return value.strip().lower() in ("1", "true", "yes", "on", "rust")


def normalize(raw: dict) -> dict:
    num_vars = int(raw.get("numVars") or raw.get("num_vars") or 0)
    if num_vars <= 0:
        raise ValueError("numVars must be positive")
    clauses = []
    ids = set()
    for clause_index, raw_clause in enumerate(raw.get("clauses") or []):
        clause_id = str(raw_clause.get("id") or f"C{clause_index + 1}")
        if not clause_id.strip():
            raise ValueError(f"clauses[{clause_index}].id must be non-empty")
        if clause_id in ids:
            raise ValueError(f"duplicate clause id {clause_id!r}")
        ids.add(clause_id)
        literals = [int(value) for value in (raw_clause.get("literals") or [])]
        if not literals:
            raise ValueError(f"clauses[{clause_index}].literals must be non-empty")
        for literal in literals:
            variable = abs(literal)
            if literal == 0 or variable < 1 or variable > num_vars:
                raise ValueError(f"clauses[{clause_index}] literal {literal} outside [1, numVars]")
        weight = float(raw_clause.get("weight", 0.0))
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError(f"clauses[{clause_index}].weight must be finite and non-negative")
        clauses.append(
            {
                "id": clause_id,
                "literals": literals,
                "weight": weight,
                "hard": bool(raw_clause.get("hard", False)),
            }
        )
    if not clauses:
        raise ValueError("clauses must be non-empty")
    return {"numVars": num_vars, "clauses": clauses}


def literal_satisfied(literal: int, assignment: list[bool]) -> bool:
    value = assignment[abs(literal) - 1]
    return value if literal > 0 else not value


def clause_satisfied(clause: dict, assignment: list[bool]) -> bool:
    return any(literal_satisfied(literal, assignment) for literal in clause["literals"])


def evaluate(problem: dict, assignment: list[bool]) -> dict:
    satisfied_soft_weight = 0.0
    unsatisfied_soft_weight = 0.0
    satisfied_clause_ids = []
    violated_hard_clause_ids = []
    for clause in problem["clauses"]:
        if clause_satisfied(clause, assignment):
            satisfied_clause_ids.append(clause["id"])
            if not clause["hard"]:
                satisfied_soft_weight += clause["weight"]
        elif clause["hard"]:
            violated_hard_clause_ids.append(clause["id"])
        else:
            unsatisfied_soft_weight += clause["weight"]
    return {
        "satisfiedSoftWeight": satisfied_soft_weight,
        "unsatisfiedSoftWeight": unsatisfied_soft_weight,
        "satisfiedClauseIds": satisfied_clause_ids,
        "violatedHardClauseIds": violated_hard_clause_ids,
    }


def output(
    status: str,
    solver: str,
    assignment: Optional[list[bool]] = None,
    evaluation: Optional[dict] = None,
    message: str = "",
) -> dict:
    return {
        "status": status,
        "solver": solver,
        "assignment": [] if assignment is None else assignment,
        "objective": None if evaluation is None else evaluation["satisfiedSoftWeight"],
        "satisfiedSoftWeight": None if evaluation is None else evaluation["satisfiedSoftWeight"],
        "unsatisfiedSoftWeight": None if evaluation is None else evaluation["unsatisfiedSoftWeight"],
        "satisfiedClauseIds": [] if evaluation is None else evaluation["satisfiedClauseIds"],
        "violatedHardClauseIds": [] if evaluation is None else evaluation["violatedHardClauseIds"],
        "message": message,
    }


def ortools_weighted_max_sat(problem: dict) -> dict:
    try:
        from ortools.sat.python import cp_model  # type: ignore
    except Exception as exc:
        return output("unavailable", "ortools:cp-sat-weighted-max-sat", None, None, str(exc))

    model = cp_model.CpModel()
    variables = [model.NewBoolVar(f"x{index + 1}") for index in range(problem["numVars"])]
    objective_terms = []
    for clause_index, clause in enumerate(problem["clauses"]):
        literals = [
            variables[abs(literal) - 1] if literal > 0 else variables[abs(literal) - 1].Not()
            for literal in clause["literals"]
        ]
        if clause["hard"]:
            model.AddBoolOr(literals)
        else:
            sat = model.NewBoolVar(f"soft_{clause_index}_{clause['id']}")
            model.AddBoolOr(literals + [sat.Not()])
            scaled = int(round(clause["weight"] * OBJECTIVE_SCALE))
            if scaled > 0:
                objective_terms.append(scaled * sat)
    model.Maximize(sum(objective_terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10.0
    solver.parameters.num_search_workers = 1
    status_code = solver.Solve(model)
    status_name = solver.StatusName(status_code).lower()
    if status_code not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return output(
            "infeasible" if status_name == "infeasible" else status_name,
            "ortools:cp-sat-weighted-max-sat",
            None,
            None,
            f"OR-Tools CP-SAT status {status_name}",
        )
    assignment = [bool(solver.Value(var)) for var in variables]
    evaluation = evaluate(problem, assignment)
    result = output(
        "optimal" if status_code == cp_model.OPTIMAL else "feasible",
        "ortools:cp-sat-weighted-max-sat",
        assignment,
        evaluation,
        f"OR-Tools CP-SAT status {status_name}",
    )
    result["objectiveBound"] = solver.BestObjectiveBound() / OBJECTIVE_SCALE
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--solver",
        choices=["auto", "ortools", "fallback", "rust-enumeration", "rust-exact"],
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
        ortools = ortools_weighted_max_sat(problem)
        print(json.dumps(ortools))
        return 0 if ortools["status"] in ("optimal", "feasible", "infeasible", "unavailable", "unsupported") else 1
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "solver": "python:weighted-max-sat-reference",
                    "assignment": [],
                    "objective": None,
                    "satisfiedSoftWeight": None,
                    "unsatisfiedSoftWeight": None,
                    "satisfiedClauseIds": [],
                    "violatedHardClauseIds": [],
                    "message": str(exc),
                }
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
