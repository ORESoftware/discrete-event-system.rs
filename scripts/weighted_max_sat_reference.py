#!/usr/bin/env python3
"""Reference bridge for small weighted partial Max-SAT instances.

The deterministic oracle enumerates all assignments for validation-scale
models. When OR-Tools is installed, the same weighted Boolean model is also
sent to CP-SAT with hard clauses enforced and soft clauses contributing to a
maximized weighted objective.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from typing import Optional


MAX_EXACT_VARS = 26
OBJECTIVE_SCALE = 1_000_000


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


def exact_weighted_max_sat(problem: dict) -> dict:
    n = problem["numVars"]
    if n > MAX_EXACT_VARS:
        return output(
            "unsupported",
            "python:exact-weighted-max-sat",
            None,
            None,
            f"exact weighted Max-SAT only practical for <= {MAX_EXACT_VARS} variables, got {n}",
        )
    best_assignment = None
    best_eval = None
    for mask in range(1 << n):
        assignment = [bool((mask >> var) & 1) for var in range(n)]
        evaluation = evaluate(problem, assignment)
        if evaluation["violatedHardClauseIds"]:
            continue
        if best_eval is None or evaluation["satisfiedSoftWeight"] > best_eval["satisfiedSoftWeight"] + 1e-9:
            best_assignment = assignment
            best_eval = evaluation
    if best_assignment is None or best_eval is None:
        return output(
            "infeasible",
            "python:exact-weighted-max-sat",
            None,
            None,
            "no assignment satisfies all hard clauses",
        )
    return output(
        "optimal",
        "python:exact-weighted-max-sat",
        best_assignment,
        best_eval,
        "exact weighted Max-SAT enumeration",
    )


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
    parser.add_argument("--solver", choices=["auto", "ortools", "fallback"], default="auto")
    args = parser.parse_args()

    try:
        problem = normalize(json.load(sys.stdin))
        exact = exact_weighted_max_sat(problem)
        if args.solver == "fallback":
            print(json.dumps(exact))
            return 0 if exact["status"] in ("optimal", "feasible", "infeasible", "unsupported") else 1

        ortools = ortools_weighted_max_sat(problem)
        if args.solver == "ortools":
            print(json.dumps(ortools))
            return 0 if ortools["status"] in ("optimal", "feasible", "infeasible", "unavailable", "unsupported") else 1

        result = dict(exact)
        result["solver"] = (
            "ortools:cp-sat-weighted-max-sat+python:exact-weighted-max-sat"
            if ortools.get("status") != "unavailable"
            else "python:exact-weighted-max-sat"
        )
        result["ortoolsStatus"] = ortools.get("status")
        result["ortoolsAssignment"] = ortools.get("assignment", [])
        result["ortoolsObjective"] = ortools.get("objective")
        result["ortoolsSatisfiedSoftWeight"] = ortools.get("satisfiedSoftWeight")
        result["ortoolsUnsatisfiedSoftWeight"] = ortools.get("unsatisfiedSoftWeight")
        result["ortoolsSatisfiedClauseIds"] = ortools.get("satisfiedClauseIds", [])
        result["ortoolsViolatedHardClauseIds"] = ortools.get("violatedHardClauseIds", [])
        result["ortoolsMessage"] = ortools.get("message")
        result["ortoolsObjectiveBound"] = ortools.get("objectiveBound")
        print(json.dumps(result))
        return 0 if result["status"] in ("optimal", "feasible", "infeasible", "unsupported") else 1
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
