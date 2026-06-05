#!/usr/bin/env python3
"""Reference bridge for small uncapacitated facility-location instances.

The deterministic open-facility subset oracle lives in Rust. This Python bridge
remains as thin adapter glue for explicit OR-Tools CP-SAT checks.
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
RUST_REFERENCE_SOLVERS = ("auto", "fallback", "rust-exact")


def exec_rust_reference(solver: str) -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    binary_name = "facility_location_reference"
    explicit = os.environ.get("FACILITY_LOCATION_REFERENCE_RUST_BIN")
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
        os.path.join(repo_root, "src", "bin", "facility_location_reference.rs"),
        os.path.join(repo_root, "src", "des", "general", "external_facility_location_reference.rs"),
        os.path.join(repo_root, "src", "des", "general", "facility_location.rs"),
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
    value = os.environ.get("FACILITY_LOCATION_REFERENCE_EXTERNAL_FALLBACK", "")
    return value.strip().lower() in ("1", "true", "yes", "on", "rust")


def normalize(raw: dict) -> dict:
    facilities = [str(value) for value in (raw.get("facilities") or raw.get("facilityIds") or [])]
    customers = [str(value) for value in (raw.get("customers") or raw.get("customerIds") or [])]
    if not facilities:
        raise ValueError("facilities must be non-empty")
    if not customers:
        raise ValueError("customers must be non-empty")
    if any(not value.strip() for value in facilities):
        raise ValueError("facilities must be non-empty strings")
    if any(not value.strip() for value in customers):
        raise ValueError("customers must be non-empty strings")
    if len(set(facilities)) != len(facilities):
        raise ValueError("facilities must be unique")
    if len(set(customers)) != len(customers):
        raise ValueError("customers must be unique")

    fixed_costs = [float(value) for value in (raw.get("fixedCosts") or [])]
    service_costs = [[float(value) for value in row] for row in (raw.get("serviceCosts") or [])]
    if len(fixed_costs) != len(facilities):
        raise ValueError("fixedCosts length must equal facilities length")
    if len(service_costs) != len(facilities):
        raise ValueError("serviceCosts row count must equal facilities length")
    for index, cost in enumerate(fixed_costs):
        if not math.isfinite(cost) or cost < 0.0:
            raise ValueError(f"fixedCosts[{index}] must be finite and >= 0")
    for facility_index, row in enumerate(service_costs):
        if len(row) != len(customers):
            raise ValueError(f"serviceCosts[{facility_index}] length must equal customers length")
        for customer_index, cost in enumerate(row):
            if not math.isfinite(cost) or cost < 0.0:
                raise ValueError(
                    f"serviceCosts[{facility_index}][{customer_index}] must be finite and >= 0"
                )
    return {
        "facilities": facilities,
        "customers": customers,
        "fixed_costs": fixed_costs,
        "service_costs": service_costs,
    }


def result(
    status: str,
    solver: str,
    problem: dict,
    open_indices: Optional[list[int]] = None,
    assignments: Optional[list[dict]] = None,
    objective: Optional[float] = None,
    message: str = "",
) -> dict:
    if open_indices is None:
        open_ids: list[str] = []
        assignments = []
        objective = None
    else:
        open_indices = sorted(set(int(index) for index in open_indices))
        open_ids = [problem["facilities"][index] for index in open_indices]
        assignments = [] if assignments is None else assignments
    return {
        "status": status,
        "solver": solver,
        "openFacilityIndices": [] if open_indices is None else open_indices,
        "openFacilities": open_ids,
        "assignments": assignments,
        "objective": objective,
        "message": message,
    }


def choose_scale(values: list[float]) -> Optional[int]:
    for scale in SCALES:
        if all(abs(round(value * scale) - value * scale) <= 1e-6 for value in values):
            return scale
    return None


def ortools_facility_location(problem: dict) -> dict:
    try:
        from ortools.sat.python import cp_model  # type: ignore
    except Exception as exc:
        return result("unavailable", "ortools:cp-sat-facility-location", problem, None, message=str(exc))

    costs = list(problem["fixed_costs"])
    for row in problem["service_costs"]:
        costs.extend(row)
    scale = choose_scale(costs)
    if scale is None:
        return result(
            "unsupported",
            "ortools:cp-sat-facility-location",
            problem,
            None,
            message="OR-Tools CP-SAT bridge requires integer-scalable costs",
        )

    facility_count = len(problem["facilities"])
    customer_count = len(problem["customers"])
    model = cp_model.CpModel()
    y = [model.NewBoolVar(f"open_f{facility}") for facility in range(facility_count)]
    x = [
        [
            model.NewBoolVar(f"assign_f{facility}_c{customer}")
            for customer in range(customer_count)
        ]
        for facility in range(facility_count)
    ]
    for customer in range(customer_count):
        model.Add(sum(x[facility][customer] for facility in range(facility_count)) == 1)
    for facility in range(facility_count):
        for customer in range(customer_count):
            model.Add(x[facility][customer] <= y[facility])
    model.Minimize(
        sum(int(round(problem["fixed_costs"][facility] * scale)) * y[facility] for facility in range(facility_count))
        + sum(
            int(round(problem["service_costs"][facility][customer] * scale)) * x[facility][customer]
            for facility in range(facility_count)
            for customer in range(customer_count)
        )
    )

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10.0
    solver.parameters.num_search_workers = 1
    status_code = solver.Solve(model)
    status_name = solver.StatusName(status_code).lower()
    if status_code not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return result(
            "infeasible" if status_name == "infeasible" else status_name,
            "ortools:cp-sat-facility-location",
            problem,
            None,
            message=f"OR-Tools CP-SAT status {status_name}",
        )

    open_indices = [facility for facility, var in enumerate(y) if solver.BooleanValue(var)]
    assignments = []
    objective = sum(problem["fixed_costs"][facility] for facility in open_indices)
    for customer in range(customer_count):
        assigned = [
            facility
            for facility in range(facility_count)
            if solver.BooleanValue(x[facility][customer])
        ]
        facility = assigned[0]
        cost = problem["service_costs"][facility][customer]
        objective += cost
        assignments.append(
            {
                "customerIndex": customer,
                "customer": problem["customers"][customer],
                "facilityIndex": facility,
                "facility": problem["facilities"][facility],
                "cost": cost,
            }
        )
    output = result(
        "optimal" if status_code == cp_model.OPTIMAL else "feasible",
        "ortools:cp-sat-facility-location",
        problem,
        open_indices,
        assignments,
        float(objective),
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
        output = ortools_facility_location(problem)
        print(json.dumps(output))
        return 0 if output["status"] in ("optimal", "feasible", "infeasible", "unavailable", "unsupported") else 1
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "solver": "python:facility-location-reference",
                    "openFacilityIndices": [],
                    "openFacilities": [],
                    "assignments": [],
                    "objective": None,
                    "message": str(exc),
                }
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
