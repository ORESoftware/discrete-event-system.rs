#!/usr/bin/env python3
"""Reference bridge for small uncapacitated facility-location instances.

The deterministic oracle enumerates open-facility subsets. When OR-Tools is
installed and costs are integer-scalable, the same model is also sent to CP-SAT
with Boolean open and assignment variables.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from typing import Optional


EPS = 1e-9
MAX_EXACT_FACILITIES = 24
SCALES = (1, 10, 100, 1000, 10000, 100000, 1000000)


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


def assignments_for(problem: dict, open_indices: list[int]) -> tuple[float, list[dict]]:
    if not open_indices:
        raise ValueError("at least one facility must be open")
    total = sum(problem["fixed_costs"][index] for index in open_indices)
    assignments = []
    for customer_index, customer in enumerate(problem["customers"]):
        best_facility = min(
            open_indices,
            key=lambda facility_index: (
                problem["service_costs"][facility_index][customer_index],
                facility_index,
            ),
        )
        cost = problem["service_costs"][best_facility][customer_index]
        total += cost
        assignments.append(
            {
                "customerIndex": customer_index,
                "customer": customer,
                "facilityIndex": best_facility,
                "facility": problem["facilities"][best_facility],
                "cost": cost,
            }
        )
    return float(total), assignments


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
        if assignments is None or objective is None:
            objective, assignments = assignments_for(problem, open_indices)
    return {
        "status": status,
        "solver": solver,
        "openFacilityIndices": [] if open_indices is None else open_indices,
        "openFacilities": open_ids,
        "assignments": assignments,
        "objective": objective,
        "message": message,
    }


def exact_facility_location(problem: dict) -> dict:
    facility_count = len(problem["facilities"])
    if facility_count > MAX_EXACT_FACILITIES:
        return result(
            "unsupported",
            "python:exact-facility-location",
            problem,
            None,
            message=(
                "exact facility-location enumeration only practical for "
                f"<= {MAX_EXACT_FACILITIES} facilities, got {facility_count}"
            ),
        )

    best_open: list[int] | None = None
    best_assignments: list[dict] | None = None
    best_objective = math.inf
    for mask in range(1, 1 << facility_count):
        open_indices = [index for index in range(facility_count) if mask & (1 << index)]
        objective, assignments = assignments_for(problem, open_indices)
        if objective < best_objective - EPS or (
            abs(objective - best_objective) <= EPS
            and (best_open is None or open_indices < best_open)
        ):
            best_open = open_indices
            best_assignments = assignments
            best_objective = objective

    if best_open is None:
        return result(
            "infeasible",
            "python:exact-facility-location",
            problem,
            None,
            message="no feasible facility subset",
        )
    return result(
        "optimal",
        "python:exact-facility-location",
        problem,
        best_open,
        best_assignments,
        best_objective,
        "exact open-facility subset enumeration",
    )


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
    parser.add_argument("--solver", choices=["auto", "ortools", "fallback"], default="auto")
    args = parser.parse_args()

    try:
        problem = normalize(json.load(sys.stdin))
        exact = exact_facility_location(problem)
        if args.solver == "fallback":
            print(json.dumps(exact))
            return 0 if exact["status"] in ("optimal", "feasible", "infeasible", "unsupported") else 1

        ortools = ortools_facility_location(problem)
        if args.solver == "ortools":
            print(json.dumps(ortools))
            return 0 if ortools["status"] in ("optimal", "feasible", "infeasible", "unavailable", "unsupported") else 1

        output = dict(exact)
        output["solver"] = (
            "ortools:cp-sat-facility-location+python:exact-facility-location"
            if ortools.get("status") != "unavailable"
            else "python:exact-facility-location"
        )
        output["ortoolsStatus"] = ortools.get("status")
        output["ortoolsOpenFacilityIndices"] = ortools.get("openFacilityIndices", [])
        output["ortoolsOpenFacilities"] = ortools.get("openFacilities", [])
        output["ortoolsAssignments"] = ortools.get("assignments", [])
        output["ortoolsObjective"] = ortools.get("objective")
        output["ortoolsMessage"] = ortools.get("message")
        output["ortoolsObjectiveBound"] = ortools.get("objectiveBound")
        print(json.dumps(output))
        return 0 if output["status"] in ("optimal", "feasible", "infeasible", "unsupported") else 1
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
