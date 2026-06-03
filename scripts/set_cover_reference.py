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
import sys
from typing import Optional


EPS = 1e-9
MAX_EXACT_SETS = 32
MAX_EXACT_ELEMENTS = 128
SCALES = (1, 10, 100, 1000, 10000, 100000, 1000000)


def popcount(mask: int) -> int:
    return bin(mask).count("1")


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


def masks(problem: dict) -> tuple[list[int], int]:
    element_index = {element: index for index, element in enumerate(problem["universe"])}
    set_masks = []
    for set_ in problem["sets"]:
        mask = 0
        for element in set_["elements"]:
            mask |= 1 << element_index[element]
        set_masks.append(mask)
    return set_masks, (1 << len(problem["universe"])) - 1


def greedy_set_cover(problem: dict) -> dict:
    set_masks, full_mask = masks(problem)
    covered = 0
    selected: list[int] = []
    while covered != full_mask:
        best = None
        best_ratio = math.inf
        best_new = 0
        for index, set_ in enumerate(problem["sets"]):
            if index in selected:
                continue
            new_bits = popcount(set_masks[index] & ~covered)
            if new_bits == 0:
                continue
            ratio = set_["cost"] / new_bits
            if ratio < best_ratio - EPS or (
                abs(ratio - best_ratio) <= EPS
                and (new_bits > best_new or (new_bits == best_new and (best is None or index < best)))
            ):
                best = index
                best_ratio = ratio
                best_new = new_bits
        if best is None:
            return result(
                "infeasible",
                "python:greedy-set-cover",
                problem,
                None,
                "greedy could not cover remaining elements",
            )
        selected.append(best)
        covered |= set_masks[best]
    return result(
        "feasible",
        "python:greedy-set-cover",
        problem,
        selected,
        "greedy weighted set cover",
    )


def exact_set_cover(problem: dict) -> dict:
    if len(problem["sets"]) > MAX_EXACT_SETS or len(problem["universe"]) > MAX_EXACT_ELEMENTS:
        return result(
            "unsupported",
            "python:exact-set-cover",
            problem,
            None,
            f"exact set-cover only practical for <= {MAX_EXACT_SETS} sets and <= {MAX_EXACT_ELEMENTS} elements, got {len(problem['sets'])} sets and {len(problem['universe'])} elements",
        )

    greedy = greedy_set_cover(problem)
    if greedy["status"] == "infeasible":
        return dict(greedy, solver="python:exact-set-cover")
    best_indices = list(greedy["selectedSetIndices"])
    best_cost = float(greedy["objective"])
    set_masks, full_mask = masks(problem)
    covering_sets: list[list[int]] = [[] for _ in problem["universe"]]
    for set_index, mask in enumerate(set_masks):
        for element_index in range(len(problem["universe"])):
            if mask & (1 << element_index):
                covering_sets[element_index].append(set_index)
    if any(not candidates for candidates in covering_sets):
        return result(
            "infeasible",
            "python:exact-set-cover",
            problem,
            None,
            "at least one universe element is uncovered by all sets",
        )

    current: list[int] = []

    def search(covered: int, current_cost: float) -> None:
        nonlocal best_indices, best_cost
        if current_cost >= best_cost - EPS:
            return
        if covered == full_mask:
            candidate = sorted(current)
            incumbent = sorted(best_indices)
            if current_cost < best_cost - EPS or (
                abs(current_cost - best_cost) <= EPS and candidate < incumbent
            ):
                best_indices = candidate
                best_cost = current_cost
            return

        uncovered = full_mask & ~covered
        chosen_candidates: Optional[list[int]] = None
        for element_index, candidates in enumerate(covering_sets):
            if not (uncovered & (1 << element_index)):
                continue
            available = [
                set_index
                for set_index in candidates
                if set_index not in current and (set_masks[set_index] & ~covered)
            ]
            if chosen_candidates is None or len(available) < len(chosen_candidates):
                chosen_candidates = available
        if not chosen_candidates:
            return
        chosen_candidates.sort(key=lambda index: (problem["sets"][index]["cost"], index))
        for set_index in chosen_candidates:
            current.append(set_index)
            search(covered | set_masks[set_index], current_cost + problem["sets"][set_index]["cost"])
            current.pop()

    search(0, 0.0)
    return result(
        "optimal",
        "python:exact-set-cover",
        problem,
        best_indices,
        "exact branch-and-bound",
    )


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
    parser.add_argument("--solver", choices=["auto", "ortools", "fallback"], default="auto")
    args = parser.parse_args()

    try:
        problem = normalize(json.load(sys.stdin))
        exact = exact_set_cover(problem)
        if args.solver == "fallback":
            print(json.dumps(exact))
            return 0 if exact["status"] in ("optimal", "feasible", "infeasible", "unsupported") else 1

        ortools = ortools_set_cover(problem)
        if args.solver == "ortools":
            print(json.dumps(ortools))
            return 0 if ortools["status"] in ("optimal", "feasible", "infeasible", "unavailable", "unsupported") else 1

        output = dict(exact)
        output["solver"] = (
            "ortools:cp-sat-set-cover+python:exact-set-cover"
            if ortools.get("status") != "unavailable"
            else "python:exact-set-cover"
        )
        output["ortoolsStatus"] = ortools.get("status")
        output["ortoolsSelectedSetIndices"] = ortools.get("selectedSetIndices", [])
        output["ortoolsSelectedSets"] = ortools.get("selectedSets", [])
        output["ortoolsObjective"] = ortools.get("objective")
        output["ortoolsCoveredElements"] = ortools.get("coveredElements", [])
        output["ortoolsMessage"] = ortools.get("message")
        output["ortoolsObjectiveBound"] = ortools.get("objectiveBound")
        print(json.dumps(output))
        return 0 if output["status"] in ("optimal", "feasible", "infeasible", "unsupported") else 1
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
