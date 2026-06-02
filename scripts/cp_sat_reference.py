#!/usr/bin/env python3
"""Reference bridge for small CP-SAT-style finite-domain models.

The bridge prefers OR-Tools CP-SAT when installed and falls back to exact
enumeration with the same JSON model contract.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import sys
from typing import Dict, List, Optional, Sequence


def objective_value(model: dict, assignment: Sequence[int]) -> Optional[int]:
    obj = model.get("objective")
    if not obj:
        return None
    return sum(int(t["coeff"]) * int(assignment[int(t["var"])]) for t in obj.get("terms", []))


def trunc_div(num: int, den: int) -> int:
    quotient = abs(num) // abs(den)
    return quotient if (num >= 0) == (den >= 0) else -quotient


def linear_bounds(model: dict, partial: Sequence[Optional[int]], terms: Sequence[dict]) -> tuple[int, int]:
    lo = hi = 0
    for term in terms:
        var = int(term["var"])
        coeff = int(term["coeff"])
        if partial[var] is not None:
            lo += coeff * int(partial[var])
            hi += coeff * int(partial[var])
            continue
        dom = model["variables"][var]["domain"]
        dmin, dmax = min(dom), max(dom)
        if coeff >= 0:
            lo += coeff * dmin
            hi += coeff * dmax
        else:
            lo += coeff * dmax
            hi += coeff * dmin
    return lo, hi


def literal_truth(partial: Sequence[Optional[int]], lit: dict) -> Optional[bool]:
    value = partial[int(lit["var"])]
    if value is None:
        return None
    return (int(value) == 1) if bool(lit.get("positive", True)) else (int(value) == 0)


def optional_presence_truth(partial: Sequence[Optional[int]], item: dict) -> Optional[bool]:
    presence = item.get("presence")
    if presence is None:
        return True
    return literal_truth(partial, presence)


def variable_interval_span(
    partial: Sequence[Optional[int]],
    item: dict,
) -> tuple[Optional[tuple[int, int]], bool]:
    if optional_presence_truth(partial, item) is not True:
        return None, True
    start = partial[int(item["start"])]
    duration = partial[int(item["duration"])]
    if start is None or duration is None:
        return None, True
    computed_end = int(start) + int(duration)
    end = partial[int(item["end"])]
    if end is not None and int(end) != computed_end:
        return None, False
    return (int(start), computed_end), True


def variable_demand_interval_span(
    partial: Sequence[Optional[int]],
    item: dict,
) -> tuple[Optional[tuple[int, int, int]], bool]:
    span, ok = variable_interval_span(partial, item)
    if not ok or span is None:
        return None, ok
    demand = partial[int(item["demand"])]
    if demand is None:
        return None, True
    return (span[0], span[1], int(demand)), True


def variable_axis_span(
    partial: Sequence[Optional[int]],
    item: dict,
    start_key: str,
    size_key: str,
    end_key: str,
) -> tuple[Optional[tuple[int, int]], bool]:
    if optional_presence_truth(partial, item) is not True:
        return None, True
    start = partial[int(item[start_key])]
    size = partial[int(item[size_key])]
    if start is None or size is None:
        return None, True
    computed_end = int(start) + int(size)
    end = partial[int(item[end_key])]
    if end is not None and int(end) != computed_end:
        return None, False
    return (int(start), computed_end), True


def variable_rectangle_span(
    partial: Sequence[Optional[int]],
    item: dict,
) -> tuple[Optional[tuple[int, int, int, int]], bool]:
    x_span, x_ok = variable_axis_span(partial, item, "x_start", "x_size", "x_end")
    if not x_ok:
        return None, False
    y_span, y_ok = variable_axis_span(partial, item, "y_start", "y_size", "y_end")
    if not y_ok:
        return None, False
    if x_span is None or y_span is None:
        return None, True
    return (x_span[0], x_span[1], y_span[0], y_span[1]), True


def interval_triplet_consistent(
    partial: Sequence[Optional[int]],
    start_key: int,
    duration_key: int,
    end_key: int,
    active: Optional[bool],
) -> bool:
    if active is not True:
        return True
    start = partial[start_key]
    duration = partial[duration_key]
    end = partial[end_key]
    if start is not None and duration is not None and end is not None:
        return int(start) + int(duration) == int(end)
    return True


def assigned_equal_or_unknown(
    partial: Sequence[Optional[int]],
    lhs: int,
    rhs: int,
) -> bool:
    if partial[lhs] is None or partial[rhs] is None:
        return True
    return int(partial[lhs]) == int(partial[rhs])


def circuit_complete_ok(selected: Sequence[dict], nodes: Sequence[int]) -> bool:
    out = {}
    incoming = {}
    for node in nodes:
        outgoing = [arc for arc in selected if int(arc["tail"]) == node]
        inbound = [arc for arc in selected if int(arc["head"]) == node]
        if len(outgoing) != 1 or len(inbound) != 1:
            return False
        out[node] = int(outgoing[0]["head"])
        incoming[node] = int(inbound[0]["tail"])

    active_nodes = [node for node in nodes if out[node] != node]
    if not active_nodes:
        return True

    start = active_nodes[0]
    current = start
    seen = []
    while True:
        if current in seen:
            return current == start and len(seen) == len(active_nodes)
        if current not in active_nodes:
            return False
        seen.append(current)
        current = out[current]


def multiple_circuit_complete_ok(selected: Sequence[dict], nodes: Sequence[int]) -> bool:
    if 0 not in nodes:
        return False
    depot_out = sum(1 for arc in selected if int(arc["tail"]) == 0)
    depot_in = sum(1 for arc in selected if int(arc["head"]) == 0)
    if depot_out != depot_in:
        return False

    for node in nodes:
        if node == 0:
            continue
        outgoing = [arc for arc in selected if int(arc["tail"]) == node]
        inbound = [arc for arc in selected if int(arc["head"]) == node]
        if len(outgoing) != 1 or len(inbound) != 1:
            return False

    for start in nodes:
        if start == 0:
            continue
        first = next((arc for arc in selected if int(arc["tail"]) == start), None)
        if first is None:
            return False
        if int(first["head"]) == start:
            continue
        current = start
        seen = []
        while True:
            if current == 0:
                break
            if current in seen:
                return False
            seen.append(current)
            next_arc = next((arc for arc in selected if int(arc["tail"]) == current), None)
            if next_arc is None:
                return False
            current = int(next_arc["head"])
    return True


def reservoir_complete_ok(events: Sequence[tuple[int, int]], min_level: int, max_level: int) -> bool:
    if not (min_level <= 0 <= max_level):
        return False
    level = 0
    sorted_events = sorted(events)
    i = 0
    while i < len(sorted_events):
        time = sorted_events[i][0]
        while i < len(sorted_events) and sorted_events[i][0] == time:
            level += sorted_events[i][1]
            i += 1
        if level < min_level or level > max_level:
            return False
    return True


def enforcement_state(partial: Sequence[Optional[int]], literals: Sequence[dict]) -> Optional[bool]:
    unknown = False
    for lit in literals:
        truth = literal_truth(partial, lit)
        if truth is False:
            return False
        if truth is None:
            unknown = True
    return None if unknown else True


def linear_partial_ok(model: dict, partial: Sequence[Optional[int]], c: dict) -> bool:
    lo, hi = linear_bounds(model, partial, c["terms"])
    rhs = int(c["rhs"])
    sense = c["sense"]
    if sense == "le" and lo > rhs:
        return False
    if sense == "ge" and hi < rhs:
        return False
    if sense == "eq" and not (lo <= rhs <= hi):
        return False
    return True


def linear_domain_partial_ok(model: dict, partial: Sequence[Optional[int]], c: dict) -> bool:
    lo, hi = linear_bounds(model, partial, c["terms"])
    return any(int(interval["lb"]) <= hi and lo <= int(interval["ub"]) for interval in c["intervals"])


def product_range(bounds: Sequence[tuple[int, int]]) -> tuple[int, int]:
    lo = 1
    hi = 1
    for next_lo, next_hi in bounds:
        candidates = [
            lo * next_lo,
            lo * next_hi,
            hi * next_lo,
            hi * next_hi,
        ]
        lo = min(candidates)
        hi = max(candidates)
    return lo, hi


def solution_hint_values(model: dict) -> list[Optional[int]]:
    values: list[Optional[int]] = [None] * len(model["variables"])
    seen = set()
    for hint in model.get("solution_hint", []):
        var = int(hint["var"])
        value = int(hint["value"])
        if var < 0 or var >= len(values):
            raise ValueError(f"solution hint variable {var} out of range")
        if var in seen:
            raise ValueError(f"duplicate solution hint for variable {var}")
        seen.add(var)
        domain = [int(v) for v in model["variables"][var]["domain"]]
        if value not in domain:
            raise ValueError(f"solution hint value {value} is outside variable {var} domain")
        values[var] = value
    return values


def choose_search_var(
    model: dict,
    partial: Sequence[Optional[int]],
    hints: Sequence[Optional[int]],
    strategies: Sequence[dict],
) -> int:
    for i, value in enumerate(hints):
        if value is not None and partial[i] is None:
            return i
    for strategy in strategies:
        chosen = choose_strategy_var(model, partial, strategy)
        if chosen is not None:
            return chosen
    return min(
        (i for i, v in enumerate(partial) if v is None),
        key=lambda i: len(model["variables"][i]["domain"]),
    )


def choose_strategy_var(
    model: dict,
    partial: Sequence[Optional[int]],
    strategy: dict,
) -> Optional[int]:
    vars_ = [int(v) for v in strategy["vars"]]
    candidates = [var for var in vars_ if partial[var] is None]
    if not candidates:
        return None
    variable_strategy = strategy.get("variable_strategy", "first")
    if variable_strategy == "first":
        return candidates[0]
    if variable_strategy == "min_domain_size":
        return min(candidates, key=lambda var: len(model["variables"][var]["domain"]))
    if variable_strategy == "max_domain_size":
        return max(candidates, key=lambda var: len(model["variables"][var]["domain"]))
    if variable_strategy == "lowest_min":
        return min(candidates, key=lambda var: min(int(v) for v in model["variables"][var]["domain"]))
    if variable_strategy == "highest_max":
        return max(candidates, key=lambda var: max(int(v) for v in model["variables"][var]["domain"]))
    raise ValueError(f"unknown decision variable strategy {variable_strategy}")


def validate_decision_strategies(model: dict) -> list[dict]:
    strategies = list(model.get("decision_strategies") or [])
    seen = set()
    for strategy in strategies:
        vars_ = [int(v) for v in strategy["vars"]]
        if not vars_:
            raise ValueError("decision strategy has no variables")
        for var in vars_:
            if var < 0 or var >= len(model["variables"]):
                raise ValueError(f"decision strategy variable {var} out of range")
            if var in seen:
                raise ValueError(f"duplicate decision strategy for variable {var}")
            seen.add(var)
    return strategies


def strategy_for_var(strategies: Sequence[dict], var: int) -> Optional[dict]:
    for strategy in strategies:
        if var in [int(v) for v in strategy["vars"]]:
            return strategy
    return None


def ordered_domain_values(
    model: dict,
    var: int,
    hints: Sequence[Optional[int]],
    strategies: Sequence[dict],
) -> list[int]:
    values = [int(v) for v in model["variables"][var]["domain"]]
    hint = hints[var]
    if hint is not None and hint in values:
        values.remove(hint)
        values.insert(0, hint)
    else:
        strategy = strategy_for_var(strategies, var)
        if strategy is not None:
            domain_strategy = strategy.get("domain_strategy", "min_value")
            if domain_strategy == "min_value":
                values.sort()
            elif domain_strategy == "max_value":
                values.sort(reverse=True)
            elif domain_strategy == "lower_half":
                values.sort()
            elif domain_strategy == "upper_half":
                values.sort(reverse=True)
            elif domain_strategy == "median_value":
                values.sort()
                median = values.pop(len(values) // 2)
                values.insert(0, median)
            else:
                raise ValueError(f"unknown decision domain strategy {domain_strategy}")
    return values


def partial_ok(model: dict, partial: Sequence[Optional[int]]) -> bool:
    for c in model.get("constraints", []):
        kind = c["kind"]
        if kind == "linear":
            if not linear_partial_ok(model, partial, c):
                return False
        elif kind == "linear_domain":
            if not linear_domain_partial_ok(model, partial, c):
                return False
        elif kind == "enforced_linear_domain":
            active = enforcement_state(partial, c["enforcement"])
            if active is True and not linear_domain_partial_ok(model, partial, c):
                return False
        elif kind == "map_domain":
            var = int(c["var"])
            bools = [int(v) for v in c["bools"]]
            offset = int(c.get("offset", 0))
            var_value = partial[var]
            true_target = None
            for i, bool_var in enumerate(bools):
                target = offset + i
                bool_value = partial[bool_var]
                if bool_value == 1:
                    if true_target is not None and true_target != target:
                        return False
                    if var_value is not None and int(var_value) != target:
                        return False
                    true_target = target
                elif bool_value == 0:
                    if var_value is not None and int(var_value) == target:
                        return False
                elif bool_value is not None:
                    return False
            if true_target is not None:
                if true_target not in [int(value) for value in model["variables"][var]["domain"]]:
                    return False
            elif var_value is None:
                domain = [int(value) for value in model["variables"][var]["domain"]]
                if not any(
                    all(candidate != offset + i or partial[bool_var] != 0 for i, bool_var in enumerate(bools))
                    for candidate in domain
                ):
                    return False
        elif kind == "enforced_linear":
            active = enforcement_state(partial, c["enforcement"])
            if active is True and not linear_partial_ok(model, partial, c):
                return False
        elif kind == "enforced_bool_or":
            active = enforcement_state(partial, c["enforcement"])
            if active is True:
                unknown = False
                satisfied = False
                for lit in c["literals"]:
                    truth = literal_truth(partial, lit)
                    if truth is True:
                        satisfied = True
                        break
                    if truth is None:
                        unknown = True
                if not satisfied and not unknown:
                    return False
        elif kind == "enforced_at_least_one":
            active = enforcement_state(partial, c["enforcement"])
            if active is True:
                unknown = False
                satisfied = False
                for lit in c["literals"]:
                    truth = literal_truth(partial, lit)
                    if truth is True:
                        satisfied = True
                        break
                    if truth is None:
                        unknown = True
                if not satisfied and not unknown:
                    return False
        elif kind == "enforced_bool_and":
            active = enforcement_state(partial, c["enforcement"])
            if active is True:
                for lit in c["literals"]:
                    truth = literal_truth(partial, lit)
                    if truth is False:
                        return False
        elif kind == "enforced_bool_xor":
            active = enforcement_state(partial, c["enforcement"])
            if active is True:
                true_count = 0
                unknown = False
                for lit in c["literals"]:
                    truth = literal_truth(partial, lit)
                    if truth is True:
                        true_count += 1
                    elif truth is None:
                        unknown = True
                if not unknown and true_count % 2 == 0:
                    return False
        elif kind == "enforced_at_most_one":
            active = enforcement_state(partial, c["enforcement"])
            if active is True:
                true_count = sum(1 for lit in c["literals"] if literal_truth(partial, lit) is True)
                if true_count > 1:
                    return False
        elif kind == "enforced_exactly_one":
            active = enforcement_state(partial, c["enforcement"])
            if active is True:
                true_count = 0
                unknown = False
                for lit in c["literals"]:
                    truth = literal_truth(partial, lit)
                    if truth is True:
                        true_count += 1
                    elif truth is None:
                        unknown = True
                if true_count > 1 or (true_count == 0 and not unknown):
                    return False
        elif kind == "all_different":
            seen = set()
            for v in c["vars"]:
                value = partial[int(v)]
                if value is None:
                    continue
                if value in seen:
                    return False
                seen.add(value)
        elif kind == "bool_or":
            unknown = False
            satisfied = False
            for lit in c["literals"]:
                value = partial[int(lit["var"])]
                if value is None:
                    unknown = True
                    continue
                truth = literal_truth(partial, lit)
                if truth:
                    satisfied = True
                    break
            if not satisfied and not unknown:
                return False
        elif kind == "bool_and":
            for lit in c["literals"]:
                truth = literal_truth(partial, lit)
                if truth is False:
                    return False
        elif kind == "bool_xor":
            true_count = 0
            unknown = False
            for lit in c["literals"]:
                truth = literal_truth(partial, lit)
                if truth is True:
                    true_count += 1
                elif truth is None:
                    unknown = True
            if not unknown and true_count % 2 == 0:
                return False
        elif kind == "at_most_one":
            true_count = sum(1 for lit in c["literals"] if literal_truth(partial, lit) is True)
            if true_count > 1:
                return False
        elif kind == "at_least_one":
            unknown = False
            satisfied = False
            for lit in c["literals"]:
                truth = literal_truth(partial, lit)
                if truth is True:
                    satisfied = True
                    break
                if truth is None:
                    unknown = True
            if not satisfied and not unknown:
                return False
        elif kind == "exactly_one":
            true_count = 0
            unknown = False
            for lit in c["literals"]:
                truth = literal_truth(partial, lit)
                if truth is True:
                    true_count += 1
                elif truth is None:
                    unknown = True
            if true_count > 1 or (true_count == 0 and not unknown):
                return False
        elif kind == "implication":
            antecedent = literal_truth(partial, c["antecedent"])
            consequent = literal_truth(partial, c["consequent"])
            if antecedent is True and consequent is False:
                return False
        elif kind == "circuit":
            arcs = c["arcs"]
            nodes = sorted({int(arc["tail"]) for arc in arcs} | {int(arc["head"]) for arc in arcs})
            for node in nodes:
                true_out = sum(
                    1
                    for arc in arcs
                    if int(arc["tail"]) == node and literal_truth(partial, arc["literal"]) is True
                )
                true_in = sum(
                    1
                    for arc in arcs
                    if int(arc["head"]) == node and literal_truth(partial, arc["literal"]) is True
                )
                if true_out > 1 or true_in > 1:
                    return False
                possible_out = sum(
                    1
                    for arc in arcs
                    if int(arc["tail"]) == node and literal_truth(partial, arc["literal"]) is not False
                )
                possible_in = sum(
                    1
                    for arc in arcs
                    if int(arc["head"]) == node and literal_truth(partial, arc["literal"]) is not False
                )
                if possible_out == 0 or possible_in == 0:
                    return False
            all_bound = True
            selected = []
            for arc in arcs:
                truth = literal_truth(partial, arc["literal"])
                if truth is True:
                    selected.append(arc)
                elif truth is None:
                    all_bound = False
            if all_bound and not circuit_complete_ok(selected, nodes):
                return False
        elif kind == "multiple_circuit":
            arcs = c["arcs"]
            nodes = sorted({int(arc["tail"]) for arc in arcs} | {int(arc["head"]) for arc in arcs})
            if 0 not in nodes:
                return False
            for node in nodes:
                true_out = sum(
                    1
                    for arc in arcs
                    if int(arc["tail"]) == node and literal_truth(partial, arc["literal"]) is True
                )
                true_in = sum(
                    1
                    for arc in arcs
                    if int(arc["head"]) == node and literal_truth(partial, arc["literal"]) is True
                )
                possible_out = sum(
                    1
                    for arc in arcs
                    if int(arc["tail"]) == node and literal_truth(partial, arc["literal"]) is not False
                )
                possible_in = sum(
                    1
                    for arc in arcs
                    if int(arc["head"]) == node and literal_truth(partial, arc["literal"]) is not False
                )
                if node == 0:
                    if true_out > possible_in or true_in > possible_out:
                        return False
                    continue
                if true_out > 1 or true_in > 1 or possible_out == 0 or possible_in == 0:
                    return False
            all_bound = True
            selected = []
            for arc in arcs:
                truth = literal_truth(partial, arc["literal"])
                if truth is True:
                    selected.append(arc)
                elif truth is None:
                    all_bound = False
            if all_bound and not multiple_circuit_complete_ok(selected, nodes):
                return False
        elif kind == "allowed_assignments":
            vars_ = [int(v) for v in c["vars"]]
            tuples = [[int(v) for v in row] for row in c["tuples"]]
            ok = False
            for row in tuples:
                if all(partial[var] is None or int(partial[var]) == value for var, value in zip(vars_, row)):
                    ok = True
                    break
            if not ok:
                return False
        elif kind == "forbidden_assignments":
            vars_ = [int(v) for v in c["vars"]]
            tuples = [[int(v) for v in row] for row in c["tuples"]]
            for row in tuples:
                if all(partial[var] is not None and int(partial[var]) == value for var, value in zip(vars_, row)):
                    return False
        elif kind == "enforced_allowed_assignments":
            active = enforcement_state(partial, c["enforcement"])
            if active is True:
                vars_ = [int(v) for v in c["vars"]]
                tuples = [[int(v) for v in row] for row in c["tuples"]]
                if not any(
                    all(partial[var] is None or int(partial[var]) == value for var, value in zip(vars_, row))
                    for row in tuples
                ):
                    return False
        elif kind == "enforced_forbidden_assignments":
            active = enforcement_state(partial, c["enforcement"])
            if active is True:
                vars_ = [int(v) for v in c["vars"]]
                tuples = [[int(v) for v in row] for row in c["tuples"]]
                for row in tuples:
                    if all(partial[var] is not None and int(partial[var]) == value for var, value in zip(vars_, row)):
                        return False
        elif kind == "inverse":
            direct = [int(v) for v in c["direct"]]
            inverse = [int(v) for v in c["inverse"]]
            n = len(direct)
            seen_direct = set()
            for i, var in enumerate(direct):
                value = partial[var]
                if value is None:
                    continue
                j = int(value)
                if j < 0 or j >= n or j in seen_direct:
                    return False
                seen_direct.add(j)
                inverse_value = partial[inverse[j]]
                if inverse_value is not None and int(inverse_value) != i:
                    return False
            seen_inverse = set()
            for j, var in enumerate(inverse):
                value = partial[var]
                if value is None:
                    continue
                i = int(value)
                if i < 0 or i >= n or i in seen_inverse:
                    return False
                seen_inverse.add(i)
                direct_value = partial[direct[i]]
                if direct_value is not None and int(direct_value) != j:
                    return False
        elif kind in ("max_equality", "min_equality"):
            target = int(c["target"])
            vars_ = [int(v) for v in c["vars"]]
            target_value = partial[target]
            if all(partial[var] is not None for var in vars_):
                values = [int(partial[var]) for var in vars_]  # type: ignore[arg-type]
                expected = max(values) if kind == "max_equality" else min(values)
                if target_value is not None and int(target_value) != expected:
                    return False
                if target_value is None and expected not in model["variables"][target]["domain"]:
                    return False
            elif target_value is not None:
                t = int(target_value)
                ranges = []
                for var in vars_:
                    if partial[var] is None:
                        dom = [int(v) for v in model["variables"][var]["domain"]]
                        ranges.append((min(dom), max(dom)))
                    else:
                        v = int(partial[var])
                        ranges.append((v, v))
                if kind == "max_equality":
                    if any(lo > t for lo, _ in ranges) or not any(lo <= t <= hi for lo, hi in ranges):
                        return False
                else:
                    if any(hi < t for _, hi in ranges) or not any(lo <= t <= hi for lo, hi in ranges):
                        return False
        elif kind == "abs_equality":
            target = int(c["target"])
            var = int(c["var"])
            target_value = partial[target]
            var_value = partial[var]
            if target_value is not None and int(target_value) < 0:
                return False
            if target_value is not None and var_value is not None:
                if int(target_value) != abs(int(var_value)):
                    return False
            elif target_value is not None:
                t = int(target_value)
                if not any(abs(int(value)) == t for value in model["variables"][var]["domain"]):
                    return False
            elif var_value is not None:
                expected = abs(int(var_value))
                if expected not in model["variables"][target]["domain"]:
                    return False
        elif kind == "multiplication_equality":
            target = int(c["target"])
            vars_ = [int(v) for v in c["vars"]]
            target_value = partial[target]
            product = 1
            all_assigned = True
            bounds = []
            for var in vars_:
                value = partial[var]
                if value is None:
                    all_assigned = False
                    dom = [int(v) for v in model["variables"][var]["domain"]]
                    bounds.append((min(dom), max(dom)))
                else:
                    value = int(value)
                    product *= value
                    bounds.append((value, value))
            if all_assigned:
                if target_value is not None:
                    if int(target_value) != product:
                        return False
                elif product not in [int(v) for v in model["variables"][target]["domain"]]:
                    return False
            else:
                product_lo, product_hi = product_range(bounds)
                if target_value is not None:
                    if not (product_lo <= int(target_value) <= product_hi):
                        return False
                else:
                    dom = [int(v) for v in model["variables"][target]["domain"]]
                    if max(dom) < product_lo or min(dom) > product_hi:
                        return False
        elif kind == "division_equality":
            target = int(c["target"])
            numerator = int(c["numerator"])
            denominator = int(c["denominator"])
            target_values = (
                [int(partial[target])]
                if partial[target] is not None
                else [int(v) for v in model["variables"][target]["domain"]]
            )
            numerator_values = (
                [int(partial[numerator])]
                if partial[numerator] is not None
                else [int(v) for v in model["variables"][numerator]["domain"]]
            )
            denominator_values = (
                [int(partial[denominator])]
                if partial[denominator] is not None
                else [int(v) for v in model["variables"][denominator]["domain"]]
            )
            if not any(
                den != 0 and trunc_div(num, den) in target_values
                for num in numerator_values
                for den in denominator_values
            ):
                return False
        elif kind == "modulo_equality":
            target = int(c["target"])
            var = int(c["var"])
            modulus = int(c["modulus"])
            target_values = (
                [int(partial[target])]
                if partial[target] is not None
                else [int(v) for v in model["variables"][target]["domain"]]
            )
            var_values = (
                [int(partial[var])]
                if partial[var] is not None
                else [int(v) for v in model["variables"][var]["domain"]]
            )
            modulus_values = (
                [int(partial[modulus])]
                if partial[modulus] is not None
                else [int(v) for v in model["variables"][modulus]["domain"]]
            )
            if not any(
                mod != 0 and value % mod in target_values
                for value in var_values
                for mod in modulus_values
            ):
                return False
        elif kind == "automaton":
            states = {int(c["starting_state"])}
            transitions = [
                (int(t["tail"]), int(t["label"]), int(t["head"]))
                for t in c["transitions"]
            ]
            for var in [int(v) for v in c["vars"]]:
                value = partial[var]
                labels = (
                    [int(value)]
                    if value is not None
                    else [int(v) for v in model["variables"][var]["domain"]]
                )
                next_states = {
                    head
                    for tail, label, head in transitions
                    if tail in states and label in labels
                }
                if not next_states:
                    return False
                states = next_states
            final_states = {int(s) for s in c["final_states"]}
            if not (states & final_states):
                return False
        elif kind == "element":
            index_var = int(c["index"])
            target_var = int(c["target"])
            values = [int(v) for v in c["values"]]
            index_value = partial[index_var]
            target_value = partial[target_var]
            if index_value is not None and target_value is not None:
                index = int(index_value)
                if index < 0 or index >= len(values) or values[index] != int(target_value):
                    return False
            elif index_value is not None:
                index = int(index_value)
                if index < 0 or index >= len(values):
                    return False
            elif target_value is not None:
                if not any(
                    0 <= int(index) < len(values) and values[int(index)] == int(target_value)
                    for index in model["variables"][index_var]["domain"]
                ):
                    return False
            elif not any(0 <= int(index) < len(values) for index in model["variables"][index_var]["domain"]):
                return False
        elif kind == "variable_element":
            index_var = int(c["index"])
            target_var = int(c["target"])
            vars_ = [int(v) for v in c["vars"]]
            index_values = (
                [int(partial[index_var])]
                if partial[index_var] is not None
                else [int(v) for v in model["variables"][index_var]["domain"]]
            )
            target_values = (
                [int(partial[target_var])]
                if partial[target_var] is not None
                else [int(v) for v in model["variables"][target_var]["domain"]]
            )
            possible = False
            for index in index_values:
                if index < 0 or index >= len(vars_):
                    continue
                selected_var = vars_[index]
                selected_values = (
                    [int(partial[selected_var])]
                    if partial[selected_var] is not None
                    else [int(v) for v in model["variables"][selected_var]["domain"]]
                )
                if set(selected_values) & set(target_values):
                    possible = True
                    break
            if not possible:
                return False
        elif kind == "alternative":
            parent_active = optional_presence_truth(partial, c)
            true_modes = 0
            unknown_modes = 0
            for mode in c["alternatives"]:
                active = optional_presence_truth(partial, mode)
                if active is True:
                    true_modes += 1
                elif active is None:
                    unknown_modes += 1
            if true_modes > 1:
                return False
            if parent_active is True and true_modes == 0 and unknown_modes == 0:
                return False
            if parent_active is False and true_modes > 0:
                return False
            if not interval_triplet_consistent(
                partial,
                int(c["start"]),
                int(c["duration"]),
                int(c["end"]),
                parent_active,
            ):
                return False
            for mode in c["alternatives"]:
                active = optional_presence_truth(partial, mode)
                if not interval_triplet_consistent(
                    partial,
                    int(mode["start"]),
                    int(mode["duration"]),
                    int(mode["end"]),
                    active,
                ):
                    return False
                if active is not True:
                    continue
                if parent_active is False:
                    return False
                if not (
                    assigned_equal_or_unknown(partial, int(c["start"]), int(mode["start"]))
                    and assigned_equal_or_unknown(partial, int(c["duration"]), int(mode["duration"]))
                    and assigned_equal_or_unknown(partial, int(c["end"]), int(mode["end"]))
                ):
                    return False
        elif kind == "no_overlap":
            intervals = c["intervals"]
            for i, a in enumerate(intervals):
                if optional_presence_truth(partial, a) is not True:
                    continue
                start_a = partial[int(a["start"])]
                if start_a is None:
                    continue
                end_a = int(start_a) + int(a["duration"])
                for b in intervals[i + 1:]:
                    if optional_presence_truth(partial, b) is not True:
                        continue
                    start_b = partial[int(b["start"])]
                    if start_b is None:
                        continue
                    end_b = int(start_b) + int(b["duration"])
                    if not (end_a <= int(start_b) or end_b <= int(start_a)):
                        return False
        elif kind == "no_overlap_variable":
            spans = []
            for interval in c["intervals"]:
                span, ok = variable_interval_span(partial, interval)
                if not ok:
                    return False
                if span is not None:
                    spans.append(span)
            for i, (start_a, end_a) in enumerate(spans):
                for start_b, end_b in spans[i + 1:]:
                    if not (end_a <= start_b or end_b <= start_a):
                        return False
        elif kind == "no_overlap_2d":
            rectangles = c["rectangles"]
            for i, a in enumerate(rectangles):
                if optional_presence_truth(partial, a) is not True:
                    continue
                x_a = partial[int(a["x_start"])]
                y_a = partial[int(a["y_start"])]
                if x_a is None or y_a is None:
                    continue
                x_end_a = int(x_a) + int(a["width"])
                y_end_a = int(y_a) + int(a["height"])
                for b in rectangles[i + 1:]:
                    if optional_presence_truth(partial, b) is not True:
                        continue
                    x_b = partial[int(b["x_start"])]
                    y_b = partial[int(b["y_start"])]
                    if x_b is None or y_b is None:
                        continue
                    x_end_b = int(x_b) + int(b["width"])
                    y_end_b = int(y_b) + int(b["height"])
                    x_disjoint = x_end_a <= int(x_b) or x_end_b <= int(x_a)
                    y_disjoint = y_end_a <= int(y_b) or y_end_b <= int(y_a)
                    if not (x_disjoint or y_disjoint):
                        return False
        elif kind == "no_overlap_2d_variable":
            spans = []
            for rectangle in c["rectangles"]:
                span, ok = variable_rectangle_span(partial, rectangle)
                if not ok:
                    return False
                if span is not None:
                    spans.append(span)
            for i, (x_start_a, x_end_a, y_start_a, y_end_a) in enumerate(spans):
                for x_start_b, x_end_b, y_start_b, y_end_b in spans[i + 1:]:
                    x_disjoint = x_end_a <= x_start_b or x_end_b <= x_start_a
                    y_disjoint = y_end_a <= y_start_b or y_end_b <= y_start_a
                    if not (x_disjoint or y_disjoint):
                        return False
        elif kind == "cumulative":
            assigned = []
            for interval in c["intervals"]:
                if optional_presence_truth(partial, interval) is not True:
                    continue
                start = partial[int(interval["start"])]
                if start is None:
                    continue
                assigned.append((int(start), int(start) + int(interval["duration"]), int(interval["demand"])))
            points = sorted({point for start, end, _ in assigned for point in (start, end)})
            capacity = int(c["capacity"])
            for t in points:
                load = sum(demand for start, end, demand in assigned if start <= t < end)
                if load > capacity:
                    return False
        elif kind == "cumulative_variable":
            assigned = []
            for interval in c["intervals"]:
                span, ok = variable_demand_interval_span(partial, interval)
                if not ok:
                    return False
                if span is not None:
                    assigned.append(span)
            capacity = partial[int(c["capacity"])]
            if capacity is not None:
                points = sorted({point for start, end, _ in assigned for point in (start, end)})
                for t in points:
                    load = sum(demand for start, end, demand in assigned if start <= t < end)
                    if load > int(capacity):
                        return False
        elif kind == "reservoir":
            all_bound = True
            active_events = []
            for event in c["events"]:
                active = True
                if event.get("active") is not None:
                    active = literal_truth(partial, event["active"])
                if active is False:
                    continue
                if active is None:
                    all_bound = False
                    continue
                time = partial[int(event["time"])]
                if time is None:
                    all_bound = False
                    continue
                active_events.append((int(time), int(event["level_change"])))
            if all_bound and not reservoir_complete_ok(
                active_events,
                int(c["min_level"]),
                int(c["max_level"]),
            ):
                return False
        else:
            raise ValueError(f"unknown constraint kind {kind}")
    return True


def enumerate_reference(model: dict) -> dict:
    n = len(model["variables"])
    partial: List[Optional[int]] = [None] * n
    hints = solution_hint_values(model)
    strategies = validate_decision_strategies(model)
    best = None
    best_obj = None
    nodes = 0

    def better(value: int, incumbent: int) -> bool:
        sense = model.get("objective", {}).get("sense", "min")
        return value < incumbent if sense == "min" else value > incumbent

    def dfs() -> None:
        nonlocal best, best_obj, nodes
        nodes += 1
        if not partial_ok(model, partial):
            return
        try:
            var = choose_search_var(model, partial, hints, strategies)
        except ValueError:
            full = [int(v) for v in partial]  # type: ignore[arg-type]
            obj = objective_value(model, full)
            if best is None or (obj is not None and best_obj is not None and better(obj, best_obj)):
                best = full
                best_obj = obj
            elif best is None:
                best = full
                best_obj = obj
            return
        for value in ordered_domain_values(model, var, hints, strategies):
            partial[var] = int(value)
            dfs()
            partial[var] = None

    dfs()
    if best is None:
        return {"status": "infeasible", "assignment": [], "objective": None, "nodes": nodes, "solver": "python:cp-enumeration"}
    return {
        "status": "optimal" if model.get("objective") else "feasible",
        "assignment": best,
        "objective": best_obj,
        "nodes": nodes,
        "solver": "python:cp-enumeration",
        "message": "dependency-free exact enumeration fallback",
    }


def enumerate_pool_reference(model: dict, max_solutions: int) -> dict:
    if max_solutions <= 0:
        raise ValueError("max_solutions must be positive")

    n = len(model["variables"])
    partial: List[Optional[int]] = [None] * n
    hints = solution_hint_values(model)
    strategies = validate_decision_strategies(model)
    solutions = []
    nodes = 0
    hit_solution_limit = False

    def dfs() -> None:
        nonlocal nodes, hit_solution_limit
        if hit_solution_limit:
            return
        nodes += 1
        if not partial_ok(model, partial):
            return
        try:
            var = choose_search_var(model, partial, hints, strategies)
        except ValueError:
            full = [int(v) for v in partial]  # type: ignore[arg-type]
            solutions.append(
                {
                    "assignment": full,
                    "objective": objective_value(model, full),
                }
            )
            if not model.get("objective") and len(solutions) >= max_solutions:
                hit_solution_limit = True
            return
        for value in ordered_domain_values(model, var, hints, strategies):
            partial[var] = int(value)
            dfs()
            partial[var] = None
            if hit_solution_limit:
                break

    dfs()
    if model.get("objective"):
        reverse = model["objective"].get("sense", "min") == "max"
        solutions.sort(
            key=lambda item: item["objective"] if item["objective"] is not None else 0,
            reverse=reverse,
        )
        if len(solutions) > max_solutions:
            solutions = solutions[:max_solutions]
            hit_solution_limit = True

    if not solutions:
        return {
            "status": "infeasible",
            "assignment": [],
            "objective": None,
            "solutions": [],
            "exhausted": not hit_solution_limit,
            "nodes": nodes,
            "solver": "python:cp-solution-enumeration",
        }

    first = solutions[0]
    return {
        "status": "optimal" if model.get("objective") else "feasible",
        "assignment": first["assignment"],
        "objective": first["objective"],
        "solutions": solutions,
        "exhausted": not hit_solution_limit,
        "nodes": nodes,
        "solver": "python:cp-solution-enumeration",
        "message": "dependency-free exact solution enumeration fallback",
    }


def model_with_assumptions(model: dict, assumptions: Sequence[dict]) -> dict:
    assumed = copy.deepcopy(model)
    constraints = assumed.setdefault("constraints", [])
    for lit in assumptions:
        constraints.append(
            {
                "kind": "linear",
                "terms": [{"var": int(lit["var"]), "coeff": 1}],
                "sense": "eq",
                "rhs": 1 if bool(lit.get("positive", True)) else 0,
            }
        )
    return assumed


def assumption_core_reference(model: dict) -> dict:
    assumptions = list(model.get("assumptions", []))
    core = list(assumptions)
    checks = 1
    full = enumerate_reference(model_with_assumptions(model, core))
    if full["status"] != "infeasible":
        return {
            "status": full["status"],
            "assumptions": [],
            "minimal": False,
            "checks": checks,
            "solver": "python:cp-assumption-core",
            "message": f"assumptions are not infeasible; status is {full['status']}",
        }

    idx = 0
    while idx < len(core):
        trial = list(core)
        del trial[idx]
        checks += 1
        if enumerate_reference(model_with_assumptions(model, trial))["status"] == "infeasible":
            core = trial
        else:
            idx += 1

    minimal = True
    for idx in range(len(core)):
        trial = list(core)
        del trial[idx]
        checks += 1
        if enumerate_reference(model_with_assumptions(model, trial))["status"] == "infeasible":
            minimal = False
            break

    return {
        "status": "infeasible",
        "assumptions": core,
        "minimal": minimal,
        "checks": checks,
        "solver": "python:cp-assumption-core",
        "message": "dependency-free exact assumption core fallback",
    }


def ortools_reference(model: dict) -> Optional[dict]:
    try:
        from ortools.sat.python import cp_model  # type: ignore
    except Exception:
        return None
    cp = cp_model.CpModel()
    xs = []
    for var in model["variables"]:
        dom = cp_model.Domain.FromValues([int(v) for v in var["domain"]])
        xs.append(cp.NewIntVarFromDomain(dom, var.get("name", f"x{len(xs)}")))
    for hint in model.get("solution_hint", []):
        cp.AddHint(xs[int(hint["var"])], int(hint["value"]))

    def variable_strategy_constant(name: str):
        mapping = {
            "first": cp_model.CHOOSE_FIRST,
            "min_domain_size": cp_model.CHOOSE_MIN_DOMAIN_SIZE,
            "max_domain_size": cp_model.CHOOSE_MAX_DOMAIN_SIZE,
            "lowest_min": cp_model.CHOOSE_LOWEST_MIN,
            "highest_max": cp_model.CHOOSE_HIGHEST_MAX,
        }
        if name not in mapping:
            raise ValueError(f"unknown decision variable strategy {name}")
        return mapping[name]

    def domain_strategy_constant(name: str):
        mapping = {
            "min_value": cp_model.SELECT_MIN_VALUE,
            "max_value": cp_model.SELECT_MAX_VALUE,
            "lower_half": cp_model.SELECT_LOWER_HALF,
            "upper_half": cp_model.SELECT_UPPER_HALF,
            "median_value": cp_model.SELECT_MEDIAN_VALUE,
        }
        if name not in mapping:
            raise ValueError(f"unknown decision domain strategy {name}")
        return mapping[name]

    decision_strategies = validate_decision_strategies(model)
    covered_strategy_vars = set()
    for strategy in decision_strategies:
        vars_ = [int(v) for v in strategy["vars"]]
        covered_strategy_vars.update(vars_)
        cp.AddDecisionStrategy(
            [xs[var] for var in vars_],
            variable_strategy_constant(strategy.get("variable_strategy", "first")),
            domain_strategy_constant(strategy.get("domain_strategy", "min_value")),
        )
    if decision_strategies:
        uncovered = [idx for idx in range(len(xs)) if idx not in covered_strategy_vars]
        if uncovered:
            cp.AddDecisionStrategy(
                [xs[idx] for idx in uncovered],
                cp_model.CHOOSE_MIN_DOMAIN_SIZE,
                cp_model.SELECT_MIN_VALUE,
            )

    def cp_literal(lit: dict):
        var = xs[int(lit["var"])]
        return var if bool(lit.get("positive", True)) else var.Not()

    def literal_expr(lit: dict):
        var = xs[int(lit["var"])]
        return var if bool(lit.get("positive", True)) else 1 - var

    def enforcement_literals(literals: Sequence[dict]):
        return [cp_literal(lit) for lit in literals]

    def fixed_size_interval(start_var, duration: int, name: str, item: dict):
        presence = item.get("presence")
        if presence is None:
            return cp.NewFixedSizeIntervalVar(start_var, duration, name)
        return cp.NewOptionalFixedSizeIntervalVar(
            start_var,
            duration,
            cp_literal(presence),
            name,
        )

    def variable_size_interval(item: dict, name: str):
        start = xs[int(item["start"])]
        size = xs[int(item["duration"])]
        end = xs[int(item["end"])]
        presence = item.get("presence")
        if presence is None:
            return cp.NewIntervalVar(start, size, end, name)
        return cp.NewOptionalIntervalVar(start, size, end, cp_literal(presence), name)

    def variable_axis_interval(
        item: dict,
        start_key: str,
        size_key: str,
        end_key: str,
        name: str,
    ):
        start = xs[int(item[start_key])]
        size = xs[int(item[size_key])]
        end = xs[int(item[end_key])]
        presence = item.get("presence")
        if presence is None:
            return cp.NewIntervalVar(start, size, end, name)
        return cp.NewOptionalIntervalVar(start, size, end, cp_literal(presence), name)

    for c in model.get("constraints", []):
        kind = c["kind"]
        if kind == "linear":
            expr = sum(int(t["coeff"]) * xs[int(t["var"])] for t in c["terms"])
            if c["sense"] == "le":
                cp.Add(expr <= int(c["rhs"]))
            elif c["sense"] == "ge":
                cp.Add(expr >= int(c["rhs"]))
            else:
                cp.Add(expr == int(c["rhs"]))
        elif kind == "linear_domain":
            expr = sum(int(t["coeff"]) * xs[int(t["var"])] for t in c["terms"])
            cp.AddLinearExpressionInDomain(
                expr,
                cp_model.Domain.FromIntervals(
                    [[int(interval["lb"]), int(interval["ub"])] for interval in c["intervals"]]
                ),
            )
        elif kind == "enforced_linear_domain":
            expr = sum(int(t["coeff"]) * xs[int(t["var"])] for t in c["terms"])
            constraint = cp.AddLinearExpressionInDomain(
                expr,
                cp_model.Domain.FromIntervals(
                    [[int(interval["lb"]), int(interval["ub"])] for interval in c["intervals"]]
                ),
            )
            constraint.OnlyEnforceIf(enforcement_literals(c["enforcement"]))
        elif kind == "map_domain":
            cp.AddMapDomain(
                xs[int(c["var"])],
                [xs[int(v)] for v in c["bools"]],
                int(c.get("offset", 0)),
            )
        elif kind == "enforced_linear":
            expr = sum(int(t["coeff"]) * xs[int(t["var"])] for t in c["terms"])
            if c["sense"] == "le":
                constraint = cp.Add(expr <= int(c["rhs"]))
            elif c["sense"] == "ge":
                constraint = cp.Add(expr >= int(c["rhs"]))
            else:
                constraint = cp.Add(expr == int(c["rhs"]))
            constraint.OnlyEnforceIf(enforcement_literals(c["enforcement"]))
        elif kind == "enforced_bool_or":
            constraint = cp.AddBoolOr([cp_literal(lit) for lit in c["literals"]])
            constraint.OnlyEnforceIf(enforcement_literals(c["enforcement"]))
        elif kind == "enforced_bool_and":
            constraint = cp.AddBoolAnd([cp_literal(lit) for lit in c["literals"]])
            constraint.OnlyEnforceIf(enforcement_literals(c["enforcement"]))
        elif kind == "enforced_bool_xor":
            constraint = cp.AddBoolXOr([cp_literal(lit) for lit in c["literals"]])
            constraint.OnlyEnforceIf(enforcement_literals(c["enforcement"]))
        elif kind == "enforced_at_most_one":
            constraint = cp.AddAtMostOne([cp_literal(lit) for lit in c["literals"]])
            constraint.OnlyEnforceIf(enforcement_literals(c["enforcement"]))
        elif kind == "enforced_at_least_one":
            constraint = cp.AddAtLeastOne([cp_literal(lit) for lit in c["literals"]])
            constraint.OnlyEnforceIf(enforcement_literals(c["enforcement"]))
        elif kind == "enforced_exactly_one":
            constraint = cp.AddExactlyOne([cp_literal(lit) for lit in c["literals"]])
            constraint.OnlyEnforceIf(enforcement_literals(c["enforcement"]))
        elif kind == "all_different":
            cp.AddAllDifferent([xs[int(v)] for v in c["vars"]])
        elif kind == "bool_or":
            lits = [cp_literal(lit) for lit in c["literals"]]
            cp.AddBoolOr(lits)
        elif kind == "bool_and":
            lits = [cp_literal(lit) for lit in c["literals"]]
            cp.AddBoolAnd(lits)
        elif kind == "bool_xor":
            lits = [cp_literal(lit) for lit in c["literals"]]
            cp.AddBoolXOr(lits)
        elif kind == "at_most_one":
            lits = [cp_literal(lit) for lit in c["literals"]]
            cp.AddAtMostOne(lits)
        elif kind == "at_least_one":
            lits = [cp_literal(lit) for lit in c["literals"]]
            cp.AddAtLeastOne(lits)
        elif kind == "exactly_one":
            lits = [cp_literal(lit) for lit in c["literals"]]
            cp.AddExactlyOne(lits)
        elif kind == "implication":
            cp.AddImplication(cp_literal(c["antecedent"]), cp_literal(c["consequent"]))
        elif kind == "circuit":
            arcs = []
            for arc in c["arcs"]:
                arcs.append((int(arc["tail"]), int(arc["head"]), cp_literal(arc["literal"])))
            cp.AddCircuit(arcs)
        elif kind == "multiple_circuit":
            arcs = []
            for arc in c["arcs"]:
                arcs.append((int(arc["tail"]), int(arc["head"]), cp_literal(arc["literal"])))
            cp.AddMultipleCircuit(arcs)
        elif kind == "allowed_assignments":
            cp.AddAllowedAssignments(
                [xs[int(v)] for v in c["vars"]],
                [[int(v) for v in row] for row in c["tuples"]],
            )
        elif kind == "forbidden_assignments":
            cp.AddForbiddenAssignments(
                [xs[int(v)] for v in c["vars"]],
                [[int(v) for v in row] for row in c["tuples"]],
            )
        elif kind == "enforced_allowed_assignments":
            constraint = cp.AddAllowedAssignments(
                [xs[int(v)] for v in c["vars"]],
                [[int(v) for v in row] for row in c["tuples"]],
            )
            constraint.OnlyEnforceIf(enforcement_literals(c["enforcement"]))
        elif kind == "enforced_forbidden_assignments":
            constraint = cp.AddForbiddenAssignments(
                [xs[int(v)] for v in c["vars"]],
                [[int(v) for v in row] for row in c["tuples"]],
            )
            constraint.OnlyEnforceIf(enforcement_literals(c["enforcement"]))
        elif kind == "inverse":
            cp.AddInverse(
                [xs[int(v)] for v in c["direct"]],
                [xs[int(v)] for v in c["inverse"]],
            )
        elif kind == "max_equality":
            cp.AddMaxEquality(
                xs[int(c["target"])],
                [xs[int(v)] for v in c["vars"]],
            )
        elif kind == "min_equality":
            cp.AddMinEquality(
                xs[int(c["target"])],
                [xs[int(v)] for v in c["vars"]],
            )
        elif kind == "abs_equality":
            cp.AddAbsEquality(
                xs[int(c["target"])],
                xs[int(c["var"])],
            )
        elif kind == "multiplication_equality":
            cp.AddMultiplicationEquality(
                xs[int(c["target"])],
                [xs[int(v)] for v in c["vars"]],
            )
        elif kind == "division_equality":
            cp.AddDivisionEquality(
                xs[int(c["target"])],
                xs[int(c["numerator"])],
                xs[int(c["denominator"])],
            )
        elif kind == "modulo_equality":
            cp.AddModuloEquality(
                xs[int(c["target"])],
                xs[int(c["var"])],
                xs[int(c["modulus"])],
            )
        elif kind == "automaton":
            cp.AddAutomaton(
                [xs[int(v)] for v in c["vars"]],
                int(c["starting_state"]),
                [int(s) for s in c["final_states"]],
                [
                    (int(t["tail"]), int(t["label"]), int(t["head"]))
                    for t in c["transitions"]
                ],
            )
        elif kind == "element":
            cp.AddElement(
                xs[int(c["index"])],
                [int(v) for v in c["values"]],
                xs[int(c["target"])],
            )
        elif kind == "variable_element":
            cp.AddElement(
                xs[int(c["index"])],
                [xs[int(v)] for v in c["vars"]],
                xs[int(c["target"])],
            )
        elif kind == "alternative":
            name = c.get("name", "alternative")
            parent_start = xs[int(c["start"])]
            parent_size = xs[int(c["duration"])]
            parent_end = xs[int(c["end"])]
            parent_presence = c.get("presence")
            if parent_presence is None:
                cp.NewIntervalVar(parent_start, parent_size, parent_end, f"{name}_parent")
            else:
                cp.NewOptionalIntervalVar(
                    parent_start,
                    parent_size,
                    parent_end,
                    cp_literal(parent_presence),
                    f"{name}_parent",
                )
            mode_literals = []
            mode_literal_exprs = []
            for i, mode in enumerate(c["alternatives"]):
                mode_name = mode.get("name", f"{name}_mode_{i}")
                presence = mode["presence"]
                active = cp_literal(presence)
                mode_literals.append(active)
                mode_literal_exprs.append(literal_expr(presence))
                cp.NewOptionalIntervalVar(
                    xs[int(mode["start"])],
                    xs[int(mode["duration"])],
                    xs[int(mode["end"])],
                    active,
                    mode_name,
                )
                cp.Add(parent_start == xs[int(mode["start"])]).OnlyEnforceIf(active)
                cp.Add(parent_size == xs[int(mode["duration"])]).OnlyEnforceIf(active)
                cp.Add(parent_end == xs[int(mode["end"])]).OnlyEnforceIf(active)
            if parent_presence is None:
                cp.AddExactlyOne(mode_literals)
            else:
                cp.Add(sum(mode_literal_exprs) == literal_expr(parent_presence))
        elif kind == "no_overlap":
            intervals = []
            for i, interval in enumerate(c["intervals"]):
                name = interval.get("name", f"interval_{i}")
                intervals.append(
                    fixed_size_interval(
                        xs[int(interval["start"])],
                        int(interval["duration"]),
                        name,
                        interval,
                    )
                )
            cp.AddNoOverlap(intervals)
        elif kind == "no_overlap_variable":
            intervals = []
            for i, interval in enumerate(c["intervals"]):
                name = interval.get("name", f"interval_{i}")
                intervals.append(variable_size_interval(interval, name))
            cp.AddNoOverlap(intervals)
        elif kind == "no_overlap_2d":
            x_intervals = []
            y_intervals = []
            for i, rectangle in enumerate(c["rectangles"]):
                name = rectangle.get("name", f"rectangle_{i}")
                x_intervals.append(
                    fixed_size_interval(
                        xs[int(rectangle["x_start"])],
                        int(rectangle["width"]),
                        f"{name}_x",
                        rectangle,
                    )
                )
                y_intervals.append(
                    fixed_size_interval(
                        xs[int(rectangle["y_start"])],
                        int(rectangle["height"]),
                        f"{name}_y",
                        rectangle,
                    )
                )
            cp.AddNoOverlap2D(x_intervals, y_intervals)
        elif kind == "no_overlap_2d_variable":
            x_intervals = []
            y_intervals = []
            for i, rectangle in enumerate(c["rectangles"]):
                name = rectangle.get("name", f"rectangle_{i}")
                x_intervals.append(
                    variable_axis_interval(
                        rectangle,
                        "x_start",
                        "x_size",
                        "x_end",
                        f"{name}_x",
                    )
                )
                y_intervals.append(
                    variable_axis_interval(
                        rectangle,
                        "y_start",
                        "y_size",
                        "y_end",
                        f"{name}_y",
                    )
                )
            cp.AddNoOverlap2D(x_intervals, y_intervals)
        elif kind == "cumulative":
            intervals = []
            demands = []
            for i, interval in enumerate(c["intervals"]):
                name = interval.get("name", f"interval_{i}")
                intervals.append(
                    fixed_size_interval(
                        xs[int(interval["start"])],
                        int(interval["duration"]),
                        name,
                        interval,
                    )
                )
                demands.append(int(interval["demand"]))
            cp.AddCumulative(intervals, demands, int(c["capacity"]))
        elif kind == "cumulative_variable":
            intervals = []
            demands = []
            for i, interval in enumerate(c["intervals"]):
                name = interval.get("name", f"interval_{i}")
                intervals.append(variable_size_interval(interval, name))
                demands.append(xs[int(interval["demand"])])
            cp.AddCumulative(intervals, demands, xs[int(c["capacity"])])
        elif kind == "reservoir":
            times = [xs[int(event["time"])] for event in c["events"]]
            level_changes = [int(event["level_change"]) for event in c["events"]]
            if any(event.get("active") is not None for event in c["events"]):
                actives = []
                for event in c["events"]:
                    active = event.get("active")
                    if active is None:
                        literal = cp.NewConstant(1)
                    else:
                        var = xs[int(active["var"])]
                        literal = var if bool(active.get("positive", True)) else var.Not()
                    actives.append(literal)
                cp.AddReservoirConstraintWithActive(
                    times,
                    level_changes,
                    actives,
                    int(c["min_level"]),
                    int(c["max_level"]),
                )
            else:
                cp.AddReservoirConstraint(
                    times,
                    level_changes,
                    int(c["min_level"]),
                    int(c["max_level"]),
                )
    if model.get("objective"):
        expr = sum(int(t["coeff"]) * xs[int(t["var"])] for t in model["objective"]["terms"])
        if model["objective"].get("sense", "min") == "min":
            cp.Minimize(expr)
        else:
            cp.Maximize(expr)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10.0
    if decision_strategies:
        solver.parameters.search_branching = cp_model.FIXED_SEARCH
        solver.parameters.cp_model_presolve = False
        solver.parameters.num_search_workers = 1
    status = solver.Solve(cp)
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        assignment = [int(solver.Value(x)) for x in xs]
        status_label = "optimal" if model.get("objective") and status == cp_model.OPTIMAL else "feasible"
        return {
            "status": status_label,
            "assignment": assignment,
            "objective": objective_value(model, assignment),
            "nodes": int(solver.NumBranches()),
            "solver": "ortools:cp-sat",
        }
    if status == cp_model.INFEASIBLE:
        return {"status": "infeasible", "assignment": [], "objective": None, "nodes": int(solver.NumBranches()), "solver": "ortools:cp-sat"}
    return {"status": "unavailable", "assignment": [], "objective": None, "nodes": int(solver.NumBranches()), "solver": "ortools:cp-sat", "message": "CP-SAT did not prove a result"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", default="auto")
    parser.add_argument("--enumerate-solutions", type=int)
    parser.add_argument("--assumption-core", action="store_true")
    args = parser.parse_args()
    model = json.load(sys.stdin)
    if args.assumption_core:
        result = assumption_core_reference(model)
        print(json.dumps(result))
        return 0

    if args.enumerate_solutions is not None:
        result = enumerate_pool_reference(model, args.enumerate_solutions)
        print(json.dumps(result))
        return 0

    result = None
    if args.solver in ("auto", "ortools", "ortools-cp-sat"):
        result = ortools_reference(model)
        if args.solver != "auto" and result is None:
            result = {"status": "unavailable", "assignment": [], "objective": None, "nodes": 0, "solver": "ortools:cp-sat", "message": "ortools is not installed"}
    if result is None:
        result = enumerate_reference(model)
    print(json.dumps(result))
    return 0 if result.get("status") != "unavailable" else 2


if __name__ == "__main__":
    raise SystemExit(main())
