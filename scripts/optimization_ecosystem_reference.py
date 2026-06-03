#!/usr/bin/env python3
"""Reference JSON adapter for non-LP/MIP optimization ecosystems.

This script is intentionally small and dependency-free. It gives the Rust
adapter runner a concrete local command for representative Java/Rust ecosystem
families while real Choco, JaCoP, CPMpy, clingo, Open-WBO, good_lp, argmin,
NLopt, and similar wrappers remain opt-in local executables.
"""

from __future__ import annotations

import argparse
import ast
import itertools
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Any


CP_TOOLS = {
    "choco-solver",
    "jacop",
    "ibm-cp-optimizer",
    "ortools-java",
    "ortools-python",
    "cpmpy",
    "pycsp3",
    "conjure",
    "savile-row",
    "picat",
    "clingo",
    "clingcon",
    "sat4j",
    "pysat",
    "open-wbo",
}
PLANNING_TOOLS = {"optaplanner", "timefold"}
MULTIOBJECTIVE_TOOLS = {"jmetal", "moea-framework", "ecj"}
CONVEX_TOOLS = {
    "cvxpy",
    "cvxopt",
    "mosek",
    "copt",
    "osqp",
    "scs",
    "clarabel",
    "ecos",
    "qpoases",
    "proxqp",
    "cosmo",
    "sdpa",
    "csdp",
}
LINEAR_TOOLS = {
    "ojalgo",
    "pyomo",
    "pulp",
    "python-mip",
    "docplex",
    "jump",
    "ampl",
    "gams",
    "symphony",
    "highs-cli",
    "glpk-cli",
    "scip-cli",
    "cbc-cli",
    "clp-cli",
    "gurobi-cli",
    "cplex-cli",
    "xpress-cli",
    "lindo-cli",
    "good-lp",
    "lp-modeler",
    "rust-linprog",
    "highs-rust",
    "scip-rust",
    "cbc-rust",
}
NATIVE_BINDING_TOOLS = {"pyscipopt", "gurobipy"}
NONLINEAR_TOOLS = {
    "argmin",
    "nlopt",
    "scipy-optimize",
    "minotaur",
    "ipopt",
    "bonmin",
    "couenne",
    "knitro",
    "baron",
    "casadi",
}
HYBRID_TOOLS = {"hexaly"}
SMT_TOOLS = {
    "z3",
    "cvc5",
    "yices",
    "bitwuzla",
    "boolector",
    "mathsat",
    "optimathsat",
    "opensmt",
    "smtinterpol",
    "princess",
}


@dataclass
class Result:
    status: str
    objective: float | None = None
    x: list[float] | None = None
    message: str = ""
    extra: dict[str, Any] | None = None


def emit(tool: str, family: str, result: Result) -> None:
    output: dict[str, Any] = {
        "kind": "optimization-ecosystem-reference-result",
        "tool": tool,
        "family": family,
        "status": result.status,
        "objective": result.objective,
        "x": result.x,
        "message": result.message,
        "backend": f"builtin:{family}",
    }
    if result.extra:
        output.update(result.extra)
    print(json.dumps(output, sort_keys=True))


def as_number(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return float(value)
    return default


def arg_tool(args_tool: str | None) -> str:
    raw = args_tool or os.environ.get("ORES_EXTERNAL_OPTIMIZATION_TOOL") or "auto"
    return raw.lower().replace("_", "-")


def tool_family(tool: str, payload_kind: str) -> str:
    if tool in HYBRID_TOOLS:
        return "hybrid-optimization"
    if tool in SMT_TOOLS:
        return "smt-omt"
    if tool in CP_TOOLS or payload_kind in {"cp-assignment", "ecosystem-cp-assignment"}:
        return "constraint-programming"
    if tool in PLANNING_TOOLS or payload_kind in {"planning-assignment", "ecosystem-planning-assignment"}:
        return "planning-metaheuristic"
    if tool in MULTIOBJECTIVE_TOOLS or payload_kind in {"multiobjective-front", "ecosystem-multiobjective"}:
        return "evolutionary-multiobjective"
    if tool in CONVEX_TOOLS:
        return "convex-optimization"
    if tool in NATIVE_BINDING_TOOLS:
        return "native-solver-binding"
    if tool in NONLINEAR_TOOLS or payload_kind in {"nonlinear-program", "ecosystem-nonlinear"}:
        return "nonlinear-optimization"
    return "linear-mip"


def row_feasible(lhs: float, sense: str, rhs: float, tol: float = 1e-9) -> bool:
    sense = sense.strip()
    if sense in {"<=", "le", "less-equal"}:
        return lhs <= rhs + tol
    if sense in {">=", "ge", "greater-equal"}:
        return lhs + tol >= rhs
    if sense in {"=", "==", "eq", "equal"}:
        return abs(lhs - rhs) <= tol
    raise ValueError(f"unsupported constraint sense {sense!r}")


def better(candidate: float, incumbent: float | None, sense: str) -> bool:
    if incumbent is None:
        return True
    if sense == "max":
        return candidate > incumbent + 1e-12
    return candidate < incumbent - 1e-12


def solve_discrete_linear(payload: dict[str, Any]) -> Result:
    objective = [as_number(v) for v in payload.get("objective", [])]
    if not objective:
        return Result("invalid", message="missing objective vector")
    sense = str(payload.get("sense", "min")).lower()
    domains = payload.get("domains")
    if not isinstance(domains, list):
        domains = [[0, 1] for _ in objective]
    value_domains: list[list[int]] = []
    for domain in domains:
        if not isinstance(domain, list) or len(domain) < 2:
            return Result("invalid", message="each domain must be [lb, ub]")
        lb = int(as_number(domain[0]))
        ub = int(as_number(domain[1]))
        if ub < lb:
            return Result("infeasible", message="empty domain")
        if ub - lb > 20:
            return Result("unsupported", message="domain too large for reference enumeration")
        value_domains.append(list(range(lb, ub + 1)))

    constraints = payload.get("constraints", [])
    if not isinstance(constraints, list):
        return Result("invalid", message="constraints must be a list")

    best_x: tuple[int, ...] | None = None
    best_obj: float | None = None
    for assignment in itertools.product(*value_domains):
        feasible = True
        for row in constraints:
            coefs = [as_number(v) for v in row.get("coefs", [])]
            if len(coefs) != len(objective):
                return Result("invalid", message="constraint coefficient length mismatch")
            lhs = sum(c * x for c, x in zip(coefs, assignment))
            if not row_feasible(lhs, str(row.get("sense", "<=")), as_number(row.get("rhs"))):
                feasible = False
                break
        if feasible:
            value = sum(c * x for c, x in zip(objective, assignment))
            if better(value, best_obj, sense):
                best_obj = value
                best_x = assignment

    if best_x is None or best_obj is None:
        return Result("infeasible", message="no feasible assignment")
    return Result("optimal", best_obj, [float(v) for v in best_x])


def solve_cp_assignment(payload: dict[str, Any]) -> Result:
    costs = payload.get("costs")
    if not isinstance(costs, list) or not costs:
        return Result("invalid", message="missing cost matrix")
    rows = [[as_number(v) for v in row] for row in costs]
    if any(len(row) != len(rows[0]) for row in rows):
        return Result("invalid", message="ragged cost matrix")
    domains = payload.get("domains")
    if not isinstance(domains, list):
        domains = [list(range(len(rows[0]))) for _ in rows]
    forbid = {tuple(pair) for pair in payload.get("forbidden", []) if isinstance(pair, list) and len(pair) == 2}
    all_different = bool(payload.get("all_different", True))

    best_x: tuple[int, ...] | None = None
    best_obj: float | None = None
    for assignment in itertools.product(*domains):
        assignment = tuple(int(v) for v in assignment)
        if all_different and len(set(assignment)) != len(assignment):
            continue
        if any((i, assignment[i]) in forbid for i in range(len(assignment))):
            continue
        if any(assignment[i] < 0 or assignment[i] >= len(rows[i]) for i in range(len(assignment))):
            continue
        value = sum(rows[i][assignment[i]] for i in range(len(assignment)))
        if better(value, best_obj, "min"):
            best_obj = value
            best_x = assignment

    if best_x is None or best_obj is None:
        return Result("infeasible", message="no feasible assignment")
    return Result("optimal", best_obj, [float(v) for v in best_x])


def solve_planning_assignment(payload: dict[str, Any]) -> Result:
    durations = [as_number(v) for v in payload.get("task_durations", [])]
    if not durations:
        return Result("invalid", message="missing task_durations")
    machines = int(as_number(payload.get("machines", 0)))
    if machines <= 0:
        return Result("invalid", message="machines must be positive")
    capacities = payload.get("capacities")
    if not isinstance(capacities, list):
        capacities = [math.inf for _ in range(machines)]
    capacities = [as_number(v, math.inf) for v in capacities]
    if len(capacities) != machines:
        return Result("invalid", message="capacity length mismatch")

    best_x: tuple[int, ...] | None = None
    best_obj: float | None = None
    for assignment in itertools.product(range(machines), repeat=len(durations)):
        loads = [0.0 for _ in range(machines)]
        for task, machine in enumerate(assignment):
            loads[machine] += durations[task]
        if any(loads[i] > capacities[i] + 1e-9 for i in range(machines)):
            continue
        value = max(loads)
        if better(value, best_obj, "min"):
            best_obj = value
            best_x = assignment

    if best_x is None or best_obj is None:
        return Result("infeasible", message="no feasible plan")
    return Result("optimal", best_obj, [float(v) for v in best_x], extra={"loads": loads_for(best_x, durations, machines)})


def loads_for(assignment: tuple[int, ...], durations: list[float], machines: int) -> list[float]:
    loads = [0.0 for _ in range(machines)]
    for task, machine in enumerate(assignment):
        loads[machine] += durations[task]
    return loads


def dominates(a: list[float], b: list[float], senses: list[str]) -> bool:
    at_least_one = False
    for av, bv, sense in zip(a, b, senses):
        if sense == "max":
            if av < bv - 1e-12:
                return False
            at_least_one = at_least_one or av > bv + 1e-12
        else:
            if av > bv + 1e-12:
                return False
            at_least_one = at_least_one or av < bv - 1e-12
    return at_least_one


def solve_multiobjective(payload: dict[str, Any]) -> Result:
    candidates = payload.get("candidates", [])
    if not isinstance(candidates, list) or not candidates:
        return Result("invalid", message="missing candidates")
    senses = [str(s).lower() for s in payload.get("senses", ["min", "min"])]
    weights = [as_number(v, 1.0) for v in payload.get("weights", [1.0 for _ in senses])]

    parsed: list[tuple[list[float], list[float]]] = []
    for candidate in candidates:
        x = [as_number(v) for v in candidate.get("x", [])]
        objectives = [as_number(v) for v in candidate.get("objectives", [])]
        if len(objectives) != len(senses):
            return Result("invalid", message="objective dimension mismatch")
        parsed.append((x, objectives))

    front: list[tuple[list[float], list[float]]] = []
    for x, objectives in parsed:
        if not any(dominates(other, objectives, senses) for _, other in parsed):
            front.append((x, objectives))
    if not front:
        return Result("infeasible", message="empty Pareto front")

    def scalar_score(objectives: list[float]) -> float:
        total = 0.0
        for value, weight, sense in zip(objectives, weights, senses):
            total += weight * (-value if sense == "max" else value)
        return total

    best_x, best_objectives = min(front, key=lambda item: scalar_score(item[1]))
    return Result(
        "optimal",
        scalar_score(best_objectives),
        [float(v) for v in best_x],
        extra={"pareto_front": [{"x": x, "objectives": y} for x, y in front]},
    )


ALLOWED_BINOPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.Pow: lambda a, b: a**b,
}
ALLOWED_UNARY = {ast.UAdd: lambda a: a, ast.USub: lambda a: -a}


def eval_expr(expr: str, env: dict[str, float]) -> float:
    def visit(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name) and node.id in env:
            return env[node.id]
        if isinstance(node, ast.BinOp) and type(node.op) in ALLOWED_BINOPS:
            return ALLOWED_BINOPS[type(node.op)](visit(node.left), visit(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in ALLOWED_UNARY:
            return ALLOWED_UNARY[type(node.op)](visit(node.operand))
        raise ValueError(f"unsupported expression node {type(node).__name__}")

    return visit(ast.parse(expr, mode="eval"))


def solve_nonlinear(payload: dict[str, Any]) -> Result:
    variables = payload.get("variables", [])
    if not isinstance(variables, list) or not variables:
        return Result("invalid", message="missing variables")
    names: list[str] = []
    domains: list[list[float]] = []
    for index, var in enumerate(variables):
        name = str(var.get("name", f"x{index}"))
        lb = as_number(var.get("lb", -5.0))
        ub = as_number(var.get("ub", 5.0))
        if ub < lb:
            return Result("infeasible", message="empty nonlinear domain")
        names.append(name)
        midpoint = 0.5 * (lb + ub)
        domains.append(sorted({lb, ub, midpoint, as_number(var.get("start", midpoint))}))

    objective_expr = str(payload.get("objective", "0"))
    constraints = payload.get("constraints", [])
    best_x: tuple[float, ...] | None = None
    best_obj: float | None = None
    best_violation = math.inf
    for point in itertools.product(*domains):
        env = {name: value for name, value in zip(names, point)}
        violation = 0.0
        for row in constraints:
            lhs = eval_expr(str(row.get("expr", "0")), env)
            rhs = as_number(row.get("rhs"))
            sense = str(row.get("sense", "<="))
            if sense in {"<=", "le"}:
                violation = max(violation, lhs - rhs)
            elif sense in {">=", "ge"}:
                violation = max(violation, rhs - lhs)
            elif sense in {"=", "==", "eq"}:
                violation = max(violation, abs(lhs - rhs))
            else:
                return Result("invalid", message=f"unsupported nonlinear sense {sense!r}")
        objective = eval_expr(objective_expr, env)
        if violation <= 1e-7 and better(objective, best_obj, str(payload.get("sense", "min")).lower()):
            best_obj = objective
            best_x = tuple(point)
        best_violation = min(best_violation, violation)

    if best_x is None or best_obj is None:
        return Result("infeasible", message=f"best constraint violation {best_violation:.3e}")
    return Result("optimal", best_obj, [float(v) for v in best_x])


def solve(tool: str, payload: dict[str, Any]) -> tuple[str, Result]:
    kind = str(payload.get("kind", ""))
    family = tool_family(tool, kind)
    if family == "constraint-programming":
        return family, solve_cp_assignment(payload)
    if family == "smt-omt":
        return family, solve_cp_assignment(payload)
    if family == "planning-metaheuristic":
        return family, solve_planning_assignment(payload)
    if family == "evolutionary-multiobjective":
        return family, solve_multiobjective(payload)
    if family == "convex-optimization":
        if kind in {"nonlinear-program", "ecosystem-nonlinear"}:
            return family, solve_nonlinear(payload)
        return family, solve_discrete_linear(payload)
    if family == "native-solver-binding":
        return family, solve_discrete_linear(payload)
    if family == "nonlinear-optimization":
        return family, solve_nonlinear(payload)
    if family == "hybrid-optimization":
        if kind in {"planning-assignment", "ecosystem-planning-assignment"}:
            return family, solve_planning_assignment(payload)
        if kind in {"nonlinear-program", "ecosystem-nonlinear"}:
            return family, solve_nonlinear(payload)
        return family, solve_discrete_linear(payload)
    return family, solve_discrete_linear(payload)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool", default=None)
    args = parser.parse_args()
    tool = arg_tool(args.tool)
    try:
        payload = json.load(sys.stdin)
        if not isinstance(payload, dict):
            raise ValueError("top-level payload must be an object")
        family, result = solve(tool, payload)
        emit(tool, family, result)
        return 0
    except Exception as exc:  # noqa: BLE001 - CLI adapter must emit JSON even on bad payloads.
        emit(tool, "unknown", Result("invalid", message=str(exc)))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
