#!/usr/bin/env python3
"""External solver oracle for des_engine math-program cross-checks."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from typing import Any


COMMAND_ALIASES = {
    "glpk": ["glpsol"],
    "highs": ["highs"],
    "scip": ["scip"],
    "cbc": ["cbc"],
    "clp": ["clp"],
    "soplex": ["soplex"],
    "lp-solve": ["lp_solve", "lp-solve", "lpsolve"],
}

COMMAND_ENV_VARS = {
    "glpk": ["GLPSOL_CMD", "GLPK_CMD", "ORES_GLPK_CMD", "ORES_GLPK_BIN"],
    "highs": ["HIGHS_CMD", "ORES_HIGHS_CMD", "ORES_HIGHS_BIN"],
    "scip": ["SCIP_CMD", "ORES_SCIP_CMD", "ORES_SCIP_BIN"],
    "cbc": ["CBC_CMD", "ORES_CBC_CMD", "ORES_CBC_BIN"],
    "clp": ["CLP_CMD", "ORES_CLP_CMD", "ORES_CLP_BIN"],
    "soplex": ["SOPLEX_CMD", "ORES_SOPLEX_CMD", "ORES_SOPLEX_BIN"],
    "lp-solve": ["LP_SOLVE_CMD", "LPSOLVE_CMD", "ORES_LP_SOLVE_CMD", "ORES_LPSOLVE_BIN"],
}

COMMERCIAL_LINEAR_CLI_SOLVERS = {"gurobi", "cplex", "xpress", "lindo"}
LINEAR_CLI_BACKEND_SOLVERS = COMMERCIAL_LINEAR_CLI_SOLVERS | {
    "highs",
    "glpk",
    "scip",
    "cbc",
    "clp",
    "soplex",
    "lp-solve",
}
LINEAR_CLI_DIRECT_ALIASES = {
    "highs": "highs",
    "highs-cli": "highs",
    "glpsol": "glpk",
    "glpk-cli": "glpk",
    "scip": "scip",
    "scip-cli": "scip",
    "cbc": "cbc",
    "cbc-cli": "cbc",
    "clp": "clp",
    "clp-cli": "clp",
    "soplex": "soplex",
    "soplex-cli": "soplex",
    "lp-solve": "lp-solve",
    "lp-solve-cli": "lp-solve",
    "lpsolve": "lp-solve",
    "lpsolve-cli": "lp-solve",
}


def _command_for(solver: str) -> str:
    for env_var in COMMAND_ENV_VARS.get(solver, []):
        value = os.environ.get(env_var)
        if value:
            return value
    for alias in COMMAND_ALIASES.get(solver, [solver]):
        path = shutil.which(alias)
        if path is not None:
            return path
    return COMMAND_ALIASES.get(solver, [solver])[0]


def _clean(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, list):
        return [_clean(item) for item in value]
    if isinstance(value, tuple):
        return [_clean(item) for item in value]
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    return value


def _status(code: int) -> str:
    if code == 0:
        return "optimal"
    if code == 1:
        return "iter-limit"
    if code == 2:
        return "infeasible"
    if code == 3:
        return "unbounded"
    return "numerical-error"


def _lp_bounds(lp: dict[str, Any]) -> list[tuple[float | None, float | None]]:
    n = len(lp.get("c", []))
    lower = lp.get("lb")
    upper = lp.get("ub")
    bounds = []
    for i in range(n):
        lo = lower[i] if lower is not None and i < len(lower) else 0.0
        hi = upper[i] if upper is not None and i < len(upper) else None
        bounds.append((lo, hi))
    return bounds


def _marginals(section: Any) -> list[float] | None:
    if section is None:
        return None
    values = getattr(section, "marginals", None)
    if values is None:
        return None
    return [float(v) for v in values]


def _clean_certificate_value(value: float) -> float:
    return 0.0 if abs(value) <= 1e-8 else float(value)


def _lp_row_counts(problem: dict[str, Any]) -> tuple[int, int]:
    return len(problem.get("A_ub") or []), len(problem.get("A_eq") or [])


def _lp_certificate_fields(
    row_duals: list[float] | None,
    reduced_costs: list[float] | None,
    ub_count: int,
    eq_count: int,
) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    if row_duals is not None and len(row_duals) >= ub_count + eq_count:
        fields["dualUB"] = [
            _clean_certificate_value(value) for value in row_duals[:ub_count]
        ]
        fields["dualEQ"] = [
            _clean_certificate_value(value)
            for value in row_duals[ub_count : ub_count + eq_count]
        ]
    if reduced_costs is not None:
        fields["reducedCosts"] = [
            _clean_certificate_value(value) for value in reduced_costs
        ]
    return fields


def _lp_basis_fields(
    var_basis: list[str | None] | None,
    row_basis: list[str | None] | None,
) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    if var_basis is not None and all(status is not None for status in var_basis):
        fields["varBasis"] = list(var_basis)
    if row_basis is not None and all(status is not None for status in row_basis):
        fields["rowBasis"] = list(row_basis)
    return fields


def _finite_float_or_none(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _nonnegative_int_or_none(value: Any) -> int | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric) or numeric < 0:
        return None
    return int(round(numeric))


def _relative_gap(best_bound: Any, objective: Any) -> float | None:
    bound = _finite_float_or_none(best_bound)
    incumbent = _finite_float_or_none(objective)
    if bound is None or incumbent is None:
        return None
    return abs(bound - incumbent) / max(1.0, abs(incumbent))


def _quality_fields(
    best_bound: Any = None,
    mip_gap: Any = None,
    nodes_explored: Any = None,
) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    bound = _finite_float_or_none(best_bound)
    gap = _finite_float_or_none(mip_gap)
    nodes = _nonnegative_int_or_none(nodes_explored)
    if bound is not None:
        fields["bestBound"] = bound
    if gap is not None:
        fields["mipGap"] = gap
    if nodes is not None:
        fields["nodesExplored"] = nodes
    return fields


def _first_attr(obj: Any, names: tuple[str, ...]) -> Any:
    for name in names:
        try:
            return getattr(obj, name)
        except Exception:
            continue
    return None


def _lp_reduced_costs_from_row_duals(
    problem: dict[str, Any], row_duals: list[float]
) -> list[float] | None:
    if problem.get("sense", "max") != "max":
        return None
    rows = [
        [float(value) for value in row]
        for row in (problem.get("A_ub") or []) + (problem.get("A_eq") or [])
    ]
    if len(row_duals) < len(rows):
        return None
    reduced_costs = []
    for col, coeff in enumerate(problem.get("c", [])):
        row_part = 0.0
        for row, dual in zip(rows, row_duals):
            if col < len(row):
                row_part += row[col] * dual
        reduced_costs.append(float(coeff) - row_part)
    return reduced_costs


def _basis_status_from_token(token: Any) -> str | None:
    text = str(token).strip().lower()
    highs_codes = {
        "0": "at_lower",
        "1": "basic",
        "2": "at_upper",
        "3": "zero",
        "4": "nonbasic",
    }
    glpk_codes = {
        "b": "basic",
        "bs": "basic",
        "l": "at_lower",
        "nl": "at_lower",
        "u": "at_upper",
        "nu": "at_upper",
        "f": "free",
        "nf": "free",
        "s": "fixed",
        "ns": "fixed",
    }
    named = {
        "basic": "basic",
        "lower": "at_lower",
        "at_lower": "at_lower",
        "upper": "at_upper",
        "at_upper": "at_upper",
        "zero": "zero",
        "nonbasic": "nonbasic",
        "free": "free",
        "fixed": "fixed",
        "superbasic": "superbasic",
    }
    return highs_codes.get(text) or glpk_codes.get(text) or named.get(text)


def _glpk_basis_status_from_code(glp: Any, code: int) -> str | None:
    return {
        glp.GLP_BS: "basic",
        glp.GLP_NL: "at_lower",
        glp.GLP_NU: "at_upper",
        glp.GLP_NF: "free",
        glp.GLP_NS: "fixed",
    }.get(code)


def _ortools_basis_status(pywraplp: Any, code: int) -> str | None:
    return {
        int(pywraplp.Solver.FREE): "free",
        int(pywraplp.Solver.AT_LOWER_BOUND): "at_lower",
        int(pywraplp.Solver.AT_UPPER_BOUND): "at_upper",
        int(pywraplp.Solver.FIXED_VALUE): "fixed",
        int(pywraplp.Solver.BASIC): "basic",
    }.get(code)


def _gurobi_var_basis_status(code: int) -> str | None:
    return {0: "basic", -1: "at_lower", -2: "at_upper", -3: "superbasic"}.get(
        code
    )


def _gurobi_row_basis_status(code: int) -> str | None:
    return {0: "basic", -1: "at_upper"}.get(code)


def _cplex_var_basis_status(status: Any, code: int) -> str | None:
    return {
        int(status.basic): "basic",
        int(status.at_lower_bound): "at_lower",
        int(status.at_upper_bound): "at_upper",
        int(status.free_nonbasic): "free",
    }.get(code)


def _cplex_row_basis_status(status: Any, code: int) -> str | None:
    return {
        int(status.basic): "basic",
        int(status.at_lower_bound): "at_upper",
        int(status.at_upper_bound): "at_lower",
        int(status.free_nonbasic): "free",
    }.get(code)


def _xpress_var_basis_status(code: int) -> str | None:
    return {1: "basic", 0: "at_lower", 2: "at_upper", 3: "superbasic"}.get(code)


def _xpress_row_basis_status(code: int) -> str | None:
    return {1: "basic", 0: "at_upper", 2: "at_lower", 3: "superbasic"}.get(code)


def _lp_row_basis_with_fixed_equalities(
    ub_statuses: list[str | None], eq_count: int
) -> list[str | None]:
    return ub_statuses + ["fixed"] * eq_count


def _solver_backend(method: str) -> tuple[str, str]:
    if ":" in method:
        family, backend = method.split(":", 1)
        return family.lower(), backend
    return "scipy", method


def _external_options(payload: dict[str, Any]) -> dict[str, Any]:
    options = payload.get("options")
    return options if isinstance(options, dict) else {}


def _linear_cli_reference_script() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    repo_relative = os.path.abspath(
        os.path.join(here, "..", "..", "scripts", "linear_cli_reference.py")
    )
    if os.path.exists(repo_relative):
        return repo_relative
    return os.path.abspath(os.path.join(os.getcwd(), "scripts", "linear_cli_reference.py"))


def _time_limit_seconds(options: dict[str, Any], default: float | None = None) -> float | None:
    value = _finite_float_or_none(options.get("timeLimitMs"))
    if value is None:
        return default
    return max(value / 1000.0, 1e-9)


def _time_limit_seconds_text(options: dict[str, Any], default: float = 60.0) -> str:
    return f"{_time_limit_seconds(options, default):.17g}"


def _time_limit_integer_seconds_text(options: dict[str, Any], default: int = 60) -> str:
    seconds = _time_limit_seconds(options)
    if seconds is None:
        return str(default)
    return str(max(1, int(math.ceil(seconds))))


def _node_limit(options: dict[str, Any]) -> int | None:
    value = _nonnegative_int_or_none(options.get("nodeLimit"))
    if value is None or value <= 0:
        return None
    return value


def _relative_gap_limit(options: dict[str, Any]) -> float | None:
    value = _finite_float_or_none(options.get("relativeGap"))
    if value is None or value < 0.0:
        return None
    return value


def _scipy_milp_options(options: dict[str, Any]) -> dict[str, Any]:
    scipy_options: dict[str, Any] = {}
    time_limit = _time_limit_seconds(options)
    node_limit = _node_limit(options)
    relative_gap = _relative_gap_limit(options)
    if time_limit is not None:
        scipy_options["time_limit"] = time_limit
    if node_limit is not None:
        scipy_options["node_limit"] = node_limit
    if relative_gap is not None:
        scipy_options["mip_rel_gap"] = relative_gap
    return scipy_options


def _scipy_mip_status(code: int, options: dict[str, Any]) -> str:
    status = _status(code)
    if code == 1:
        if _time_limit_seconds(options) is not None:
            return "time-limit"
        if _node_limit(options) is not None:
            return "node-limit"
    return status


def _limited_status(status: str, options: dict[str, Any]) -> str:
    if status == "iter-limit":
        if _time_limit_seconds(options) is not None:
            return "time-limit"
        if _node_limit(options) is not None:
            return "node-limit"
    return status


def _integer_value(value: Any, name: str) -> int:
    numeric = float(value)
    rounded = round(numeric)
    if not math.isfinite(numeric) or abs(numeric - rounded) > 1e-9:
        raise RuntimeError(f"CP-SAT oracle requires integer-scaled {name}, got {value}")
    return int(rounded)


def _mip_start(problem: dict[str, Any]) -> list[float] | None:
    start = problem.get("mipStart")
    n = len(problem.get("c", []))
    if not isinstance(start, list) or len(start) != n:
        return None
    values = [float(value) for value in start]
    if not all(math.isfinite(value) for value in values):
        return None
    return values


def _mip_rows(problem: dict[str, Any]) -> list[tuple[list[float], float]]:
    rows = [
        ([float(v) for v in row], float(bound))
        for row, bound in zip(problem.get("A", []), problem.get("b", []))
    ]
    for lazy in problem.get("lazyConstraints") or []:
        rows.append((
            [float(v) for v in lazy.get("coefs", [])],
            float(lazy.get("rhs", 0.0)),
        ))
    return rows


def _mip_row_arrays(problem: dict[str, Any]) -> tuple[list[list[float]], list[float]]:
    rows = _mip_rows(problem)
    return [row for row, _ in rows], [bound for _, bound in rows]


def _mip_for_linear_cli_bridge(problem: dict[str, Any]) -> dict[str, Any]:
    bridge_problem = dict(problem)
    if "A" in problem and "a" not in bridge_problem:
        bridge_problem["a"] = problem["A"]
    if "integerVars" in problem and "integer_vars" not in bridge_problem:
        bridge_problem["integer_vars"] = problem["integerVars"]

    rows = list(bridge_problem.get("a") or [])
    rhs = list(bridge_problem.get("b") or [])
    for lazy in problem.get("lazyConstraints") or []:
        rows.append(lazy.get("coefs", []))
        rhs.append(lazy.get("rhs", 0.0))
    bridge_problem["a"] = rows
    bridge_problem["b"] = rhs
    return bridge_problem


def solve_linear_cli_bridge(
    problem: dict[str, Any],
    kind: str,
    solver: str,
    options: dict[str, Any],
) -> dict[str, Any]:
    raw = {"lp": problem} if kind == "lp" else _mip_for_linear_cli_bridge(problem)
    time_limit = _time_limit_seconds(options, 60.0) or 60.0
    commands = [
        sys.executable,
        _linear_cli_reference_script(),
        "--kind",
        kind,
        "--solver",
        solver,
        "--time-limit",
        f"{time_limit:.17g}",
    ]

    passthrough = [
        ("nodeLimit", "--node-limit", int),
        ("solutionLimit", "--solution-limit", int),
        ("solutionPoolSize", "--solution-pool-size", int),
        ("relativeGap", "--relative-gap", float),
        ("absoluteGap", "--absolute-gap", float),
        ("objectiveLimit", "--objective-limit", float),
        ("threads", "--threads", int),
        ("randomSeed", "--random-seed", int),
        ("presolve", "--presolve", str),
        ("cuts", "--cuts", str),
        ("heuristics", "--heuristics", str),
        ("branchRule", "--branch-rule", str),
        ("nodeSelection", "--node-selection", str),
    ]
    for key, flag, cast in passthrough:
        if key in options and options[key] is not None:
            commands.extend([flag, str(cast(options[key]))])
    if isinstance(options.get("branchPriorities"), list):
        commands.extend(["--branch-priorities", json.dumps(options["branchPriorities"])])
    if isinstance(problem.get("mipStart"), list):
        commands.extend(["--mip-start", json.dumps(problem["mipStart"])])

    try:
        run = subprocess.run(
            commands,
            input=json.dumps(raw, allow_nan=True),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=max(time_limit + 15.0, 20.0),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "status": "time-limit",
            "x": [],
            "objective": None,
            "message": f"{solver}:cli bridge timed out: {exc}",
        }

    if run.returncode != 0:
        return {
            "status": "numerical-error",
            "x": [],
            "objective": None,
            "message": f"{solver}:cli bridge exited with {run.returncode}: {run.stderr.strip()}",
        }

    lines = [line for line in run.stdout.splitlines() if line.strip()]
    if not lines:
        return {
            "status": "numerical-error",
            "x": [],
            "objective": None,
            "message": f"{solver}:cli bridge produced no JSON: {run.stderr.strip()}",
        }
    result = json.loads(lines[-1])
    if result.get("status") == "unavailable":
        result["status"] = "numerical-error"
    return result


def _linear_cli_bridge_solver(family: str, backend: str, integer: bool) -> str | None:
    solver = LINEAR_CLI_DIRECT_ALIASES.get(family)
    if solver is None and backend == "cli" and family in LINEAR_CLI_BACKEND_SOLVERS:
        solver = family
    if solver == "clp" and integer:
        return None
    return solver


def solve_lp(payload: dict[str, Any], method: str) -> dict[str, Any]:
    family, backend = _solver_backend(method)
    if family == "ortools":
        return solve_ortools(payload, backend, integer=False)
    cli_solver = _linear_cli_bridge_solver(family, backend, integer=False)
    if cli_solver is not None:
        return solve_linear_cli_bridge(
            payload["lp"], "lp", cli_solver, _external_options(payload)
        )
    if family == "gurobi":
        return solve_gurobi_lp(payload)
    if family == "cplex":
        return solve_cplex_lp(payload)
    if family == "xpress":
        return solve_xpress_lp(payload)
    if family == "glpk":
        return solve_glpk_lp(payload)

    from scipy.optimize import linprog

    lp = payload["lp"]
    sense = lp.get("sense", "max")
    c = [float(v) for v in lp.get("c", [])]
    scipy_c = [-v for v in c] if sense == "max" else c
    result = linprog(
        scipy_c,
        A_ub=lp.get("A_ub"),
        b_ub=lp.get("b_ub"),
        A_eq=lp.get("A_eq"),
        b_eq=lp.get("b_eq"),
        bounds=_lp_bounds(lp),
        method=method,
    )
    objective = None
    if result.fun is not None and math.isfinite(float(result.fun)):
        objective = -float(result.fun) if sense == "max" else float(result.fun)
    sign = -1.0 if sense == "max" else 1.0
    dual_ub = _marginals(getattr(result, "ineqlin", None))
    dual_eq = _marginals(getattr(result, "eqlin", None))
    if dual_ub is not None:
        dual_ub = [sign * v for v in dual_ub]
    if dual_eq is not None:
        dual_eq = [sign * v for v in dual_eq]
    lower = _marginals(getattr(result, "lower", None))
    upper = _marginals(getattr(result, "upper", None))
    reduced_costs = None
    if lower is not None and upper is not None:
        reduced_costs = [sign * (lo + hi) for lo, hi in zip(lower, upper)]
    return {
        "status": _status(int(result.status)),
        "x": [float(v) for v in result.x] if result.x is not None else [],
        "objective": objective,
        "dualUB": dual_ub,
        "dualEQ": dual_eq,
        "reducedCosts": reduced_costs,
        "message": str(result.message),
    }


def solve_qp(payload: dict[str, Any], method: str) -> dict[str, Any]:
    family, _backend = _solver_backend(method)
    if family == "ortools":
        raise RuntimeError("OR-Tools linear_solver does not expose a continuous QP oracle")
    if family == "glpk":
        raise RuntimeError("GLPK oracle supports LP and MIP models, not QP")
    if family == "gurobi":
        return solve_gurobi_qp(payload)
    if family == "cplex":
        return solve_cplex_qp(payload)
    if family == "xpress":
        return solve_xpress_qp(payload)

    try:
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, minimize
    except Exception as exc:
        raise RuntimeError(f"scipy QP unavailable: {exc}") from exc

    qp = payload["qp"]
    if any(bool(value) for value in qp.get("integerVars", [])):
        raise RuntimeError("SciPy QP oracle does not support integer variables")
    sense = qp.get("sense", "max")
    c = np.array([float(v) for v in qp.get("c", [])], dtype=float)
    sign = -1.0 if sense == "max" else 1.0
    hessian = np.zeros((len(c), len(c)), dtype=float)
    for term in qp.get("quadratic", []):
        i = int(term["i"])
        j = int(term["j"])
        coeff = float(term["coeff"])
        if i == j:
            hessian[i, i] += 2.0 * coeff
        else:
            hessian[i, j] += coeff
            hessian[j, i] += coeff

    def objective(x: Any) -> float:
        return float(sign * (np.dot(c, x) + 0.5 * np.dot(x, hessian @ x)))

    def gradient(x: Any) -> Any:
        return sign * (c + hessian @ x)

    lower = qp.get("lb") or [None] * len(c)
    upper = qp.get("ub") or [None] * len(c)
    bounds = Bounds(
        [
            -math.inf if value is None else float(value)
            for value in lower
        ],
        [
            math.inf if value is None else float(value)
            for value in upper
        ],
    )
    x0 = []
    for lo, hi in zip(bounds.lb, bounds.ub):
        if math.isfinite(lo) and math.isfinite(hi):
            x0.append(0.5 * (lo + hi))
        elif math.isfinite(lo):
            x0.append(lo)
        elif math.isfinite(hi):
            x0.append(hi)
        else:
            x0.append(0.0)
    constraints = []
    if qp.get("A_ub"):
        a_ub = np.array(qp.get("A_ub"), dtype=float)
        b_ub = np.array(qp.get("b_ub"), dtype=float)
        constraints.append(LinearConstraint(a_ub, -math.inf * np.ones(len(b_ub)), b_ub))
    if qp.get("A_eq"):
        a_eq = np.array(qp.get("A_eq"), dtype=float)
        b_eq = np.array(qp.get("b_eq"), dtype=float)
        constraints.append(LinearConstraint(a_eq, b_eq, b_eq))

    scipy_method = "SLSQP" if method.lower().startswith("highs") else method
    result = minimize(
        objective,
        np.array(x0, dtype=float),
        jac=gradient,
        bounds=bounds,
        constraints=constraints,
        method=scipy_method,
    )
    status = "optimal" if bool(result.success) else "numerical-error"
    objective_value = None
    if result.fun is not None and math.isfinite(float(result.fun)):
        objective_value = float(result.fun) / sign
    return {
        "status": status,
        "x": [float(v) for v in result.x] if result.x is not None else [],
        "objective": objective_value,
        "message": str(result.message),
    }


def solve_conic(payload: dict[str, Any], method: str) -> dict[str, Any]:
    family, _backend = _solver_backend(method)
    if family == "ortools":
        raise RuntimeError("OR-Tools linear_solver does not expose a continuous conic oracle")
    if family == "glpk":
        raise RuntimeError("GLPK oracle supports LP and MIP models, not conic models")
    if family == "gurobi":
        return solve_gurobi_conic(payload)
    if family == "cplex":
        return solve_cplex_conic(payload)
    if family == "xpress":
        return solve_xpress_conic(payload)

    try:
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint, minimize
    except Exception as exc:
        raise RuntimeError(f"scipy conic unavailable: {exc}") from exc

    conic = payload["conic"]
    if any(bool(value) for value in conic.get("integerVars", [])):
        raise RuntimeError("SciPy conic oracle does not support integer variables")
    sense = conic.get("sense", "max")
    c = np.array([float(v) for v in conic.get("c", [])], dtype=float)
    sign = -1.0 if sense == "max" else 1.0
    hessian = np.zeros((len(c), len(c)), dtype=float)
    for term in conic.get("quadratic", []):
        i = int(term["i"])
        j = int(term["j"])
        coeff = float(term["coeff"])
        if i == j:
            hessian[i, i] += 2.0 * coeff
        else:
            hessian[i, j] += coeff
            hessian[j, i] += coeff

    def objective(x: Any) -> float:
        return float(sign * (np.dot(c, x) + 0.5 * np.dot(x, hessian @ x)))

    def gradient(x: Any) -> Any:
        return sign * (c + hessian @ x)

    lower = conic.get("lb") or [None] * len(c)
    upper = conic.get("ub") or [None] * len(c)
    bounds = Bounds(
        [-math.inf if value is None else float(value) for value in lower],
        [math.inf if value is None else float(value) for value in upper],
    )
    x0 = []
    for lo, hi in zip(bounds.lb, bounds.ub):
        if math.isfinite(lo) and math.isfinite(hi):
            x0.append(0.5 * (lo + hi))
        elif math.isfinite(lo):
            x0.append(lo)
        elif math.isfinite(hi):
            x0.append(hi)
        else:
            x0.append(0.0)

    constraints = []
    if conic.get("A_ub"):
        a_ub = np.array(conic.get("A_ub"), dtype=float)
        b_ub = np.array(conic.get("b_ub"), dtype=float)
        constraints.append(LinearConstraint(a_ub, -math.inf * np.ones(len(b_ub)), b_ub))
    if conic.get("A_eq"):
        a_eq = np.array(conic.get("A_eq"), dtype=float)
        b_eq = np.array(conic.get("b_eq"), dtype=float)
        constraints.append(LinearConstraint(a_eq, b_eq, b_eq))

    def sparse_value(coeffs: Any, constant: float, x: Any) -> float:
        return float(constant + sum(float(coef) * x[int(idx)] for idx, coef in coeffs))

    for cone in conic.get("soc", []):
        terms = cone.get("terms", [])
        rhs_coeffs = cone.get("rhsCoeffs", [])
        rhs_constant = float(cone.get("rhsConstant", 0.0))

        def cone_slack(x: Any, terms: Any = terms, rhs_coeffs: Any = rhs_coeffs, rhs_constant: float = rhs_constant) -> float:
            values = [
                sparse_value(term.get("coeffs", []), float(term.get("constant", 0.0)), x)
                for term in terms
            ]
            rhs = sparse_value(rhs_coeffs, rhs_constant, x)
            return float(rhs - np.linalg.norm(np.array(values, dtype=float)))

        constraints.append(NonlinearConstraint(cone_slack, 0.0, math.inf))

    for row in conic.get("quadraticConstraints", []):
        quadratic = row.get("quadratic", [])
        linear = row.get("linear", [])
        rhs = float(row.get("rhs", 0.0))
        row_sense = row.get("sense", "<=")

        def quadratic_expr(
            x: Any,
            quadratic: Any = quadratic,
            linear: Any = linear,
        ) -> float:
            value = 0.0
            for term in quadratic:
                value += (
                    float(term["coeff"])
                    * x[int(term["i"])]
                    * x[int(term["j"])]
                )
            value += sum(float(coef) * x[int(idx)] for idx, coef in linear)
            return float(value)

        def quadratic_slack(
            x: Any,
            row_sense: str = row_sense,
            rhs: float = rhs,
            quadratic_expr: Any = quadratic_expr,
        ) -> float:
            expr = quadratic_expr(x)
            if row_sense == ">=":
                return float(expr - rhs)
            if row_sense in {"=", "=="}:
                return float(-abs(expr - rhs))
            return float(rhs - expr)

        def quadratic_jac(
            x: Any,
            row_sense: str = row_sense,
            rhs: float = rhs,
            quadratic: Any = quadratic,
            linear: Any = linear,
            quadratic_expr: Any = quadratic_expr,
        ) -> Any:
            jac = np.zeros(len(c), dtype=float)
            for term in quadratic:
                i = int(term["i"])
                j = int(term["j"])
                coeff = float(term["coeff"])
                if i == j:
                    jac[i] += 2.0 * coeff * x[i]
                else:
                    jac[i] += coeff * x[j]
                    jac[j] += coeff * x[i]
            for idx, coef in linear:
                jac[int(idx)] += float(coef)
            if row_sense == ">=":
                return jac
            if row_sense in {"=", "=="}:
                expr = quadratic_expr(x)
                if abs(expr - rhs) <= 1e-12:
                    return np.zeros(len(c), dtype=float)
                return -math.copysign(1.0, expr - rhs) * jac
            return -jac

        constraints.append(NonlinearConstraint(quadratic_slack, 0.0, math.inf, jac=quadratic_jac))

    scipy_method = "SLSQP" if method.lower().startswith("highs") else method
    result = minimize(
        objective,
        np.array(x0, dtype=float),
        jac=gradient,
        bounds=bounds,
        constraints=constraints,
        method=scipy_method,
    )
    status = "optimal" if bool(result.success) else "numerical-error"
    objective_value = None
    if result.fun is not None and math.isfinite(float(result.fun)):
        objective_value = float(result.fun) / sign
    return {
        "status": status,
        "x": [float(v) for v in result.x] if result.x is not None else [],
        "objective": objective_value,
        "message": str(result.message),
    }


def solve_mip(payload: dict[str, Any], method: str) -> dict[str, Any]:
    family, backend = _solver_backend(method)
    if family == "ortools":
        normalized_backend = backend.upper().replace("_", "-")
        if normalized_backend in {"CP-SAT", "CPSAT"}:
            return solve_ortools_cp_sat(payload)
        return solve_ortools(payload, backend, integer=True)
    cli_solver = _linear_cli_bridge_solver(family, backend, integer=True)
    if cli_solver is not None:
        return solve_linear_cli_bridge(
            payload["mip"], "mip", cli_solver, _external_options(payload)
        )
    if family == "gurobi":
        return solve_gurobi_mip(payload)
    if family == "cplex":
        return solve_cplex_mip(payload)
    if family == "xpress":
        return solve_xpress_mip(payload)
    if family == "glpk":
        return solve_glpk_mip(payload)

    try:
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, milp
    except Exception as exc:
        raise RuntimeError(f"scipy milp unavailable: {exc}") from exc

    mip = payload["mip"]
    sense = mip.get("sense", "max")
    c = np.array([float(v) for v in mip.get("c", [])], dtype=float)
    scipy_c = -c if sense == "max" else c
    rows, bounds = _mip_row_arrays(mip)
    a = np.array(rows, dtype=float)
    b = np.array(bounds, dtype=float)
    lower = np.zeros(len(c), dtype=float)
    upper = np.array(
        [
            math.inf if value is None or not math.isfinite(float(value)) else float(value)
            for value in (mip.get("ub") or [math.inf] * len(c))
        ],
        dtype=float,
    )
    constraints = LinearConstraint(a, -math.inf * np.ones(len(b)), b)
    integrality = np.array([1 if v else 0 for v in mip.get("integerVars", [])], dtype=int)
    external_options = _external_options(payload)
    milp_options = _scipy_milp_options(external_options)
    result = milp(
        c=scipy_c,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=constraints,
        options=milp_options or None,
    )
    objective = None
    if result.fun is not None and math.isfinite(float(result.fun)):
        objective = -float(result.fun) if sense == "max" else float(result.fun)
    best_bound = getattr(result, "mip_dual_bound", None)
    if best_bound is not None and sense == "max":
        best_bound = -float(best_bound)
    parsed = {
        "status": _scipy_mip_status(int(result.status), external_options),
        "x": [float(v) for v in result.x] if result.x is not None else [],
        "objective": objective,
        "message": str(result.message),
    }
    parsed.update(
        _quality_fields(
            best_bound=best_bound,
            mip_gap=getattr(result, "mip_gap", None),
            nodes_explored=getattr(result, "mip_node_count", None),
        )
    )
    return parsed


def solve_ortools_cp_sat(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        from ortools.sat.python import cp_model
    except Exception as exc:
        raise RuntimeError(f"OR-Tools CP-SAT unavailable: {exc}") from exc

    mip = payload["mip"]
    c = [_integer_value(v, f"objective coefficient {i}") for i, v in enumerate(mip.get("c", []))]
    integer_vars = [bool(v) for v in mip.get("integerVars", [])]
    if len(integer_vars) != len(c) or not all(integer_vars):
        raise RuntimeError("CP-SAT oracle requires every compiled variable to be integer")

    upper = mip.get("ub") or [None] * len(c)
    model = cp_model.CpModel()
    variables = []
    for i in range(len(c)):
        if i >= len(upper) or upper[i] is None:
            raise RuntimeError(f"CP-SAT oracle requires finite upper bound for variable {i}")
        hi = _integer_value(upper[i], f"upper bound {i}")
        if hi < 0:
            raise RuntimeError(f"CP-SAT oracle requires non-negative upper bound for variable {i}")
        variables.append(model.NewIntVar(0, hi, f"x{i}"))

    for row_index, (row, bound) in enumerate(_mip_rows(mip)):
        terms = []
        for var_index, (var, coef) in enumerate(zip(variables, row)):
            coef_i = _integer_value(coef, f"row {row_index} coefficient {var_index}")
            if coef_i:
                terms.append(coef_i * var)
        model.Add(sum(terms) <= _integer_value(bound, f"row {row_index} bound"))

    objective_terms = [coef * var for coef, var in zip(c, variables) if coef]
    if mip.get("sense", "max") == "max":
        model.Maximize(sum(objective_terms))
    else:
        model.Minimize(sum(objective_terms))
    start = _mip_start(mip)
    if start is not None:
        for i, (var, value) in enumerate(zip(variables, start)):
            model.AddHint(var, _integer_value(value, f"MIP start {i}"))

    solver = cp_model.CpSolver()
    options = _external_options(payload)
    time_limit = _time_limit_seconds(options)
    node_limit = _node_limit(options)
    relative_gap = _relative_gap_limit(options)
    if time_limit is not None:
        solver.parameters.max_time_in_seconds = time_limit
    if node_limit is not None:
        solver.parameters.max_number_of_conflicts = node_limit
    if relative_gap is not None:
        solver.parameters.relative_gap_limit = relative_gap
    status_code = solver.Solve(model)
    status = {
        cp_model.OPTIMAL: "optimal",
        cp_model.FEASIBLE: "iter-limit",
        cp_model.INFEASIBLE: "infeasible",
        cp_model.MODEL_INVALID: "numerical-error",
        cp_model.UNKNOWN: "numerical-error",
    }.get(status_code, "numerical-error")
    status = _limited_status(status, options)
    x = [float(solver.Value(var)) for var in variables] if status in {"optimal", "iter-limit"} else []
    objective_value = float(solver.ObjectiveValue()) if status in {"optimal", "iter-limit"} else None
    parsed = {
        "status": status,
        "x": x,
        "objective": objective_value,
        "message": f"ortools:CP-SAT status={status_code}",
    }
    if status in {"optimal", "iter-limit"}:
        best_bound = float(solver.BestObjectiveBound())
        parsed.update(
            _quality_fields(
                best_bound=best_bound,
                mip_gap=_relative_gap(best_bound, objective_value),
                nodes_explored=solver.NumBranches(),
            )
        )
    return parsed


def _cli_bounds(problem: dict[str, Any], n: int) -> list[tuple[float | None, float | None]]:
    lower = problem.get("lb")
    upper = problem.get("ub")
    bounds = []
    for i in range(n):
        lo = lower[i] if lower is not None and i < len(lower) else 0.0
        hi = upper[i] if upper is not None and i < len(upper) else None
        bounds.append((None if lo is None else float(lo), None if hi is None else float(hi)))
    return bounds


def _lp_number(value: float) -> str:
    if not math.isfinite(value):
        raise RuntimeError(f"LP writer cannot encode non-finite value {value}")
    return f"{value:.17g}"


def _lp_expr(coeffs: list[float], names: list[str]) -> str:
    parts = []
    for idx, coef in enumerate(coeffs):
        coef = float(coef)
        if abs(coef) <= 1e-12:
            continue
        sign = "+" if coef >= 0.0 else "-"
        parts.append(f" {sign} {_lp_number(abs(coef))} {names[idx]}")
    if not parts:
        return f"0 {names[0]}"
    expr = "".join(parts).strip()
    return expr[2:] if expr.startswith("+ ") else expr


def _write_lp_file(problem: dict[str, Any], integer: bool, path: str) -> list[str]:
    c = [float(v) for v in problem.get("c", [])]
    names = [f"x{i}" for i in range(len(c))]
    rows: list[tuple[list[float], str, float]] = []
    if problem.get("A") is not None:
        for row, bound in _mip_rows(problem):
            rows.append(([float(v) for v in row], "<=", float(bound)))
    else:
        for row, bound in zip(problem.get("A_ub") or [], problem.get("b_ub") or []):
            rows.append(([float(v) for v in row], "<=", float(bound)))
        for row, bound in zip(problem.get("A_eq") or [], problem.get("b_eq") or []):
            rows.append(([float(v) for v in row], "=", float(bound)))

    integer_vars = [bool(v) for v in problem.get("integerVars", [])] if integer else []
    bounds = _cli_bounds(problem, len(c))
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("Maximize\n" if problem.get("sense", "max") == "max" else "Minimize\n")
        handle.write(f" obj: {_lp_expr(c, names)}\n")
        handle.write("Subject To\n")
        if rows:
            for row_idx, (row, sense, rhs) in enumerate(rows):
                handle.write(f" c{row_idx}: {_lp_expr(row, names)} {sense} {_lp_number(rhs)}\n")
        else:
            handle.write(f" c0: 0 {names[0]} <= 0\n")
        handle.write("Bounds\n")
        for name, (lo, hi) in zip(names, bounds):
            if lo is None and hi is None:
                handle.write(f" {name} free\n")
            elif lo is None:
                handle.write(f" {name} <= {_lp_number(float(hi))}\n")
            elif hi is None:
                handle.write(f" {_lp_number(float(lo))} <= {name}\n")
            else:
                handle.write(f" {_lp_number(float(lo))} <= {name} <= {_lp_number(float(hi))}\n")
        general_names = [name for name, is_integer in zip(names, integer_vars) if is_integer]
        if general_names:
            handle.write("Generals\n")
            for name in general_names:
                handle.write(f" {name}\n")
        handle.write("End\n")
    return names


def _parse_first_float(tokens: list[str]) -> float | None:
    for token in tokens:
        try:
            return float(token)
        except ValueError:
            continue
    return None


def _glpk_status_from_tokens(tokens: list[str]) -> str:
    if len(tokens) >= 6 and tokens[1] == "bas":
        return "optimal" if tokens[4] == "f" and tokens[5] == "f" else "numerical-error"
    if len(tokens) >= 5 and tokens[1] == "mip":
        if tokens[4] == "o":
            return "optimal"
        if tokens[4] == "n":
            return "infeasible"
        if tokens[4] == "u":
            return "numerical-error"
    return "numerical-error"


def _parse_glpk_solution(
    path: str,
    n: int,
    ub_count: int = 0,
    eq_count: int = 0,
    include_certificates: bool = False,
) -> dict[str, Any]:
    status = "numerical-error"
    objective = None
    x = [0.0] * n
    row_duals = [0.0] * (ub_count + eq_count)
    reduced_costs = [0.0] * n
    var_basis: list[str | None] = [None] * n
    row_basis: list[str | None] = [None] * (ub_count + eq_count)
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            tokens = line.split()
            if not tokens:
                continue
            if tokens[0] == "s":
                status = _glpk_status_from_tokens(tokens)
                objective = _parse_first_float(list(reversed(tokens)))
            elif tokens[0] == "j" and len(tokens) >= 3:
                idx = int(tokens[1]) - 1
                if 0 <= idx < n:
                    if include_certificates:
                        status_token = _basis_status_from_token(tokens[2])
                        if status_token is not None:
                            var_basis[idx] = status_token
                    value = _parse_first_float(tokens[2:])
                    if value is not None:
                        x[idx] = value
                    if include_certificates:
                        dual = _parse_first_float(tokens[4:] if len(tokens) >= 5 else tokens[3:])
                        if dual is not None:
                            reduced_costs[idx] = dual
            elif tokens[0] == "i" and len(tokens) >= 5 and include_certificates:
                idx = int(tokens[1]) - 1
                if 0 <= idx < len(row_duals):
                    status_token = _basis_status_from_token(tokens[2])
                    if status_token is not None:
                        row_basis[idx] = status_token
                    dual = _parse_first_float(tokens[4:])
                    if dual is not None:
                        row_duals[idx] = dual
    parsed = {"status": status, "x": x, "objective": objective, "solver": "glpk-cli"}
    if include_certificates and status == "optimal":
        parsed.update(_lp_certificate_fields(row_duals, reduced_costs, ub_count, eq_count))
        parsed.update(_lp_basis_fields(var_basis, row_basis))
    return parsed


def _parse_glpk_report(path: str, names: list[str]) -> dict[str, Any]:
    name_to_index = {name: idx for idx, name in enumerate(names)}
    status = "numerical-error"
    objective = None
    x = [0.0] * len(names)
    in_columns = False
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            upper = stripped.upper()
            if upper.startswith("STATUS:"):
                if "OPTIMAL" in upper:
                    status = "optimal"
                elif "INFEASIBLE" in upper or "NO PRIMAL FEASIBLE" in upper:
                    status = "infeasible"
                elif "UNBOUNDED" in upper:
                    status = "unbounded"
                elif "TIME LIMIT" in upper:
                    status = "time-limit"
            elif upper.startswith("OBJECTIVE:"):
                objective = _parse_first_float(
                    stripped.replace("=", " = ").split("=")[1].split()
                    if "=" in stripped
                    else stripped.split()[1:]
                )
            elif "Column name" in line:
                in_columns = True
            elif in_columns:
                tokens = stripped.split()
                if len(tokens) >= 3 and tokens[0].isdigit() and tokens[1] in name_to_index:
                    value_tokens = tokens[3:] if tokens[2] == "*" else tokens[2:]
                    value = _parse_first_float(value_tokens)
                    if value is not None:
                        x[name_to_index[tokens[1]]] = value
    return {"status": status, "x": x, "objective": objective, "solver": "glpk-cli"}


def solve_glpsol_cli(
    problem: dict[str, Any], integer: bool, options: dict[str, Any] | None = None
) -> dict[str, Any]:
    options = options or {}
    with tempfile.TemporaryDirectory(prefix="des-glpk-cli-") as tmp:
        model_path = f"{tmp}/model.lp"
        solution_path = f"{tmp}/solution.txt"
        report_path = f"{tmp}/report.txt"
        names = _write_lp_file(problem, integer, model_path)
        commands = [
            _command_for("glpk"),
            "--lp",
            model_path,
            "--tmlim",
            _time_limit_integer_seconds_text(options),
            "--output",
            report_path,
            "--write",
            solution_path,
        ]
        relative_gap = _relative_gap_limit(options)
        if integer and relative_gap is not None:
            commands.extend(["--mipgap", f"{relative_gap:.17g}"])
        result = subprocess.run(commands, check=False, capture_output=True, text=True)
        if result.returncode != 0:
            return {
                "status": "numerical-error",
                "x": [],
                "objective": None,
                "solver": "glpk-cli",
                "message": result.stderr or result.stdout,
            }
        ub_count, eq_count = _lp_row_counts(problem) if not integer else (0, 0)
        solution = _parse_glpk_solution(
            solution_path,
            len(names),
            ub_count,
            eq_count,
            include_certificates=not integer,
        )
        parsed = _parse_glpk_report(report_path, names)
        if parsed["status"] == "numerical-error":
            parsed = solution
        else:
            for key in ("dualUB", "dualEQ", "reducedCosts", "varBasis", "rowBasis"):
                if key in solution:
                    parsed[key] = solution[key]
        parsed["status"] = _limited_status(parsed["status"], options)
        parsed["message"] = result.stderr or result.stdout
        return parsed


def _parse_scip_solution(path: str, names: list[str]) -> dict[str, Any]:
    name_to_index = {name: idx for idx, name in enumerate(names)}
    status = "numerical-error"
    objective = None
    x = [0.0] * len(names)
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("solution status:"):
                lowered = stripped.lower()
                if "optimal" in lowered:
                    status = "optimal"
                elif "infeasible" in lowered:
                    status = "infeasible"
                elif "unbounded" in lowered:
                    status = "unbounded"
                elif "time limit" in lowered:
                    status = "time-limit"
                elif "node limit" in lowered:
                    status = "node-limit"
                elif "gap limit" in lowered:
                    status = "iter-limit"
            elif stripped.startswith("objective value:"):
                objective = _parse_first_float(stripped.split()[2:])
            else:
                tokens = stripped.split()
                if tokens and tokens[0] in name_to_index:
                    value = _parse_first_float(tokens[1:])
                    if value is not None:
                        x[name_to_index[tokens[0]]] = value
    return {"status": status, "x": x, "objective": objective, "solver": "scip-cli"}


def _parse_scip_dual_fields(output: str, problem: dict[str, Any]) -> dict[str, Any]:
    ub_count, eq_count = _lp_row_counts(problem)
    row_count = ub_count + eq_count
    if row_count == 0:
        return {}
    row_duals = [0.0] * row_count
    saw_finite_dual = False
    saw_bad_dual = False
    for line in output.splitlines():
        tokens = line.split()
        if len(tokens) != 2 or not tokens[0].startswith("c"):
            continue
        try:
            row_idx = int(tokens[0][1:])
        except ValueError:
            continue
        if not 0 <= row_idx < row_count:
            continue
        try:
            value = float(tokens[1])
        except ValueError:
            saw_bad_dual = True
            continue
        if not math.isfinite(value):
            saw_bad_dual = True
            continue
        row_duals[row_idx] = value
        saw_finite_dual = True
    if not saw_finite_dual or saw_bad_dual:
        return {}
    reduced_costs = _lp_reduced_costs_from_row_duals(problem, row_duals)
    return _lp_certificate_fields(row_duals, reduced_costs, ub_count, eq_count)


def solve_scip_cli(
    problem: dict[str, Any], integer: bool, options: dict[str, Any] | None = None
) -> dict[str, Any]:
    options = options or {}
    with tempfile.TemporaryDirectory(prefix="des-scip-cli-") as tmp:
        model_path = f"{tmp}/model.lp"
        solution_path = f"{tmp}/solution.txt"
        names = _write_lp_file(problem, integer, model_path)
        commands = [
            f"set limits time {_time_limit_seconds_text(options)}",
            f"read {model_path}",
            "optimize",
        ]
        node_limit = _node_limit(options)
        relative_gap = _relative_gap_limit(options)
        if integer and node_limit is not None:
            commands.insert(1, f"set limits nodes {node_limit}")
        if integer and relative_gap is not None:
            commands.insert(1, f"set limits gap {relative_gap:.17g}")
        if not integer:
            commands.insert(0, "set presolving maxrounds 0")
            commands.append("display dualsolution")
        commands.extend(
            [
                f"write solution {solution_path}",
                "quit",
            ]
        )
        scip_cmd = [_command_for("scip")] if not integer else [_command_for("scip"), "-q"]
        for command in commands:
            scip_cmd.extend(["-c", command])
        result = subprocess.run(
            scip_cmd,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return {
                "status": "numerical-error",
                "x": [],
                "objective": None,
                "solver": "scip-cli",
                "message": result.stderr or result.stdout,
            }
        parsed = _parse_scip_solution(solution_path, names)
        if not integer and parsed["status"] == "optimal":
            parsed.update(_parse_scip_dual_fields(result.stdout, problem))
        parsed["status"] = _limited_status(parsed["status"], options)
        parsed["message"] = (
            result.stderr
            or ("scip-cli status=optimal" if parsed["status"] == "optimal" else result.stdout)
        )
        if integer:
            parsed.update(_scip_quality_fields(result.stdout))
        return parsed


def _scip_quality_fields(output: str) -> dict[str, Any]:
    best_bound = None
    mip_gap = None
    nodes_explored = None
    for raw_line in output.splitlines():
        line = raw_line.strip()
        lowered = line.lower()
        if "dual bound" in lowered and ":" in line:
            best_bound = _last_float_token(line)
        elif lowered.startswith("gap") and ":" in line:
            value = _last_float_token(line)
            if value is not None:
                mip_gap = value / 100.0 if "%" in line else value
        elif "solving nodes" in lowered and ":" in line:
            nodes_explored = _last_float_token(line)
    return _quality_fields(best_bound, mip_gap, nodes_explored)


def _highs_status_from_text(text: str) -> str:
    upper = text.upper()
    if "OPTIMAL" in upper:
        return "optimal"
    if "INFEASIBLE" in upper:
        return "infeasible"
    if "UNBOUNDED" in upper:
        return "unbounded"
    if "TIME LIMIT" in upper or "TIME-LIMIT" in upper:
        return "time-limit"
    if "NODE LIMIT" in upper or "NODE-LIMIT" in upper:
        return "node-limit"
    return "numerical-error"


def _parse_highs_solution(
    path: str,
    names: list[str],
    output: str,
    problem: dict[str, Any],
    integer: bool,
) -> dict[str, Any]:
    name_to_index = {name: idx for idx, name in enumerate(names)}
    ub_count, eq_count = _lp_row_counts(problem) if not integer else (0, 0)
    status = "numerical-error"
    objective = None
    x = [0.0] * len(names)
    row_duals = [0.0] * (ub_count + eq_count)
    reduced_costs = [0.0] * len(names)
    var_basis: list[str | None] = [None] * len(names)
    row_basis: list[str | None] = [None] * (ub_count + eq_count)
    section: str | None = None
    in_columns = False
    in_rows = False
    pending_model_status = False
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            if pending_model_status:
                status = _highs_status_from_text(stripped)
                pending_model_status = False
                continue
            if stripped == "Model status":
                pending_model_status = True
            elif stripped == "# Primal solution values":
                section = "primal"
                in_columns = False
                in_rows = False
            elif stripped.startswith("# Dual solution values") or stripped.startswith("# Basis"):
                section = "dual" if stripped.startswith("# Dual solution values") else "basis"
                in_columns = False
                in_rows = False
            elif stripped.startswith("Objective "):
                objective = _parse_first_float(stripped.split()[1:])
            elif section in {"primal", "dual", "basis"} and stripped.startswith("# Columns"):
                in_columns = True
                in_rows = False
            elif section in {"primal", "dual", "basis"} and stripped.startswith("# Rows"):
                in_columns = False
                in_rows = True
            elif in_columns:
                tokens = stripped.split()
                if len(tokens) >= 2 and tokens[0] in name_to_index:
                    value = _parse_first_float(tokens[1:])
                    if value is not None:
                        if section == "primal":
                            x[name_to_index[tokens[0]]] = value
                        elif section == "dual" and not integer:
                            reduced_costs[name_to_index[tokens[0]]] = value
                    if section == "basis" and not integer:
                        basis_status = _basis_status_from_token(tokens[1])
                        if basis_status is not None:
                            var_basis[name_to_index[tokens[0]]] = basis_status
            elif in_rows and section in {"dual", "basis"} and not integer:
                tokens = stripped.split()
                if len(tokens) >= 2 and tokens[0].startswith("c"):
                    try:
                        idx = int(tokens[0][1:])
                    except ValueError:
                        idx = -1
                    if 0 <= idx < len(row_duals):
                        if section == "dual":
                            value = _parse_first_float(tokens[1:])
                            if value is not None:
                                row_duals[idx] = value
                        else:
                            basis_status = _basis_status_from_token(tokens[1])
                            if basis_status is not None:
                                row_basis[idx] = basis_status
    if status == "numerical-error":
        status = _highs_status_from_text(output)
    parsed = {"status": status, "x": x, "objective": objective, "solver": "highs-cli"}
    if not integer and status == "optimal":
        parsed.update(_lp_certificate_fields(row_duals, reduced_costs, ub_count, eq_count))
        parsed.update(_lp_basis_fields(var_basis, row_basis))
    if integer:
        parsed.update(_highs_quality_fields(output))
    return parsed


def _highs_quality_fields(output: str) -> dict[str, Any]:
    best_bound = None
    mip_gap = None
    nodes_explored = None
    for raw_line in output.splitlines():
        line = raw_line.strip()
        lowered = line.lower()
        if lowered.startswith(("dual bound", "best bound")):
            best_bound = _last_float_token(line)
        elif lowered.startswith("gap"):
            value = _last_float_token(line)
            if value is not None:
                mip_gap = value / 100.0 if "%" in line else value
        elif lowered.startswith("nodes"):
            nodes_explored = _last_float_token(line)
    return _quality_fields(best_bound, mip_gap, nodes_explored)


def solve_highs_cli(
    problem: dict[str, Any], integer: bool, options: dict[str, Any] | None = None
) -> dict[str, Any]:
    options = options or {}
    with tempfile.TemporaryDirectory(prefix="des-highs-cli-") as tmp:
        model_path = f"{tmp}/model.lp"
        solution_path = f"{tmp}/solution.txt"
        options_path = f"{tmp}/options.txt"
        log_path = f"{tmp}/highs.log"
        names = _write_lp_file(problem, integer, model_path)
        node_limit = _node_limit(options)
        relative_gap = _relative_gap_limit(options)
        with open(options_path, "w", encoding="utf-8") as handle:
            handle.write(f"time_limit = {_time_limit_seconds_text(options)}\n")
            handle.write(f"log_file = {log_path}\n")
            if integer and node_limit is not None:
                handle.write(f"mip_max_nodes = {node_limit}\n")
            if integer and relative_gap is not None:
                handle.write(f"mip_rel_gap = {relative_gap:.17g}\n")
        commands = [
            _command_for("highs"),
            "--model_file",
            model_path,
            "--solution_file",
            solution_path,
            "--options_file",
            options_path,
        ]
        result = subprocess.run(commands, check=False, capture_output=True, text=True)
        output = f"{result.stdout}\n{result.stderr}"
        if result.returncode != 0:
            return {
                "status": "numerical-error",
                "x": [],
                "objective": None,
                "solver": "highs-cli",
                "message": result.stderr or result.stdout,
            }
        try:
            parsed = _parse_highs_solution(solution_path, names, output, problem, integer)
        except FileNotFoundError:
            parsed = {
                "status": _highs_status_from_text(output),
                "x": [],
                "objective": None,
                "solver": "highs-cli",
            }
        parsed["status"] = _limited_status(parsed["status"], options)
        parsed["message"] = result.stderr or result.stdout
        return parsed


def _cbc_status_from_text(text: str) -> str:
    lowered = text.lower()
    if "optimal" in lowered:
        return "optimal"
    if "infeasible" in lowered:
        return "infeasible"
    if "unbounded" in lowered:
        return "unbounded"
    if "node limit" in lowered or "stopped on nodes" in lowered:
        return "node-limit"
    if "stopped on time" in lowered or "time limit" in lowered:
        return "time-limit"
    return "numerical-error"


def _last_float_token(text: str) -> float | None:
    for token in reversed(text.replace(",", "").split()):
        token = token.strip().strip("%")
        value = _finite_float_or_none(token)
        if value is not None:
            return value
    return None


def _cbc_quality_fields(output: str) -> dict[str, Any]:
    best_bound = None
    mip_gap = None
    nodes_explored = None
    for raw_line in output.splitlines():
        line = raw_line.strip()
        lowered = line.lower()
        if lowered.startswith(("lower bound:", "upper bound:")):
            best_bound = _last_float_token(line)
        elif lowered.startswith("gap:"):
            value = _last_float_token(line)
            if value is not None:
                mip_gap = value / 100.0 if "%" in line else value
        elif lowered.startswith("enumerated nodes:"):
            nodes_explored = _last_float_token(line)
    return _quality_fields(best_bound, mip_gap, nodes_explored)


def _parse_cbc_basis(
    path: str, names: list[str], ub_count: int, eq_count: int
) -> dict[str, Any]:
    name_to_index = {name: idx for idx, name in enumerate(names)}
    var_basis: list[str | None] = [None] * len(names)
    row_basis: list[str | None] = ["basic"] * ub_count + ["fixed"] * eq_count
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            tokens = line.split()
            if not tokens or tokens[0] in {"NAME", "ENDATA"}:
                continue
            code = tokens[0].upper()
            if len(tokens) >= 2 and tokens[1] in name_to_index:
                idx = name_to_index[tokens[1]]
                if code in {"BS", "XL", "XU"}:
                    var_basis[idx] = "basic"
                elif code == "LL":
                    var_basis[idx] = "at_lower"
                elif code == "UL":
                    var_basis[idx] = "at_upper"
                elif code == "FX":
                    var_basis[idx] = "fixed"
                elif code == "FR":
                    var_basis[idx] = "free"
            if code in {"XL", "XU"} and len(tokens) >= 3 and tokens[2].startswith("c"):
                try:
                    row_idx = int(tokens[2][1:])
                except ValueError:
                    continue
                if 0 <= row_idx < ub_count:
                    row_basis[row_idx] = "at_lower" if code == "XL" else "at_upper"
    return _lp_basis_fields(var_basis, row_basis)


def _parse_cbc_solution(
    path: str,
    names: list[str],
    output: str,
    ub_count: int = 0,
    eq_count: int = 0,
    include_certificates: bool = False,
    basis_path: str | None = None,
) -> dict[str, Any]:
    name_to_index = {name: idx for idx, name in enumerate(names)}
    status = "numerical-error"
    objective = None
    x = [0.0] * len(names)
    row_duals = [0.0] * (ub_count + eq_count)
    reduced_costs = [0.0] * len(names)
    with open(path, encoding="utf-8") as handle:
        for line_idx, line in enumerate(handle):
            stripped = line.strip()
            if not stripped:
                continue
            if line_idx == 0:
                status = _cbc_status_from_text(stripped)
                if "objective value" in stripped:
                    objective = _parse_first_float(stripped.split("objective value", 1)[1].split())
                continue
            tokens = stripped.split()
            if len(tokens) >= 3 and tokens[1] in name_to_index:
                idx = name_to_index[tokens[1]]
                value = _parse_first_float(tokens[2:])
                if value is not None:
                    x[idx] = value
                if include_certificates and len(tokens) >= 4:
                    reduced_cost = _parse_first_float(tokens[3:])
                    if reduced_cost is not None:
                        reduced_costs[idx] = reduced_cost
            elif include_certificates and len(tokens) >= 4 and tokens[1].startswith("c"):
                try:
                    row_idx = int(tokens[1][1:])
                except ValueError:
                    continue
                if 0 <= row_idx < len(row_duals):
                    dual = _parse_first_float(tokens[3:])
                    if dual is not None:
                        row_duals[row_idx] = dual
    if status == "numerical-error":
        status = _cbc_status_from_text(output)
    parsed = {"status": status, "x": x, "objective": objective, "solver": "cbc-cli"}
    if include_certificates and status == "optimal":
        parsed.update(_lp_certificate_fields(row_duals, reduced_costs, ub_count, eq_count))
        if basis_path is not None:
            try:
                parsed.update(_parse_cbc_basis(basis_path, names, ub_count, eq_count))
            except FileNotFoundError:
                pass
    return parsed


def solve_cbc_cli(
    problem: dict[str, Any], integer: bool, options: dict[str, Any] | None = None
) -> dict[str, Any]:
    options = options or {}
    with tempfile.TemporaryDirectory(prefix="des-cbc-cli-") as tmp:
        model_path = f"{tmp}/model.lp"
        solution_path = f"{tmp}/solution.txt"
        basis_path = f"{tmp}/basis.bas"
        names = _write_lp_file(problem, integer, model_path)
        ub_count = len(problem.get("A_ub") or [])
        eq_count = len(problem.get("A_eq") or [])
        commands = [_command_for("cbc"), model_path, "sec", _time_limit_seconds_text(options)]
        node_limit = _node_limit(options)
        relative_gap = _relative_gap_limit(options)
        if integer and node_limit is not None:
            commands.extend(["maxNodes", str(node_limit)])
        if integer and relative_gap is not None:
            commands.extend(["ratioGap", f"{relative_gap:.17g}"])
        if not integer:
            commands.extend(["printingOptions", "all"])
        commands.extend(["solve", "solution", solution_path])
        if not integer:
            commands.extend(["basisOut", basis_path])
        commands.append("quit")
        result = subprocess.run(
            commands,
            check=False,
            capture_output=True,
            text=True,
        )
        output = f"{result.stdout}\n{result.stderr}"
        if result.returncode != 0:
            return {
                "status": "numerical-error",
                "x": [],
                "objective": None,
                "solver": "cbc-cli",
                "message": result.stderr or result.stdout,
            }
        try:
            parsed = _parse_cbc_solution(
                solution_path,
                names,
                output,
                ub_count,
                eq_count,
                include_certificates=not integer,
                basis_path=basis_path if not integer else None,
            )
        except FileNotFoundError:
            parsed = {
                "status": _cbc_status_from_text(output),
                "x": [],
                "objective": None,
                "solver": "cbc-cli",
            }
        if integer:
            parsed.update(_cbc_quality_fields(output))
        parsed["status"] = _limited_status(parsed["status"], options)
        parsed["message"] = result.stderr or result.stdout
        return parsed


def _import_cplex() -> Any:
    try:
        import cplex
    except Exception as exc:
        raise RuntimeError(f"CPLEX unavailable: {exc}") from exc
    return cplex


def _cplex_quiet(cpx: Any) -> None:
    cpx.set_log_stream(None)
    cpx.set_results_stream(None)
    cpx.set_warning_stream(None)
    cpx.set_error_stream(None)


def _cplex_status(cpx: Any, status_code: int) -> str:
    status = cpx.solution.status
    if status_code in {
        status.optimal,
        status.optimal_tolerance,
        status.MIP_optimal,
        status.MIP_optimal_infeasible,
    }:
        return "optimal"
    if status_code in {status.infeasible, status.MIP_infeasible, status.fail_infeasible}:
        return "infeasible"
    if status_code in {status.unbounded, status.MIP_unbounded}:
        return "unbounded"
    if status_code in {
        status.abort_iteration_limit,
    }:
        return "iter-limit"
    if status_code in {
        status.node_limit_feasible,
        status.node_limit_infeasible,
    }:
        return "node-limit"
    if status_code in {
        status.abort_time_limit,
        status.MIP_time_limit_feasible,
        status.MIP_time_limit_infeasible,
    }:
        return "time-limit"
    return "numerical-error"


def _cplex_variables(cplex_mod: Any, cpx: Any, problem: dict[str, Any], integer: bool) -> None:
    c = [float(v) for v in problem.get("c", [])]
    lower = problem.get("lb")
    upper = problem.get("ub")
    integer_vars = problem.get("integerVars") if integer else [False] * len(c)
    lb = []
    ub = []
    types = []
    names = []
    for i in range(len(c)):
        lb.append(0.0 if lower is None or i >= len(lower) or lower[i] is None else float(lower[i]))
        if upper is not None and i < len(upper) and upper[i] is not None:
            ub.append(float(upper[i]))
        else:
            ub.append(cplex_mod.infinity)
        types.append(cpx.variables.type.integer if integer_vars[i] else cpx.variables.type.continuous)
        names.append(f"x{i}")
    cpx.variables.add(obj=c, lb=lb, ub=ub, types="".join(types), names=names)


def _cplex_sparse_pair(cplex_mod: Any, coeffs: dict[int, float]) -> Any:
    items = [(idx, value) for idx, value in sorted(coeffs.items()) if abs(value) > 1e-12]
    return cplex_mod.SparsePair(
        ind=[idx for idx, _ in items],
        val=[float(value) for _, value in items],
    )


def _cplex_sparse_triple(cplex_mod: Any, coeffs: dict[tuple[int, int], float]) -> Any:
    items = [
        (i, j, value)
        for (i, j), value in sorted(coeffs.items())
        if abs(value) > 1e-12
    ]
    return cplex_mod.SparseTriple(
        ind1=[i for i, _, _ in items],
        ind2=[j for _, j, _ in items],
        val=[float(value) for _, _, value in items],
    )


def _cplex_add_dense_linear_rows(cplex_mod: Any, cpx: Any, problem: dict[str, Any]) -> None:
    for row, bound in zip(problem.get("A_ub") or [], problem.get("b_ub") or []):
        coeffs = {i: float(coef) for i, coef in enumerate(row) if coef}
        cpx.linear_constraints.add(
            lin_expr=[_cplex_sparse_pair(cplex_mod, coeffs)],
            senses="L",
            rhs=[float(bound)],
        )
    for row, bound in zip(problem.get("A_eq") or [], problem.get("b_eq") or []):
        coeffs = {i: float(coef) for i, coef in enumerate(row) if coef}
        cpx.linear_constraints.add(
            lin_expr=[_cplex_sparse_pair(cplex_mod, coeffs)],
            senses="E",
            rhs=[float(bound)],
        )


def _cplex_add_mip_rows(cplex_mod: Any, cpx: Any, mip: dict[str, Any]) -> None:
    for row, bound in _mip_rows(mip):
        coeffs = {i: float(coef) for i, coef in enumerate(row) if coef}
        cpx.linear_constraints.add(
            lin_expr=[_cplex_sparse_pair(cplex_mod, coeffs)],
            senses="L",
            rhs=[float(bound)],
        )


def _cplex_set_objective(cpx: Any, problem: dict[str, Any]) -> None:
    if problem.get("sense", "max") == "max":
        cpx.objective.set_sense(cpx.objective.sense.maximize)
    else:
        cpx.objective.set_sense(cpx.objective.sense.minimize)


def _cplex_set_quadratic_objective(cpx: Any, problem: dict[str, Any]) -> None:
    coeffs: dict[tuple[int, int], float] = {}
    for term in problem.get("quadratic", []):
        i = int(term["i"])
        j = int(term["j"])
        coeff = float(term["coeff"])
        if i == j:
            coeffs[(i, i)] = coeffs.get((i, i), 0.0) + 2.0 * coeff
        else:
            coeffs[(i, j)] = coeffs.get((i, j), 0.0) + coeff
            coeffs[(j, i)] = coeffs.get((j, i), 0.0) + coeff
    for (i, j), coeff in coeffs.items():
        cpx.objective.set_quadratic_coefficients(i, j, coeff)


def _cplex_apply_options(cpx: Any, options: dict[str, Any], integer: bool = False) -> None:
    time_limit = _time_limit_seconds(options)
    if time_limit is not None:
        try:
            cpx.parameters.timelimit.set(time_limit)
        except Exception:
            pass
    if not integer:
        return
    node_limit = _node_limit(options)
    relative_gap = _relative_gap_limit(options)
    if node_limit is not None:
        try:
            cpx.parameters.mip.limits.nodes.set(node_limit)
        except Exception:
            pass
    if relative_gap is not None:
        try:
            cpx.parameters.mip.tolerances.mipgap.set(relative_gap)
        except Exception:
            pass


def _cplex_finish(
    cpx: Any,
    solver_name: str,
    lp_problem: dict[str, Any] | None = None,
    include_quality: bool = False,
) -> dict[str, Any]:
    cpx.solve()
    status_code = int(cpx.solution.get_status())
    status = _cplex_status(cpx, status_code)
    has_solution = status in {"optimal", "iter-limit", "time-limit", "node-limit"}
    x = [float(v) for v in cpx.solution.get_values()] if has_solution else []
    objective = float(cpx.solution.get_objective_value()) if has_solution else None
    parsed = {
        "status": status,
        "x": x,
        "objective": objective,
        "solver": solver_name,
        "message": f"{solver_name} status={status_code}",
    }
    if lp_problem is not None and status == "optimal":
        ub_count, eq_count = _lp_row_counts(lp_problem)
        row_duals = [float(v) for v in cpx.solution.get_dual_values()]
        reduced_costs = [float(v) for v in cpx.solution.get_reduced_costs()]
        parsed.update(_lp_certificate_fields(row_duals, reduced_costs, ub_count, eq_count))
        basis_status = cpx.solution.basis.status
        col_basis, row_basis = cpx.solution.basis.get_basis()
        parsed.update(
            _lp_basis_fields(
                [
                    _cplex_var_basis_status(basis_status, int(code))
                    for code in col_basis
                ],
                _lp_row_basis_with_fixed_equalities(
                    [
                        _cplex_row_basis_status(basis_status, int(code))
                        for code in row_basis[:ub_count]
                    ],
                    eq_count,
                ),
            )
        )
    if include_quality:
        best_bound = None
        mip_gap = None
        nodes_explored = None
        try:
            best_bound = cpx.solution.MIP.get_best_objective()
        except Exception:
            pass
        try:
            mip_gap = cpx.solution.MIP.get_mip_relative_gap()
        except Exception:
            pass
        progress = _first_attr(cpx.solution, ("progress",))
        if progress is not None:
            try:
                nodes_explored = progress.get_num_nodes_processed()
            except Exception:
                pass
        parsed.update(_quality_fields(best_bound, mip_gap, nodes_explored))
    return parsed


def solve_cplex_lp(payload: dict[str, Any]) -> dict[str, Any]:
    cplex_mod = _import_cplex()
    lp = payload["lp"]
    cpx = cplex_mod.Cplex()
    _cplex_quiet(cpx)
    _cplex_variables(cplex_mod, cpx, lp, integer=False)
    _cplex_add_dense_linear_rows(cplex_mod, cpx, lp)
    _cplex_set_objective(cpx, lp)
    cpx.set_problem_type(cpx.problem_type.LP)
    _cplex_apply_options(cpx, _external_options(payload), integer=False)
    return _cplex_finish(cpx, "cplex:lp", lp)


def solve_cplex_mip(payload: dict[str, Any]) -> dict[str, Any]:
    cplex_mod = _import_cplex()
    mip = payload["mip"]
    cpx = cplex_mod.Cplex()
    _cplex_quiet(cpx)
    _cplex_variables(cplex_mod, cpx, mip, integer=True)
    _cplex_add_mip_rows(cplex_mod, cpx, mip)
    _cplex_set_objective(cpx, mip)
    _cplex_apply_options(cpx, _external_options(payload), integer=True)
    start = _mip_start(mip)
    if start is not None:
        try:
            cpx.MIP_starts.add(
                cplex_mod.SparsePair(ind=list(range(len(start))), val=start),
                cpx.MIP_starts.effort_level.auto,
                "mip-start",
            )
        except Exception:
            pass
    return _cplex_finish(cpx, "cplex:mip", include_quality=True)


def solve_cplex_qp(payload: dict[str, Any]) -> dict[str, Any]:
    cplex_mod = _import_cplex()
    qp = payload["qp"]
    cpx = cplex_mod.Cplex()
    _cplex_quiet(cpx)
    _cplex_variables(
        cplex_mod,
        cpx,
        qp,
        integer=any(bool(value) for value in qp.get("integerVars", [])),
    )
    _cplex_add_dense_linear_rows(cplex_mod, cpx, qp)
    _cplex_set_objective(cpx, qp)
    _cplex_set_quadratic_objective(cpx, qp)
    _cplex_apply_options(
        cpx,
        _external_options(payload),
        integer=any(bool(value) for value in qp.get("integerVars", [])),
    )
    return _cplex_finish(
        cpx,
        "cplex:qp",
        include_quality=any(bool(value) for value in qp.get("integerVars", [])),
    )


def _cplex_affine_square(
    coeffs: list[tuple[int, float]],
    constant: float,
    sign: float,
    linear: dict[int, float],
    quadratic: dict[tuple[int, int], float],
) -> float:
    for idx, coef in coeffs:
        linear[idx] = linear.get(idx, 0.0) + sign * 2.0 * constant * coef
    for pos, (i, coef_i) in enumerate(coeffs):
        quadratic[(i, i)] = quadratic.get((i, i), 0.0) + sign * coef_i * coef_i
        for j, coef_j in coeffs[pos + 1:]:
            key = (min(i, j), max(i, j))
            quadratic[key] = quadratic.get(key, 0.0) + sign * 2.0 * coef_i * coef_j
    return sign * constant * constant


def _cplex_add_quadratic_constraints(cplex_mod: Any, cpx: Any, problem: dict[str, Any]) -> None:
    for row in problem.get("quadraticConstraints", []):
        linear = {
            int(idx): float(coef)
            for idx, coef in row.get("linear", [])
            if float(coef)
        }
        quadratic: dict[tuple[int, int], float] = {}
        for term in row.get("quadratic", []):
            i = int(term["i"])
            j = int(term["j"])
            key = (min(i, j), max(i, j))
            quadratic[key] = quadratic.get(key, 0.0) + float(term["coeff"])
        sense = row.get("sense", "<=")
        cpx.quadratic_constraints.add(
            lin_expr=_cplex_sparse_pair(cplex_mod, linear),
            quad_expr=_cplex_sparse_triple(cplex_mod, quadratic),
            sense="G" if sense == ">=" else "E" if sense in {"=", "=="} else "L",
            rhs=float(row.get("rhs", 0.0)),
        )


def _cplex_add_soc_constraints(cplex_mod: Any, cpx: Any, problem: dict[str, Any]) -> None:
    for cone in problem.get("soc", []):
        linear: dict[int, float] = {}
        quadratic: dict[tuple[int, int], float] = {}
        constant = 0.0
        for term in cone.get("terms", []):
            coeffs = [
                (int(idx), float(coef))
                for idx, coef in term.get("coeffs", [])
                if float(coef)
            ]
            constant += _cplex_affine_square(
                coeffs,
                float(term.get("constant", 0.0)),
                1.0,
                linear,
                quadratic,
            )

        rhs_coeffs = [
            (int(idx), float(coef))
            for idx, coef in cone.get("rhsCoeffs", [])
            if float(coef)
        ]
        rhs_constant = float(cone.get("rhsConstant", 0.0))
        if rhs_coeffs or rhs_constant < 0.0:
            cpx.linear_constraints.add(
                lin_expr=[_cplex_sparse_pair(cplex_mod, dict(rhs_coeffs))],
                senses="G",
                rhs=[-rhs_constant],
            )
        constant += _cplex_affine_square(rhs_coeffs, rhs_constant, -1.0, linear, quadratic)
        cpx.quadratic_constraints.add(
            lin_expr=_cplex_sparse_pair(cplex_mod, linear),
            quad_expr=_cplex_sparse_triple(cplex_mod, quadratic),
            sense="L",
            rhs=-constant,
        )


def solve_cplex_conic(payload: dict[str, Any]) -> dict[str, Any]:
    cplex_mod = _import_cplex()
    conic = payload["conic"]
    cpx = cplex_mod.Cplex()
    _cplex_quiet(cpx)
    _cplex_variables(
        cplex_mod,
        cpx,
        conic,
        integer=any(bool(value) for value in conic.get("integerVars", [])),
    )
    _cplex_add_dense_linear_rows(cplex_mod, cpx, conic)
    _cplex_set_objective(cpx, conic)
    _cplex_set_quadratic_objective(cpx, conic)
    _cplex_add_quadratic_constraints(cplex_mod, cpx, conic)
    _cplex_add_soc_constraints(cplex_mod, cpx, conic)
    _cplex_apply_options(
        cpx,
        _external_options(payload),
        integer=any(bool(value) for value in conic.get("integerVars", [])),
    )
    return _cplex_finish(
        cpx,
        "cplex:conic",
        include_quality=any(bool(value) for value in conic.get("integerVars", [])),
    )


def _import_glpk() -> Any:
    try:
        import swiglpk as glp
    except Exception as exc:
        raise RuntimeError(f"GLPK unavailable: {exc}") from exc
    return glp


def _glpk_bound_type(glp: Any, lo: float | None, hi: float | None) -> tuple[int, float, float]:
    if lo is None and hi is None:
        return glp.GLP_FR, 0.0, 0.0
    if lo is None:
        return glp.GLP_UP, 0.0, float(hi)
    if hi is None:
        return glp.GLP_LO, float(lo), 0.0
    if abs(float(lo) - float(hi)) <= 1e-12:
        return glp.GLP_FX, float(lo), float(hi)
    return glp.GLP_DB, float(lo), float(hi)


def _glpk_status(glp: Any, status_code: int) -> str:
    if status_code == glp.GLP_OPT:
        return "optimal"
    if status_code == glp.GLP_FEAS:
        return "iter-limit"
    if status_code in {glp.GLP_NOFEAS, glp.GLP_INFEAS}:
        return "infeasible"
    if status_code == glp.GLP_UNBND:
        return "unbounded"
    return "numerical-error"


def _glpk_create_problem(glp: Any, problem: dict[str, Any], integer: bool) -> Any:
    model = glp.glp_create_prob()
    glp.glp_set_obj_dir(model, glp.GLP_MAX if problem.get("sense", "max") == "max" else glp.GLP_MIN)
    c = [float(v) for v in problem.get("c", [])]
    lower = problem.get("lb") if not integer else None
    upper = problem.get("ub")
    integer_vars = problem.get("integerVars") if integer else [False] * len(c)
    glp.glp_add_cols(model, len(c))
    for i, coef in enumerate(c, start=1):
        lo = 0.0 if lower is None or i - 1 >= len(lower) or lower[i - 1] is None else float(lower[i - 1])
        hi = None if upper is None or i - 1 >= len(upper) else upper[i - 1]
        btype, lb, ub = _glpk_bound_type(glp, lo, None if hi is None else float(hi))
        glp.glp_set_col_bnds(model, i, btype, lb, ub)
        if integer_vars[i - 1]:
            if lo == 0.0 and hi is not None and abs(float(hi) - 1.0) <= 1e-12:
                glp.glp_set_col_kind(model, i, glp.GLP_BV)
            else:
                glp.glp_set_col_kind(model, i, glp.GLP_IV)
        glp.glp_set_obj_coef(model, i, coef)
    return model


def _glpk_add_matrix_rows(
    glp: Any,
    model: Any,
    rows: list[list[float]],
    bounds: list[float],
    equality: bool,
) -> list[tuple[int, int, float]]:
    if not rows:
        return []
    offset = glp.glp_get_num_rows(model)
    glp.glp_add_rows(model, len(rows))
    triplets = []
    for local_i, (row, bound) in enumerate(zip(rows, bounds), start=1):
        row_idx = offset + local_i
        if equality:
            glp.glp_set_row_bnds(model, row_idx, glp.GLP_FX, float(bound), float(bound))
        else:
            glp.glp_set_row_bnds(model, row_idx, glp.GLP_UP, 0.0, float(bound))
        for col_idx, coef in enumerate(row, start=1):
            value = float(coef)
            if value:
                triplets.append((row_idx, col_idx, value))
    return triplets


def _glpk_load_rows(
    glp: Any,
    model: Any,
    ub_rows: list[list[float]],
    ub_bounds: list[float],
    eq_rows: list[list[float]] | None = None,
    eq_bounds: list[float] | None = None,
) -> None:
    triplets = _glpk_add_matrix_rows(glp, model, ub_rows, ub_bounds, equality=False)
    if eq_rows:
        triplets.extend(_glpk_add_matrix_rows(glp, model, eq_rows, eq_bounds or [], equality=True))
    ne = len(triplets)
    ia = glp.intArray(ne + 1)
    ja = glp.intArray(ne + 1)
    ar = glp.doubleArray(ne + 1)
    for pos, (row_idx, col_idx, value) in enumerate(triplets, start=1):
        ia[pos] = row_idx
        ja[pos] = col_idx
        ar[pos] = value
    glp.glp_load_matrix(model, ne, ia, ja, ar)


def _glpk_simplex(glp: Any, model: Any) -> str:
    params = glp.glp_smcp()
    glp.glp_init_smcp(params)
    params.msg_lev = glp.GLP_MSG_OFF
    ret = glp.glp_simplex(model, params)
    if ret != 0:
        return "numerical-error"
    return _glpk_status(glp, glp.glp_get_status(model))


def solve_glpk_lp(payload: dict[str, Any]) -> dict[str, Any]:
    glp = _import_glpk()
    lp = payload["lp"]
    model = _glpk_create_problem(glp, lp, integer=False)
    try:
        _glpk_load_rows(
            glp,
            model,
            lp.get("A_ub") or [],
            lp.get("b_ub") or [],
            lp.get("A_eq") or [],
            lp.get("b_eq") or [],
        )
        status = _glpk_simplex(glp, model)
        n = len(lp.get("c", []))
        x = [glp.glp_get_col_prim(model, i) for i in range(1, n + 1)] if status == "optimal" else []
        objective = float(glp.glp_get_obj_val(model)) if x else None
        parsed = {
            "status": status,
            "x": [float(value) for value in x],
            "objective": objective,
            "solver": "glpk:lp",
            "message": f"glpk:lp status={glp.glp_get_status(model)}",
        }
        if status == "optimal":
            ub_count, eq_count = _lp_row_counts(lp)
            row_duals = [
                float(glp.glp_get_row_dual(model, i))
                for i in range(1, ub_count + eq_count + 1)
            ]
            reduced_costs = [
                float(glp.glp_get_col_dual(model, i)) for i in range(1, n + 1)
            ]
            parsed.update(
                _lp_certificate_fields(row_duals, reduced_costs, ub_count, eq_count)
            )
            var_basis = [
                _glpk_basis_status_from_code(glp, glp.glp_get_col_stat(model, i))
                for i in range(1, n + 1)
            ]
            row_basis = [
                _glpk_basis_status_from_code(glp, glp.glp_get_row_stat(model, i))
                for i in range(1, ub_count + eq_count + 1)
            ]
            parsed.update(_lp_basis_fields(var_basis, row_basis))
        return parsed
    finally:
        glp.glp_delete_prob(model)


def solve_glpk_mip(payload: dict[str, Any]) -> dict[str, Any]:
    glp = _import_glpk()
    mip = payload["mip"]
    options = _external_options(payload)
    model = _glpk_create_problem(glp, mip, integer=True)
    try:
        rows, bounds = _mip_row_arrays(mip)
        _glpk_load_rows(glp, model, rows, bounds)
        relaxation_status = _glpk_simplex(glp, model)
        if relaxation_status != "optimal":
            return {
                "status": relaxation_status,
                "x": [],
                "objective": None,
                "solver": "glpk:mip",
                "message": f"glpk:mip relaxation_status={glp.glp_get_status(model)}",
            }
        params = glp.glp_iocp()
        glp.glp_init_iocp(params)
        params.msg_lev = glp.GLP_MSG_OFF
        time_limit = _time_limit_seconds(options)
        relative_gap = _relative_gap_limit(options)
        if time_limit is not None:
            params.tm_lim = max(1, int(math.ceil(time_limit * 1000.0)))
        if relative_gap is not None:
            params.mip_gap = relative_gap
        ret = glp.glp_intopt(model, params)
        status = _glpk_status(glp, glp.glp_mip_status(model))
        if ret != 0 and status == "numerical-error" and time_limit is not None:
            status = "time-limit"
        status = _limited_status(status, options)
        x = [glp.glp_mip_col_val(model, i) for i in range(1, len(mip.get("c", [])) + 1)] if status in {"optimal", "iter-limit", "time-limit"} else []
        objective = float(glp.glp_mip_obj_val(model)) if x else None
        parsed = {
            "status": status,
            "x": [float(value) for value in x],
            "objective": objective,
            "solver": "glpk:mip",
            "message": f"glpk:mip status={glp.glp_mip_status(model)}",
        }
        try:
            parsed.update(_quality_fields(mip_gap=glp.glp_mip_gap(model)))
        except Exception:
            pass
        return parsed
    finally:
        glp.glp_delete_prob(model)


def _import_xpress() -> Any:
    try:
        import warnings

        warnings.simplefilter("ignore")
        import xpress as xp
    except Exception as exc:
        raise RuntimeError(f"Xpress unavailable: {exc}") from exc

    try:
        from pathlib import Path

        import xpresslibs

        auth = Path(xpresslibs.__file__).parent / "bin" / "community-xpauth.xpr"
        if auth.exists():
            xp.init(str(auth))
    except Exception:
        pass
    return xp


def _xpress_status(model: Any, options: dict[str, Any] | None = None) -> str:
    options = options or {}
    solstatus = int(model.attributes.solstatus)
    if solstatus == 1:
        return "optimal"
    if solstatus == 2:
        return _limited_status("iter-limit", options)
    if solstatus == 3:
        return "infeasible"
    if solstatus == 4:
        return "unbounded"
    return "numerical-error"


def _xpress_variables(xp: Any, model: Any, problem: dict[str, Any], integer: bool) -> Any:
    c = problem.get("c", [])
    lower = problem.get("lb")
    upper = problem.get("ub")
    integer_vars = problem.get("integerVars") if integer else [False] * len(c)
    variables = []
    for i in range(len(c)):
        lo = 0.0 if lower is None or i >= len(lower) or lower[i] is None else float(lower[i])
        hi = xp.infinity
        if upper is not None and i < len(upper) and upper[i] is not None:
            hi = float(upper[i])
        vartype = xp.integer if integer_vars[i] else xp.continuous
        variables.append(xp.var(name=f"x{i}", lb=lo, ub=hi, vartype=vartype))
    if variables:
        model.addVariable(*variables)
    return variables


def _xpress_linear_expr(_xp: Any, variables: Any, coeffs: Any, constant: float = 0.0) -> Any:
    expr = float(constant)
    for idx, coef in coeffs:
        value = float(coef)
        if value:
            expr += value * variables[int(idx)]
    return expr


def _xpress_add_linear_rows(xp: Any, model: Any, variables: Any, problem: dict[str, Any]) -> None:
    for row, bound in zip(problem.get("A_ub") or [], problem.get("b_ub") or []):
        expr = 0.0
        for var, coef in zip(variables, row):
            value = float(coef)
            if value:
                expr += value * var
        model.addConstraint(expr <= float(bound))
    for row, bound in zip(problem.get("A_eq") or [], problem.get("b_eq") or []):
        expr = 0.0
        for var, coef in zip(variables, row):
            value = float(coef)
            if value:
                expr += value * var
        model.addConstraint(expr == float(bound))


def _xpress_add_mip_rows(xp: Any, model: Any, variables: Any, mip: dict[str, Any]) -> None:
    for row, bound in _mip_rows(mip):
        expr = 0.0
        for var, coef in zip(variables, row):
            value = float(coef)
            if value:
                expr += value * var
        model.addConstraint(expr <= float(bound))


def _xpress_objective(_xp: Any, variables: Any, problem: dict[str, Any]) -> Any:
    objective = 0.0
    for var, coef in zip(variables, problem.get("c", [])):
        value = float(coef)
        if value:
            objective += value * var
    for term in problem.get("quadratic", []):
        objective += (
            float(term["coeff"])
            * variables[int(term["i"])]
            * variables[int(term["j"])]
        )
    return objective


def _xpress_apply_options(model: Any, options: dict[str, Any], integer: bool = False) -> None:
    time_limit = _time_limit_seconds(options)
    if time_limit is not None:
        try:
            model.controls.maxtime = time_limit
        except Exception:
            pass
    if not integer:
        return
    node_limit = _node_limit(options)
    relative_gap = _relative_gap_limit(options)
    if node_limit is not None:
        try:
            model.controls.maxnode = node_limit
        except Exception:
            pass
    if relative_gap is not None:
        try:
            model.controls.miprelstop = relative_gap
        except Exception:
            pass


def _xpress_finish(
    model: Any,
    variables: Any,
    solver_name: str,
    lp_problem: dict[str, Any] | None = None,
    include_quality: bool = False,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    model.solve()
    status = _xpress_status(model, options)
    try:
        x = (
            model.getSolution(variables)
            if status in {"optimal", "iter-limit", "time-limit", "node-limit"}
            else []
        )
    except Exception:
        x = []
    objective = float(model.attributes.objval) if x else None
    parsed = {
        "status": status,
        "x": [float(value) for value in x],
        "objective": objective,
        "solver": solver_name,
        "message": (
            f"{solver_name} solvestatus={model.attributes.solvestatus} "
            f"solstatus={model.attributes.solstatus}"
        ),
    }
    if lp_problem is not None and status == "optimal":
        ub_count, eq_count = _lp_row_counts(lp_problem)
        row_duals = [float(value) for value in model.getDuals()]
        reduced_costs = [float(value) for value in model.getRCost()]
        parsed.update(_lp_certificate_fields(row_duals, reduced_costs, ub_count, eq_count))
        row_basis_raw, col_basis_raw = model.getBasis()
        parsed.update(
            _lp_basis_fields(
                [_xpress_var_basis_status(int(code)) for code in col_basis_raw],
                _lp_row_basis_with_fixed_equalities(
                    [
                        _xpress_row_basis_status(int(code))
                        for code in row_basis_raw[:ub_count]
                    ],
                    eq_count,
                ),
            )
        )
    if include_quality:
        attrs = _first_attr(model, ("attributes",))
        if attrs is not None:
            parsed.update(
                _quality_fields(
                    _first_attr(attrs, ("bestbound", "mipbestbound", "best_bound")),
                    _first_attr(attrs, ("mipgap", "relmipgap", "gap")),
                    _first_attr(attrs, ("nodes", "nodesprocessed", "mipnodes")),
                )
            )
    return parsed


def solve_xpress_lp(payload: dict[str, Any]) -> dict[str, Any]:
    with contextlib.redirect_stdout(sys.stderr):
        xp = _import_xpress()
        lp = payload["lp"]
        model = xp.problem()
        model.controls.outputlog = 0
        variables = _xpress_variables(xp, model, lp, integer=False)
        _xpress_add_linear_rows(xp, model, variables, lp)
        model.setObjective(
            _xpress_objective(xp, variables, lp),
            sense=xp.maximize if lp.get("sense", "max") == "max" else xp.minimize,
        )
        _xpress_apply_options(model, _external_options(payload), integer=False)
        return _xpress_finish(
            model, variables, "xpress:lp", lp, options=_external_options(payload)
        )


def solve_xpress_mip(payload: dict[str, Any]) -> dict[str, Any]:
    with contextlib.redirect_stdout(sys.stderr):
        xp = _import_xpress()
        mip = payload["mip"]
        model = xp.problem()
        model.controls.outputlog = 0
        variables = _xpress_variables(xp, model, mip, integer=True)
        _xpress_add_mip_rows(xp, model, variables, mip)
        model.setObjective(
            _xpress_objective(xp, variables, mip),
            sense=xp.maximize if mip.get("sense", "max") == "max" else xp.minimize,
        )
        _xpress_apply_options(model, _external_options(payload), integer=True)
        start = _mip_start(mip)
        if start is not None and hasattr(model, "addmipsol"):
            try:
                model.addmipsol(start, variables)
            except Exception:
                pass
        return _xpress_finish(
            model,
            variables,
            "xpress:mip",
            include_quality=True,
            options=_external_options(payload),
        )


def solve_xpress_qp(payload: dict[str, Any]) -> dict[str, Any]:
    with contextlib.redirect_stdout(sys.stderr):
        xp = _import_xpress()
        qp = payload["qp"]
        model = xp.problem()
        model.controls.outputlog = 0
        variables = _xpress_variables(
            xp,
            model,
            qp,
            integer=any(bool(value) for value in qp.get("integerVars", [])),
        )
        _xpress_add_linear_rows(xp, model, variables, qp)
        model.setObjective(
            _xpress_objective(xp, variables, qp),
            sense=xp.maximize if qp.get("sense", "max") == "max" else xp.minimize,
        )
        _xpress_apply_options(
            model,
            _external_options(payload),
            integer=any(bool(value) for value in qp.get("integerVars", [])),
        )
        return _xpress_finish(
            model,
            variables,
            "xpress:qp",
            include_quality=any(bool(value) for value in qp.get("integerVars", [])),
            options=_external_options(payload),
        )


def _xpress_add_quadratic_constraints(xp: Any, model: Any, variables: Any, problem: dict[str, Any]) -> None:
    for row in problem.get("quadraticConstraints", []):
        expr = 0.0
        for term in row.get("quadratic", []):
            expr += (
                float(term["coeff"])
                * variables[int(term["i"])]
                * variables[int(term["j"])]
            )
        expr += _xpress_linear_expr(xp, variables, row.get("linear", []))
        rhs = float(row.get("rhs", 0.0))
        sense = row.get("sense", "<=")
        if sense == ">=":
            model.addConstraint(expr >= rhs)
        elif sense in {"=", "=="}:
            model.addConstraint(expr == rhs)
        else:
            model.addConstraint(expr <= rhs)


def _xpress_add_soc_constraints(xp: Any, model: Any, variables: Any, problem: dict[str, Any]) -> None:
    for cone in problem.get("soc", []):
        lhs = 0.0
        for term in cone.get("terms", []):
            expr = _xpress_linear_expr(
                xp,
                variables,
                term.get("coeffs", []),
                float(term.get("constant", 0.0)),
            )
            lhs += expr * expr
        rhs = _xpress_linear_expr(
            xp,
            variables,
            cone.get("rhsCoeffs", []),
            float(cone.get("rhsConstant", 0.0)),
        )
        model.addConstraint(rhs >= 0.0)
        model.addConstraint(lhs <= rhs * rhs)


def solve_xpress_conic(payload: dict[str, Any]) -> dict[str, Any]:
    with contextlib.redirect_stdout(sys.stderr):
        xp = _import_xpress()
        conic = payload["conic"]
        model = xp.problem()
        model.controls.outputlog = 0
        variables = _xpress_variables(
            xp,
            model,
            conic,
            integer=any(bool(value) for value in conic.get("integerVars", [])),
        )
        _xpress_add_linear_rows(xp, model, variables, conic)
        _xpress_add_quadratic_constraints(xp, model, variables, conic)
        _xpress_add_soc_constraints(xp, model, variables, conic)
        model.setObjective(
            _xpress_objective(xp, variables, conic),
            sense=xp.maximize if conic.get("sense", "max") == "max" else xp.minimize,
        )
        _xpress_apply_options(
            model,
            _external_options(payload),
            integer=any(bool(value) for value in conic.get("integerVars", [])),
        )
        return _xpress_finish(
            model,
            variables,
            "xpress:conic",
            include_quality=any(bool(value) for value in conic.get("integerVars", [])),
            options=_external_options(payload),
        )


def _import_gurobi() -> Any:
    try:
        import gurobipy as gp
    except Exception as exc:
        raise RuntimeError(f"Gurobi unavailable: {exc}") from exc
    return gp


def _gurobi_status(gp: Any, status_code: int) -> str:
    status_map = {
        gp.GRB.OPTIMAL: "optimal",
        gp.GRB.INFEASIBLE: "infeasible",
        gp.GRB.UNBOUNDED: "unbounded",
        gp.GRB.TIME_LIMIT: "time-limit",
        gp.GRB.ITERATION_LIMIT: "iter-limit",
        gp.GRB.NODE_LIMIT: "node-limit",
    }
    return status_map.get(status_code, "numerical-error")


def _gurobi_linear_expr(gp: Any, variables: Any, coeffs: Any, constant: float = 0.0) -> Any:
    expr = gp.LinExpr(float(constant))
    for idx, coef in coeffs:
        value = float(coef)
        if value:
            expr.addTerms(value, variables[int(idx)])
    return expr


def _gurobi_add_linear_rows(gp: Any, model: Any, variables: Any, problem: dict[str, Any]) -> list[Any]:
    constraints = []
    for row, bound in zip(problem.get("A_ub") or [], problem.get("b_ub") or []):
        expr = gp.LinExpr()
        for var, coef in zip(variables, row):
            value = float(coef)
            if value:
                expr.addTerms(value, var)
        constraints.append(model.addConstr(expr <= float(bound)))
    for row, bound in zip(problem.get("A_eq") or [], problem.get("b_eq") or []):
        expr = gp.LinExpr()
        for var, coef in zip(variables, row):
            value = float(coef)
            if value:
                expr.addTerms(value, var)
        constraints.append(model.addConstr(expr == float(bound)))
    return constraints


def _gurobi_add_mip_rows(gp: Any, model: Any, variables: Any, mip: dict[str, Any]) -> None:
    for row, bound in _mip_rows(mip):
        expr = gp.LinExpr()
        for var, coef in zip(variables, row):
            value = float(coef)
            if value:
                expr.addTerms(value, var)
        model.addConstr(expr <= float(bound))


def _gurobi_variables(gp: Any, model: Any, problem: dict[str, Any], integer: bool) -> Any:
    c = problem.get("c", [])
    lower = problem.get("lb")
    upper = problem.get("ub")
    integer_vars = problem.get("integerVars") if integer else [False] * len(c)
    variables = []
    for i in range(len(c)):
        lo = 0.0 if lower is None or i >= len(lower) or lower[i] is None else float(lower[i])
        hi = gp.GRB.INFINITY
        if upper is not None and i < len(upper) and upper[i] is not None:
            hi = float(upper[i])
        vtype = gp.GRB.INTEGER if integer_vars[i] else gp.GRB.CONTINUOUS
        variables.append(model.addVar(lb=lo, ub=hi, vtype=vtype, name=f"x{i}"))
    return variables


def _gurobi_objective(gp: Any, variables: Any, problem: dict[str, Any]) -> Any:
    objective = gp.QuadExpr()
    for var, coef in zip(variables, problem.get("c", [])):
        value = float(coef)
        if value:
            objective += value * var
    for term in problem.get("quadratic", []):
        objective += (
            float(term["coeff"])
            * variables[int(term["i"])]
            * variables[int(term["j"])]
        )
    return objective


def _gurobi_apply_options(model: Any, options: dict[str, Any], integer: bool = False) -> None:
    time_limit = _time_limit_seconds(options)
    if time_limit is not None:
        model.Params.TimeLimit = time_limit
    if not integer:
        return
    node_limit = _node_limit(options)
    relative_gap = _relative_gap_limit(options)
    if node_limit is not None:
        model.Params.NodeLimit = node_limit
    if relative_gap is not None:
        model.Params.MIPGap = relative_gap


def _gurobi_finish(
    gp: Any,
    model: Any,
    variables: Any,
    solver_name: str,
    lp_problem: dict[str, Any] | None = None,
    linear_constraints: list[Any] | None = None,
    include_quality: bool = False,
) -> dict[str, Any]:
    model.optimize()
    status = _gurobi_status(gp, int(model.Status))
    x = [
        float(var.X) for var in variables
    ] if status in {"optimal", "iter-limit", "time-limit", "node-limit"} and model.SolCount else []
    objective = float(model.ObjVal) if x else None
    parsed = {
        "status": status,
        "x": x,
        "objective": objective,
        "solver": solver_name,
        "message": f"{solver_name} status={model.Status}",
    }
    if lp_problem is not None and linear_constraints is not None and status == "optimal":
        ub_count, eq_count = _lp_row_counts(lp_problem)
        row_duals = [float(constraint.Pi) for constraint in linear_constraints]
        reduced_costs = [float(var.RC) for var in variables]
        parsed.update(_lp_certificate_fields(row_duals, reduced_costs, ub_count, eq_count))
        parsed.update(
            _lp_basis_fields(
                [_gurobi_var_basis_status(int(var.VBasis)) for var in variables],
                _lp_row_basis_with_fixed_equalities(
                    [
                        _gurobi_row_basis_status(int(constraint.CBasis))
                        for constraint in linear_constraints[:ub_count]
                    ],
                    eq_count,
                ),
            )
        )
    if include_quality:
        parsed.update(
            _quality_fields(
                _first_attr(model, ("ObjBound",)),
                _first_attr(model, ("MIPGap",)),
                _first_attr(model, ("NodeCount",)),
            )
        )
    return parsed


def solve_gurobi_lp(payload: dict[str, Any]) -> dict[str, Any]:
    with contextlib.redirect_stdout(sys.stderr):
        gp = _import_gurobi()
        lp = payload["lp"]
        model = gp.Model()
        model.Params.OutputFlag = 0
        variables = _gurobi_variables(gp, model, lp, integer=False)
        linear_constraints = _gurobi_add_linear_rows(gp, model, variables, lp)
        model.setObjective(
            _gurobi_objective(gp, variables, lp),
            gp.GRB.MAXIMIZE if lp.get("sense", "max") == "max" else gp.GRB.MINIMIZE,
        )
        _gurobi_apply_options(model, _external_options(payload), integer=False)
        return _gurobi_finish(gp, model, variables, "gurobi:lp", lp, linear_constraints)


def solve_gurobi_mip(payload: dict[str, Any]) -> dict[str, Any]:
    with contextlib.redirect_stdout(sys.stderr):
        gp = _import_gurobi()
        mip = payload["mip"]
        model = gp.Model()
        model.Params.OutputFlag = 0
        variables = _gurobi_variables(gp, model, mip, integer=True)
        _gurobi_add_mip_rows(gp, model, variables, mip)
        model.setObjective(
            _gurobi_objective(gp, variables, mip),
            gp.GRB.MAXIMIZE if mip.get("sense", "max") == "max" else gp.GRB.MINIMIZE,
        )
        _gurobi_apply_options(model, _external_options(payload), integer=True)
        start = _mip_start(mip)
        if start is not None:
            for var, value in zip(variables, start):
                var.Start = value
        return _gurobi_finish(gp, model, variables, "gurobi:mip", include_quality=True)


def solve_gurobi_qp(payload: dict[str, Any]) -> dict[str, Any]:
    with contextlib.redirect_stdout(sys.stderr):
        gp = _import_gurobi()
        qp = payload["qp"]
        model = gp.Model()
        model.Params.OutputFlag = 0
        variables = _gurobi_variables(
            gp,
            model,
            qp,
            integer=any(bool(value) for value in qp.get("integerVars", [])),
        )
        _gurobi_add_linear_rows(gp, model, variables, qp)
        model.setObjective(
            _gurobi_objective(gp, variables, qp),
            gp.GRB.MAXIMIZE if qp.get("sense", "max") == "max" else gp.GRB.MINIMIZE,
        )
        _gurobi_apply_options(
            model,
            _external_options(payload),
            integer=any(bool(value) for value in qp.get("integerVars", [])),
        )
        return _gurobi_finish(
            gp,
            model,
            variables,
            "gurobi:qp",
            include_quality=any(bool(value) for value in qp.get("integerVars", [])),
        )


def _gurobi_add_quadratic_constraints(gp: Any, model: Any, variables: Any, problem: dict[str, Any]) -> None:
    for row in problem.get("quadraticConstraints", []):
        expr = gp.QuadExpr()
        for term in row.get("quadratic", []):
            expr += (
                float(term["coeff"])
                * variables[int(term["i"])]
                * variables[int(term["j"])]
            )
        expr += _gurobi_linear_expr(gp, variables, row.get("linear", []))
        rhs = float(row.get("rhs", 0.0))
        sense = row.get("sense", "<=")
        if sense == ">=":
            model.addQConstr(expr >= rhs)
        elif sense in {"=", "=="}:
            model.addQConstr(expr == rhs)
        else:
            model.addQConstr(expr <= rhs)


def _gurobi_add_soc_constraints(gp: Any, model: Any, variables: Any, problem: dict[str, Any]) -> None:
    for cone in problem.get("soc", []):
        lhs = gp.QuadExpr()
        for term in cone.get("terms", []):
            expr = _gurobi_linear_expr(
                gp,
                variables,
                term.get("coeffs", []),
                float(term.get("constant", 0.0)),
            )
            lhs += expr * expr
        rhs = _gurobi_linear_expr(
            gp,
            variables,
            cone.get("rhsCoeffs", []),
            float(cone.get("rhsConstant", 0.0)),
        )
        model.addConstr(rhs >= 0.0)
        model.addQConstr(lhs <= rhs * rhs)


def solve_gurobi_conic(payload: dict[str, Any]) -> dict[str, Any]:
    with contextlib.redirect_stdout(sys.stderr):
        gp = _import_gurobi()
        conic = payload["conic"]
        model = gp.Model()
        model.Params.OutputFlag = 0
        model.Params.NonConvex = 2
        variables = _gurobi_variables(
            gp,
            model,
            conic,
            integer=any(bool(value) for value in conic.get("integerVars", [])),
        )
        _gurobi_add_linear_rows(gp, model, variables, conic)
        _gurobi_add_quadratic_constraints(gp, model, variables, conic)
        _gurobi_add_soc_constraints(gp, model, variables, conic)
        model.setObjective(
            _gurobi_objective(gp, variables, conic),
            gp.GRB.MAXIMIZE if conic.get("sense", "max") == "max" else gp.GRB.MINIMIZE,
        )
        _gurobi_apply_options(
            model,
            _external_options(payload),
            integer=any(bool(value) for value in conic.get("integerVars", [])),
        )
        return _gurobi_finish(
            gp,
            model,
            variables,
            "gurobi:conic",
            include_quality=any(bool(value) for value in conic.get("integerVars", [])),
        )


def solve_ortools(payload: dict[str, Any], backend: str, integer: bool) -> dict[str, Any]:
    try:
        from ortools.linear_solver import pywraplp
    except Exception as exc:
        raise RuntimeError(f"OR-Tools unavailable: {exc}") from exc

    problem = payload["mip"] if integer else payload["lp"]
    solver = pywraplp.Solver.CreateSolver(backend)
    if solver is None:
        raise RuntimeError(f"OR-Tools backend unavailable: {backend}")

    c = [float(v) for v in problem.get("c", [])]
    upper = problem.get("ub")
    lower = problem.get("lb") if not integer else None
    integer_vars = problem.get("integerVars") if integer else [False] * len(c)
    variables = []
    for i in range(len(c)):
        lo = 0.0 if lower is None or i >= len(lower) or lower[i] is None else float(lower[i])
        hi = solver.infinity()
        if upper is not None and i < len(upper) and upper[i] is not None:
            hi = float(upper[i])
        if integer_vars[i]:
            variables.append(solver.IntVar(lo, hi, f"x{i}"))
        else:
            variables.append(solver.NumVar(lo, hi, f"x{i}"))

    linear_constraints = []
    if integer:
        for row, bound in _mip_rows(problem):
            constraint = solver.RowConstraint(-solver.infinity(), float(bound), "")
            for var, coef in zip(variables, row):
                if coef:
                    constraint.SetCoefficient(var, float(coef))
    else:
        for row, bound in zip(problem.get("A_ub") or [], problem.get("b_ub") or []):
            constraint = solver.RowConstraint(-solver.infinity(), float(bound), "")
            linear_constraints.append(constraint)
            for var, coef in zip(variables, row):
                if coef:
                    constraint.SetCoefficient(var, float(coef))
        for row, bound in zip(problem.get("A_eq") or [], problem.get("b_eq") or []):
            constraint = solver.RowConstraint(float(bound), float(bound), "")
            linear_constraints.append(constraint)
            for var, coef in zip(variables, row):
                if coef:
                    constraint.SetCoefficient(var, float(coef))

    objective = solver.Objective()
    for var, coef in zip(variables, c):
        objective.SetCoefficient(var, float(coef))
    if problem.get("sense", "max") == "max":
        objective.SetMaximization()
    else:
        objective.SetMinimization()
    start = _mip_start(problem) if integer else None
    if start is not None and hasattr(solver, "SetHint"):
        solver.SetHint(variables, start)
    options = _external_options(payload)
    time_limit = _time_limit_seconds(options)
    if time_limit is not None and hasattr(solver, "SetTimeLimit"):
        solver.SetTimeLimit(int(math.ceil(time_limit * 1000.0)))
    if integer:
        specific = []
        relative_gap = _relative_gap_limit(options)
        node_limit = _node_limit(options)
        if relative_gap is not None:
            specific.append(f"limits/gap = {relative_gap:.17g}")
        if node_limit is not None:
            specific.append(f"limits/nodes = {node_limit}")
        if specific and hasattr(solver, "SetSolverSpecificParametersAsString"):
            try:
                solver.SetSolverSpecificParametersAsString("\n".join(specific))
            except Exception:
                pass

    status_code = solver.Solve()
    status = {
        pywraplp.Solver.OPTIMAL: "optimal",
        pywraplp.Solver.FEASIBLE: "iter-limit",
        pywraplp.Solver.INFEASIBLE: "infeasible",
        pywraplp.Solver.UNBOUNDED: "unbounded",
        pywraplp.Solver.ABNORMAL: "numerical-error",
        pywraplp.Solver.NOT_SOLVED: "numerical-error",
    }.get(status_code, "numerical-error")
    status = _limited_status(status, options)
    x = [var.solution_value() for var in variables] if status in {"optimal", "iter-limit"} else []
    objective_value = objective.Value() if status in {"optimal", "iter-limit"} else None
    parsed = {
        "status": status,
        "x": x,
        "objective": objective_value,
        "message": f"ortools:{backend} status={status_code}",
    }
    if not integer and status == "optimal":
        try:
            ub_count, eq_count = _lp_row_counts(problem)
            row_duals = [float(constraint.dual_value()) for constraint in linear_constraints]
            reduced_costs = [float(var.reduced_cost()) for var in variables]
            parsed.update(
                _lp_certificate_fields(row_duals, reduced_costs, ub_count, eq_count)
            )
            parsed.update(
                _lp_basis_fields(
                    [
                        _ortools_basis_status(pywraplp, int(var.basis_status()))
                        for var in variables
                    ],
                    [
                        _ortools_basis_status(
                            pywraplp, int(constraint.basis_status())
                        )
                        for constraint in linear_constraints
                    ],
                )
            )
        except Exception as exc:
            parsed["message"] = f"{parsed['message']}; certificate extraction failed: {exc}"
    if integer and status in {"optimal", "iter-limit"}:
        best_bound = None
        nodes_explored = None
        try:
            best_bound = objective.BestBound()
        except Exception:
            pass
        try:
            nodes_explored = solver.nodes()
        except Exception:
            pass
        parsed.update(
            _quality_fields(
                best_bound,
                _relative_gap(best_bound, objective_value),
                nodes_explored,
            )
        )
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="highs")
    args = parser.parse_args()

    try:
        payload = json.load(sys.stdin)
        kind = payload.get("kind")
        if kind == "lp":
            result = solve_lp(payload, args.method)
        elif kind == "qp":
            result = solve_qp(payload, args.method)
        elif kind == "conic":
            result = solve_conic(payload, args.method)
        elif kind == "mip":
            result = solve_mip(payload, args.method)
        else:
            raise ValueError(f"unknown kind: {kind}")
    except Exception as exc:
        result = {
            "status": "numerical-error",
            "x": [],
            "objective": None,
            "message": str(exc),
        }

    print(json.dumps(_clean(result)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
