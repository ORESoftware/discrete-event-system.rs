#!/usr/bin/env python3
"""Reference bridge for small nonlinear optimization validation payloads.

The bridge keeps heavyweight nonlinear solvers optional. It accepts a compact
JSON expression model over variables named x0, x1, ... or by the names supplied
in the variable list, prefers SciPy SLSQP when available, and delegates fallback
bounded grid-plus-pattern search to the Rust reference binary.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import subprocess
import sys
from typing import Any, Callable


ALLOWED_FUNCS: dict[str, Callable[..., float]] = {
    "abs": abs,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "exp": math.exp,
    "log": math.log,
    "sqrt": math.sqrt,
    "pow": pow,
    "min": min,
    "max": max,
}

SCIPY_BRIDGE_SOLVERS = (
    "scipy",
    "ipopt",
    "bonmin",
    "minotaur",
    "couenne",
    "symphony",
    "knitro",
    "mosek",
    "baron",
    "copt",
)

PACKAGE_BRIDGE_SOLVERS = ("casadi", "nlopt", "nlopt-cli")


def result(
    status: str,
    solver: str,
    x: list[float] | None = None,
    objective: float | None = None,
    message: str = "",
    iterations: int = 0,
) -> dict[str, Any]:
    return {
        "status": status,
        "solver": solver,
        "x": x or [],
        "objective": objective,
        "message": message,
        "iterations": iterations,
    }


def rust_reference_command() -> list[str]:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    binary_name = "nonlinear_validation_reference"
    explicit = os.environ.get("NONLINEAR_VALIDATION_REFERENCE_RUST_BIN")
    if explicit:
        return [explicit]
    local_binary = os.path.join(repo_root, "target", "debug", binary_name)
    if local_rust_binary_is_current(repo_root, local_binary):
        return [local_binary]
    return ["cargo", "run", "--quiet", "--bin", binary_name, "--"]


def local_rust_binary_is_current(repo_root: str, binary_path: str) -> bool:
    if not os.path.exists(binary_path):
        return False
    binary_mtime = os.path.getmtime(binary_path)
    source_paths = [
        os.path.join(repo_root, "src", "bin", "nonlinear_validation_reference.rs"),
        os.path.join(
            repo_root,
            "src",
            "des",
            "general",
            "external_nonlinear_validation_reference.rs",
        ),
    ]
    return all(
        not os.path.exists(source_path) or os.path.getmtime(source_path) <= binary_mtime
        for source_path in source_paths
    )


def rust_fallback_reference(payload: dict[str, Any], solver: str = "fallback") -> dict[str, Any]:
    command = rust_reference_command()
    cwd = None
    if command[0] == "cargo":
        script_dir = os.path.dirname(os.path.abspath(__file__))
        cwd = os.path.dirname(script_dir)
    completed = subprocess.run(
        [*command, "--solver", solver],
        input=json.dumps(payload),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=cwd,
        check=False,
    )
    try:
        parsed = json.loads(completed.stdout)
    except Exception as exc:
        return result(
            "failed",
            "rust:nonlinear-validation-reference",
            [],
            None,
            f"failed to parse Rust nonlinear validation output: {exc}; stderr={completed.stderr.strip()}",
        )
    if completed.returncode != 0 and not parsed.get("message"):
        parsed["message"] = completed.stderr.strip()
    return parsed


class SafeExpression:
    def __init__(self, expression: str) -> None:
        self.expression = expression
        self.tree = ast.parse(expression, mode="eval")

    def eval(self, env: dict[str, float]) -> float:
        value = self._eval_node(self.tree.body, env)
        out = float(value)
        if not math.isfinite(out):
            raise ValueError(f"expression {self.expression!r} produced a non-finite value")
        return out

    def _eval_node(self, node: ast.AST, env: dict[str, float]) -> float:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
                return float(node.value)
            raise ValueError(f"unsupported constant in {self.expression!r}")
        if isinstance(node, ast.Name):
            if node.id in env:
                return env[node.id]
            raise ValueError(f"unknown variable {node.id!r} in {self.expression!r}")
        if isinstance(node, ast.UnaryOp):
            value = self._eval_node(node.operand, env)
            if isinstance(node.op, ast.USub):
                return -value
            if isinstance(node.op, ast.UAdd):
                return value
            raise ValueError(f"unsupported unary operator in {self.expression!r}")
        if isinstance(node, ast.BinOp):
            left = self._eval_node(node.left, env)
            right = self._eval_node(node.right, env)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Pow):
                return left**right
            raise ValueError(f"unsupported binary operator in {self.expression!r}")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            func = ALLOWED_FUNCS.get(node.func.id)
            if func is None:
                raise ValueError(f"unsupported function {node.func.id!r} in {self.expression!r}")
            args = [self._eval_node(arg, env) for arg in node.args]
            return float(func(*args))
        raise ValueError(f"unsupported expression node in {self.expression!r}")


def normalize(payload: dict[str, Any]) -> dict[str, Any]:
    variables = payload.get("variables")
    if not isinstance(variables, list) or not variables:
        variables = [{"name": f"x{idx}"} for idx in range(int(payload.get("dimension", 0)))]
    if not variables:
        raise ValueError("nonlinear payload needs variables or dimension")
    names: list[str] = []
    lb: list[float] = []
    ub: list[float] = []
    x0: list[float] = []
    for idx, raw in enumerate(variables):
        item = raw if isinstance(raw, dict) else {"name": str(raw)}
        name = str(item.get("name") or f"x{idx}")
        lower = float(item.get("lb", item.get("lower", -10.0)))
        upper = float(item.get("ub", item.get("upper", 10.0)))
        if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
            raise ValueError(f"variable {name} needs finite ordered bounds")
        start = float(item.get("start", item.get("initial", 0.5 * (lower + upper))))
        start = min(max(start, lower), upper)
        names.append(name)
        lb.append(lower)
        ub.append(upper)
        x0.append(start)
    objective = str(payload.get("objective", "0"))
    constraints = payload.get("constraints", [])
    if not isinstance(constraints, list):
        raise ValueError("constraints must be an array")
    return {
        "names": names,
        "lb": lb,
        "ub": ub,
        "x0": x0,
        "sense": str(payload.get("sense", "min")).lower(),
        "objective": SafeExpression(objective),
        "constraints": [
            {
                "expr": SafeExpression(str(item.get("expr", item.get("expression", "0")))),
                "sense": str(item.get("sense", "<=")),
                "rhs": float(item.get("rhs", 0.0)),
                "name": str(item.get("name", f"c{idx}")),
            }
            for idx, item in enumerate(constraints)
            if isinstance(item, dict)
        ],
    }


def env_for(model: dict[str, Any], x: list[float]) -> dict[str, float]:
    env = {f"x{idx}": value for idx, value in enumerate(x)}
    env.update({name: value for name, value in zip(model["names"], x)})
    return env


def objective_value(model: dict[str, Any], x: list[float]) -> float:
    value = model["objective"].eval(env_for(model, x))
    return -value if model["sense"] in ("max", "maximize") else value


def public_objective(model: dict[str, Any], x: list[float]) -> float:
    value = model["objective"].eval(env_for(model, x))
    return value


def constraint_violation(model: dict[str, Any], x: list[float]) -> float:
    env = env_for(model, x)
    total = 0.0
    for constraint in model["constraints"]:
        lhs = constraint["expr"].eval(env)
        rhs = constraint["rhs"]
        sense = constraint["sense"]
        if sense in ("<=", "le", "less-equal"):
            total += max(0.0, lhs - rhs) ** 2
        elif sense in (">=", "ge", "greater-equal"):
            total += max(0.0, rhs - lhs) ** 2
        elif sense in ("=", "==", "eq"):
            total += (lhs - rhs) ** 2
        else:
            raise ValueError(f"unsupported constraint sense {sense!r}")
    return math.sqrt(total)


def feasible(model: dict[str, Any], x: list[float], tol: float = 1e-6) -> bool:
    return constraint_violation(model, x) <= tol


def scipy_reference(payload: dict[str, Any], solver_label: str) -> dict[str, Any] | None:
    try:
        import numpy as np  # type: ignore
        from scipy.optimize import Bounds, NonlinearConstraint, minimize  # type: ignore
    except Exception:
        return None
    model = normalize(payload)
    constraints = []
    for constraint in model["constraints"]:
        expr = constraint["expr"]
        rhs = constraint["rhs"]
        if constraint["sense"] in ("<=", "le", "less-equal"):
            lower = -np.inf
            upper = rhs
        elif constraint["sense"] in (">=", "ge", "greater-equal"):
            lower = rhs
            upper = np.inf
        else:
            lower = rhs
            upper = rhs

        def fun(x, expr=expr):
            return float(expr.eval(env_for(model, [float(value) for value in x])))

        constraints.append(NonlinearConstraint(fun, lower, upper))

    def fun(x):
        return float(objective_value(model, [float(value) for value in x]))

    res = minimize(
        fun,
        np.array(model["x0"], dtype=float),
        bounds=Bounds(model["lb"], model["ub"]),
        constraints=constraints,
        method="SLSQP",
        options={"ftol": 1e-10, "maxiter": 500},
    )
    x = [float(value) for value in res.x] if getattr(res, "x", None) is not None else []
    if res.success and feasible(model, x):
        return result("optimal", solver_label, x, public_objective(model, x), str(res.message), int(res.nit))
    fallback = rust_fallback_reference(payload)
    if fallback["status"] == "optimal":
        fallback["solver"] = f"{solver_label}+fallback"
        fallback["message"] = f"{res.message}; fallback recovered feasible solution"
        return fallback
    return result("infeasible", solver_label, x, public_objective(model, x) if x else None, str(res.message), int(res.nit))


def package_reference(payload: dict[str, Any], package: str) -> dict[str, Any] | None:
    if package == "casadi":
        try:
            import casadi  # type: ignore # noqa: F401
        except Exception:
            return None
        scipy = scipy_reference(payload, "casadi:scipy-bridge")
        return scipy
    if package == "nlopt":
        try:
            import nlopt  # type: ignore # noqa: F401
        except Exception:
            return None
        scipy = scipy_reference(payload, "nlopt:scipy-bridge")
        return scipy
    return None


def dispatch(payload: dict[str, Any], requested: str) -> dict[str, Any]:
    solver = requested.strip().lower().replace("_", "-")
    if solver == "auto":
        return rust_fallback_reference(payload, solver)
    if solver in SCIPY_BRIDGE_SOLVERS:
        scipy = scipy_reference(payload, "scipy:SLSQP" if solver in ("auto", "scipy") else f"{solver}:scipy-bridge")
        if scipy is not None:
            return scipy
        if solver != "auto":
            return rust_fallback_reference(payload, solver)
    if solver in PACKAGE_BRIDGE_SOLVERS:
        package = "nlopt" if solver in ("nlopt", "nlopt-cli") else solver
        package_result = package_reference(payload, package)
        if package_result is not None:
            return package_result
        return rust_fallback_reference(payload, solver)
    return rust_fallback_reference(payload)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", default="auto")
    args = parser.parse_args()
    try:
        payload = json.load(sys.stdin)
        if str(payload.get("kind", "nonlinear-validation")).replace("_", "-") not in (
            "nonlinear-validation",
            "nlp-validation",
        ):
            raise ValueError("payload kind must be nonlinear-validation or nlp-validation")
        print(json.dumps(dispatch(payload, args.solver)))
    except Exception as exc:
        print(json.dumps(result("failed", args.solver, [], None, str(exc))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
