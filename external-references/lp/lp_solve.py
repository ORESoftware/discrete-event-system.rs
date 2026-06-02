#!/usr/bin/env python3
"""Small scipy.linprog bridge for des_engine's LP external solver.

The Rust side writes {"lp": ..., "method": "..."} on stdin and expects a
single JSON object on stdout. Keep this script dependency-light so it can serve
as an optional validation oracle whenever Python + SciPy are available.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from typing import Any


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


def _array_or_none(value: Any) -> Any:
    if value is None:
        return None
    return value


def _bounds(lp: dict[str, Any]) -> list[tuple[float | None, float | None]]:
    n = len(lp.get("c", []))
    lower = lp.get("lb")
    upper = lp.get("ub")
    bounds = []
    for i in range(n):
        lo = lower[i] if lower is not None and i < len(lower) else 0.0
        hi = upper[i] if upper is not None and i < len(upper) else None
        bounds.append((lo, hi))
    return bounds


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


def _marginals(section: Any) -> Any:
    if section is None:
        return None
    values = getattr(section, "marginals", None)
    if values is None:
        return None
    return [float(v) for v in values]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="highs")
    args = parser.parse_args()

    try:
        from scipy.optimize import linprog
    except Exception as exc:  # pragma: no cover - depends on host environment
        print(
            json.dumps(
                {
                    "status": "numerical-error",
                    "x": [],
                    "objective": None,
                    "message": f"scipy unavailable: {exc}",
                }
            )
        )
        return 0

    try:
        payload = json.load(sys.stdin)
        lp = payload["lp"]
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "numerical-error",
                    "x": [],
                    "objective": None,
                    "message": f"invalid LP payload: {exc}",
                }
            )
        )
        return 0

    sense = lp.get("sense", "max")
    c = [float(v) for v in lp.get("c", [])]
    scipy_c = [-v for v in c] if sense == "max" else c

    try:
        result = linprog(
            scipy_c,
            A_ub=_array_or_none(lp.get("A_ub")),
            b_ub=_array_or_none(lp.get("b_ub")),
            A_eq=_array_or_none(lp.get("A_eq")),
            b_eq=_array_or_none(lp.get("b_eq")),
            bounds=_bounds(lp),
            method=args.method,
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "numerical-error",
                    "x": [],
                    "objective": None,
                    "message": f"scipy linprog failed: {exc}",
                }
            )
        )
        return 0

    x = [float(v) for v in result.x] if result.x is not None else []
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
        reduced_costs = [sign * (a + b) for a, b in zip(lower, upper)]

    print(
        json.dumps(
            _clean(
                {
                    "status": _status(int(result.status)),
                    "x": x,
                    "objective": objective,
                    "dualUB": dual_ub,
                    "dualEQ": dual_eq,
                    "reducedCosts": reduced_costs,
                    "iters": getattr(result, "nit", None),
                    "message": str(result.message),
                }
            )
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
