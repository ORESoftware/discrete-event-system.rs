#!/usr/bin/env python3
"""Direct LP/MIP bridge for installed solver CLIs.

The Rust optimization suite already cross-checks through Python APIs such as
SciPy and OR-Tools. This bridge exercises actual command-line solvers
(`highs`, `glpsol`, `scip`, `cbc`, LP-only `clp`, and optional commercial
CLIs such as `gurobi_cl`, `cplex`, FICO Xpress `optimizer`, and LINDO
`runlindo`) on the same small validation models by writing a solver-readable
LP/MPS file, invoking the solver, and parsing the primal solution.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from typing import Optional, Sequence

from ip_mip_reference import expand_source_features
from lp_solve import dot, normalize_lp


COMMAND_ALIASES = {
    "glpk": ["glpsol"],
    "highs": ["highs"],
    "scip": ["scip"],
    "cbc": ["cbc"],
    "clp": ["clp"],
    "gurobi": ["gurobi_cl"],
    "cplex": ["cplex"],
    "xpress": ["optimizer", "xpress"],
    "lindo": ["runlindo", "lindo", "lindoapi"],
}

COMMAND_ENV_VARS = {
    "glpk": ["GLPSOL_CMD", "GLPK_CMD", "ORES_GLPK_CMD"],
    "highs": ["HIGHS_CMD", "ORES_HIGHS_CMD"],
    "scip": ["SCIP_CMD", "ORES_SCIP_CMD"],
    "cbc": ["CBC_CMD", "ORES_CBC_CMD"],
    "clp": ["CLP_CMD", "ORES_CLP_CMD"],
    "gurobi": ["GUROBI_CL_CMD", "GUROBI_CMD", "ORES_GUROBI_CMD"],
    "cplex": ["CPLEX_CMD", "ORES_CPLEX_CMD"],
    "xpress": ["XPRESS_CMD", "XPRESS_OPTIMIZER_CMD", "ORES_XPRESS_CMD"],
    "lindo": ["RUNLINDO_CMD", "LINDO_CMD", "LINDOAPI_CMD", "ORES_LINDO_CMD"],
}

SUPPORTED_SOLVERS = {
    "glpk",
    "highs",
    "scip",
    "cbc",
    "clp",
    "gurobi",
    "cplex",
    "xpress",
    "lindo",
}


def basis_status_from_token(token: object) -> Optional[str]:
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


def status_payload(status: str, solver: str, message: str = "") -> dict:
    return {
        "status": status,
        "solver": solver,
        "x": [],
        "objective": None,
        "message": message,
    }


def var_name(index: int) -> str:
    return f"x{index}"


def finite(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def term_expr(coefs: Sequence[float], names: Sequence[str]) -> str:
    parts: list[str] = []
    for coef, name in zip(coefs, names):
        coef = float(coef)
        if abs(coef) <= 1e-12:
            continue
        sign = "-" if coef < 0 else "+"
        mag = abs(coef)
        body = name if abs(mag - 1.0) <= 1e-12 else f"{mag:.12g} {name}"
        if not parts:
            parts.append(f"- {body}" if sign == "-" else body)
        else:
            parts.append(f"{sign} {body}")
    return " ".join(parts) if parts else f"0 {names[0] if names else 'x0'}"


def normalize_mip(raw: dict) -> tuple[str, list[float], list[list[float]], list[float], list[Optional[float]], list[Optional[float]], list[bool]]:
    p = expand_source_features(raw)
    c = [float(v) for v in p["c"]]
    n = len(c)
    rows = [[float(v) for v in row] for row in p.get("a", [])]
    rhs = [float(v) for v in p.get("b", [])]
    if len(rows) != len(rhs):
        raise ValueError("MIP row/RHS length mismatch")
    for row in rows:
        if len(row) != n:
            raise ValueError("MIP row length mismatch")
    lb_raw = p.get("lb")
    ub_raw = p.get("ub")
    lbs = [0.0] * n if lb_raw is None else [0.0 if v is None else float(v) for v in lb_raw]
    ubs = [None] * n if ub_raw is None else [finite(v) for v in ub_raw]
    ints = [bool(v) for v in p.get("integer_vars", [False] * n)]
    if len(lbs) != n or len(ubs) != n or len(ints) != n:
        raise ValueError("MIP bound/integrality vector length mismatch")
    return p.get("sense", "max"), c, rows, rhs, lbs, ubs, ints


def write_cplex_lp(
    path: str,
    sense: str,
    c: Sequence[float],
    le_rows: Sequence[Sequence[float]],
    le_rhs: Sequence[float],
    eq_rows: Sequence[Sequence[float]],
    eq_rhs: Sequence[float],
    lbs: Sequence[Optional[float]],
    ubs: Sequence[Optional[float]],
    integer_vars: Sequence[bool],
) -> list[str]:
    n = len(c)
    names = [var_name(i) for i in range(n)]
    binary_vars = [
        i
        for i, is_int in enumerate(integer_vars)
        if is_int
        and (lbs[i] is None or abs(float(lbs[i])) <= 1e-12)
        and ubs[i] is not None
        and abs(float(ubs[i]) - 1.0) <= 1e-12
    ]
    binary_set = set(binary_vars)
    general_vars = [i for i, is_int in enumerate(integer_vars) if is_int and i not in binary_set]

    with open(path, "w", encoding="utf-8") as f:
        f.write("Maximize\n" if sense == "max" else "Minimize\n")
        f.write(f" obj: {term_expr(c, names)}\n")
        f.write("Subject To\n")
        for i, (row, rhs) in enumerate(zip(le_rows, le_rhs)):
            f.write(f" c{i}: {term_expr(row, names)} <= {float(rhs):.12g}\n")
        offset = len(le_rows)
        for i, (row, rhs) in enumerate(zip(eq_rows, eq_rhs)):
            f.write(f" e{i}: {term_expr(row, names)} = {float(rhs):.12g}\n")
        if not le_rows and not eq_rows:
            f.write(" c0: 0 x0 <= 0\n")
        f.write("Bounds\n")
        for i, name in enumerate(names):
            if i in binary_set:
                continue
            lb = lbs[i]
            ub = ubs[i]
            if lb is None and ub is None:
                f.write(f" {name} free\n")
            elif lb is None:
                f.write(f" {name} <= {float(ub):.12g}\n")
            elif ub is None:
                f.write(f" {float(lb):.12g} <= {name}\n")
            else:
                f.write(f" {float(lb):.12g} <= {name} <= {float(ub):.12g}\n")
        if general_vars:
            f.write("General\n")
            f.write(" " + " ".join(names[i] for i in general_vars) + "\n")
        if binary_vars:
            f.write("Binary\n")
            f.write(" " + " ".join(names[i] for i in binary_vars) + "\n")
        f.write("End\n")
    return names


def write_free_mps(
    path: str,
    sense: str,
    c: Sequence[float],
    le_rows: Sequence[Sequence[float]],
    le_rhs: Sequence[float],
    eq_rows: Sequence[Sequence[float]],
    eq_rhs: Sequence[float],
    lbs: Sequence[Optional[float]],
    ubs: Sequence[Optional[float]],
    integer_vars: Sequence[bool],
) -> list[str]:
    n = len(c)
    names = [var_name(i) for i in range(n)]
    rows = [("L", f"c{i}", row, rhs) for i, (row, rhs) in enumerate(zip(le_rows, le_rhs))]
    rows.extend(("E", f"e{i}", row, rhs) for i, (row, rhs) in enumerate(zip(eq_rows, eq_rhs)))
    if not rows:
        rows = [("L", "c0", [0.0] * n, 0.0)]

    with open(path, "w", encoding="utf-8") as f:
        f.write("NAME          ORESCLI\n")
        f.write("OBJSENSE\n")
        f.write(" MAX\n" if sense == "max" else " MIN\n")
        f.write("ROWS\n")
        f.write(" N  OBJ\n")
        for row_sense, row_name, _, _ in rows:
            f.write(f" {row_sense}  {row_name}\n")

        f.write("COLUMNS\n")
        in_integer = False
        marker = 0
        for idx, name in enumerate(names):
            if integer_vars[idx] and not in_integer:
                f.write(f"    MARK{marker:04d}  'MARKER'                 'INTORG'\n")
                marker += 1
                in_integer = True
            elif not integer_vars[idx] and in_integer:
                f.write(f"    MARK{marker:04d}  'MARKER'                 'INTEND'\n")
                marker += 1
                in_integer = False

            entries: list[tuple[str, float]] = []
            if abs(float(c[idx])) > 1e-12:
                entries.append(("OBJ", float(c[idx])))
            for _, row_name, row, _ in rows:
                value = float(row[idx])
                if abs(value) > 1e-12:
                    entries.append((row_name, value))
            if not entries:
                entries.append(("OBJ", 0.0))
            for pos in range(0, len(entries), 2):
                first_name, first_value = entries[pos]
                line = f"    {name}  {first_name}  {first_value:.17g}"
                if pos + 1 < len(entries):
                    second_name, second_value = entries[pos + 1]
                    line += f"  {second_name}  {second_value:.17g}"
                f.write(line + "\n")
        if in_integer:
            f.write(f"    MARK{marker:04d}  'MARKER'                 'INTEND'\n")

        f.write("RHS\n")
        for _, row_name, _, rhs in rows:
            f.write(f"    RHS1  {row_name}  {float(rhs):.17g}\n")

        f.write("BOUNDS\n")
        for idx, name in enumerate(names):
            lb = lbs[idx]
            ub = ubs[idx]
            is_binary = (
                integer_vars[idx]
                and lb is not None
                and abs(float(lb)) <= 1e-12
                and ub is not None
                and abs(float(ub) - 1.0) <= 1e-12
            )
            if is_binary:
                f.write(f" BV BND1  {name}\n")
                continue
            if lb is None and ub is None:
                f.write(f" FR BND1  {name}\n")
            elif lb is None:
                f.write(f" MI BND1  {name}\n")
            elif abs(float(lb)) > 1e-12:
                f.write(f" LO BND1  {name}  {float(lb):.17g}\n")
            if ub is not None:
                f.write(f" UP BND1  {name}  {float(ub):.17g}\n")
        f.write("ENDATA\n")
    return names


def parse_highs_solution(
    path: str,
    n: int,
    le_count: int = 0,
    eq_count: int = 0,
) -> tuple[str, list[float], dict[str, object]]:
    x = [0.0] * n
    status = "unknown"
    dual_columns: list[Optional[float]] = [None] * n
    dual_rows: dict[str, float] = {}
    var_basis: list[Optional[str]] = [None] * n
    row_basis: dict[str, str] = {}
    section: Optional[str] = None
    block: Optional[str] = None
    remaining = 0
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f]
    for i, line in enumerate(lines):
        if line == "Model status" and i + 1 < len(lines):
            status = lines[i + 1].lower()
        if line == "# Primal solution values":
            section = "primal"
            block = None
            continue
        if line == "# Dual solution values":
            section = "dual"
            block = None
            continue
        if line.startswith("# Basis"):
            section = "basis"
            block = None
            continue
        if section is None:
            continue
        if line.startswith("# Columns"):
            block = "columns"
            remaining = int(line.split()[2])
            continue
        if line.startswith("# Rows"):
            block = "rows"
            remaining = int(line.split()[2])
            continue
        if not line or line.startswith("#") or block is None:
            continue
        parts = line.split()
        if len(parts) >= 2:
            value = float(parts[1]) if _is_number(parts[1]) else None
            if block == "columns" and parts[0].startswith("x") and parts[0][1:].isdigit():
                idx = int(parts[0][1:])
                if 0 <= idx < n:
                    if value is not None and section == "primal":
                        x[idx] = value
                    elif value is not None and section == "dual":
                        dual_columns[idx] = value
                    elif section == "basis":
                        basis_status = basis_status_from_token(parts[1])
                        if basis_status is not None:
                            var_basis[idx] = basis_status
            elif section == "dual" and block == "rows":
                if value is not None:
                    dual_rows[parts[0]] = value
            elif section == "basis" and block == "rows":
                basis_status = basis_status_from_token(parts[1])
                if basis_status is not None:
                    row_basis[parts[0]] = basis_status
        remaining -= 1
        if remaining <= 0:
            block = None

    fields: dict[str, object] = {}
    if all(value is not None for value in dual_columns):
        fields["reducedCosts"] = [float(value) for value in dual_columns if value is not None]
    dual_ub = [dual_rows.get(f"c{i}") for i in range(le_count)]
    dual_eq = [dual_rows.get(f"e{i}") for i in range(eq_count)]
    if all(value is not None for value in dual_ub):
        fields["dualUB"] = [float(value) for value in dual_ub if value is not None]
    if all(value is not None for value in dual_eq):
        fields["dualEQ"] = [float(value) for value in dual_eq if value is not None]
    if all(value is not None for value in var_basis):
        fields["varBasis"] = [str(value) for value in var_basis if value is not None]
    row_statuses = [row_basis.get(f"c{i}") for i in range(le_count)]
    row_statuses.extend(row_basis.get(f"e{i}") for i in range(eq_count))
    if all(value is not None for value in row_statuses):
        fields["rowBasis"] = [str(value) for value in row_statuses if value is not None]
    return status, x, fields


def parse_glpk_solution(
    path: str,
    n: int,
    le_count: int = 0,
    eq_count: int = 0,
) -> tuple[str, list[float], dict[str, object]]:
    x = [0.0] * n
    status = "unknown"
    row_duals: list[Optional[float]] = [None] * (le_count + eq_count)
    reduced_costs: list[Optional[float]] = [None] * n
    var_basis: list[Optional[str]] = [None] * n
    row_basis: list[Optional[str]] = [None] * (le_count + eq_count)
    in_named_columns = False
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 3 and parts[0] == "c" and parts[1] == "Status:":
                status = " ".join(parts[2:]).lower()
            elif len(parts) >= 2 and parts[0] == "Status:":
                status = " ".join(parts[1:]).lower()
            elif "Column name" in line:
                in_named_columns = True
            elif in_named_columns and stripped_starts(line, ("Integer feasibility", "KKT.", "End of output")):
                in_named_columns = False
            elif in_named_columns and len(parts) >= 3 and parts[0].isdigit() and parts[1].startswith("x"):
                suffix = parts[1][1:]
                if suffix.isdigit():
                    idx = int(suffix)
                    if 0 <= idx < n:
                        for token in parts[2:]:
                            if token != "*" and _is_number(token):
                                x[idx] = float(token)
                                break
            elif len(parts) >= 3 and parts[0] == "j":
                idx = int(parts[1]) - 1
                if 0 <= idx < n:
                    if len(parts) >= 4 and not _is_number(parts[2]):
                        x[idx] = float(parts[3])
                        status_token = basis_status_from_token(parts[2])
                        if status_token is not None:
                            var_basis[idx] = status_token
                        if len(parts) >= 5 and _is_number(parts[4]):
                            reduced_costs[idx] = float(parts[4])
                    else:
                        x[idx] = float(parts[2])
            elif len(parts) >= 5 and parts[0] == "i":
                idx = int(parts[1]) - 1
                if 0 <= idx < len(row_duals):
                    status_token = basis_status_from_token(parts[2])
                    if status_token is not None:
                        row_basis[idx] = status_token
                    if _is_number(parts[4]):
                        row_duals[idx] = float(parts[4])

    fields: dict[str, object] = {}
    if all(value is not None for value in reduced_costs):
        fields["reducedCosts"] = [
            float(value) for value in reduced_costs if value is not None
        ]
    dual_ub = row_duals[:le_count]
    dual_eq = row_duals[le_count:]
    if all(value is not None for value in dual_ub):
        fields["dualUB"] = [float(value) for value in dual_ub if value is not None]
    if all(value is not None for value in dual_eq):
        fields["dualEQ"] = [float(value) for value in dual_eq if value is not None]
    if all(value is not None for value in var_basis):
        fields["varBasis"] = [str(value) for value in var_basis if value is not None]
    if all(value is not None for value in row_basis):
        fields["rowBasis"] = [str(value) for value in row_basis if value is not None]
    return status, x, fields


def parse_scip_solution(path: str, n: int) -> tuple[str, list[float]]:
    x = [0.0] * n
    status = "unknown"
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("solution status:"):
                status = stripped.split(":", 1)[1].strip().lower()
            elif stripped.startswith("x"):
                parts = stripped.split()
                if len(parts) >= 2 and parts[0][1:].isdigit():
                    idx = int(parts[0][1:])
                    if 0 <= idx < n:
                        x[idx] = float(parts[1])
    return status, x


def parse_cbc_basis(
    path: str,
    n: int,
    le_count: int,
    eq_count: int,
) -> dict[str, object]:
    var_basis: list[Optional[str]] = [None] * n
    row_basis: list[Optional[str]] = ["basic"] * le_count + ["fixed"] * eq_count
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if not parts or parts[0] in {"NAME", "ENDATA"}:
                continue
            code = parts[0].upper()
            if len(parts) >= 2 and parts[1].startswith("x") and parts[1][1:].isdigit():
                idx = int(parts[1][1:])
                if 0 <= idx < n:
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
            if code in {"XL", "XU"} and len(parts) >= 3 and parts[2].startswith("c"):
                suffix = parts[2][1:]
                if suffix.isdigit():
                    row_idx = int(suffix)
                    if 0 <= row_idx < le_count:
                        row_basis[row_idx] = "at_lower" if code == "XL" else "at_upper"

    fields: dict[str, object] = {}
    if all(value is not None for value in var_basis):
        fields["varBasis"] = [str(value) for value in var_basis if value is not None]
    if all(value is not None for value in row_basis):
        fields["rowBasis"] = [str(value) for value in row_basis if value is not None]
    return fields


def parse_cbc_solution(
    path: str,
    n: int,
    le_count: int = 0,
    eq_count: int = 0,
    basis_path: Optional[str] = None,
) -> tuple[str, list[float], dict[str, object]]:
    x = [0.0] * n
    status = "unknown"
    row_duals: list[Optional[float]] = [None] * (le_count + eq_count)
    reduced_costs: list[Optional[float]] = [None] * n
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            stripped = line.strip()
            if line_no == 0 and stripped:
                status = stripped.lower()
                continue
            parts = stripped.split()
            if parts and parts[0] == "**":
                parts = parts[1:]
            if len(parts) >= 3 and parts[0].lstrip("-").isdigit() and parts[1].startswith("x"):
                suffix = parts[1][1:]
                if suffix.isdigit():
                    idx = int(suffix)
                    if 0 <= idx < n:
                        x[idx] = float(parts[2])
                        if len(parts) >= 4 and _is_number(parts[3]):
                            reduced_costs[idx] = float(parts[3])
            elif len(parts) >= 4 and parts[0].lstrip("-").isdigit():
                row_name = parts[1]
                if row_name.startswith("c") and row_name[1:].isdigit():
                    idx = int(row_name[1:])
                elif row_name.startswith("e") and row_name[1:].isdigit():
                    idx = le_count + int(row_name[1:])
                else:
                    idx = -1
                if 0 <= idx < len(row_duals) and _is_number(parts[3]):
                    row_duals[idx] = float(parts[3])

    fields: dict[str, object] = {}
    if all(value is not None for value in reduced_costs):
        fields["reducedCosts"] = [
            float(value) for value in reduced_costs if value is not None
        ]
    dual_ub = row_duals[:le_count]
    dual_eq = row_duals[le_count:]
    if all(value is not None for value in dual_ub):
        fields["dualUB"] = [float(value) for value in dual_ub if value is not None]
    if all(value is not None for value in dual_eq):
        fields["dualEQ"] = [float(value) for value in dual_eq if value is not None]
    if basis_path is not None and os.path.exists(basis_path):
        fields.update(parse_cbc_basis(basis_path, n, le_count, eq_count))
    return status, x, fields


def parse_named_solution(path: str, n: int, default_status: str = "optimal") -> tuple[str, list[float]]:
    x = [0.0] * n
    status = default_status
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                lower = stripped.lower()
                if "objective" in lower and "value" in lower:
                    status = default_status
                continue
            parts = stripped.split()
            if len(parts) >= 2 and parts[0].startswith("x") and parts[0][1:].isdigit():
                idx = int(parts[0][1:])
                if 0 <= idx < n and _is_number(parts[1]):
                    x[idx] = float(parts[1])
    return status, x


def parse_cplex_solution(path: str, n: int) -> tuple[str, list[float]]:
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        return parse_named_solution(path, n)

    x = [0.0] * n
    status = "optimal"
    for elem in root.iter():
        tag = elem.tag.rsplit("}", 1)[-1]
        if tag == "header":
            status = (
                elem.attrib.get("solutionStatusString")
                or elem.attrib.get("solutionStatus")
                or status
            ).lower()
        elif tag == "variable":
            name = elem.attrib.get("name", "")
            value = elem.attrib.get("value")
            if name.startswith("x") and name[1:].isdigit() and value is not None:
                idx = int(name[1:])
                if 0 <= idx < n:
                    x[idx] = float(value)
    return status, x


def parse_xpress_solution(path: str, n: int) -> tuple[str, list[float]]:
    x = [0.0] * n
    status = "optimal"
    header_path = os.path.splitext(path)[0] + ".hdr"
    if os.path.exists(header_path):
        with open(header_path, "r", encoding="utf-8", errors="replace") as f:
            header = f.read().lower()
        if "infeas" in header:
            status = "infeasible"
        elif "unbounded" in header:
            status = "unbounded"
        elif "optimal" in header:
            status = "optimal"

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    for line in text.splitlines():
        fields = _split_xpress_solution_line(line)
        if len(fields) < 2:
            continue
        for name_pos, name in enumerate(fields):
            name = name.strip().strip('"')
            if not (name.startswith("x") and name[1:].isdigit()):
                continue
            idx = int(name[1:])
            if 0 <= idx < n:
                for value in fields[name_pos + 1 :]:
                    value = value.strip().strip('"')
                    if _is_number(value):
                        x[idx] = float(value)
                        break
                break
    return status, x


def parse_lindo_solution(path: str, n: int) -> tuple[str, list[float]]:
    x = [0.0] * n
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    lower = text.lower()
    if "infeasible" in lower or "no feasible" in lower:
        status = "infeasible"
    elif "unbounded" in lower:
        status = "unbounded"
    elif "optimal" in lower or "objective" in lower:
        status = "optimal"
    else:
        status = "unknown"

    for line in text.splitlines():
        parsed = _parse_named_value_line(line, n)
        if parsed is not None:
            idx, value = parsed
            x[idx] = value
    return status, x


def _parse_named_value_line(line: str, n: int) -> Optional[tuple[int, float]]:
    match = re.search(r"\bx(\d+)\b", line, flags=re.IGNORECASE)
    if match is None:
        return None
    idx = int(match.group(1))
    if not 0 <= idx < n:
        return None
    after = line[match.end() :]
    for token in re.split(r"[\s,;:=]+", after):
        token = token.strip()
        if _is_number(token):
            return idx, float(token)
    before = line[: match.start()]
    numeric_before = [
        float(token)
        for token in re.split(r"[\s,;:=]+", before)
        if token.strip() and _is_number(token.strip())
    ]
    if numeric_before:
        return idx, numeric_before[-1]
    return None


def _split_xpress_solution_line(line: str) -> list[str]:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return []
    for delimiter in (";", ","):
        if delimiter in stripped:
            return [
                field.strip()
                for field in next(csv.reader([stripped], delimiter=delimiter))
                if field.strip()
            ]
    return stripped.split()


def _is_number(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False


def stripped_starts(text: str, prefixes: Sequence[str]) -> bool:
    stripped = text.strip()
    return any(stripped.startswith(prefix) for prefix in prefixes)


def classify_status(status: str, stdout: str, stderr: str) -> str:
    parsed = status.lower()
    if "primal infeasible" in parsed or ("infeasible" in parsed and "dual" not in parsed):
        return "infeasible"
    if "dual infeasible" in parsed or "unbounded" in parsed:
        return "unbounded"
    if "optimal" in parsed:
        return "optimal"

    text = f"{stdout}\n{stderr}".lower()
    infeasible_markers = (
        "no primal feasible",
        "primal infeasible",
        "linear relaxation infeasible",
        "no feasible solution",
        "no solution exists",
        "integer infeasible",
        "problem has no feasible",
    )
    if any(marker in text for marker in infeasible_markers):
        return "infeasible"
    unbounded_markers = (
        "has unbounded solution",
        "linear relaxation unbounded",
        "dual infeasible",
        "unbounded",
    )
    if any(marker in text for marker in unbounded_markers):
        return "unbounded"
    return "unknown"


def _first_float_after_colon(line: str) -> Optional[float]:
    text = line.split(":", 1)[1] if ":" in line else line
    return _first_float(text)


def _first_float(text: str) -> Optional[float]:
    for token in re.split(r"[\s,()]+", text):
        cleaned = token.strip().rstrip("%")
        if not cleaned:
            continue
        try:
            return float(cleaned)
        except ValueError:
            continue
    return None


def parse_lp_iterations(solver: str, kind: str, stdout: str, stderr: str) -> dict:
    if kind != "lp":
        return {}
    text = f"{stdout}\n{stderr}"
    iterations: Optional[int] = None

    if solver == "highs":
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.lower().startswith("simplex") and "iterations" in stripped.lower():
                value = _first_float_after_colon(stripped)
                if value is not None:
                    iterations = int(round(value))
    elif solver == "glpk":
        for line in text.splitlines():
            match = re.match(r"^\*?\s*(\d+):\s+obj\b", line.strip())
            if match is not None:
                iterations = int(match.group(1))
    elif solver in {"cbc", "clp"}:
        for line in text.splitlines():
            match = re.search(r"-\s+(\d+)\s+iterations\b", line.strip(), flags=re.IGNORECASE)
            if match is not None:
                iterations = int(match.group(1))

    if iterations is not None and iterations >= 0:
        return {"iterations": iterations}
    return {}


def parse_mip_quality(
    solver: str,
    kind: str,
    objective: Optional[float],
    stdout: str,
    stderr: str,
) -> dict:
    if kind != "mip":
        return {}
    text = f"{stdout}\n{stderr}"
    lowered = text.lower()
    fields = {}

    best_bound = None
    mip_gap = None
    nodes_explored = None

    if solver == "highs":
        for line in text.splitlines():
            stripped = line.strip()
            lowered_line = stripped.lower()
            if lowered_line.startswith("dual bound"):
                best_bound = _first_float_after_colon(stripped)
            elif lowered_line.startswith("gap"):
                value = _first_float(stripped)
                if value is not None:
                    mip_gap = value / 100.0 if "%" in stripped else value
            elif lowered_line.startswith("nodes"):
                value = _first_float_after_colon(stripped)
                if value is not None:
                    nodes_explored = int(round(value))
    elif solver == "cbc":
        for line in text.splitlines():
            stripped = line.strip()
            lowered_line = stripped.lower()
            if lowered_line.startswith("enumerated nodes:"):
                value = _first_float_after_colon(stripped)
                if value is not None:
                    nodes_explored = int(round(value))
            elif lowered_line.startswith("lower bound:"):
                best_bound = _first_float_after_colon(stripped)
            elif "gap:" in lowered_line:
                value = _first_float_after_colon(stripped)
                if value is not None:
                    mip_gap = value / 100.0 if "%" in stripped else value
    elif solver == "scip":
        for line in text.splitlines():
            stripped = line.strip()
            lowered_line = stripped.lower()
            if lowered_line.startswith("solving nodes"):
                value = _first_float_after_colon(stripped)
                if value is not None:
                    nodes_explored = int(round(value))
            elif lowered_line.startswith("dual bound"):
                best_bound = _first_float_after_colon(stripped)
            elif lowered_line.startswith("gap"):
                value = _first_float_after_colon(stripped)
                if value is not None:
                    mip_gap = value / 100.0 if "%" in stripped else value

    if best_bound is not None and math.isfinite(best_bound):
        fields["bestBound"] = best_bound
    if mip_gap is None and best_bound is not None and objective is not None:
        mip_gap = abs(best_bound - objective) / max(1.0, abs(objective))
    if mip_gap is not None and math.isfinite(mip_gap):
        fields["mipGap"] = max(0.0, mip_gap)
    if nodes_explored is not None and nodes_explored >= 0:
        fields["nodesExplored"] = nodes_explored
    return fields


def solver_available(solver: str) -> bool:
    return solver_command(solver) is not None


def solver_command(solver: str) -> Optional[str]:
    configured_any = False
    for env_var in COMMAND_ENV_VARS.get(solver, []):
        configured = os.environ.get(env_var)
        if configured and configured.strip():
            configured_any = True
            resolved = shutil.which(configured)
            if resolved is not None:
                return resolved
    if configured_any:
        return None
    for command in COMMAND_ALIASES.get(solver, [solver]):
        resolved = shutil.which(command)
        if resolved is not None:
            return resolved
    return None


def run_solver(
    solver: str,
    kind: str,
    sense: str,
    model_path: str,
    solution_path: str,
    time_limit: float,
    node_limit: Optional[int] = None,
    relative_gap: Optional[float] = None,
) -> tuple[str, str]:
    command = solver_command(solver)
    if command is None:
        raise ValueError(f"{solver} executable not found")
    if solver == "highs":
        options_path = None
        if kind == "mip" and (node_limit is not None or relative_gap is not None):
            options_path = solution_path + ".options"
            with open(options_path, "w", encoding="utf-8") as f:
                if node_limit is not None:
                    f.write(f"mip_max_nodes = {int(node_limit)}\n")
                if relative_gap is not None:
                    f.write(f"mip_rel_gap = {float(relative_gap):.17g}\n")
        cmd = [
            command,
            "--model_file",
            model_path,
            "--solution_file",
            solution_path,
            "--time_limit",
            str(time_limit),
        ]
        if options_path is not None:
            cmd.extend(["--options_file", options_path])
    elif solver == "glpk":
        if kind == "lp":
            cmd = [
                command,
                "--lp",
                model_path,
                "--output",
                solution_path + ".report",
                "--write",
                solution_path,
                "--tmlim",
                str(max(1, int(math.ceil(time_limit)))),
            ]
        else:
            cmd = [
                command,
                "--lp",
                model_path,
                "-o",
                solution_path,
                "--tmlim",
                str(max(1, int(math.ceil(time_limit)))),
            ]
            if relative_gap is not None:
                cmd.extend(["--mipgap", f"{float(relative_gap):.17g}"])
    elif solver == "scip":
        cmd = [command]
        if kind == "mip":
            if node_limit is not None:
                cmd.extend(["-c", f"set limits nodes {int(node_limit)}"])
            if relative_gap is not None:
                cmd.extend(["-c", f"set limits gap {float(relative_gap):.17g}"])
        if kind == "lp":
            cmd.append("-q")
        cmd.extend([
            "-c",
            f"read {model_path}",
            "-c",
            f"set limits time {time_limit}",
            "-c",
            "optimize",
            "-c",
            f"write solution {solution_path}",
        ])
        if kind == "mip":
            cmd.extend([
                "-c",
                "display statistics",
            ])
        cmd.extend([
            "-c",
            "quit",
        ])
    elif solver == "cbc":
        cmd = [
            command,
            model_path,
            "-seconds",
            str(time_limit),
        ]
        if kind == "lp":
            cmd.extend(["-printingOptions", "all"])
        elif node_limit is not None:
            cmd.extend(["-maxNodes", str(int(node_limit))])
        if kind == "mip" and relative_gap is not None:
            cmd.extend(["-ratioGap", f"{float(relative_gap):.17g}"])
        cmd.extend([
            "-solve",
            "-solution",
            solution_path,
        ])
        if kind == "lp":
            cmd.extend(["-basisOut", solution_path + ".basis"])
    elif solver == "clp":
        cmd = [
            command,
            model_path,
            "-seconds",
            str(time_limit),
            "-printingOptions",
            "all",
            "-solve",
            "-solution",
            solution_path,
            "-basisOut",
            solution_path + ".basis",
        ]
    elif solver == "gurobi":
        cmd = [
            command,
            f"ResultFile={solution_path}",
            f"TimeLimit={time_limit}",
            *([f"NodeLimit={int(node_limit)}"] if kind == "mip" and node_limit is not None else []),
            *([f"MIPGap={float(relative_gap):.17g}"] if kind == "mip" and relative_gap is not None else []),
            model_path,
        ]
    elif solver == "cplex":
        cmd = [
            command,
            "-c",
            f"read {model_path}",
            f"set timelimit {time_limit}",
            *([f"set mip limits nodes {int(node_limit)}"] if kind == "mip" and node_limit is not None else []),
            *([f"set mip tolerances mipgap {float(relative_gap):.17g}"] if kind == "mip" and relative_gap is not None else []),
            "optimize",
            f"write {solution_path}",
            "quit",
        ]
    elif solver == "xpress":
        script_path = os.path.join(os.path.dirname(model_path), "xpress_commands.txt")
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(f"MAXTIME = -{max(1, int(math.ceil(time_limit)))}\n")
            f.write(f"readprob -l {model_path}\n")
            f.write("mipoptimize\n" if kind == "mip" else "lpoptimize\n")
            f.write(f"writesol {solution_path} -npa\n")
            f.write("quit\n")
        cmd = [command, f"@{script_path}"]
    elif solver == "lindo":
        cmd = [
            command,
            model_path,
            "-sol",
            "-max" if sense == "max" else "-min",
        ]
        if kind == "mip":
            cmd.append("-mip")
        else:
            cmd.append("-lp")
    else:
        raise ValueError(f"unknown CLI solver '{solver}'")
    run = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        check=False,
        cwd=os.path.dirname(model_path),
    )
    return run.stdout, run.stderr


def solve(kind: str, solver: str, raw: dict, time_limit: float) -> dict:
    if not solver_available(solver):
        return status_payload("unavailable", f"{solver}:cli", f"{solver} executable not found")
    if solver not in SUPPORTED_SOLVERS:
        return status_payload(
            "unavailable",
            f"{solver}:cli",
            f"{solver} executable found, but this bridge does not yet know the non-interactive solve command",
        )

    if kind == "lp":
        lp = raw.get("lp", raw)
        sense, c, a_ub, b_ub, a_eq, b_eq, lbs, ubs = normalize_lp(lp)
        integer_vars = [False] * len(c)
    elif kind == "mip":
        if solver == "clp":
            return status_payload("unavailable", "clp:cli", "CLP is LP-only")
        sense, c, a_ub, b_ub, lbs, ubs, integer_vars = normalize_mip(raw)
        a_eq, b_eq = [], []
    else:
        raise ValueError("kind must be 'lp' or 'mip'")

    with tempfile.TemporaryDirectory(prefix="ores-linear-cli-") as tmp:
        if solver == "lindo":
            model_path = os.path.join(tmp, "model.mps")
            solution_path = os.path.join(tmp, "model.sol")
            write_free_mps(model_path, sense, c, a_ub, b_ub, a_eq, b_eq, lbs, ubs, integer_vars)
        else:
            model_path = os.path.join(tmp, "model.lp")
            solution_path = (
                os.path.join(tmp, "xpress_solution")
                if solver == "xpress"
                else os.path.join(tmp, f"{solver}.sol")
            )
            write_cplex_lp(model_path, sense, c, a_ub, b_ub, a_eq, b_eq, lbs, ubs, integer_vars)
        stdout, stderr = run_solver(solver, kind, sense, model_path, solution_path, time_limit)
        if solver == "xpress":
            solution_path = _first_existing_path(
                [solution_path + ".asc", solution_path, solution_path + ".sol"]
            ) or solution_path
        elif solver == "lindo":
            solution_path = _first_existing_path(
                [solution_path, os.path.splitext(model_path)[0] + ".sol"]
            ) or solution_path
        if not os.path.exists(solution_path):
            classified = classify_status("", stdout, stderr)
            if classified in ("infeasible", "unbounded"):
                return status_payload(classified, f"{solver}:cli", (stderr or stdout).strip())
            return status_payload("unavailable", f"{solver}:cli", (stderr or stdout).strip())
        certificate_fields: dict[str, object] = {}
        if solver == "highs":
            status, x, certificate_fields = parse_highs_solution(
                solution_path,
                len(c),
                len(a_ub),
                len(a_eq),
            )
        elif solver == "glpk":
            status, x, certificate_fields = parse_glpk_solution(
                solution_path,
                len(c),
                len(a_ub),
                len(a_eq),
            )
        elif solver == "scip":
            status, x = parse_scip_solution(solution_path, len(c))
        elif solver == "gurobi":
            status, x = parse_named_solution(solution_path, len(c))
        elif solver == "cplex":
            status, x = parse_cplex_solution(solution_path, len(c))
        elif solver == "xpress":
            status, x = parse_xpress_solution(solution_path, len(c))
        elif solver == "lindo":
            status, x = parse_lindo_solution(solution_path, len(c))
        else:
            status, x, certificate_fields = parse_cbc_solution(
                solution_path,
                len(c),
                len(a_ub),
                len(a_eq),
                solution_path + ".basis" if kind == "lp" else None,
            )

    classified = classify_status(status, stdout, stderr)
    if classified != "optimal":
        return status_payload(
            classified if classified in ("infeasible", "unbounded") else "unavailable",
            f"{solver}:cli",
            status,
        )
    objective = dot(c, x)
    result = {
        "status": "optimal",
        "solver": f"{solver}:cli",
        "x": x,
        "objective": objective,
        "message": status,
    }
    if kind == "lp":
        result.update(certificate_fields)
        result.update(parse_lp_iterations(solver, kind, stdout, stderr))
    result.update(parse_mip_quality(solver, kind, objective, stdout, stderr))
    return result


def _first_existing_path(paths: Sequence[str]) -> Optional[str]:
    for path in paths:
        if os.path.exists(path):
            return path
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["lp", "mip"], required=True)
    parser.add_argument("--solver", choices=sorted(COMMAND_ALIASES.keys()), required=True)
    parser.add_argument("--problem")
    parser.add_argument("--time-limit", type=float, default=10.0)
    args = parser.parse_args()
    try:
        if args.problem:
            with open(args.problem, "r", encoding="utf-8") as f:
                raw = json.load(f)
        else:
            raw = json.load(sys.stdin)
        print(json.dumps(solve(args.kind, args.solver, raw, args.time_limit), allow_nan=True))
        return 0
    except Exception as exc:
        print(json.dumps(status_payload("numerical-error", f"{args.solver}:cli", str(exc)), allow_nan=True))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
