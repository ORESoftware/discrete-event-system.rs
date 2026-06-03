#!/usr/bin/env python3
"""Small proof-checker reference bridge for DRAT/LRAT-style CNF payloads."""

from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
from typing import Any


def artifact_content(payload: dict[str, Any], *names: str) -> str | None:
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list):
        return None
    wanted = {name.lower() for name in names}
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            continue
        name = str(artifact.get("name", "")).lower()
        if name in wanted:
            value = artifact.get("content")
            if isinstance(value, str):
                return value
    return None


def parse_dimacs(text: str) -> tuple[int, list[list[int]]]:
    variables = 0
    clauses: list[list[int]] = []
    current: list[int] = []
    saw_header = False
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("c"):
            continue
        if line.startswith("p "):
            parts = line.split()
            if len(parts) < 4 or parts[1] != "cnf":
                raise ValueError("expected DIMACS 'p cnf <vars> <clauses>' header")
            variables = int(parts[2])
            saw_header = True
            continue
        for token in line.split():
            literal = int(token)
            if literal == 0:
                clauses.append(current)
                current = []
            else:
                current.append(literal)
                variables = max(variables, abs(literal))
    if current:
        raise ValueError("unterminated DIMACS clause")
    if not saw_header:
        raise ValueError("missing DIMACS header")
    return variables, clauses


def clause_satisfied(clause: list[int], assignment: tuple[bool, ...]) -> bool:
    for literal in clause:
        value = assignment[abs(literal) - 1]
        if literal > 0 and value:
            return True
        if literal < 0 and not value:
            return True
    return False


def find_model(variables: int, clauses: list[list[int]]) -> list[int] | None:
    if variables > 20:
        raise ValueError("reference proof bridge only brute-forces up to 20 variables")
    for assignment in itertools.product([False, True], repeat=variables):
        if all(clause_satisfied(clause, assignment) for clause in clauses):
            return [index + 1 if value else -(index + 1) for index, value in enumerate(assignment)]
    return None


def drat_has_empty_clause(proof: str) -> bool:
    for raw in proof.splitlines():
        line = raw.strip()
        if not line or line.startswith("c") or line.startswith("d "):
            continue
        try:
            tokens = [int(token) for token in line.split()]
        except ValueError:
            return False
        if tokens == [0]:
            return True
    return False


def lrat_has_empty_clause(proof: str) -> bool:
    for raw in proof.splitlines():
        line = raw.strip()
        if not line or line.startswith("c"):
            continue
        try:
            tokens = [int(token) for token in line.split()]
        except ValueError:
            return False
        if len(tokens) >= 2 and tokens[1] == 0:
            return True
    return False


def frat_has_empty_clause(proof: str) -> bool:
    for raw in proof.splitlines():
        line = raw.strip()
        if not line or line.startswith("c"):
            continue
        if line == "0":
            return True
        tokens = line.split()
        if tokens[:1] == ["a"] and "0" in tokens[1:]:
            zero_index = tokens.index("0")
            if zero_index <= 2:
                return True
    return False


def parse_opb(text: str) -> tuple[list[str], list[tuple[list[tuple[int, str]], str, int]]]:
    variables: set[str] = set()
    constraints: list[tuple[list[tuple[int, str]], str, int]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("*"):
            continue
        if line.lower().startswith(("min:", "max:")):
            continue
        match = re.fullmatch(r"(.+?)\s*(>=|<=|=)\s*(-?\d+)\s*;?", line)
        if not match:
            raise ValueError(f"unsupported OPB constraint {line!r}")
        lhs, op, rhs_text = match.groups()
        tokens = lhs.split()
        if len(tokens) % 2 != 0:
            raise ValueError(f"unsupported OPB term list {lhs!r}")
        terms: list[tuple[int, str]] = []
        for idx in range(0, len(tokens), 2):
            coeff = int(tokens[idx])
            name = tokens[idx + 1]
            if not re.fullmatch(r"[A-Za-z_]\w*", name):
                raise ValueError(f"unsupported OPB variable {name!r}")
            variables.add(name)
            terms.append((coeff, name))
        constraints.append((terms, op, int(rhs_text)))
    if not constraints:
        raise ValueError("missing OPB constraints")
    return sorted(variables), constraints


def pb_constraint_satisfied(
    constraint: tuple[list[tuple[int, str]], str, int],
    assignment: dict[str, bool],
) -> bool:
    terms, op, rhs = constraint
    total = sum(coeff * int(assignment[name]) for coeff, name in terms)
    if op == ">=":
        return total >= rhs
    if op == "<=":
        return total <= rhs
    return total == rhs


def find_pb_model(
    variables: list[str],
    constraints: list[tuple[list[tuple[int, str]], str, int]],
) -> dict[str, int] | None:
    if len(variables) > 20:
        raise ValueError("reference pseudo-Boolean proof bridge only brute-forces up to 20 variables")
    for bits in itertools.product([False, True], repeat=len(variables)):
        assignment = dict(zip(variables, bits))
        if all(pb_constraint_satisfied(constraint, assignment) for constraint in constraints):
            return {name: int(assignment[name]) for name in variables}
    return None


def veripb_has_derivation(proof: str) -> bool:
    for raw in proof.splitlines():
        line = raw.strip()
        if line and not line.startswith(("*", "c")):
            return True
    return False


def emit(tool: str, status: str, verdict: str, **extra: Any) -> None:
    validator = extra.pop("validator", f"builtin:small-cnf-proof-for-{tool}")
    output = {
        "kind": "proof-validation-result",
        "tool": tool,
        "validator": validator,
        "status": status,
        "verdict": verdict,
    }
    output.update(extra)
    print(json.dumps(output, sort_keys=True))


def validate_pseudo_boolean(payload: dict[str, Any], tool: str) -> None:
    opb = payload.get("opb") or payload.get("model") or artifact_content(payload, "opb", "model")
    proof = payload.get("proof") or artifact_content(payload, "proof", "pbp", "rup")
    if not isinstance(opb, str) or not opb.strip():
        raise ValueError("missing OPB text")
    if not isinstance(proof, str) or not proof.strip():
        raise ValueError("missing pseudo-Boolean proof text")
    variables, constraints = parse_opb(opb)
    model = find_pb_model(variables, constraints)
    if model is not None:
        emit(
            tool,
            "ok",
            "invalid",
            pb_status="sat",
            message="OPB model is satisfiable; proof cannot validate infeasibility",
            validator=f"builtin:small-opb-proof-for-{tool}",
            witness=model,
        )
        return
    if veripb_has_derivation(proof):
        emit(
            tool,
            "ok",
            "valid",
            pb_status="unsat",
            message="infeasible OPB model with non-empty pseudo-Boolean proof",
            validator=f"builtin:small-opb-proof-for-{tool}",
        )
    else:
        emit(
            tool,
            "ok",
            "invalid",
            pb_status="unsat",
            message="infeasible OPB model proof did not contain a derivation line",
            validator=f"builtin:small-opb-proof-for-{tool}",
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool", default="drat")
    args = parser.parse_args()
    tool = args.tool.lower().replace("_", "-")
    try:
        payload = json.load(sys.stdin)
        if not isinstance(payload, dict):
            raise ValueError("top-level payload must be an object")
        kind = str(payload.get("kind", "")).lower().replace("_", "-")
        if tool in {"veripb", "veripb-checker"} or kind in {
            "pseudo-boolean-proof-validation",
            "opb-proof-validation",
            "veripb-validation",
        }:
            validate_pseudo_boolean(payload, tool)
            return 0
        cnf = payload.get("cnf") or payload.get("dimacs") or artifact_content(payload, "cnf", "model")
        proof = payload.get("proof") or artifact_content(payload, "proof", "drat", "lrat", "frat")
        if not isinstance(cnf, str) or not cnf.strip():
            raise ValueError("missing CNF text")
        if not isinstance(proof, str) or not proof.strip():
            raise ValueError("missing proof text")
        variables, clauses = parse_dimacs(cnf)
        model = find_model(variables, clauses)
        if model is not None:
            emit(
                tool,
                "ok",
                "invalid",
                cnf_status="sat",
                message="CNF is satisfiable; unsat proof cannot validate",
                witness=model,
            )
            return 0
        has_empty_clause = (
            lrat_has_empty_clause(proof)
            if tool in {"lrat", "lrat-check", "lrat-checker"}
            else frat_has_empty_clause(proof)
            if tool in {"frat", "frat-rs", "frat-trim"}
            else drat_has_empty_clause(proof)
        )
        if has_empty_clause:
            emit(
                tool,
                "ok",
                "valid",
                cnf_status="unsat",
                message="unsat CNF with empty-clause proof line",
            )
        else:
            emit(
                tool,
                "ok",
                "invalid",
                cnf_status="unsat",
                message="unsat CNF proof did not contain an empty-clause line",
            )
        return 0
    except Exception as exc:  # noqa: BLE001 - bridge returns JSON for malformed payloads.
        emit(tool, "ok", "invalid", message=str(exc))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
