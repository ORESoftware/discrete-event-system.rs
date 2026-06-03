#!/usr/bin/env python3
"""Small proof-checker reference bridge for DRAT/LRAT-style CNF payloads."""

from __future__ import annotations

import argparse
import itertools
import json
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


def emit(tool: str, status: str, verdict: str, **extra: Any) -> None:
    output = {
        "kind": "proof-validation-result",
        "tool": tool,
        "validator": f"builtin:small-cnf-proof-for-{tool}",
        "status": status,
        "verdict": verdict,
    }
    output.update(extra)
    print(json.dumps(output, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool", default="drat")
    args = parser.parse_args()
    tool = args.tool.lower().replace("_", "-")
    try:
        payload = json.load(sys.stdin)
        if not isinstance(payload, dict):
            raise ValueError("top-level payload must be an object")
        cnf = payload.get("cnf") or payload.get("dimacs") or artifact_content(payload, "cnf", "model")
        proof = payload.get("proof") or artifact_content(payload, "proof", "drat", "lrat")
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
