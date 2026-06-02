#!/usr/bin/env python3
"""Reference bridge for formal-model and benchmark validation payloads.

This script gives the Rust suite a normalized oracle for TLA+/PRISM, Alloy,
Promela, SMV, UPPAAL, CBMC-style C assertions, deductive/program-verifier
exports, security-protocol models, mCRL2, Maude, and benchmark manifests
without requiring heavyweight model checkers or datasets. When a simple local
PRISM command is available it can be used; otherwise the built-in path performs
structural checks that catch malformed generated payloads.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


def result(
    status: str,
    verdict: str,
    validator: str,
    message: str = "",
    checks: list[dict[str, Any]] | None = None,
    stdout: str = "",
    stderr: str = "",
) -> dict[str, Any]:
    return {
        "status": status,
        "verdict": verdict,
        "validator": validator,
        "message": message,
        "checks": checks or [],
        "stdout": stdout,
        "stderr": stderr,
    }


def normalize_tool(tool: str | None) -> str:
    return (tool or "auto").strip().lower().replace("_", "-")


def adapter_env(tool: str) -> str:
    return "ORES_" + re.sub(r"[^A-Z0-9]+", "_", tool.upper()).strip("_") + "_ADAPTER"


def first_command(tool: str, aliases: list[str]) -> str | None:
    configured = os.environ.get(adapter_env(tool))
    if configured and shutil.which(configured):
        return configured
    for alias in aliases:
        found = shutil.which(alias)
        if found:
            return found
    return None


def check(name: str, passed: bool, message: str = "") -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "message": "" if passed else message}


def verdict_from_checks(checks: list[dict[str, Any]]) -> str:
    return "valid" if checks and all(item["passed"] for item in checks) else "invalid"


def balanced(text: str, left: str, right: str) -> bool:
    depth = 0
    for char in text:
        if char == left:
            depth += 1
        elif char == right:
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def validate_tla(payload: dict[str, Any]) -> dict[str, Any]:
    module = str(payload.get("module") or payload.get("model") or payload.get("text") or "")
    expected_invariants = [str(item) for item in payload.get("expected_invariants", [])]
    expected_temporal = [str(item) for item in payload.get("expected_temporal_properties", [])]
    header = re.search(r"----\s+MODULE\s+([A-Za-z_]\w*)\s+----", module)
    checks = [
        check("module-header", header is not None, "missing TLA+ module header"),
        check("module-terminator", module.rstrip().endswith("===="), "missing final ===="),
        check("init-definition", "Init ==" in module, "missing Init definition"),
        check("next-definition", "Next ==" in module, "missing Next definition"),
        check("spec-definition", "Spec ==" in module, "missing Spec definition"),
    ]
    for invariant in expected_invariants:
        checks.append(check(f"invariant:{invariant}", f"{invariant} ==" in module, "invariant definition missing"))
    for temporal in expected_temporal:
        checks.append(
            check(f"temporal:{temporal}", f"{temporal} ==" in module, "temporal property definition missing")
        )
    module_name = header.group(1) if header else ""
    validator = "builtin:tla-structural"
    message = f"module={module_name}" if module_name else ""
    return result("ok", verdict_from_checks(checks), validator, message, checks)


def infer_prism_verdict(stdout: str, stderr: str, exit_success: bool) -> str:
    text = f"{stdout}\n{stderr}".lower()
    if not exit_success:
        return "failure"
    if "false" in text and "result" in text:
        return "invalid"
    if "true" in text and "result" in text:
        return "valid"
    if "error" in text:
        return "failure"
    return "valid"


def run_prism(command: str, model: str, properties: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="ores-prism-") as tmp:
        model_path = Path(tmp) / "model.pm"
        props_path = Path(tmp) / "props.pctl"
        model_path.write_text(model, encoding="utf-8")
        props_path.write_text(properties, encoding="utf-8")
        completed = subprocess.run(
            [command, str(model_path), str(props_path)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    verdict = infer_prism_verdict(completed.stdout, completed.stderr, completed.returncode == 0)
    return result(
        "ok" if completed.returncode == 0 else "failed",
        verdict,
        command,
        "",
        [],
        completed.stdout,
        completed.stderr,
    )


def validate_prism(payload: dict[str, Any], tool: str) -> dict[str, Any]:
    model = str(payload.get("model") or payload.get("module") or payload.get("text") or "")
    properties = str(payload.get("properties") or payload.get("props") or "")
    command = first_command(tool, ["prism"]) if tool in ("auto", "prism") else None
    if command and model.strip() and properties.strip():
        return run_prism(command, model, properties)
    model_type = model.splitlines()[0].strip().lower() if model.strip() else ""
    module_count = len(re.findall(r"(?m)^\s*module\s+[A-Za-z_]\w*", model))
    endmodule_count = len(re.findall(r"(?m)^\s*endmodule\s*$", model))
    checks = [
        check("model-type", model_type in ("dtmc", "ctmc", "mdp", "pta"), "unknown PRISM model type"),
        check("module-present", module_count > 0, "no PRISM module found"),
        check("module-balanced", module_count == endmodule_count, "module/endmodule count mismatch"),
        check("command-present", "->" in model, "no transition command found"),
        check("properties-present", bool(properties.strip()), "no PRISM properties supplied"),
    ]
    return result("ok", verdict_from_checks(checks), "builtin:prism-structural", "", checks)


def validate_promela(payload: dict[str, Any]) -> dict[str, Any]:
    model = str(payload.get("model") or payload.get("promela") or payload.get("text") or "")
    properties = payload.get("properties", payload.get("ltl", []))
    if isinstance(properties, str):
        properties = [properties]
    expected_ltl = [str(item) for item in payload.get("expected_ltl_properties", [])]
    checks = [
        check("process-present", bool(re.search(r"\b(init|proctype)\b", model)), "missing init/proctype"),
        check("braces-balanced", balanced(model, "{", "}"), "Promela braces are not balanced"),
        check("statement-terminator", ";" in model or "->" in model, "no Promela statements found"),
    ]
    for prop in properties:
        prop_text = str(prop)
        checks.append(
            check(
                f"ltl:{prop_text[:24]}",
                "ltl" in prop_text.lower() or "<>" in prop_text or "[]" in prop_text,
                "malformed LTL property",
            )
        )
    for name in expected_ltl:
        checks.append(check(f"expected-ltl:{name}", name in model, "expected LTL property missing"))
    return result("ok", verdict_from_checks(checks), "builtin:promela-structural", "", checks)


def validate_smv(payload: dict[str, Any]) -> dict[str, Any]:
    model = str(payload.get("model") or payload.get("smv") or payload.get("text") or "")
    properties = payload.get("properties", [])
    if isinstance(properties, str):
        properties = [properties]
    property_text = "\n".join(str(item) for item in properties)
    combined = f"{model}\n{property_text}"
    checks = [
        check("module-main", bool(re.search(r"(?mi)^\s*MODULE\s+main\b", model)), "missing MODULE main"),
        check("var-section", bool(re.search(r"(?mi)^\s*VAR\b", model)), "missing VAR section"),
        check(
            "state-update",
            bool(re.search(r"(?mi)^\s*(ASSIGN|INIT|TRANS)\b", model)),
            "missing ASSIGN/INIT/TRANS section",
        ),
        check(
            "property-present",
            bool(re.search(r"\b(CTLSPEC|LTLSPEC|INVARSPEC|SPEC)\b", combined)),
            "missing nuXmv/SMV property",
        ),
    ]
    return result("ok", verdict_from_checks(checks), "builtin:smv-structural", "", checks)


def validate_cbmc(payload: dict[str, Any]) -> dict[str, Any]:
    source = str(payload.get("source") or payload.get("model") or payload.get("c") or payload.get("text") or "")
    expected_assertions = [str(item) for item in payload.get("expected_assertions", [])]
    checks = [
        check("main-function", bool(re.search(r"\b(?:int|void)\s+main\s*\(", source)), "missing main function"),
        check("braces-balanced", balanced(source, "{", "}"), "C braces are not balanced"),
        check(
            "assertion-present",
            "__CPROVER_assert" in source or re.search(r"\bassert\s*\(", source) is not None,
            "missing C/CBMC assertion",
        ),
    ]
    for assertion in expected_assertions:
        checks.append(check(f"assertion:{assertion}", assertion in source, "expected assertion text missing"))
    return result("ok", verdict_from_checks(checks), "builtin:cbmc-structural", "", checks)


def validate_alloy(payload: dict[str, Any]) -> dict[str, Any]:
    model = str(payload.get("model") or payload.get("alloy") or payload.get("text") or "")
    commands = payload.get("commands", payload.get("properties", []))
    if isinstance(commands, str):
        commands = [commands]
    combined = f"{model}\n" + "\n".join(str(item) for item in commands)
    checks = [
        check("module-or-signature", "module " in model or re.search(r"\bsig\s+\w+", model) is not None, "missing module/sig"),
        check("signature-present", re.search(r"\bsig\s+\w+", model) is not None, "missing Alloy signature"),
        check("braces-balanced", balanced(model, "{", "}"), "Alloy braces are not balanced"),
        check(
            "predicate-or-fact",
            re.search(r"\b(pred|fact|assert)\s+\w+", model) is not None,
            "missing pred/fact/assert",
        ),
        check("command-present", re.search(r"\b(run|check)\s+\w+", combined) is not None, "missing run/check command"),
    ]
    return result("ok", verdict_from_checks(checks), "builtin:alloy-structural", "", checks)


def validate_uppaal(payload: dict[str, Any]) -> dict[str, Any]:
    model = str(payload.get("model") or payload.get("xml") or payload.get("text") or "")
    queries = payload.get("queries", payload.get("properties", payload.get("query", [])))
    if isinstance(queries, str):
        queries = [queries]
    query_text = "\n".join(str(item) for item in queries)
    checks = [
        check("nta-root", "<nta" in model and "</nta>" in model, "missing UPPAAL nta root"),
        check("template-present", "<template" in model and "</template>" in model, "missing template"),
        check("location-present", "<location" in model, "missing location"),
        check("transition-present", "<transition" in model, "missing transition"),
        check("query-present", bool(query_text.strip()), "missing UPPAAL query"),
        check(
            "query-operator",
            bool(re.search(r"\b[AE]\s*(<>|\[\])", query_text)),
            "missing UPPAAL temporal operator",
        ),
    ]
    return result("ok", verdict_from_checks(checks), "builtin:uppaal-structural", "", checks)


def validate_mcrl2(payload: dict[str, Any]) -> dict[str, Any]:
    spec = str(payload.get("model") or payload.get("mcrl2") or payload.get("spec") or payload.get("text") or "")
    properties = payload.get("properties", payload.get("formulae", []))
    if isinstance(properties, str):
        properties = [properties]
    formula_text = "\n".join(str(item) for item in properties)
    checks = [
        check("action-section", bool(re.search(r"(?m)^\s*act\b", spec)), "missing mCRL2 act section"),
        check("process-section", bool(re.search(r"(?m)^\s*proc\b", spec)), "missing mCRL2 proc section"),
        check("init-section", bool(re.search(r"(?m)^\s*init\b", spec)), "missing mCRL2 init section"),
        check("semicolon-present", ";" in spec, "missing mCRL2 statement terminators"),
        check(
            "property-or-modal-operator",
            bool(formula_text.strip()) or any(token in spec for token in ("[", "<", "mu ", "nu ")),
            "missing mCRL2 modal property/formula",
        ),
    ]
    return result("ok", verdict_from_checks(checks), "builtin:mcrl2-structural", "", checks)


def validate_maude(payload: dict[str, Any]) -> dict[str, Any]:
    module = str(payload.get("model") or payload.get("maude") or payload.get("module") or payload.get("text") or "")
    commands = payload.get("commands", payload.get("properties", []))
    if isinstance(commands, str):
        commands = [commands]
    command_text = "\n".join(str(item) for item in commands)
    checks = [
        check(
            "module-header",
            bool(re.search(r"\b(fmod|mod)\s+\w+\s+is\b", module)),
            "missing Maude module header",
        ),
        check("module-terminator", "endfm" in module or "endm" in module, "missing Maude module terminator"),
        check("operator-or-rule", bool(re.search(r"\b(op|eq|rl|crl)\b", module)), "missing Maude op/equation/rule"),
        check("brackets-balanced", balanced(module, "[", "]"), "Maude brackets are not balanced"),
        check(
            "command-present",
            bool(command_text.strip()) or re.search(r"\b(search|red|rew)\b", module) is not None,
            "missing Maude command/search",
        ),
    ]
    return result("ok", verdict_from_checks(checks), "builtin:maude-structural", "", checks)


def validate_program_verifier(payload: dict[str, Any], tool: str) -> dict[str, Any]:
    source = str(
        payload.get("source")
        or payload.get("model")
        or payload.get("program")
        or payload.get("spec")
        or payload.get("text")
        or ""
    )
    language = normalize_tool(str(payload.get("language") or tool))
    contracts = payload.get("contracts", payload.get("properties", payload.get("expected_contracts", [])))
    if isinstance(contracts, str):
        contracts = [contracts]
    contract_text = "\n".join(str(item) for item in contracts)
    combined = f"{source}\n{contract_text}"
    has_contract = bool(
        re.search(
            r"\b(requires|ensures|invariant|assert|assume|lemma|theorem|goal|predicate|claim)\b",
            combined,
        )
        or "/*@" in source
        or "//@" in source
        or "#[kani::proof]" in source
    )
    checks = [
        check("source-present", bool(source.strip()), "missing program/verifier source"),
        check("braces-balanced", balanced(source, "{", "}"), "program braces are not balanced"),
        check("contract-or-assertion", has_contract, "missing contract/assertion/proof obligation"),
    ]
    if language in ("dafny",):
        checks.extend(
            [
                check(
                    "dafny-declaration",
                    bool(re.search(r"\b(method|function|predicate|lemma|class)\b", source)),
                    "missing Dafny declaration",
                ),
                check(
                    "dafny-spec",
                    bool(re.search(r"\b(requires|ensures|invariant|assert)\b", combined)),
                    "missing Dafny specification",
                ),
            ]
        )
    elif language in ("why3", "whyml"):
        checks.extend(
            [
                check("why3-module", bool(re.search(r"\bmodule\s+\w+", source)), "missing Why3 module"),
                check(
                    "why3-obligation",
                    bool(re.search(r"\b(goal|lemma|let|requires|ensures|invariant)\b", combined)),
                    "missing Why3 proof obligation",
                ),
            ]
        )
    elif language in ("frama-c", "framac"):
        checks.extend(
            [
                check(
                    "c-function",
                    bool(re.search(r"\b(?:int|void|double|float|char)\s+\w+\s*\(", source)),
                    "missing C function",
                ),
                check("acsl-contract", "/*@" in source or "//@" in source, "missing ACSL annotation"),
            ]
        )
    elif language in ("kani", "rust"):
        checks.extend(
            [
                check("rust-function", bool(re.search(r"\bfn\s+\w+\s*\(", source)), "missing Rust function"),
                check(
                    "kani-harness-or-assert",
                    "#[kani::proof]" in source or "kani::" in source or "assert!" in source,
                    "missing Kani proof harness/assertion",
                ),
            ]
        )
    elif language in ("esbmc", "cbmc", "cpa-checker", "cpachecker", "jbmc"):
        checks.append(
            check(
                "bounded-model-assertion",
                "__CPROVER_assert" in source or re.search(r"\bassert\s*\(", source) is not None,
                "missing bounded-model-checker assertion",
            )
        )
    elif language in ("coq", "isabelle", "lean", "pvs", "acl2"):
        checks.append(
            check(
                "proof-declaration",
                bool(re.search(r"\b(Theorem|Lemma|theorem|lemma|Definition|def)\b", source)),
                "missing proof assistant theorem/lemma",
            )
        )
    return result("ok", verdict_from_checks(checks), "builtin:program-verifier-structural", language, checks)


def validate_security_protocol(payload: dict[str, Any], tool: str) -> dict[str, Any]:
    model = str(
        payload.get("model")
        or payload.get("protocol")
        or payload.get("source")
        or payload.get("spec")
        or payload.get("text")
        or ""
    )
    properties = payload.get("properties", payload.get("queries", payload.get("lemmas", [])))
    if isinstance(properties, str):
        properties = [properties]
    property_text = "\n".join(str(item) for item in properties)
    combined = f"{model}\n{property_text}"
    checks = [
        check("model-present", bool(model.strip()), "missing security-protocol model"),
        check("braces-balanced", balanced(model, "{", "}"), "protocol braces are not balanced"),
        check(
            "actor-or-process",
            bool(re.search(r"\b(role|process|principal|rule|protocol|theory)\b", model, re.IGNORECASE)),
            "missing role/process/rule/protocol declaration",
        ),
        check(
            "query-or-lemma",
            bool(re.search(r"\b(query|lemma|claim|confidentiality|authentication|secrecy)\b", combined, re.IGNORECASE)),
            "missing security query/lemma/claim",
        ),
    ]
    if tool == "tamarin":
        checks.extend(
            [
                check("tamarin-theory", bool(re.search(r"\btheory\s+\w+\s+begin\b", model)), "missing theory begin"),
                check("tamarin-end", bool(re.search(r"\bend\b", model)), "missing theory end"),
                check("tamarin-rule", bool(re.search(r"\brule\s+\w+", model)), "missing Tamarin rule"),
            ]
        )
    elif tool in ("proverif", "cryptoverif", "deepsec"):
        checks.append(
            check(
                "applied-pi-shape",
                bool(re.search(r"\b(free|fun|event|query|process)\b", combined)),
                "missing applied-pi/protocol declarations",
            )
        )
    elif tool == "scyther":
        checks.extend(
            [
                check("scyther-protocol", bool(re.search(r"\bprotocol\s+\w+", model)), "missing protocol declaration"),
                check("scyther-role", bool(re.search(r"\brole\s+\w+", model)), "missing role declaration"),
                check("scyther-claim", "claim" in combined, "missing claim"),
            ]
        )
    elif tool == "verifpal":
        checks.append(
            check(
                "verifpal-query",
                "queries" in combined.lower() or "confidentiality?" in combined.lower(),
                "missing Verifpal query block",
            )
        )
    return result("ok", verdict_from_checks(checks), "builtin:security-protocol-structural", tool, checks)


def validate_benchmark_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    suite = str(payload.get("suite") or "").strip()
    entries = payload.get("entries")
    require_paths = bool(payload.get("require_paths", False))
    root_dir = Path(str(payload.get("root_dir") or "."))
    checks = [
        check("suite-present", bool(suite), "missing benchmark suite"),
        check("entries-array", isinstance(entries, list), "entries must be an array"),
    ]
    names: set[str] = set()
    if isinstance(entries, list):
        checks.append(check("entries-nonempty", len(entries) > 0, "manifest has no entries"))
        for idx, entry in enumerate(entries):
            if not isinstance(entry, dict):
                checks.append(check(f"entry:{idx}:object", False, "entry must be an object"))
                continue
            name = str(entry.get("name") or "").strip()
            family = str(entry.get("family") or "").strip()
            fmt = str(entry.get("format") or "").strip().lower()
            path = str(entry.get("path") or "").strip()
            checks.append(check(f"entry:{idx}:name", bool(name), "missing entry name"))
            checks.append(check(f"entry:{idx}:family", bool(family), "missing entry family"))
            checks.append(check(f"entry:{idx}:format", bool(fmt), "missing entry format"))
            checks.append(check(f"entry:{idx}:path", bool(path), "missing entry path"))
            checks.append(check(f"entry:{idx}:unique", not name or name not in names, "duplicate entry name"))
            if name:
                names.add(name)
            if fmt:
                checks.append(
                    check(
                        f"entry:{idx}:format-known",
                        fmt in {"lp", "mps", "nl", "osil", "json", "dzn", "qplib", "cnf", "fzn"},
                        f"unrecognized benchmark format {fmt!r}",
                    )
                )
            if require_paths and path:
                checks.append(
                    check(
                        f"entry:{idx}:path-exists",
                        (root_dir / path).is_file(),
                        f"benchmark file not found: {root_dir / path}",
                    )
                )
    return result("ok", verdict_from_checks(checks), "builtin:benchmark-manifest", "", checks)


def dispatch(payload: dict[str, Any], tool_override: str | None = None) -> dict[str, Any]:
    tool = normalize_tool(tool_override or payload.get("tool"))
    kind = normalize_tool(str(payload.get("kind", "")))
    if kind in ("tla-validation", "tla-plus-validation") or tool in ("tlc", "apalache", "tla"):
        return validate_tla(payload)
    if kind in ("prism-validation", "prism-model-validation") or tool in ("prism", "storm"):
        return validate_prism(payload, "prism" if tool == "storm" else tool)
    if kind in ("alloy-validation", "kodkod-validation") or tool in ("alloy", "kodkod"):
        return validate_alloy(payload)
    if kind in ("promela-validation", "spin-validation") or tool == "spin":
        return validate_promela(payload)
    if kind in ("smv-validation", "nuxmv-validation") or tool == "nuxmv":
        return validate_smv(payload)
    if kind in ("uppaal-validation", "uppaal-xml-validation") or tool == "uppaal":
        return validate_uppaal(payload)
    if kind in ("cbmc-validation", "c-bounded-model-validation") or tool == "cbmc":
        return validate_cbmc(payload)
    if kind in ("program-verifier-validation", "deductive-verification") or tool in (
        "dafny",
        "frama-c",
        "framac",
        "why3",
        "whyml",
        "kani",
        "esbmc",
        "cpa-checker",
        "cpachecker",
        "jbmc",
        "coq",
        "isabelle",
        "lean",
        "pvs",
        "acl2",
    ):
        return validate_program_verifier(payload, tool)
    if kind in ("security-protocol-validation", "protocol-verification") or tool in (
        "tamarin",
        "proverif",
        "cryptoverif",
        "deepsec",
        "scyther",
        "verifpal",
        "sapic",
        "sapic-plus",
    ):
        return validate_security_protocol(payload, tool)
    if kind in ("mcrl2-validation", "mcrl2-spec-validation") or tool == "mcrl2":
        return validate_mcrl2(payload)
    if kind in ("maude-validation", "maude-module-validation") or tool == "maude":
        return validate_maude(payload)
    if kind == "external-benchmark-manifest" or tool in ("benchmark", "miplib", "qplib", "minlplib"):
        return validate_benchmark_manifest(payload)
    return result("unavailable", "unknown", tool, f"unknown formal/benchmark payload kind {kind!r}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool", default=None)
    args = parser.parse_args()
    try:
        payload = json.load(sys.stdin)
        print(json.dumps(dispatch(payload, args.tool)))
    except Exception as exc:
        print(json.dumps(result("failed", "failure", args.tool or "formal-benchmark", str(exc))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
