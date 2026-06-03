#!/usr/bin/env python3
"""Reference bridge for output/data validators.

The bridge intentionally keeps local dependencies optional. JSON Schema,
OpenAPI, XML/XSD/Schematron, Protobuf-style message schemas, Avro record
schemas, Pydantic-style model specs, and table schemas have small
dependency-free validators for the common run-artifact checks used by the Rust
suite, and richer package-backed validators can be selected when installed in
the caller's Python environment.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import sys
import xml.etree.ElementTree as ET
from typing import Any


def result(status: str, verdict: str, validator: str, message: str = "", errors: list[str] | None = None) -> dict:
    return {
        "status": status,
        "verdict": verdict,
        "validator": validator,
        "message": message,
        "errors": errors or [],
    }


def json_type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int) and not isinstance(value, bool):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def matches_json_type(value: Any, expected: str) -> bool:
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "string":
        return isinstance(value, str)
    if expected == "array":
        return isinstance(value, list)
    if expected == "object":
        return isinstance(value, dict)
    if expected == "null":
        return value is None
    return True


def validate_builtin_schema(schema: dict, instance: Any, path: str = "$") -> list[str]:
    errors: list[str] = []
    expected_type = schema.get("type")
    if isinstance(expected_type, list):
        if not any(matches_json_type(instance, item) for item in expected_type):
            errors.append(f"{path}: expected one of {expected_type}, got {json_type_name(instance)}")
            return errors
    elif isinstance(expected_type, str) and not matches_json_type(instance, expected_type):
        errors.append(f"{path}: expected {expected_type}, got {json_type_name(instance)}")
        return errors

    if "const" in schema and instance != schema["const"]:
        errors.append(f"{path}: expected constant {schema['const']!r}")
    if "enum" in schema and instance not in schema["enum"]:
        errors.append(f"{path}: value {instance!r} is not in enum")

    if isinstance(instance, (int, float)) and not isinstance(instance, bool):
        if "minimum" in schema and float(instance) < float(schema["minimum"]):
            errors.append(f"{path}: value is below minimum {schema['minimum']}")
        if "maximum" in schema and float(instance) > float(schema["maximum"]):
            errors.append(f"{path}: value is above maximum {schema['maximum']}")
        if "exclusiveMinimum" in schema and float(instance) <= float(schema["exclusiveMinimum"]):
            errors.append(f"{path}: value is not above exclusiveMinimum {schema['exclusiveMinimum']}")
        if "exclusiveMaximum" in schema and float(instance) >= float(schema["exclusiveMaximum"]):
            errors.append(f"{path}: value is not below exclusiveMaximum {schema['exclusiveMaximum']}")

    if isinstance(instance, str):
        if "minLength" in schema and len(instance) < int(schema["minLength"]):
            errors.append(f"{path}: string is shorter than minLength {schema['minLength']}")
        if "maxLength" in schema and len(instance) > int(schema["maxLength"]):
            errors.append(f"{path}: string is longer than maxLength {schema['maxLength']}")

    if isinstance(instance, list):
        if "minItems" in schema and len(instance) < int(schema["minItems"]):
            errors.append(f"{path}: array has fewer than minItems {schema['minItems']}")
        if "maxItems" in schema and len(instance) > int(schema["maxItems"]):
            errors.append(f"{path}: array has more than maxItems {schema['maxItems']}")
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for idx, item in enumerate(instance):
                errors.extend(validate_builtin_schema(item_schema, item, f"{path}[{idx}]"))

    if isinstance(instance, dict):
        required = schema.get("required", [])
        if isinstance(required, list):
            for key in required:
                if key not in instance:
                    errors.append(f"{path}: missing required property {key!r}")
        properties = schema.get("properties", {})
        if isinstance(properties, dict):
            for key, property_schema in properties.items():
                if key in instance and isinstance(property_schema, dict):
                    errors.extend(validate_builtin_schema(property_schema, instance[key], f"{path}.{key}"))
        if schema.get("additionalProperties") is False and isinstance(properties, dict):
            known = set(properties.keys())
            for key in instance:
                if key not in known:
                    errors.append(f"{path}: unexpected property {key!r}")

    return errors


def jsonschema_reference(payload: dict) -> dict:
    schema = payload.get("schema", {})
    instance = payload.get("instance")
    try:
        import jsonschema  # type: ignore
    except Exception:
        if not isinstance(schema, dict):
            return result("failed", "invalid", "builtin:json-schema-subset", "schema must be an object")
        errors = validate_builtin_schema(schema, instance)
        verdict = "valid" if not errors else "invalid"
        return result("ok", verdict, "builtin:json-schema-subset", errors[0] if errors else "", errors)

    validator = jsonschema.validators.validator_for(schema)
    validator.check_schema(schema)
    errors = sorted(validator(schema).iter_errors(instance), key=lambda error: list(error.path))
    messages = [error.message for error in errors]
    verdict = "valid" if not errors else "invalid"
    return result("ok", verdict, "python:jsonschema", messages[0] if messages else "", messages)


def frictionless_reference(payload: dict) -> dict:
    if str(payload.get("kind", "")).strip().lower().replace("_", "-") == "table-validation":
        fallback = table_reference(payload)
        fallback["validator"] = "builtin:table-schema-subset-for-frictionless"
        return fallback
    try:
        import frictionless  # type: ignore
    except Exception:
        return result("unavailable", "unknown", "python:frictionless", "frictionless is not installed")
    target = payload.get("path") or payload.get("resource") or payload.get("instance")
    if target is None:
        return result("failed", "invalid", "python:frictionless", "payload needs path, resource, or instance")
    report = frictionless.validate(target)
    valid = bool(getattr(report, "valid", False))
    errors = [str(error) for task in getattr(report, "tasks", []) for error in getattr(task, "errors", [])]
    return result("ok", "valid" if valid else "invalid", "python:frictionless", errors[0] if errors else "", errors)


def parse_csv_rows(text: str) -> list[dict[str, Any]]:
    reader = csv.DictReader(io.StringIO(text))
    return [dict(row) for row in reader]


def table_rows(payload: dict) -> list[dict[str, Any]]:
    source = payload.get("rows", payload.get("data", payload.get("instance")))
    if source is None:
        source = payload.get("csv", payload.get("text"))
    if isinstance(source, str):
        return parse_csv_rows(source)
    if not isinstance(source, list):
        raise ValueError("table-validation payload needs rows, data, instance, csv, or text")
    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(source):
        if not isinstance(row, dict):
            raise ValueError(f"row {idx} must be an object")
        rows.append(row)
    return rows


def table_column_specs(schema: dict) -> dict[str, dict[str, Any]]:
    columns = schema.get("columns", {})
    if isinstance(columns, dict):
        return {
            str(name): spec if isinstance(spec, dict) else {"type": str(spec)}
            for name, spec in columns.items()
        }
    if isinstance(columns, list):
        specs = {}
        for item in columns:
            if isinstance(item, str):
                specs[item] = {}
            elif isinstance(item, dict) and "name" in item:
                specs[str(item["name"])] = item
        return specs
    return {}


def missing_cell(value: Any) -> bool:
    return value is None or value == ""


def parse_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        out = float(value)
    elif isinstance(value, str):
        try:
            out = float(value)
        except ValueError:
            return None
    else:
        return None
    return out if math.isfinite(out) else None


def matches_table_type(value: Any, expected: str) -> bool:
    if expected == "number":
        return parse_number(value) is not None
    if expected == "integer":
        number = parse_number(value)
        return number is not None and number.is_integer()
    if expected == "boolean":
        if isinstance(value, bool):
            return True
        return isinstance(value, str) and value.strip().lower() in ("true", "false", "0", "1")
    if expected == "string":
        return isinstance(value, str)
    return True


def validate_table_schema(schema: dict, rows: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    min_rows = schema.get("min_rows", schema.get("minRows"))
    max_rows = schema.get("max_rows", schema.get("maxRows"))
    if min_rows is not None and len(rows) < int(min_rows):
        errors.append(f"table: expected at least {min_rows} rows, got {len(rows)}")
    if max_rows is not None and len(rows) > int(max_rows):
        errors.append(f"table: expected at most {max_rows} rows, got {len(rows)}")
    columns = table_column_specs(schema)
    known_columns = set(columns)
    if schema.get("additionalColumns") is False or schema.get("additional_columns") is False:
        for idx, row in enumerate(rows):
            for key in row:
                if key not in known_columns:
                    errors.append(f"row {idx}: unexpected column {key!r}")

    unique_values: dict[str, set[Any]] = {
        name: set() for name, spec in columns.items() if bool(spec.get("unique", False))
    }
    for idx, row in enumerate(rows):
        for name, spec in columns.items():
            value = row.get(name)
            required = bool(spec.get("required", False)) or name in schema.get("required", [])
            if missing_cell(value):
                if required:
                    errors.append(f"row {idx}.{name}: required value is missing")
                continue
            expected_type = spec.get("type")
            if isinstance(expected_type, str) and not matches_table_type(value, expected_type):
                errors.append(f"row {idx}.{name}: expected {expected_type}, got {value!r}")
                continue
            if "enum" in spec and value not in spec["enum"]:
                errors.append(f"row {idx}.{name}: value {value!r} is not in enum")
            number = parse_number(value)
            if number is not None:
                if "minimum" in spec and number < float(spec["minimum"]):
                    errors.append(f"row {idx}.{name}: value is below minimum {spec['minimum']}")
                if "maximum" in spec and number > float(spec["maximum"]):
                    errors.append(f"row {idx}.{name}: value is above maximum {spec['maximum']}")
            if isinstance(value, str):
                if "minLength" in spec and len(value) < int(spec["minLength"]):
                    errors.append(f"row {idx}.{name}: string is shorter than minLength {spec['minLength']}")
                if "maxLength" in spec and len(value) > int(spec["maxLength"]):
                    errors.append(f"row {idx}.{name}: string is longer than maxLength {spec['maxLength']}")
            if name in unique_values:
                if value in unique_values[name]:
                    errors.append(f"row {idx}.{name}: duplicate value {value!r}")
                unique_values[name].add(value)
    return errors


def table_reference(payload: dict) -> dict:
    schema = payload.get("schema", payload.get("expectations", {}))
    if not isinstance(schema, dict):
        return result("failed", "invalid", "builtin:table-schema-subset", "schema must be an object")
    rows = table_rows(payload)
    errors = validate_table_schema(schema, rows)
    verdict = "valid" if not errors else "invalid"
    return result("ok", verdict, "builtin:table-schema-subset", errors[0] if errors else "", errors)


def object_field_specs(schema: dict) -> dict[str, dict[str, Any]]:
    fields = schema.get("fields", {})
    if isinstance(fields, dict):
        return {
            str(name): spec if isinstance(spec, dict) else {"type": str(spec)}
            for name, spec in fields.items()
        }
    if isinstance(fields, list):
        specs = {}
        for item in fields:
            if isinstance(item, str):
                specs[item] = {}
            elif isinstance(item, dict) and "name" in item:
                specs[str(item["name"])] = item
        return specs
    return {}


def matches_protobuf_scalar(value: Any, expected: str) -> bool:
    expected = expected.lower()
    if expected in ("int32", "sint32", "sfixed32"):
        return isinstance(value, int) and not isinstance(value, bool) and -(2**31) <= value < 2**31
    if expected in ("uint32", "fixed32"):
        return isinstance(value, int) and not isinstance(value, bool) and 0 <= value < 2**32
    if expected in ("int64", "sint64", "sfixed64"):
        return isinstance(value, int) and not isinstance(value, bool) and -(2**63) <= value < 2**63
    if expected in ("uint64", "fixed64"):
        return isinstance(value, int) and not isinstance(value, bool) and 0 <= value < 2**64
    if expected in ("double", "float"):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))
    if expected in ("bool", "boolean"):
        return isinstance(value, bool)
    if expected == "string":
        return isinstance(value, str)
    if expected == "bytes":
        return isinstance(value, str)
    if expected in ("message", "object"):
        return isinstance(value, dict)
    return True


def validate_protobuf_field(name: str, spec: dict[str, Any], value: Any, path: str) -> list[str]:
    errors: list[str] = []
    if spec.get("repeated") is True:
        if not isinstance(value, list):
            return [f"{path}.{name}: expected repeated field array"]
        for idx, item in enumerate(value):
            errors.extend(validate_protobuf_field(name, {**spec, "repeated": False}, item, f"{path}.{name}[{idx}]"))
        return errors
    enum_values = spec.get("enum")
    if isinstance(enum_values, list) and value not in enum_values:
        errors.append(f"{path}.{name}: value {value!r} is not in enum")
        return errors
    expected_type = str(spec.get("type", "string"))
    if not matches_protobuf_scalar(value, expected_type):
        errors.append(f"{path}.{name}: expected protobuf {expected_type}, got {json_type_name(value)}")
    nested = spec.get("fields")
    if isinstance(nested, (dict, list)) and isinstance(value, dict):
        errors.extend(validate_protobuf_message({"fields": nested}, value, f"{path}.{name}"))
    return errors


def validate_protobuf_oneofs(schema: dict, message: dict[str, Any], path: str) -> list[str]:
    errors: list[str] = []
    oneofs = schema.get("oneof", schema.get("oneofs", []))
    if isinstance(oneofs, dict):
        groups = oneofs.values()
    elif isinstance(oneofs, list):
        groups = oneofs
    else:
        groups = []
    for group in groups:
        if isinstance(group, dict):
            fields = [str(field) for field in group.get("fields", [])]
            name = str(group.get("name", "oneof"))
        else:
            fields = [str(field) for field in group]
            name = "oneof"
        present = [field for field in fields if field in message and message[field] is not None]
        if len(present) != 1:
            errors.append(f"{path}: oneof {name!r} expected exactly one of {fields}, got {present}")
    return errors


def validate_protobuf_message(schema: dict, message: dict[str, Any], path: str = "$") -> list[str]:
    errors: list[str] = []
    fields = object_field_specs(schema)
    required = set(str(name) for name in schema.get("required", []))
    known = set(fields)
    for name, spec in fields.items():
        value = message.get(name)
        is_required = bool(spec.get("required", False)) or name in required
        if value is None:
            if is_required:
                errors.append(f"{path}: missing required protobuf field {name!r}")
            continue
        errors.extend(validate_protobuf_field(name, spec, value, path))
    if schema.get("additionalFields") is False or schema.get("additional_fields") is False:
        for name in message:
            if name not in known:
                errors.append(f"{path}: unexpected protobuf field {name!r}")
    errors.extend(validate_protobuf_oneofs(schema, message, path))
    return errors


def protobuf_reference(payload: dict) -> dict:
    schema = payload.get("schema", payload.get("descriptor", {}))
    message = payload.get("message", payload.get("instance", payload.get("data")))
    if not isinstance(schema, dict):
        return result("failed", "invalid", "builtin:protobuf-conformance-subset", "schema must be an object")
    if not isinstance(message, dict):
        return result("failed", "invalid", "builtin:protobuf-conformance-subset", "message must be an object")
    errors = validate_protobuf_message(schema, message)
    verdict = "valid" if not errors else "invalid"
    return result("ok", verdict, "builtin:protobuf-conformance-subset", errors[0] if errors else "", errors)


def validate_avro_value(schema: Any, value: Any, path: str = "$") -> list[str]:
    errors: list[str] = []
    if isinstance(schema, list):
        branch_errors = [validate_avro_value(branch, value, path) for branch in schema]
        if any(not branch for branch in branch_errors):
            return []
        errors.append(f"{path}: value did not match any Avro union branch")
        return errors
    if isinstance(schema, str):
        if schema == "null" and value is not None:
            errors.append(f"{path}: expected null")
        elif schema == "boolean" and not isinstance(value, bool):
            errors.append(f"{path}: expected boolean")
        elif schema in ("int", "long") and (not isinstance(value, int) or isinstance(value, bool)):
            errors.append(f"{path}: expected {schema}")
        elif schema in ("float", "double") and (
            not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value))
        ):
            errors.append(f"{path}: expected {schema}")
        elif schema in ("bytes", "string") and not isinstance(value, str):
            errors.append(f"{path}: expected {schema}")
        return errors
    if not isinstance(schema, dict):
        return [f"{path}: unsupported Avro schema shape"]
    schema_type = schema.get("type")
    if isinstance(schema_type, (list, dict)):
        return validate_avro_value(schema_type, value, path)
    if schema_type == "record":
        if not isinstance(value, dict):
            return [f"{path}: expected record object"]
        fields = schema.get("fields", [])
        if not isinstance(fields, list):
            return [f"{path}: record fields must be a list"]
        known = set()
        for field in fields:
            if not isinstance(field, dict) or "name" not in field:
                errors.append(f"{path}: invalid Avro field")
                continue
            name = str(field["name"])
            known.add(name)
            if name not in value:
                if "default" not in field:
                    errors.append(f"{path}: missing Avro field {name!r}")
                continue
            errors.extend(validate_avro_value(field.get("type", "string"), value[name], f"{path}.{name}"))
        if schema.get("additionalFields") is False or schema.get("additional_fields") is False:
            for name in value:
                if name not in known:
                    errors.append(f"{path}: unexpected Avro field {name!r}")
        return errors
    if schema_type == "array":
        if not isinstance(value, list):
            return [f"{path}: expected Avro array"]
        item_schema = schema.get("items", "string")
        for idx, item in enumerate(value):
            errors.extend(validate_avro_value(item_schema, item, f"{path}[{idx}]"))
        return errors
    if schema_type == "map":
        if not isinstance(value, dict):
            return [f"{path}: expected Avro map"]
        value_schema = schema.get("values", "string")
        for key, item in value.items():
            errors.extend(validate_avro_value(value_schema, item, f"{path}.{key}"))
        return errors
    if schema_type == "enum":
        symbols = schema.get("symbols", [])
        if value not in symbols:
            return [f"{path}: value {value!r} is not in Avro enum"]
        return []
    return validate_avro_value(str(schema_type), value, path)


def avro_reference(payload: dict) -> dict:
    schema = payload.get("schema", {})
    instance = payload.get("record", payload.get("instance", payload.get("data")))
    errors = validate_avro_value(schema, instance)
    verdict = "valid" if not errors else "invalid"
    return result("ok", verdict, "builtin:avro-schema-subset", errors[0] if errors else "", errors)


def openapi_reference(payload: dict, validator: str = "builtin:openapi-structural") -> dict:
    spec = payload.get("spec", payload.get("schema", payload.get("openapi", {})))
    if not isinstance(spec, dict):
        return result("failed", "invalid", validator, "OpenAPI spec must be an object")
    errors: list[str] = []
    version = spec.get("openapi", spec.get("swagger"))
    if not version:
        errors.append("$.openapi: missing OpenAPI/Swagger version")
    info = spec.get("info")
    if not isinstance(info, dict) or not info.get("title"):
        errors.append("$.info.title: missing API title")
    paths = spec.get("paths")
    if not isinstance(paths, dict) or not paths:
        errors.append("$.paths: expected non-empty object")
    else:
        valid_methods = {"get", "put", "post", "delete", "patch", "head", "options", "trace"}
        for path, operations in paths.items():
            if not str(path).startswith("/"):
                errors.append(f"$.paths.{path}: path must start with '/'")
            if not isinstance(operations, dict) or not operations:
                errors.append(f"$.paths.{path}: expected operations object")
                continue
            for method, operation in operations.items():
                if str(method).lower() not in valid_methods:
                    continue
                if not isinstance(operation, dict):
                    errors.append(f"$.paths.{path}.{method}: operation must be an object")
                    continue
                responses = operation.get("responses")
                if not isinstance(responses, dict) or not responses:
                    errors.append(f"$.paths.{path}.{method}.responses: missing responses")
    verdict = "valid" if not errors else "invalid"
    return result("ok", verdict, validator, errors[0] if errors else "", errors)


def xml_reference(payload: dict, validator: str = "builtin:xml-structural") -> dict:
    xml_text = str(payload.get("xml") or payload.get("instance") or payload.get("document") or payload.get("text") or "")
    schema_text = str(payload.get("schema") or payload.get("xsd") or payload.get("schematron") or "")
    required_elements = [str(item) for item in payload.get("required_elements", [])]
    errors: list[str] = []
    root = None
    try:
        root = ET.fromstring(xml_text)
    except Exception as exc:
        errors.append(f"xml: not well-formed: {exc}")
    if root is not None:
        for name in required_elements:
            found = root.tag == name or any(element.tag == name for element in root.iter())
            if not found:
                errors.append(f"xml: missing required element {name!r}")
    if schema_text:
        lower = schema_text.lower()
        if "schematron" in validator:
            if "<schema" not in lower or ("<assert" not in lower and "<report" not in lower):
                errors.append("schematron: expected schema with assert/report rules")
        elif "xsd" in validator or "xml-schema" in validator:
            if "<xs:schema" not in lower and "<xsd:schema" not in lower and "<schema" not in lower:
                errors.append("xsd: expected schema root")
    verdict = "valid" if not errors else "invalid"
    return result("ok", verdict, validator, errors[0] if errors else "", errors)


def pydantic_reference(payload: dict) -> dict:
    model = payload.get("model", payload.get("schema", {}))
    instance = payload.get("instance", payload.get("data", {}))
    if not isinstance(model, dict):
        return result("failed", "invalid", "builtin:pydantic-model-subset", "model must be an object")
    if not isinstance(instance, dict):
        return result("failed", "invalid", "builtin:pydantic-model-subset", "instance must be an object")
    fields = model.get("fields", model.get("properties", model))
    if not isinstance(fields, dict):
        return result("failed", "invalid", "builtin:pydantic-model-subset", "fields must be an object")
    schema = {"type": "object", "properties": {}, "required": []}
    for name, spec in fields.items():
        if isinstance(spec, str):
            field_spec = {"type": spec}
        elif isinstance(spec, dict):
            field_spec = dict(spec)
        else:
            field_spec = {}
        if field_spec.get("required", False):
            schema["required"].append(str(name))
        field_type = str(field_spec.get("type", "string")).lower()
        if field_type in ("int", "integer"):
            field_spec["type"] = "integer"
        elif field_type in ("float", "double", "number"):
            field_spec["type"] = "number"
        elif field_type in ("bool", "boolean"):
            field_spec["type"] = "boolean"
        elif field_type in ("list", "array"):
            field_spec["type"] = "array"
        elif field_type in ("dict", "object"):
            field_spec["type"] = "object"
        else:
            field_spec["type"] = "string"
        schema["properties"][str(name)] = field_spec
    errors = validate_builtin_schema(schema, instance)
    verdict = "valid" if not errors else "invalid"
    return result("ok", verdict, "builtin:pydantic-model-subset", errors[0] if errors else "", errors)


def package_unavailable(tool: str) -> dict:
    return result("unavailable", "unknown", f"python:{tool}", f"{tool} adapter package is not installed")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool", default="json-schema")
    args = parser.parse_args()
    payload = json.load(sys.stdin)
    tool = args.tool.strip().lower().replace("_", "-")
    try:
        kind = str(payload.get("kind", "")).strip().lower().replace("_", "-")
        if tool in ("json-schema", "jsonschema"):
            print(json.dumps(jsonschema_reference(payload)))
        elif tool in ("ajv", "ajv-cli", "check-jsonschema", "cue"):
            value = jsonschema_reference(payload)
            value["validator"] = f"builtin:json-schema-subset-for-{tool}"
            print(json.dumps(value))
        elif tool in (
            "openapi",
            "openapi-validator",
            "spectral",
            "openapi-spec-validator",
            "redocly-cli",
            "asyncapi-cli",
        ) or kind == "openapi-validation":
            validator = (
                f"builtin:openapi-structural-for-{tool}"
                if tool
                in ("spectral", "openapi-spec-validator", "redocly-cli", "asyncapi-cli")
                else "builtin:openapi-structural"
            )
            print(json.dumps(openapi_reference(payload, validator)))
        elif tool in ("xml", "xmllint", "xml-schema", "xsd", "python-xmlschema") or kind in ("xml-validation", "xsd-validation"):
            xml_validator = {
                "xmllint": "builtin:xml-schema-structural-for-xmllint",
                "python-xmlschema": "builtin:xml-schema-structural-for-python-xmlschema",
            }.get(tool, "builtin:xml-schema-structural")
            print(json.dumps(xml_reference(payload, xml_validator)))
        elif tool in ("schematron", "jing", "saxon") or kind == "schematron-validation":
            print(json.dumps(xml_reference(payload, "builtin:schematron-structural")))
        elif tool in ("pydantic", "zod", "valibot", "marshmallow", "cerberus") or kind == "pydantic-validation":
            value = pydantic_reference(payload)
            if tool != "pydantic":
                value["validator"] = f"builtin:pydantic-model-subset-for-{tool}"
            print(json.dumps(value))
        elif tool in ("table", "table-schema", "tabular", "csv-validator") or kind == "table-validation":
            value = table_reference(payload)
            if tool in (
                "frictionless",
                "pandera",
                "dbt",
                "whylogs",
                "great-expectations",
                "soda-core",
                "evidently",
                "deepchecks",
                "parquet-tools",
                "apache-arrow",
                "deequ",
                "tensorflow-data-validation",
                "openrefine",
            ):
                value["validator"] = f"builtin:table-schema-subset-for-{tool}"
            print(json.dumps(value))
        elif tool in ("protobuf", "protobuf-conformance", "protoc") or kind == "protobuf-validation":
            print(json.dumps(protobuf_reference(payload)))
        elif tool in ("avro", "avro-tools", "apache-avro") or kind == "avro-validation":
            print(json.dumps(avro_reference(payload)))
        elif tool == "frictionless":
            print(json.dumps(frictionless_reference(payload)))
        elif tool in (
            "pandera",
            "dbt",
            "whylogs",
            "great-expectations",
            "soda-core",
            "evidently",
            "deepchecks",
            "parquet-tools",
            "apache-arrow",
            "deequ",
            "tensorflow-data-validation",
            "openrefine",
            "yamllint",
            "graphql-schema",
        ):
            print(json.dumps(package_unavailable(tool)))
        else:
            print(json.dumps(result("unavailable", "unknown", tool, f"unknown output validator '{tool}'")))
    except Exception as exc:
        print(json.dumps(result("failed", "failure", tool, str(exc))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
