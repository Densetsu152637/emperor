"""Imperium's dispatch contract, vendored under `vendor/imperium` (see its SOURCE file).

Holds the tool schemas the model is offered, the root system prompt it runs with, and a small
JSON Schema validator for the keywords those schemas use. The validator refuses any keyword it
does not know, so a contract update that needs more fails loudly instead of validating loosely.
"""

import json
from functools import cache
from importlib import resources
from typing import Any

VENDOR = resources.files("emperor.eval") / "vendor" / "imperium"

# Every tool in the contract, in the order the model is offered them.
ALL_TOOLS = [
    "spawn_agent",
    "await_tasks",
    "update_plan",
    "delegate_chunks",
    "ask_user",
    "web_search",
    "web_fetch",
]

CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    """Rough token count (4 characters per token); good enough for limits and overflow checks."""
    return -(-len(text) // CHARS_PER_TOKEN)


@cache
def tool_schemas() -> dict[str, dict]:
    """Tool name -> its exported definition (name, description, parameters, status)."""
    index = json.loads((VENDOR / "tools" / "index.json").read_text(encoding="utf-8"))
    schemas = {}
    for entry in index["tools"]:
        name = entry["name"]
        schemas[name] = json.loads((VENDOR / "tools" / f"{name}.json").read_text(encoding="utf-8"))
    return schemas


@cache
def prompt(name: str) -> str:
    return (VENDOR / "prompts" / f"{name}.md").read_text(encoding="utf-8")


def contract_source() -> dict[str, str]:
    source = {}
    for line in (VENDOR / "SOURCE").read_text(encoding="utf-8").splitlines():
        key, _, value = line.partition(":")
        source[key.strip()] = value.strip()
    return source


def system_prompt(context_tokens: int) -> str:
    """The stage-1 root prompt plus the context-budget line that tells the model its tier."""
    return (
        f"{prompt('emperor-root').rstrip()}\n\ncontext budget: {context_tokens // 1024}K tokens\n"
    )


def openai_tools(names: list[str]) -> list[dict]:
    """The named tools in OpenAI `tools` format."""
    schemas = tool_schemas()
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": schemas[name]["description"],
                "parameters": schemas[name]["parameters"],
            },
        }
        for name in names
    ]


class SchemaError(ValueError):
    """The schema itself uses something this validator does not support."""


_KNOWN_KEYWORDS = {
    "type",
    "properties",
    "required",
    "items",
    "enum",
    "minItems",
    "description",
}
_TYPES = {
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    "boolean": lambda v: isinstance(v, bool),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, int | float) and not isinstance(v, bool),
    "null": lambda v: v is None,
}


def validate(instance: Any, schema: dict, path: str = "$") -> list[str]:
    """Return every violation of `schema` by `instance` (empty when valid)."""
    unknown = set(schema) - _KNOWN_KEYWORDS
    if unknown:
        raise SchemaError(f"unsupported schema keywords at {path}: {sorted(unknown)}")
    errors: list[str] = []
    if "type" in schema:
        types = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
        if not any(_TYPES[t](instance) for t in types):
            return [f"{path}: expected {'/'.join(types)}, got {type(instance).__name__}"]
    if "enum" in schema and instance not in schema["enum"]:
        errors.append(f"{path}: {instance!r} is not one of {schema['enum']}")
    if isinstance(instance, dict):
        for key in schema.get("required", []):
            if key not in instance:
                errors.append(f"{path}: missing required property {key!r}")
        for key, sub in schema.get("properties", {}).items():
            if key in instance:
                errors.extend(validate(instance[key], sub, f"{path}.{key}"))
    if isinstance(instance, list):
        if len(instance) < schema.get("minItems", 0):
            errors.append(f"{path}: fewer than {schema['minItems']} items")
        if "items" in schema:
            for i, item in enumerate(instance):
                errors.extend(validate(item, schema["items"], f"{path}[{i}]"))
    return errors


def check_call(name: str, raw_arguments: str, offered: list[str]) -> tuple[dict | None, list[str]]:
    """Parse and validate one tool call; returns the arguments (if they parse) and any errors."""
    if name not in offered:
        return None, [f"tool {name!r} was not offered"]
    try:
        arguments = json.loads(raw_arguments) if raw_arguments.strip() else {}
    except json.JSONDecodeError as error:
        return None, [f"arguments are not JSON: {error.msg}"]
    return arguments, validate(arguments, tool_schemas()[name]["parameters"])
