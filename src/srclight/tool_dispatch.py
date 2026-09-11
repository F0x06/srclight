"""Turn shell tokens into typed MCP tool arguments.

The CLI dispatches to tools it has never heard of, so it cannot declare
options ahead of time. It reads each tool's JSON Schema instead — the same
schema an MCP client validates against — which is why a tool added to the
server needs no CLI change, and why a rejected argument is rejected here for
the same reason and in the same words it would be over MCP.

Deliberately free of Click, MCP and I/O: this is the fiddly part, and it is
worth being able to test it on its own.
"""

from __future__ import annotations

from typing import Any

_TRUE = {"true", "1", "yes", "on"}
_FALSE = {"false", "0", "no", "off"}


class ToolArgumentError(Exception):
    """A usage error whose message is meant for the person who typed it."""


def parse_cli_pairs(tokens: list[str]) -> dict[str, str]:
    """Turn ``["--key", "value", "--flag"]`` into ``{"key": "value", "flag": "true"}``.

    A bare ``--flag`` (nothing after it, or another option next) reads as
    "true" so boolean arguments behave the way a shell user expects. Dashes in
    names become underscores, since MCP argument names are snake_case.
    """
    pairs: dict[str, str] = {}
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if not token.startswith("--"):
            raise ToolArgumentError(
                f"unexpected argument {token!r}: pass tool arguments as --name value"
            )
        body = token[2:]
        if "=" in body:
            name, value = body.split("=", 1)
            i += 1
        else:
            name = body
            nxt = tokens[i + 1] if i + 1 < len(tokens) else None
            # "--limit -1": a lone dash-number is a value, not an option.
            if nxt is not None and (not nxt.startswith("--")):
                value = nxt
                i += 2
            else:
                value = "true"
                i += 1
        pairs[name.replace("-", "_")] = value
    return pairs


def _accepted_types(prop: dict[str, Any]) -> set[str]:
    """The JSON Schema types a property accepts, ignoring null."""
    if "type" in prop:
        return {prop["type"]}
    types = set()
    for branch in prop.get("anyOf", []):
        if isinstance(branch, dict) and "type" in branch:
            types.add(branch["type"])
    types.discard("null")
    return types


def coerce_arguments(schema: dict[str, Any], raw: dict[str, str]) -> dict[str, object]:
    """Convert string values to the types the schema declares.

    Omitted optionals are left out rather than filled in, so the tool's own
    defaults apply — restating them here would freeze them at the value they
    had the day this was written.
    """
    properties: dict[str, Any] = schema.get("properties", {})
    required: list[str] = list(schema.get("required", []))

    unknown = sorted(set(raw) - set(properties))
    if unknown:
        accepted = ", ".join(f"--{name}" for name in sorted(properties))
        raise ToolArgumentError(
            f"unknown argument(s): {', '.join(unknown)}. Accepted: {accepted or '(none)'}"
        )

    missing = [name for name in required if name not in raw]
    if missing:
        raise ToolArgumentError(f"missing required argument(s): {', '.join(missing)}")

    out: dict[str, object] = {}
    for name, value in raw.items():
        types = _accepted_types(properties[name])
        if "integer" in types:
            try:
                out[name] = int(value)
            except ValueError:
                raise ToolArgumentError(
                    f"--{name} expects an integer, got {value!r}"
                ) from None
        elif "number" in types:
            try:
                out[name] = float(value)
            except ValueError:
                raise ToolArgumentError(
                    f"--{name} expects a number, got {value!r}"
                ) from None
        elif "boolean" in types:
            lowered = value.strip().lower()
            if lowered in _TRUE:
                out[name] = True
            elif lowered in _FALSE:
                out[name] = False
            else:
                raise ToolArgumentError(
                    f"--{name} expects true or false, got {value!r}"
                )
        elif "array" in types:
            items_type = _array_item_type(properties[name])
            if items_type is not None and items_type != "string":
                raise ToolArgumentError(
                    f"--{name} is a list of {items_type}, which cannot be "
                    f"expressed on the command line"
                )
            out[name] = [item.strip() for item in value.split(",")]
        elif "object" in types:
            # No non-scalar besides array exists in the registry today, but
            # an object would have no command-line spelling either.
            raise ToolArgumentError(
                f"--{name} is an object, which cannot be expressed on the command line"
            )
        else:
            out[name] = value
    return out


def _array_item_type(prop: dict[str, Any]) -> str | None:
    """The declared ``items.type`` of an array-typed property, if any."""
    if prop.get("type") == "array":
        items = prop.get("items")
        if isinstance(items, dict):
            return items.get("type")
        return None
    for branch in prop.get("anyOf", []):
        if isinstance(branch, dict) and branch.get("type") == "array":
            items = branch.get("items")
            if isinstance(items, dict):
                return items.get("type")
            return None
    return None


def format_tool_help(name: str, description: str, schema: dict[str, Any]) -> str:
    """Render a tool's usage from its schema, in the shape of --help output."""
    properties: dict[str, Any] = schema.get("properties", {})
    required: set[str] = set(schema.get("required", []))

    summary = (description or "").strip().split("\n\n")[0].strip()
    lines = [f"Usage: srclight tool {name} [ARGUMENTS]", ""]
    if summary:
        lines += [summary, ""]
    lines.append(
        "Arguments are derived from the running server's schema for this "
        "tool and may change as that schema changes."
    )
    lines.append("")
    if not properties:
        lines.append("This tool takes no arguments.")
        return "\n".join(lines)

    lines.append("Arguments:")
    for arg in sorted(properties):
        prop = properties[arg]
        types = _accepted_types(prop) or {"string"}
        if "array" in types:
            item_type = _array_item_type(prop) or "string"
            kind = f"comma-separated {item_type} list"
        else:
            kind = "|".join(sorted(types))
        if arg in required:
            note = "required"
        else:
            note = f"default: {prop.get('default')!r}"
        lines.append(f"  --{arg} <{kind}>  ({note})")
    return "\n".join(lines)
