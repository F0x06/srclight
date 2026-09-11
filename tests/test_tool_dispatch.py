"""Turning shell tokens into typed MCP tool arguments."""

import pytest

from srclight.tool_dispatch import (
    ToolArgumentError,
    coerce_arguments,
    format_tool_help,
    parse_cli_pairs,
)

# The real shape, copied from mcp.list_tools() for find_pattern.
FIND_PATTERN_SCHEMA = {
    "type": "object",
    "title": "find_patternArguments",
    "required": ["pattern"],
    "properties": {
        "pattern": {"title": "Pattern", "type": "string"},
        "project": {"anyOf": [{"type": "string"}, {"type": "null"}],
                    "default": None, "title": "Project"},
        "limit": {"default": 50, "title": "Limit", "type": "integer"},
        "verbose": {"default": False, "title": "Verbose", "type": "boolean"},
    },
}


class TestParseCliPairs:
    def test_separate_value(self):
        assert parse_cli_pairs(["--pattern", "TODO"]) == {"pattern": "TODO"}

    def test_equals_form(self):
        assert parse_cli_pairs(["--pattern=TODO"]) == {"pattern": "TODO"}

    def test_dashes_become_underscores(self):
        assert parse_cli_pairs(["--symbol-name", "x"]) == {"symbol_name": "x"}

    def test_bare_flag_is_true(self):
        assert parse_cli_pairs(["--verbose"]) == {"verbose": "true"}

    def test_bare_flag_before_another_flag(self):
        assert parse_cli_pairs(["--verbose", "--limit", "5"]) == {
            "verbose": "true", "limit": "5",
        }

    def test_value_may_look_negative(self):
        assert parse_cli_pairs(["--limit", "-1"]) == {"limit": "-1"}

    def test_a_bare_word_is_rejected(self):
        with pytest.raises(ToolArgumentError) as e:
            parse_cli_pairs(["pattern", "TODO"])
        assert "pattern" in str(e.value)


class TestCoerceArguments:
    def test_string_passes_through(self):
        out = coerce_arguments(FIND_PATTERN_SCHEMA, {"pattern": "TODO"})
        assert out == {"pattern": "TODO"}

    def test_integer_is_converted(self):
        out = coerce_arguments(FIND_PATTERN_SCHEMA, {"pattern": "x", "limit": "80"})
        assert out["limit"] == 80

    def test_bad_integer_is_reported_with_the_key(self):
        with pytest.raises(ToolArgumentError) as e:
            coerce_arguments(FIND_PATTERN_SCHEMA, {"pattern": "x", "limit": "many"})
        assert "limit" in str(e.value)

    def test_boolean_accepts_both_spellings(self):
        assert coerce_arguments(FIND_PATTERN_SCHEMA,
                                {"pattern": "x", "verbose": "true"})["verbose"] is True
        assert coerce_arguments(FIND_PATTERN_SCHEMA,
                                {"pattern": "x", "verbose": "false"})["verbose"] is False

    def test_nullable_string_is_accepted(self):
        out = coerce_arguments(FIND_PATTERN_SCHEMA, {"pattern": "x", "project": "alpha"})
        assert out["project"] == "alpha"

    def test_omitted_optionals_are_left_out(self):
        """The tool's own defaults must apply — we do not restate them."""
        out = coerce_arguments(FIND_PATTERN_SCHEMA, {"pattern": "x"})
        assert out == {"pattern": "x"}

    def test_missing_required_names_the_key(self):
        with pytest.raises(ToolArgumentError) as e:
            coerce_arguments(FIND_PATTERN_SCHEMA, {})
        assert "pattern" in str(e.value)

    def test_unknown_key_lists_what_is_accepted(self):
        with pytest.raises(ToolArgumentError) as e:
            coerce_arguments(FIND_PATTERN_SCHEMA, {"pattern": "x", "patern": "typo"})
        message = str(e.value)
        assert "patern" in message and "pattern" in message


class TestFormatToolHelp:
    def test_help_shows_arguments_types_and_requiredness(self):
        text = format_tool_help("find_pattern", "Search for structural patterns.",
                                FIND_PATTERN_SCHEMA)
        assert "find_pattern" in text
        assert "--pattern" in text and "required" in text
        assert "--limit" in text and "50" in text
