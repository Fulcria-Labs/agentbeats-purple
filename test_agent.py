"""Tests for the AgentBeats Purple Agent tool parsing and execution logic."""

import json
import pytest
from agent import (
    parse_tools_from_input,
    execute_local_tool,
    _is_tool_def,
    _normalize_tool_defs,
    _func_signature_to_tool,
)


class TestToolParsing:
    """Test tool definition extraction from user input."""

    def test_parse_openai_format_in_json_block(self):
        """Tools in ```json blocks with OpenAI format should be extracted."""
        user_input = """Here is your task. Use the following tools:

```json
[
    {
        "type": "function",
        "function": {
            "name": "search_flights",
            "description": "Search for available flights",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {"type": "string"},
                    "destination": {"type": "string"}
                },
                "required": ["origin", "destination"]
            }
        }
    }
]
```

Find me a flight from NYC to LAX."""

        tools, cleaned = parse_tools_from_input(user_input)
        assert len(tools) == 1
        assert tools[0]["type"] == "function"
        assert tools[0]["function"]["name"] == "search_flights"
        assert "Find me a flight" in cleaned

    def test_parse_shorthand_format(self):
        """Shorthand tool defs (name + parameters) should be normalized."""
        user_input = """<tools>
[{"name": "get_weather", "description": "Get weather info", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}]
</tools>

What is the weather in London?"""

        tools, cleaned = parse_tools_from_input(user_input)
        assert len(tools) == 1
        assert tools[0]["type"] == "function"
        assert tools[0]["function"]["name"] == "get_weather"
        assert "What is the weather" in cleaned

    def test_parse_function_signature_format(self):
        """Python function signatures should be converted to tool defs."""
        user_input = """You have the following tools:

def search_customer(customer_id: str, include_history: bool) -> dict:
    \"\"\"Search for a customer by their ID and optionally include order history.\"\"\"

Find customer C12345."""

        tools, cleaned = parse_tools_from_input(user_input)
        assert len(tools) == 1
        assert tools[0]["function"]["name"] == "search_customer"
        params = tools[0]["function"]["parameters"]
        assert "customer_id" in params["properties"]
        assert params["properties"]["customer_id"]["type"] == "string"
        assert params["properties"]["include_history"]["type"] == "boolean"

    def test_no_tools_in_plain_text(self):
        """Plain text input should yield no tools."""
        user_input = "What is the capital of France?"
        tools, cleaned = parse_tools_from_input(user_input)
        assert len(tools) == 0
        assert cleaned == user_input

    def test_tools_tag_with_single_object(self):
        """<tools> tag with a single object (not array) should work."""
        user_input = """<tools>
{"name": "calculate", "description": "Do math", "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}}}
</tools>

Calculate 2 + 2."""

        tools, cleaned = parse_tools_from_input(user_input)
        assert len(tools) == 1
        assert tools[0]["function"]["name"] == "calculate"

    def test_multiple_function_signatures(self):
        """Multiple Python function defs should all be extracted."""
        user_input = """Available tools:

def get_balance(account_id: str) -> float:
    \"\"\"Get account balance.\"\"\"

def transfer_funds(from_id: str, to_id: str, amount: float) -> dict:
    \"\"\"Transfer funds between accounts.\"\"\"

Transfer $100 from A to B."""

        tools, cleaned = parse_tools_from_input(user_input)
        assert len(tools) == 2
        names = {t["function"]["name"] for t in tools}
        assert "get_balance" in names
        assert "transfer_funds" in names


class TestToolDefValidation:
    """Test tool definition validation helpers."""

    def test_is_tool_def_openai_format(self):
        assert _is_tool_def({"type": "function", "function": {"name": "test"}})

    def test_is_tool_def_shorthand(self):
        assert _is_tool_def({"name": "test", "parameters": {}})

    def test_is_not_tool_def(self):
        assert not _is_tool_def({"key": "value"})
        assert not _is_tool_def("string")
        assert not _is_tool_def(42)

    def test_normalize_openai_format(self):
        items = [{"type": "function", "function": {"name": "test"}}]
        result = _normalize_tool_defs(items)
        assert len(result) == 1
        assert result[0] == items[0]

    def test_normalize_shorthand(self):
        items = [{"name": "test", "description": "A test", "parameters": {"type": "object"}}]
        result = _normalize_tool_defs(items)
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "test"


class TestFuncSignatureConversion:
    """Test Python function signature to OpenAI tool conversion."""

    def test_basic_conversion(self):
        tool = _func_signature_to_tool(
            "my_func", "name: str, count: int", "Does something."
        )
        assert tool["function"]["name"] == "my_func"
        assert tool["function"]["description"] == "Does something."
        props = tool["function"]["parameters"]["properties"]
        assert props["name"]["type"] == "string"
        assert props["count"]["type"] == "integer"

    def test_with_defaults(self):
        tool = _func_signature_to_tool(
            "my_func", "name: str, limit: int = 10", "Has defaults."
        )
        required = tool["function"]["parameters"].get("required", [])
        assert "name" in required
        assert "limit" not in required

    def test_no_params(self):
        tool = _func_signature_to_tool("refresh", "", "Refresh data.")
        assert tool["function"]["parameters"]["properties"] == {}


class TestLocalToolExecution:
    """Test local tool execution fallback."""

    def test_respond_tool(self):
        result = execute_local_tool("respond", {"content": "Hello!"})
        assert result == "Hello!"

    def test_done_tool(self):
        result = execute_local_tool("done", {"message": "Task complete"})
        assert result == "Task complete"

    def test_unknown_tool(self):
        result = execute_local_tool("search", {"query": "test"})
        parsed = json.loads(result)
        assert parsed["status"] == "executed"
        assert parsed["tool"] == "search"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
