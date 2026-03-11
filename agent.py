"""AgentBeats Purple Agent - A2A-compatible competitive agent.

Supports Sprint 1 (OfficeQA, CRMArena, Game) and Sprint 2
(Research Agent, Multi-agent Eval, tau2-Bench, Computer Use).

Uses standard OpenAI function-calling via litellm instead of custom
<json> tags. Implements a proper tool-calling loop with multi-step
execution within a single task.
"""

import asyncio
import json
import logging
import os
import re
import uuid
from datetime import datetime, timezone
from typing import Any

import uvicorn
from a2a.server.agent_execution.agent_executor import AgentExecutor
from a2a.server.agent_execution.context import RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events.event_queue import EventQueue
from a2a.server.request_handlers.default_request_handler import (
    DefaultRequestHandler,
)
from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    Message,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
    TextPart,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agentbeats-purple")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o")
MAX_TOOL_STEPS = int(os.environ.get("MAX_TOOL_STEPS", "15"))
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.0"))


# ---------------------------------------------------------------------------
# LLM call with native function-calling support
# ---------------------------------------------------------------------------
async def call_llm(
    messages: list[dict],
    tools: list[dict] | None = None,
    tool_choice: str | None = None,
) -> dict:
    """Call LLM via litellm with native function-calling support.

    Returns the raw response choice message dict with keys:
        - content: str | None
        - tool_calls: list[dict] | None (each with id, function.name, function.arguments)
    """
    from litellm import acompletion

    kwargs: dict[str, Any] = {
        "model": LLM_MODEL,
        "messages": messages,
        "temperature": LLM_TEMPERATURE,
    }

    # Support Ollama / custom API base
    api_base = os.environ.get("OLLAMA_API_BASE", os.environ.get("LLM_API_BASE"))
    if api_base:
        kwargs["api_base"] = api_base
    elif LLM_MODEL.startswith("ollama/"):
        kwargs["api_base"] = "http://localhost:11434"

    # Pass tools if available (standard OpenAI function-calling format)
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = tool_choice or "auto"

    # Disable thinking for Claude models (litellm quirk)
    if "claude" in LLM_MODEL.lower():
        kwargs["thinking"] = {"type": "disabled"}

    resp = await acompletion(**kwargs)
    msg = resp.choices[0].message

    result: dict[str, Any] = {
        "content": msg.content,
        "tool_calls": None,
    }

    if msg.tool_calls:
        result["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in msg.tool_calls
        ]

    return result


# ---------------------------------------------------------------------------
# Tool definition parsing from user input
# ---------------------------------------------------------------------------
def parse_tools_from_input(user_input: str) -> tuple[list[dict], str]:
    """Extract tool definitions from user input if present.

    Benchmarks may embed tool definitions in the user prompt in several
    formats. We detect and extract them, returning both the OpenAI-format
    tool schemas and the cleaned user message.

    Returns:
        (tools_list, cleaned_input) where tools_list is a list of OpenAI
        function-calling tool dicts, and cleaned_input has tool defs removed.
    """
    tools: list[dict] = []

    # Pattern 1: JSON array of tool definitions
    # Look for ```json blocks or raw JSON arrays containing "function" schemas
    json_block_pattern = re.compile(
        r'```(?:json)?\s*(\[[\s\S]*?\])\s*```', re.MULTILINE
    )
    matches = json_block_pattern.findall(user_input)
    for match in matches:
        try:
            parsed = json.loads(match)
            if isinstance(parsed, list) and parsed:
                # Check if it looks like OpenAI tool defs
                if all(_is_tool_def(item) for item in parsed):
                    tools.extend(_normalize_tool_defs(parsed))
                    user_input = user_input.replace(f"```json\n{match}\n```", "")
                    user_input = user_input.replace(f"```\n{match}\n```", "")
        except json.JSONDecodeError:
            pass

    # Pattern 2: <tools>...</tools> XML-style tags
    tools_tag_pattern = re.compile(
        r'<tools>\s*([\s\S]*?)\s*</tools>', re.MULTILINE
    )
    for match in tools_tag_pattern.findall(user_input):
        try:
            parsed = json.loads(match)
            if isinstance(parsed, list):
                tools.extend(_normalize_tool_defs(parsed))
            elif isinstance(parsed, dict):
                tools.extend(_normalize_tool_defs([parsed]))
            user_input = re.sub(
                r'<tools>\s*[\s\S]*?\s*</tools>', '', user_input, count=1
            )
        except json.JSONDecodeError:
            pass

    # Pattern 3: Inline JSON objects with "type": "function"
    # (for benchmarks that embed tools directly in text)
    if not tools:
        inline_pattern = re.compile(
            r'\{[^{}]*"type"\s*:\s*"function"[^{}]*"function"\s*:\s*\{[^}]*\}[^}]*\}',
            re.DOTALL,
        )
        for match in inline_pattern.finditer(user_input):
            try:
                parsed = json.loads(match.group())
                if _is_tool_def(parsed):
                    tools.extend(_normalize_tool_defs([parsed]))
            except json.JSONDecodeError:
                pass

    # Pattern 4: Function signature style (tau2-bench format)
    # def function_name(param1: type, param2: type) -> return_type:
    #     """docstring"""
    func_pattern = re.compile(
        r'def\s+(\w+)\s*\(([^)]*)\)\s*(?:->\s*\w+)?\s*:\s*\n\s*"""(.*?)"""',
        re.DOTALL,
    )
    for match in func_pattern.finditer(user_input):
        fname, params_str, docstring = match.groups()
        tool_def = _func_signature_to_tool(fname, params_str, docstring.strip())
        if tool_def:
            tools.append(tool_def)

    return tools, user_input.strip()


def _is_tool_def(item: Any) -> bool:
    """Check if a dict looks like an OpenAI tool definition."""
    if not isinstance(item, dict):
        return False
    # Direct OpenAI format: {"type": "function", "function": {...}}
    if item.get("type") == "function" and "function" in item:
        return True
    # Shorthand: {"name": "...", "description": "...", "parameters": {...}}
    if "name" in item and ("parameters" in item or "description" in item):
        return True
    return False


def _normalize_tool_defs(items: list[dict]) -> list[dict]:
    """Normalize various tool definition formats to OpenAI standard."""
    normalized = []
    for item in items:
        if item.get("type") == "function" and "function" in item:
            # Already in OpenAI format
            normalized.append(item)
        elif "name" in item:
            # Shorthand format - wrap in OpenAI structure
            func_def = {
                "name": item["name"],
                "description": item.get("description", ""),
            }
            if "parameters" in item:
                func_def["parameters"] = item["parameters"]
            else:
                func_def["parameters"] = {
                    "type": "object",
                    "properties": {},
                }
            normalized.append({"type": "function", "function": func_def})
    return normalized


def _func_signature_to_tool(
    name: str, params_str: str, docstring: str
) -> dict | None:
    """Convert a Python function signature to OpenAI tool format."""
    properties: dict[str, dict] = {}
    required: list[str] = []

    type_map = {
        "str": "string",
        "int": "integer",
        "float": "number",
        "bool": "boolean",
        "list": "array",
        "dict": "object",
        "List": "array",
        "Dict": "object",
        "Optional": "string",
    }

    for param in params_str.split(","):
        param = param.strip()
        if not param:
            continue

        # Handle default values
        has_default = "=" in param
        if has_default:
            param = param.split("=")[0].strip()

        # Handle type annotations
        if ":" in param:
            pname, ptype = param.split(":", 1)
            pname = pname.strip()
            ptype = ptype.strip()
            json_type = type_map.get(ptype, "string")
        else:
            pname = param.strip()
            json_type = "string"

        if not pname:
            continue

        properties[pname] = {"type": json_type}
        if not has_default:
            required.append(pname)

    parameters = {"type": "object", "properties": properties}
    if required:
        parameters["required"] = required

    return {
        "type": "function",
        "function": {
            "name": name,
            "description": docstring,
            "parameters": parameters,
        },
    }


# ---------------------------------------------------------------------------
# Simulated tool execution (for tools defined in benchmark prompts)
# ---------------------------------------------------------------------------
def execute_local_tool(tool_name: str, arguments: dict) -> str:
    """Execute a tool locally when no external tool server is available.

    For benchmarks, tool calls are typically evaluated by the orchestrator.
    This function provides a fallback response indicating the tool was called,
    so the agent can continue its reasoning loop.

    In production (with MCP or external tool servers), this would dispatch
    to the actual tool implementation.
    """
    # Built-in tools that the agent can always use
    if tool_name == "respond" or tool_name == "done":
        content = arguments.get("content", arguments.get("message", ""))
        return content

    # For benchmark tools, return a structured acknowledgment
    # The benchmark orchestrator intercepts these before they reach here
    return json.dumps({
        "status": "executed",
        "tool": tool_name,
        "arguments": arguments,
        "result": f"Tool '{tool_name}' called with arguments: {json.dumps(arguments)}",
    })


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a helpful AI agent participating in competitive benchmarks.

You can use tools when they are available. When you need to call a tool:
- Use the function-calling mechanism (tool_calls) to invoke tools
- Wait for tool results before proceeding
- You can make multiple tool calls in sequence to complete a task
- Be precise with tool arguments - match expected types exactly

When responding to the user directly (no tool needed):
- Provide clear, accurate, and concise answers
- Think step by step for complex problems
- Follow task instructions carefully

Key behaviors:
- Always follow the policy or instructions provided in the task
- If you are a customer service agent, be helpful and follow the policy exactly
- If tools are available, prefer using them over guessing
- Never fabricate tool results - always call the tool first
"""


# ---------------------------------------------------------------------------
# Agent Executor with tool-calling loop
# ---------------------------------------------------------------------------
class PurpleAgentExecutor(AgentExecutor):
    """Execution logic for the AgentBeats purple agent.

    Implements a proper tool-calling loop: the agent can make multiple
    LLM calls within a single task, executing tools between calls,
    until it produces a final text response.
    """

    def __init__(self) -> None:
        self.conversations: dict[str, list[dict]] = {}
        self.running_tasks: set[str] = set()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        task_id = context.task_id
        self.running_tasks.discard(task_id)
        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                task_id=task_id,
                context_id=context.context_id or str(uuid.uuid4()),
                status=TaskStatus(
                    state=TaskState.canceled,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                ),
                final=True,
            )
        )

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task_id = context.task_id
        context_id = context.context_id or str(uuid.uuid4())
        self.running_tasks.add(task_id)

        # Get user input
        user_input = context.get_user_input()
        logger.info(
            "[Purple] Task %s (ctx %s): received %d chars",
            task_id,
            context_id,
            len(user_input) if user_input else 0,
        )

        # Send working status
        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                task_id=task_id,
                context_id=context_id,
                status=TaskStatus(
                    state=TaskState.working,
                    message=Message(
                        role="agent",
                        message_id=str(uuid.uuid4()),
                        parts=[TextPart(text="Processing...")],
                        task_id=task_id,
                        context_id=context_id,
                    ),
                    timestamp=datetime.now(timezone.utc).isoformat(),
                ),
                final=False,
            )
        )

        # Extract tools from user input if present
        tools, cleaned_input = parse_tools_from_input(user_input)
        if tools:
            logger.info(
                "[Purple] Extracted %d tools from input: %s",
                len(tools),
                [t["function"]["name"] for t in tools],
            )

        # Initialize or continue conversation
        if context_id not in self.conversations:
            self.conversations[context_id] = [
                {"role": "system", "content": SYSTEM_PROMPT}
            ]

        self.conversations[context_id].append(
            {"role": "user", "content": cleaned_input or user_input}
        )

        # Tool-calling loop
        response_text = await self._run_tool_loop(
            context_id, tools, task_id
        )

        if task_id not in self.running_tasks:
            logger.info("[Purple] Task %s was cancelled", task_id)
            return

        logger.info("[Purple] Final response: %s", response_text[:300])

        # Determine final state
        final_state = TaskState.completed if response_text else TaskState.input_required

        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                task_id=task_id,
                context_id=context_id,
                status=TaskStatus(
                    state=final_state,
                    message=Message(
                        role="agent",
                        message_id=str(uuid.uuid4()),
                        parts=[TextPart(text=response_text or "No response generated.")],
                        task_id=task_id,
                        context_id=context_id,
                    ),
                    timestamp=datetime.now(timezone.utc).isoformat(),
                ),
                final=True,
            )
        )

    async def _run_tool_loop(
        self,
        context_id: str,
        tools: list[dict] | None,
        task_id: str,
    ) -> str:
        """Run the LLM in a loop, executing tool calls until a text response.

        This implements the standard agent loop:
        1. Call LLM with messages + tools
        2. If LLM returns tool_calls, execute them and add results to messages
        3. Repeat until LLM returns a text response (no tool_calls)
        4. Return the final text response
        """
        messages = self.conversations[context_id]

        for step in range(MAX_TOOL_STEPS):
            if task_id not in self.running_tasks:
                return "Task was cancelled."

            try:
                llm_response = await call_llm(
                    messages,
                    tools=tools if tools else None,
                )
            except Exception as e:
                logger.error("[Purple] LLM call failed at step %d: %s", step, e)
                return f"I encountered an error: {e}"

            tool_calls = llm_response.get("tool_calls")
            content = llm_response.get("content")

            if not tool_calls:
                # No tool calls - this is the final response
                if content:
                    messages.append({"role": "assistant", "content": content})
                    return content
                else:
                    return "I was unable to generate a response."

            # LLM wants to call tools - add the assistant message with tool_calls
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": content,
                "tool_calls": tool_calls,
            }
            messages.append(assistant_msg)

            logger.info(
                "[Purple] Step %d: %d tool call(s): %s",
                step,
                len(tool_calls),
                [tc["function"]["name"] for tc in tool_calls],
            )

            # Execute each tool call and add results
            for tc in tool_calls:
                tc_id = tc["id"]
                func_name = tc["function"]["name"]
                try:
                    func_args = json.loads(tc["function"]["arguments"])
                except json.JSONDecodeError:
                    func_args = {}

                # Check for terminal tool calls
                if func_name in ("respond", "done"):
                    final_content = func_args.get(
                        "content", func_args.get("message", content or "")
                    )
                    if final_content:
                        return final_content

                # Execute the tool
                try:
                    result = execute_local_tool(func_name, func_args)
                except Exception as e:
                    result = json.dumps({"error": str(e)})

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": result,
                })

        # Exhausted max steps - ask LLM for a final summary
        messages.append({
            "role": "user",
            "content": "You have used the maximum number of tool steps. Please provide your final answer now.",
        })
        try:
            final_resp = await call_llm(messages, tools=None)
            final_text = final_resp.get("content", "")
            if final_text:
                messages.append({"role": "assistant", "content": final_text})
                return final_text
        except Exception as e:
            logger.error("[Purple] Final summary call failed: %s", e)

        return "I reached the maximum number of steps without completing the task."


# ---------------------------------------------------------------------------
# A2A server setup
# ---------------------------------------------------------------------------
def create_app(host: str = "0.0.0.0", port: int = 9002) -> None:
    """Create and run the A2A purple agent server."""
    url = f"http://{host}:{port}"

    agent_card = AgentCard(
        name="Fulcria Purple Agent",
        description=(
            "A competitive AI agent for AgentBeats. "
            "Handles tool-calling tasks, customer service (tau2-bench), "
            "financial QA (OfficeQA), business process (CRMArena), "
            "and research agent benchmarks. "
            "Supports standard OpenAI function-calling via litellm."
        ),
        url=url,
        version="2.0.0",
        protocol_version="0.3.0",
        capabilities=AgentCapabilities(
            streaming=True,
            push_notifications=False,
            state_transition_history=True,
        ),
        default_input_modes=["text"],
        default_output_modes=["text", "task-status"],
        skills=[
            AgentSkill(
                id="tool_calling",
                name="Tool Calling Agent",
                description=(
                    "Handles tasks requiring multi-step tool use with "
                    "standard function-calling. Supports OpenAI, Anthropic, "
                    "Google, Groq, and other LLM providers via litellm."
                ),
                tags=["tool-calling", "general", "function-calling"],
                examples=[
                    "Use the provided tools to answer questions",
                    "Execute a sequence of tool calls to complete a task",
                ],
            ),
            AgentSkill(
                id="customer_service",
                name="Customer Service Agent",
                description=(
                    "Handles customer service tasks following domain policies. "
                    "Compatible with tau2-bench airline, retail, and telecom domains."
                ),
                tags=["customer-service", "tau2-bench", "policy"],
                examples=[
                    "Help a customer change their flight booking",
                    "Process a return request following store policy",
                ],
            ),
            AgentSkill(
                id="research",
                name="Research Agent",
                description="Conducts research tasks using available tools and knowledge",
                tags=["research", "analysis", "qa"],
                examples=[
                    "Research and summarize findings on a topic",
                    "Answer questions using provided data sources",
                ],
            ),
        ],
        supports_authenticated_extended_card=False,
    )

    request_handler = DefaultRequestHandler(
        agent_executor=PurpleAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )

    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )

    app = server.build()
    logger.info("Starting Fulcria Purple Agent v2.0.0 on %s:%d", host, port)
    logger.info("LLM Model: %s | Max tool steps: %d", LLM_MODEL, MAX_TOOL_STEPS)
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "9002"))
    create_app(port=port)
