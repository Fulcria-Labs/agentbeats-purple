"""AgentBeats Purple Agent - A2A-compatible competitive agent for Sprint 1."""

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime, timezone

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

# LLM backend configuration
LLM_BACKEND = os.environ.get("LLM_BACKEND", "litellm")  # litellm, anthropic, google
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o")


async def call_llm(messages: list[dict], system_prompt: str | None = None) -> str:
    """Call the configured LLM backend."""
    if LLM_BACKEND == "anthropic":
        import anthropic

        client = anthropic.AsyncAnthropic()
        sys_msgs = [m for m in messages if m["role"] == "system"]
        chat_msgs = [m for m in messages if m["role"] != "system"]
        system = system_prompt or (sys_msgs[0]["content"] if sys_msgs else "")
        resp = await client.messages.create(
            model=LLM_MODEL or "claude-sonnet-4-20250514",
            max_tokens=4096,
            system=system,
            messages=chat_msgs,
        )
        return resp.content[0].text

    elif LLM_BACKEND == "google":
        import google.generativeai as genai

        model = genai.GenerativeModel(LLM_MODEL or "gemini-2.0-flash")
        # Convert messages to Gemini format
        parts = []
        for m in messages:
            parts.append(f"{m['role']}: {m['content']}")
        response = await asyncio.to_thread(
            model.generate_content, "\n".join(parts)
        )
        return response.text

    else:
        # Default: litellm (supports openai, anthropic, google, ollama, etc.)
        from litellm import acompletion

        full_messages = messages.copy()
        if system_prompt and not any(m["role"] == "system" for m in full_messages):
            full_messages.insert(0, {"role": "system", "content": system_prompt})
        kwargs = {"model": LLM_MODEL, "messages": full_messages, "temperature": 0.0}
        # Support Ollama via api_base
        api_base = os.environ.get("OLLAMA_API_BASE", os.environ.get("LLM_API_BASE"))
        if api_base:
            kwargs["api_base"] = api_base
        elif LLM_MODEL.startswith("ollama/"):
            kwargs["api_base"] = "http://localhost:11434"
        resp = await acompletion(**kwargs)
        return resp.choices[0].message.content


SYSTEM_PROMPT = """You are an AI agent participating in the AgentBeats competition.
You receive tasks that may include tool definitions and user queries.
When tools are provided, you must respond in the specified JSON format.

Key rules:
1. When given tools, respond ONLY with JSON wrapped in <json>...</json> tags
2. The JSON must contain "name" (tool name or "respond") and "kwargs" (arguments)
3. Be precise with tool arguments - match the expected types exactly
4. When responding directly, use {"name": "respond", "kwargs": {"content": "your message"}}
5. Think step by step before acting
6. Follow instructions carefully - the assessment depends on exact behavior
"""


class PurpleAgentExecutor(AgentExecutor):
    """Execution logic for the AgentBeats purple agent."""

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
            task_id, context_id, len(user_input) if user_input else 0,
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

        # Maintain conversation history per context
        if context_id not in self.conversations:
            self.conversations[context_id] = [
                {"role": "system", "content": SYSTEM_PROMPT}
            ]

        self.conversations[context_id].append(
            {"role": "user", "content": user_input}
        )

        try:
            response_text = await call_llm(
                self.conversations[context_id],
                system_prompt=SYSTEM_PROMPT,
            )
        except Exception as e:
            logger.error("[Purple] LLM call failed: %s", e)
            response_text = json.dumps({
                "name": "respond",
                "kwargs": {"content": f"Error: {e}"}
            })
            response_text = f"<json>{response_text}</json>"

        self.conversations[context_id].append(
            {"role": "assistant", "content": response_text}
        )

        if task_id not in self.running_tasks:
            logger.info("[Purple] Task %s was cancelled", task_id)
            return

        logger.info("[Purple] Response: %s", response_text[:200])

        # Send final response
        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                task_id=task_id,
                context_id=context_id,
                status=TaskStatus(
                    state=TaskState.input_required,
                    message=Message(
                        role="agent",
                        message_id=str(uuid.uuid4()),
                        parts=[TextPart(text=response_text)],
                        task_id=task_id,
                        context_id=context_id,
                    ),
                    timestamp=datetime.now(timezone.utc).isoformat(),
                ),
                final=True,
            )
        )


def create_app(host: str = "0.0.0.0", port: int = 9002) -> None:
    """Create and run the A2A purple agent server."""
    url = f"http://{host}:{port}"

    agent_card = AgentCard(
        name="Fulcria Purple Agent",
        description="A competitive AI agent for AgentBeats Sprint 1. "
        "Handles tool-calling tasks, financial QA (OfficeQA), "
        "and business process benchmarks.",
        url=url,
        version="1.0.0",
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
                description="Handles tasks requiring tool use with structured JSON responses",
                tags=["tool-calling", "general", "finance", "business"],
                examples=[
                    "Answer financial questions using provided tools",
                    "Complete business process tasks step by step",
                ],
            ),
            AgentSkill(
                id="office_qa",
                name="Office QA",
                description="Answers financial and office-related questions",
                tags=["finance", "qa", "office"],
                examples=[
                    "What was the revenue last quarter?",
                    "Calculate the profit margin",
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
    logger.info("Starting Fulcria Purple Agent on %s:%d", host, port)
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "9002"))
    create_app(port=port)
