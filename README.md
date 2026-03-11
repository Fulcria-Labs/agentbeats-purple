# Fulcria Purple Agent - AgentBeats Sprint 1

A2A-compatible competitive agent for the [AgentBeats](https://agentbeats.dev) hackathon.

## Tracks
- Finance Agent (OfficeQA)
- Business Process Agent (CRMArena)
- Game Agent (Minecraft)

## Quick Start

```bash
# Copy and configure environment
cp .env.example .env
# Edit .env with your LLM API key

# Install dependencies
pip install -r requirements.txt

# Run the agent
python agent.py
```

## Docker

```bash
docker build -t fulcria-purple-agent .
docker run -p 9002:9002 --env-file .env fulcria-purple-agent
```

## LLM Backends

Supports multiple backends via environment variables:
- `litellm` (default) - routes to OpenAI, Anthropic, Google, etc.
- `anthropic` - direct Anthropic SDK
- `google` - direct Google Generative AI SDK

## Architecture

The agent implements the A2A protocol v0.3.0:
- Receives tasks from green agents (evaluators)
- Maintains conversation history per context
- Responds with structured JSON for tool-calling benchmarks
- Supports streaming and task cancellation
