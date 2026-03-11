# Fulcria Purple Agent - AgentBeats

A2A-compatible competitive agent for the [AgentBeats](https://agentbeats.dev) hackathon by UC Berkeley RDI.

## Architecture

The agent implements a proper **tool-calling loop** using standard OpenAI function-calling via [litellm](https://docs.litellm.ai/):

1. Receives a task via A2A protocol
2. Parses tool definitions from the input (supports multiple formats)
3. Calls the LLM with tools in OpenAI function-calling format
4. If the LLM returns `tool_calls`, executes them and feeds results back
5. Repeats steps 3-4 until the LLM produces a final text response
6. Returns the response via A2A protocol

### Supported Tool Definition Formats

The agent auto-detects tools embedded in benchmark prompts:

- **OpenAI format**: `{"type": "function", "function": {"name": "...", ...}}`
- **Shorthand format**: `{"name": "...", "description": "...", "parameters": {...}}`
- **XML tags**: `<tools>[...]</tools>`
- **Python signatures**: `def func_name(param: type) -> return_type: """docstring"""`

### Supported LLM Providers

Via litellm, the agent works with 100+ providers:

| Provider | Model Format | Free Tier |
|----------|-------------|-----------|
| OpenAI | `gpt-4o`, `gpt-4o-mini` | No |
| Anthropic | `claude-sonnet-4-20250514` | No |
| Google | `gemini/gemini-2.5-flash` | 500 RPD |
| Groq | `groq/llama-3.3-70b-versatile` | 1K RPD |
| DeepSeek | `deepseek/deepseek-chat` | 10M tokens |
| Cerebras | `cerebras/llama3.3-70b` | 1M tok/day |
| Ollama | `ollama/llama3.2` | Local |

## Quick Start

```bash
# Configure
cp .env.example .env
# Edit .env with your LLM API key and model

# Install
pip install -r requirements.txt

# Run
python agent.py

# Test
python -m pytest test_agent.py -v
```

## Docker

```bash
docker build -t fulcria-purple-agent .
docker run -p 9002:9002 \
  -e LLM_MODEL=groq/llama-3.3-70b-versatile \
  -e GROQ_API_KEY=your-key \
  fulcria-purple-agent
```

## Tracks

### Sprint 1 (Complete)
- Finance Agent (OfficeQA)
- Business Process Agent (CRMArena)
- Game Agent (Minecraft)

### Sprint 2 (Mar 23 - Apr 12)
- Research Agent
- Multi-agent Eval
- tau2-Bench (customer service: airline, retail, telecom) - $5K prizes
- Computer Use & Web Agent

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_MODEL` | `gpt-4o` | litellm model identifier |
| `LLM_TEMPERATURE` | `0.0` | Sampling temperature |
| `MAX_TOOL_STEPS` | `15` | Max tool-calling loop iterations |
| `PORT` | `9002` | Server port |
| `LLM_API_BASE` | - | Custom API base URL |

## Testing

```bash
python -m pytest test_agent.py -v
```

17 unit tests covering tool parsing, format normalization, and local execution.
