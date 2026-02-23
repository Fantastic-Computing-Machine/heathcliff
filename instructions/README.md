# Instructions Directory

This directory contains centralized prompt templates and instructions for the Heathcliff AI assistant.

## Files

### `prompts.py`

Core prompt templates for the agent, including:

- **`build_system_prompt(master_info)`**: Builds the main system instruction that defines Heathcliff's behavior, emphasizing:
  - Concise, voice-optimized responses
  - **Single-pass tool execution** to prevent redundant API calls
  - Context-aware decision making
  - Natural conversation flow
  - Tool usage rules (never call the same tool twice, check feedback first)

Prompt context architecture:
- **Long-term memories** from Mem0 are injected into `USER_PROMPT_TEMPLATE` under `Long-term Memory Context`.
- **Current date/time metadata** is injected into `USER_PROMPT_TEMPLATE` for each invocation.
- **Semantic history pairs** (past conversations from all sessions, retrieved by similarity to the current query) appear as `HumanMessage`/`AIMessage` objects before the recent history.
- **Recent chat pairs** (last N turns from the current session) appear chronologically right before the current user message.

## Anti-Redundancy Features

The prompts in this directory are specifically designed to **reduce redundant tool calls** and lower API costs:

1. **Explicit tool usage rules** in the system prompt:
   - "NEVER call the same tool multiple times for one user request"
   - "ALWAYS check tool feedback first"
   - Decision flowchart before any tool call

2. **Combined with agent logic**:
   - Max iteration limit (default: 3) in `core/agent_core.py`
   - Duplicate tool call filtering
   - Iteration count tracking

## Usage

The prompts are imported and used by `core/agent_core.py`:

```python
from instructions.prompts import build_system_prompt

# System prompt is built with master info from config:
system_prompt = build_system_prompt(master_info=config.master)
```

## Customization

To modify Heathcliff's behavior:

1. Edit `build_system_prompt()` in `prompts.py`
2. Adjust response style, tool usage rules, or voice optimization guidelines
3. Changes take effect immediately on agent restart

## Best Practices

- Keep system prompt concise for token efficiency
- Emphasize critical rules (like "no duplicate tool calls") multiple times
- Use clear formatting (headers, bullet points) for LLM clarity
- Test prompt changes with verbose logging (`LOG_LEVEL=DEBUG`)
