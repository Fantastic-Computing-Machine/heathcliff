# Instructions Directory

This directory contains centralized prompt templates and instructions for the Heathcliff AI assistant.

## Files

### `prompts.py`

Core prompt templates for the agent, including:

- **`SYSTEM_PROMPT`**: Main system instruction that defines Heathcliff's behavior, emphasizing:
  - Concise, voice-optimized responses
  - **Single-pass tool execution** to prevent redundant API calls
  - Context-aware decision making
  - Natural conversation flow

- **`CONTEXT_TEMPLATE`**: Format for organizing context sections (memories, chat history, tool results)

- **`build_full_prompt()`**: Helper function to assemble complete prompts

## Anti-Redundancy Features

The prompts in this directory are specifically designed to **reduce redundant tool calls** and lower API costs:

1. **Explicit tool usage rules** in `SYSTEM_PROMPT`:
   - "NEVER call the same tool multiple times for one user request"
   - "ALWAYS check tool feedback first"
   - Decision flowchart before any tool call

2. **Clear tool result presentation**:
   - Tool feedback is prominently displayed with checkmarks
   - Messages emphasize "DO NOT re-call these tools"

3. **Combined with agent logic**:
   - Max iteration limit (default: 3) in `core/agent_core.py`
   - Duplicate tool call filtering
   - Iteration count tracking

## Usage

The prompts are imported and used by `core/agent_core.py`:

```python
from instructions.prompts import SYSTEM_PROMPT

# In _build_prompt_template():
ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("system", "Tool feedback (DO NOT re-call these tools):\n{tool_results_block}"),
    # ...
])
```

## Customization

To modify Heathcliff's behavior:

1. Edit `SYSTEM_PROMPT` in `prompts.py`
2. Adjust response style, tool usage rules, or voice optimization guidelines
3. Changes take effect immediately on agent restart

## Best Practices

- Keep system prompt concise for token efficiency
- Emphasize critical rules (like "no duplicate tool calls") multiple times
- Use clear formatting (headers, bullet points) for LLM clarity
- Test prompt changes with verbose logging (`LOG_LEVEL=DEBUG`)
