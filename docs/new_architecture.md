# Heathcliff New Architecture

## Core Direction

Heathcliff is a personal butler-style AI assistant: a Jarvis-like system that can understand broad user intent, coordinate across personal services, and execute useful work through specialized capabilities.

The redesign should be explicitly object-oriented and service-based, not a loose collection of functions.

```txt
Heathcliff should use concrete classes, typed data structures, explicit service boundaries,
controlled singletons, and caching where it improves runtime behavior.
```

The target style is:

```txt
Typed OOP services + LangGraph orchestration.
```

LangGraph should own workflow control: routing, fan-out, joins, handoffs, approvals, and final synthesis. Concrete Python services should own runtime responsibilities: context building, agent registration, persistence, memory, approvals, tool access, and external integrations.

## Runtime Shape

One user-facing turn should run through a `HeathcliffRuntime` service. The runtime owns the request lifecycle, while individual services own their specific responsibilities.

```txt
HeathcliffRuntime
  - owns one user-facing conversation turn
  - coordinates context loading, planning, execution, synthesis, and persistence

ContextBuilder
  - builds SystemMessage / HumanMessage / AIMessage inputs
  - selects recent history, semantic history, memories, and subagent context

AgentRegistry
  - stores available specialist agents
  - exposes agent metadata for routing

PromptRegistry
  - stores prompt templates and versions
  - separates stable instructions from dynamic context
  - exposes Heathcliff and specialist prompts

HandoffResolver
  - validates handoff requests
  - prevents invalid loops and excessive delegation

ApprovalService
  - centralizes user approval before side effects

ConversationManager
  - persists conversation history in Chroma
  - retrieves chronological and semantic history as LangChain messages

MemoryManager
  - uses Mem0 for memory extraction/search behavior
  - extracts, categorizes, stores, and retrieves long-term memories
```

The runtime should coordinate these services, not absorb their responsibilities.

```txt
User request
  -> HeathcliffRuntime
  -> ContextBuilder
  -> HeathcliffAgent
  -> LangGraph execution
  -> specialist agents
  -> synthesis / approval / persistence
```

## Persistence And Memory Services

Conversation history and extracted memories are separate concepts and should live in separate Chroma collections.

```txt
heathcliff_conversations
  - raw multimodal conversation messages
  - chronological history retrieval
  - semantic history retrieval

heathcliff_memories
  - extracted personal/relevant memories
  - categorized long-term facts
  - source links to conversation/message ids
```

DB responsibilities should live in a dedicated `db` module:

```txt
db/
  base.py
    - ChromaConnection singleton
    - collection lifecycle: create, get, delete, destroy
    - generic CRUD/query helpers
    - shared serialization utilities

  conversation_manager.py
    - save user and assistant messages
    - store searchable text projections
    - store full multimodal message payloads
    - retrieve recent chronological message pairs
    - retrieve semantically relevant message pairs
    - reconstruct list[HumanMessage | AIMessage]

  memory_manager.py
    - use Mem0 for memory extraction/search behavior
    - extract personal/relevant memories asynchronously
    - categorize, deduplicate, and upsert memories
    - link memories to source conversation/message records
```

Conversation records should store both search-friendly text and the full message payload:

```txt
ConversationMessageRecord
  - id
  - conversation_id
  - turn_id
  - message_index
  - role
  - searchable_text
  - message_payload
  - artifact_uris
  - created_at
  - metadata
```

`searchable_text` is used for embeddings and semantic search. `message_payload` preserves the full LangChain-compatible message content, including multimodal content blocks.

```python
HumanMessage(
    content=[
        {"type": "text", "text": "What is in this image?"},
        {"type": "image", "source": {"type": "url", "url": image_url}},
    ]
)
```

Memories are not generic chat summaries. They are durable personal or relevant facts extracted from conversations.

```txt
Memory categories
  - identity / profile
  - preferences
  - routines
  - relationships
  - projects / goals
  - commitments
  - constraints
  - important facts
  - corrections / updates
```

Async memory creation flow:

```txt
1. ConversationManager saves the completed turn.
2. HeathcliffRuntime returns the response to the user.
3. MemoryManager receives the turn in the background.
4. Mem0 extracts candidate memories.
5. MemoryManager categorizes and deduplicates candidates.
6. MemoryManager upserts accepted memories into heathcliff_memories.
7. Each memory stores source conversation/message ids.
```

## Multi-Agent Topology

Heathcliff should act as the top-level orchestrator, not as one giant agent with every tool attached. The specialist agents should own domain-specific tools, prompts, and execution behavior.

```txt
Heathcliff Agent
  -> task-specific agents with focused tools
```

Initial specialist set:

```txt
HeathcliffAgent
  - owns persona, routing behavior, and final response synthesis
  - knows about specialist agents, not every low-level tool

Specialist agents
  - Email Agent
  - Calendar Agent
  - Music Agent
  - Info / Research Agent
  - Contacts Agent
  - Comms Agent
  - Memory Agent
  - Skills Agent
```

Each specialist should be represented by a concrete class with a focused prompt, a focused tool set, and a structured output contract.

```txt
BaseSpecialistAgent
  - name
  - description
  - allowed tools
  - build task-specific instructions
  - run task packet
  - return AgentResult
```

## Fan-Out With Shared State

The architecture should support planner-led fan-out. Heathcliff should not broadcast every request to every agent. It should select only the agents needed for the current request.

```txt
User request
  -> Heathcliff Agent
  -> plan / route
  -> fan out to relevant specialist agents
  -> gather results in shared state
  -> synthesize final response
  -> request approval when needed
  -> execute side effects after approval
```

This is inspired by ADK's multi-agent and parallel-agent patterns:

- Use a parent/coordinator agent to organize specialist agents.
- Run independent subtasks concurrently when possible.
- Store each specialist's result in shared state.
- Merge gathered results in a synthesis step.
- Keep state writes explicit to avoid race conditions.

The shared state should be structured around concrete data objects.

```txt
ContextBundle
TaskPlan
TaskPacket
AgentResult
HandoffRequest
ApprovalRequest
ConversationMessageRecord
MemoryRecord
ToolExecutionRecord
```

Agents should write to their own result namespace. The gather/synthesis step should read across namespaces.

```txt
shared_state.agent_results["calendar"]
shared_state.agent_results["research"]
shared_state.agent_results["email"]
```

## Controlled Handoffs

Specialist agents may request help from another specialist when it would improve the current task. Handoffs should be controlled by Heathcliff, not performed as unrestricted agent-to-agent calls.

```txt
Calendar Agent:
  "I need attendee email addresses."

HandoffRequest:
  target_agent = "contacts"
  reason = "Resolve attendee names to email addresses"
```

The handoff flow should remain centralized:

```txt
Specialist Agent
  -> emits HandoffRequest
  -> HandoffResolver validates the request
  -> Heathcliff dispatches the requested agent
  -> result is written to shared state
  -> original task continues or synthesis proceeds
```

This keeps collaboration possible while avoiding unbounded loops, hidden side effects, and tangled agent behavior.

## Context Management

Heathcliff should receive broad orchestration context. Specialist agents should receive narrow execution context.

### Heathcliff Context

Heathcliff's model input should be assembled as a normal LangChain message list:

```python
from langchain.messages import SystemMessage, HumanMessage, AIMessage

messages = [
    SystemMessage(heathcliff_system_prompt),
    HumanMessage(previous_user_message_1),
    AIMessage(previous_assistant_message_1),
    HumanMessage(previous_user_message_2),
    AIMessage(previous_assistant_message_2),
    HumanMessage(current_message_bundle),
]

response = model.invoke(messages)
```

The final message order should be:

```txt
1. SystemMessage
   - Heathcliff persona
   - specialist agent information
   - routing rules
   - guardrails
   - limitations
   - approval policy

2. History as real message objects
   - past N chronological user/assistant message pairs
   - past M semantically relevant user/assistant message pairs
   - deduplicated before invocation
   - passed as HumanMessage / AIMessage objects

3. Current HumanMessage
   - current user query
   - relevant user memories
   - current runtime context
   - active task state
   - selected subagent context and results
   - citations / provenance when available
```

Conversation history must not be serialized into the current `HumanMessage`.

Do this:

```txt
SystemMessage(...)
HumanMessage(previous user turn)
AIMessage(previous assistant turn)
HumanMessage(current user query + current-turn context)
```

Do not do this:

```txt
HumanMessage("
Previous conversation:
User: ...
Assistant: ...

Current user query:
...
")
```

The `ContextBuilder` should request history from `ConversationManager` as real LangChain messages. It should not know Chroma collection details or storage serialization details.

### Subagent Context

Subagent outputs are first-class context inputs for Heathcliff.

For example, a Research Agent may return a large body of research that should inform the current answer. That full output should live in shared state, while the current `HumanMessage` should receive the selected parts needed for the next model call.

Recommended subagent result shape:

```python
class AgentResult(BaseModel):
    agent_name: str
    status: str
    summary: str
    key_facts: list[str]
    detailed_context: str
    sources: list[dict[str, str]]
    handoff_requests: list[dict]
    proposed_actions: list[dict]
```

The `ContextBuilder` decides what to include in the prompt:

```txt
Always consider:
  - summary
  - key facts
  - sources / citations

Conditionally include:
  - excerpts from detailed_context
  - raw tool results
  - long research notes
```

The goal is to make subagent work available to Heathcliff without flooding every model call with raw text.

### Memory Context

The `ContextBuilder` should request relevant memories from `MemoryManager`. It should receive categorized memory records, not raw extraction artifacts.

Context assembly should combine:

```txt
Conversation context
  - recent chronological message pairs
  - semantically relevant message pairs

Memory context
  - relevant categorized memories
  - source/provenance when useful
```

### Specialist Context

Specialist agents should not receive Heathcliff's entire context by default. Each specialist should receive a focused task packet:

```txt
TaskPacket
  - task id
  - user goal
  - specific instruction
  - relevant current-turn context
  - relevant memories
  - relevant upstream subagent results
  - allowed tools
  - expected output schema
```

Specialist instructions should be strict but not restrictive:

- Do the assigned task.
- Use only the allowed domain tools.
- Return structured output.
- Report uncertainty clearly.
- Ask for missing information when needed.
- Emit a handoff request when another specialist can help.

## Prompting And Grounding

Prompts should be treated as architecture, not scattered strings. Heathcliff and every specialist should have a clear, versioned prompt with explicit responsibilities, grounding rules, and output expectations.

Prompt definitions should live in a dedicated module:

```txt
prompts/
  base.py
    - PromptTemplate base class
    - PromptRegistry singleton
    - prompt version metadata
    - shared formatting helpers

  heathcliff.py
    - Heathcliff persona
    - orchestration instructions
    - routing rules
    - grounding rules
    - approval rules
    - response style

  specialists/
    email.py
    calendar.py
    research.py
    music.py
    contacts.py
    comms.py
    memory.py
```

Prompt structure should keep stable instructions first and dynamic context later. This improves clarity and keeps reusable prompt prefixes cache-friendly.

```txt
Stable prompt prefix
  - identity / role
  - responsibilities
  - available agents or tools
  - routing and execution rules
  - guardrails and limitations
  - output contract

Dynamic prompt context
  - current user query
  - selected memories
  - retrieved history
  - active task state
  - subagent results
  - tool outputs and citations
```

Use Markdown headings for readable hierarchy and XML-style tags for variable context boundaries.

```txt
<identity>
...
</identity>

<routing_rules>
...
</routing_rules>

<current_user_query>
...
</current_user_query>

<retrieved_context>
...
</retrieved_context>
```

### Heathcliff Prompt

The Heathcliff prompt should teach orchestration, not domain execution. Heathcliff should know which specialist agents exist, when to call them, when to fan out, when to ask a clarification, when to ask approval, and how to synthesize a final answer.

```txt
<identity>
You are Heathcliff, a personal butler-style AI assistant.
</identity>

<responsibilities>
  - understand the user's intent
  - answer directly when no specialist is needed
  - route or fan out to specialist agents when useful
  - synthesize specialist results
  - ask for clarification when required inputs are missing
  - request approval before side effects
</responsibilities>

<available_agents>
  - Calendar Agent: calendar reads, availability, event proposals, event execution
  - Email Agent: email search, drafting, sending after approval
  - Research Agent: source-backed research and factual grounding
  - Memory Agent: memory search and memory management
</available_agents>

<grounding_rules>
  - ground factual claims in memories, conversation history, tool output, or cited research
  - do not invent missing facts
  - if evidence is insufficient, say what is missing
  - if sources conflict, report the conflict
</grounding_rules>

<approval_rules>
  - never perform side effects without user approval
  - side effects include sending messages, creating/updating/deleting events, purchases, and account changes
</approval_rules>
```

### Specialist Prompts

Specialist prompts should be narrow, strict, and operational. They should include the agent's role, allowed tools, task boundaries, handoff rules, failure behavior, and expected structured output.

```txt
<role>
You are the Calendar Agent.
</role>

<task_scope>
Handle calendar availability, calendar reads, event proposals, and approved event execution.
</task_scope>

<rules>
  - use only calendar tools
  - do not answer unrelated questions
  - request handoff if contacts, email, or research context is needed
  - never create, update, or delete events without approval
</rules>

<output_contract>
Return AgentResult with status, summary, key facts, proposed actions, and handoff requests.
</output_contract>
```

Specialist prompts should not copy Heathcliff's full persona or full orchestration policy. They should receive focused `TaskPacket` context and return structured `AgentResult` data.

### Grounding Rules

Grounding should be a system behavior, not a style preference.

```txt
Grounding sources
  - conversation history from ConversationManager
  - categorized memories from MemoryManager
  - tool outputs
  - cited research sources
  - explicit user input in the current turn
```

For factual or actionable responses:

- Prefer grounded facts over model knowledge.
- Cite research sources when the answer depends on external research.
- Keep source/provenance available for memory-derived claims when useful.
- Do not treat missing evidence as evidence of absence.
- Ask a clarification when the task cannot be completed safely.
- Return uncertainty explicitly when a specialist cannot verify a claim.

### Prompt Quality Rules

- Define the agent's role and responsibilities explicitly.
- Put stable instructions before dynamic context.
- Use concise, concrete rules instead of broad prose.
- Use examples only where they improve routing or output reliability.
- Keep output contracts structured and typed.
- Keep specialist prompts short enough to preserve task focus.
- Evaluate prompt changes with scenario tests before treating them as stable.

## Lifecycle And Caching

Singletons should be used only for stable shared resources:

```txt
Good singleton candidates:
  - Config
  - Chroma connection/client
  - model registry / model factory
  - agent registry
  - prompt registry
  - tool registry
  - memory manager
  - Langfuse/tracing client
```

Per-turn mutable state should stay request-scoped:

```txt
Do not use singletons for:
  - current conversation state
  - active task plan
  - per-turn LangGraph state
  - current user message bundle
  - agent result buffers
```

Caching should be explicit and bounded:

```txt
Cache:
  - initialized model clients
  - compiled LangGraph graphs
  - stable prompt prefixes
  - OAuth/service clients
  - Chroma collections
  - loaded tool lists
  - stable config/profile reads
  - within-turn context-builder results
```

Avoid caching volatile or safety-sensitive decisions:

```txt
Avoid caching:
  - permission decisions
  - live calendar/email/music state without a short TTL
  - user intent classifications across unrelated turns
  - volatile weather/news/search results without a TTL
```

## Architecture Rules

- Heathcliff owns planning, routing, synthesis, memory policy, and approval policy.
- Specialist agents own domain execution and domain-specific tools.
- Agents write results into their own shared-state namespace.
- Agents may request handoffs, but Heathcliff decides whether to route them.
- Parallel work is allowed only when subtasks are independent.
- Side-effect actions require approval before execution.
- Shared state should carry structured results, not raw internal reasoning traces.
- History should be passed as real LangChain message objects, not serialized into the current prompt.
- Prompts should separate stable instructions from dynamic context.
- Heathcliff prompts should focus on orchestration; specialist prompts should focus on execution.
- Factual and actionable responses should be grounded in memories, history, tool outputs, or cited research.

## Sources

- ADK multi-agent systems: https://adk.dev/agents/multi-agents/
- ADK parallel agents: https://adk.dev/agents/workflow-agents/parallel-agents/
- LangChain multi-agent patterns: https://docs.langchain.com/oss/python/langchain/multi-agent
- LangChain messages: https://docs.langchain.com/oss/python/langchain/messages
- LangChain subagent prompts: https://docs.langchain.com/oss/python/deepagents/subagents
- OpenAI prompt engineering: https://developers.openai.com/api/docs/guides/prompt-engineering
- OpenAI prompt guidance: https://developers.openai.com/api/docs/guides/prompt-guidance
- Google Gemini prompting strategies: https://ai.google.dev/gemini-api/docs/prompting-strategies
- Google Gemini context caching: https://ai.google.dev/gemini-api/docs/caching
- Anthropic prompt engineering overview: https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/overview
- Anthropic Claude prompting best practices: https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices
