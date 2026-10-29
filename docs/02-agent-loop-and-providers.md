# 02 — Agent loop and providers

## Canonical message model (`loom.core.messages`)

The core uses one provider-neutral format, close to Anthropic's content blocks. Adapters convert at the edge.

```python
Role = Literal["user", "assistant"]

@dataclass
class TextBlock:        text: str
@dataclass
class ToolUseBlock:     id: str; name: str; input: dict[str, Any]
@dataclass
class ToolResultBlock:  tool_use_id: str; content: str | list[TextBlock | ImageBlock]; is_error: bool = False
@dataclass
class ImageBlock:       media_type: str; data_b64: str
@dataclass
class ThinkingBlock:    text: str; provider: str; opaque: dict[str, Any]  # e.g. Anthropic signature

Block = TextBlock | ToolUseBlock | ToolResultBlock | ImageBlock | ThinkingBlock

@dataclass
class Message:
    role: Role
    content: list[Block]
    meta: dict[str, Any] = field(default_factory=dict)  # model id, usage, timestamps
```

Rules:
- Tool results are carried in a `user` message (Anthropic style). The OpenAI adapter converts them to `role="tool"` messages. Tool messages can't carry images in Chat Completions, so image parts of a tool result go into a user message right after the tool messages (or are replaced by a text notice when `images = false`).
- `ThinkingBlock`s are kept verbatim and replayed, unchanged, on every request to the adapter that produced them (`provider` field), including after switching between models of that provider: the Anthropic API silently drops blocks the current model can't read. Other adapters drop them.
- **History is append-only.** Current Anthropic models bind thinking blocks to the exact conversation that produced them ("preserved thinking"); editing an earlier turn invalidates later thinking. Only compaction (M6) may rewrite history, and it does so by replacing a prefix with a summary, never by editing messages in place.
- Every `ToolUseBlock` must be followed by a matching `ToolResultBlock` in the next message. On cancel, the loop inserts `ToolResultBlock(content="Interrupted by user", is_error=True)`.

## Provider interface (`loom.providers.base`)

```python
class Provider(Protocol):
    name: str
    def __init__(self, profile: ModelProfile, secrets: SecretStore) -> None: ...
    async def stream(self, req: ModelRequest) -> AsyncIterator[ProviderEvent]: ...
    def count_tokens_estimate(self, messages: list[Message]) -> int: ...  # cheap heuristic is fine

@dataclass
class ModelRequest:
    model: str
    system: str
    messages: list[Message]
    tools: list[ToolSchema]        # name, description, JSON schema
    max_tokens: int
    temperature: float | None = None   # never sent to Anthropic (current API rejects sampling params)
    thinking: ThinkingConfig | None = None
    cache: bool = True             # provider-specific prompt caching hints

ProviderEvent =
    TextDelta(text) | ThinkingDelta(text) |
    ToolUseStart(id, name) | ToolUseInputDelta(id, partial_json) | ToolUseEnd(id, input) |
    MessageEnd(stop_reason, usage: Usage, final: Message) |
    ProviderError(kind: Literal["rate_limit","overloaded","context_too_long","auth","other"], message, retry_after)

@dataclass(frozen=True)
class ThinkingConfig:              # adapters map to their own API shape
    effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None
    budget_tokens: int | None = None     # only for older models that still take a fixed budget
```

Thinking is configured per model (`[models.<alias>] thinking = { effort = "medium" }`); the loop passes `profile.thinking` in every request when `capabilities.thinking` is true. `None` means "provider default", which is **not** necessarily "off": current Anthropic models (e.g. `claude-sonnet-5-5`) think adaptively by default and reject a fixed `budget_tokens`.

Anthropic mapping: `effort` → `thinking={"type": "adaptive", "display": "summarized"}` + `output_config={"effort": …}`; `budget_tokens` → `thinking={"type": "enabled", "budget_tokens": …}` (older models only). `display: "summarized"` is always sent when thinking is configured so the TUI has something to show (the API default returns empty thinking text). OpenAI-compatible mapping: `effort` → `reasoning_effort` when the server supports it; otherwise ignored.

### Adapters

| Adapter | SDK | Notes |
|---|---|---|
| `anthropic` | `anthropic` (async client, Messages streaming, SDK retries off: `max_retries=0`) | Prompt caching: two `cache_control` breakpoints, on the system prompt (tools render before it, so it covers them) and on the last block of the last message (max 4 allowed). Preserve thinking blocks. No `temperature`/`top_p`/`top_k`. Tool definitions don't set `eager_input_streaming`: with it the server stops validating tool input and a truncated string could still pass loom's schema check and be written to disk. Without an API key secret, the SDK's own credential chain is used (`ANTHROPIC_API_KEY`, `ant auth login` profile). |
| `openai_compat` | `openai` (async client, Chat Completions streaming, `max_retries=0`) | `base_url` configurable, so it covers OpenAI, OpenRouter, Ollama, vLLM, LM Studio, llama.cpp server. Accumulate streamed `tool_calls` deltas by index. The widest common denominator; the default for any non-Anthropic endpoint. |
| `openai_responses` | `openai` (async client, Responses API streaming) | For OpenAI models where reasoning features need the Responses API. Maps `function_call` / `function_call_output` items to `ToolUseBlock` / `ToolResultBlock`. Reasoning items are stored as `ThinkingBlock(provider="openai_responses", opaque=<item>)` and replayed only to this adapter (encrypted reasoning content via `include=["reasoning.encrypted_content"]` with `store=false`, so loom stays stateless and never depends on `previous_response_id`). |
| `fake` | none | Scripted list of responses for tests; can emit tool calls. |

Any other API is added by an extension (`api.add_provider(name, factory)`, see `08-skills-extensions-subagents.md`) implementing the same `Provider` protocol.

### Model profiles and capabilities

Each configured model (`[models.<alias>]` in `09-config.md`) resolves to a `ModelProfile` that the loop and adapters consult. Adapters must never send something the profile says the model can't handle.

```python
@dataclass(frozen=True)
class ModelCapabilities:
    tools: bool = True              # native tool calling
    parallel_tool_calls: bool = True
    images: bool = False            # accepts ImageBlock input
    thinking: bool = False          # supports a reasoning/thinking config
    prompt_cache: bool = False      # adapter may add cache hints
    streaming: bool = True
    system_role: Literal["system", "developer", "user"] = "system"

@dataclass(frozen=True)
class ModelProfile:
    alias: str
    provider: str                   # adapter name
    model: str                      # provider's model id
    base_url: str | None
    api_key_secret: str | None      # Keychain secret name; None = no auth (local)
    context_window: int
    max_tokens: int
    capabilities: ModelCapabilities
    headers: dict[str, str]         # extra HTTP headers (values may reference secrets)
    extra_body: dict[str, Any]      # merged into the request body as-is
    thinking: ThinkingConfig | None = None   # None = thinking off
    pricing: Pricing | None = None  # USD per MTok: input, output, cache_read, cache_write; None = cost unknown
    timeout_s: float = 600
    verify_tls: bool = True
    ca_bundle: Path | None = None
```

Behavior driven by capabilities:

| Capability false | What loom does |
|---|---|
| `images` | `ImageBlock`s are replaced with `TextBlock("[image omitted: model does not accept images]")`; `read` on an image returns a text notice. |
| `thinking` | `ThinkingConfig` is not sent, even if the model entry configures `thinking` (a notice says it was ignored). |
| `prompt_cache` | No cache hints are added. |
| `parallel_tool_calls` | Request sets `parallel_tool_calls=false` (OpenAI-compatible) and the loop still handles one call per response. |
| `streaming` | Adapter makes a non-streaming call and emits the whole response as one `TextDelta` + `ToolUseEnd` events. |
| `tools` | **Text tool-call fallback** (below). |

`system_role`: some servers/models reject a `system` message; the adapter then sends the system prompt as a `developer` message, or prepends it to the first user message.

**Defaults:** built-in profiles for known providers (Anthropic models: images, thinking, prompt_cache true; OpenAI models: images true) live in `loom/providers/profiles.py`. The default model is `claude-sonnet-5-5` (alias `sonnet`): 1M context window, 128K max output (loom requests 64K), $2 / $10 / $0.20 / $2.50 per MTok (input / output / cache read / 5-minute cache write = 1.25 × input), per Anthropic's model table as of 2026-10. Because the context window is 1M, `compact_at = 0.75` means turns can reach ~750K input tokens (~$1.50 each uncached); users who care about cost should lower `context_window` for this alias. Built-in `pricing` and `context_window` values are copied from the provider's docs at implementation time, never guessed; when pricing is unknown, cost is shown as `—`. Unknown `openai_compat` models default to `tools=true`, `streaming=true`, `parallel_tool_calls=true`, everything else false; the user overrides per model in config. `loom models test <alias>` runs a short probe (plain reply, one tool call, optional image) and prints which capabilities actually work, suggesting config lines to add.

### Text tool-call fallback (models without native tool calling)

When `capabilities.tools = false`, the adapter:

1. Appends a section to the system prompt listing each tool's name, description and JSON schema, and instructs the model to call a tool by emitting exactly:
   ```
   <tool_call>{"name": "execute", "input": {"code": "..."}}</tool_call>
   ```
2. Parses `<tool_call>…</tool_call>` blocks out of the streamed text (stream text before the first tag normally; buffer once a tag opens) and emits them as `ToolUseStart/ToolUseEnd` events with generated ids. Invalid JSON produces a tool result error asking the model to resend valid JSON.
3. Sends tool results back as a user message: `<tool_result id="…">…</tool_result>`.

The canonical history is unchanged (real `ToolUseBlock`/`ToolResultBlock`), so switching to a native-tools model mid-session works. Code mode makes this fallback practical: there are only six tools to describe.

### Custom headers, extra parameters, timeouts

- `headers`: added to every request (gateways, corporate proxies, OpenRouter's `HTTP-Referer`/`X-Title`). A value of the form `"secret:<name>"` is resolved from the Keychain at request time; header values are redacted in logs.
- `extra_body`: merged into the request body without validation, for server-specific options (`top_k`, `repetition_penalty`, Ollama `options.num_ctx`, vLLM `guided_json`, OpenRouter `provider` routing). Keys loom sets itself (`model`, `messages`, `tools`, `stream`) cannot be overridden; config validation rejects them.
- `timeout_s`: per-model request timeout (local models on slow hardware may need more).
- TLS: `verify_tls` (default true) and optional `ca_bundle` path, for self-hosted servers with internal CAs.

### Retries

The loop (not the adapter) retries `rate_limit` and `overloaded` errors with exponential backoff (1s, 2s, 4s, 8s, max 4 tries, honoring `retry_after`). `context_too_long` triggers an immediate compaction, then one retry. Other errors end the turn with an `ErrorEvent`.

## Agent loop (`loom.core.agent.AgentLoop`)

```python
async def run_turn(self, user_input: str, attachments: list[Block] = ()) -> None:
    self.history.append(Message("user", [TextBlock(user_input), *attachments]))
    self.bus.emit(TurnStarted())
    for step in range(self.cfg.max_steps):          # default 50
        await self.compaction.maybe_compact(self.history)
        req = self.context.build_request(self.history, self.tools.schemas())
        msg = await self._stream_model(req)          # emits TextDelta etc. to the bus
        self.history.append(msg)
        calls = [b for b in msg.content if isinstance(b, ToolUseBlock)]
        if not calls:
            break
        if msg.meta["stop_reason"] == "max_tokens":   # tool input may be truncated
            self.history.append(Message("user", truncated_results(calls)))  # error results, not run
            continue
        results = await self.tools.run_all(calls)    # sequential by default
        self.history.append(Message("user", results))
    else:
        self.bus.emit(Notice("Step limit reached; stopping."))
    self.bus.emit(TurnFinished(usage=self.usage))
```

- **Tool execution order:** sequential. Exception: consecutive `read` calls may run concurrently.
- **Cancellation:** the TUI cancels the turn's `asyncio.Task`. The loop catches `CancelledError`, interrupts any running kernel execution (`KernelClient.interrupt()`), kills running bash processes, cancels in-flight broker calls (pending approvals resolve as deny), logs `TurnCancelled`, and returns. History repair:
  - Cancelled **while streaming**: keep the partial assistant text (with `meta.interrupted = true`); drop `ToolUseBlock`s whose input was not complete. If nothing remains, drop the assistant message.
  - Cancelled **during a tool batch**: completed calls keep their real results; the running and not-yet-started calls get `ToolResultBlock(content="Interrupted by user", is_error=True)`.
- **Steering:** not in v1 (see `00-overview.md` Non-goals). Input is not accepted while a turn runs.

## Event bus (`loom.core.events`)

A simple in-process pub/sub. Subscribers are async callables; the bus awaits them in order. Events are frozen dataclasses:

```
TurnStarted, TextDelta, ThinkingDelta, AssistantMessageDone,
ToolCallStarted(id, name, input), ToolCallOutput(id, chunk),   # live streaming from bash/execute
ToolCallFinished(id, result_preview, is_error, duration_ms),
ApprovalRequested(request_id, tool, args_preview, effect),     # TUI must answer via bus.respond()
KernelStatus(state: "starting"|"idle"|"busy"|"dead"),
Compacted(before_tokens, after_tokens), Notice(text), ErrorEvent(text),
UsageUpdated(input_tokens, output_tokens, cache_read, context_pct),
TurnCancelled, TurnFinished
```

`ApprovalRequested` is the only event that expects a reply. It carries a `request_id`; the TUI calls `bus.respond(request_id, decision)`, which resolves an `asyncio.Future` the broker is awaiting.

## System prompt (`core/prompts/system.md`)

Keep it short (target < 1,000 tokens). It contains:

1. Identity and working directory, OS, date.
2. Tool guidance — the important part:
   - Use `execute` for anything with more than ~2 steps, loops, data processing, or external systems.
   - Use `search_tools` to find functions before calling them.
   - Aggregate in the source system (e.g. SPL `stats`) and in code; **print summaries, never raw bulk data**.
   - Variables persist in the kernel; refer to them in later steps instead of re-fetching.
   - Use `read`/`edit`/`write` for files, `bash` for shell commands.
3. Context files (AGENTS.md / CLAUDE.md contents), appended.
4. Skills list (name, description, path), appended.

The system prompt must stay byte-stable within a session so prompt caching works. Dynamic state is **not** put in it: the kernel inventory travels in the compaction summary (`07-…`), and kernel restarts are reported by the `[kernel restarted …]` prefix of the next `execute` result.
