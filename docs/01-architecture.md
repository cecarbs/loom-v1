# 01 — Architecture

## Components

| Component | Package | Runs where | Responsibility |
|---|---|---|---|
| CLI | `loom.cli` | host | Parse args, load config, start session, start TUI |
| Agent loop | `loom.core` | host | Turn loop, message history, tool dispatch, compaction, emits `AgentEvent`s |
| Providers | `loom.providers` | host | Adapters from canonical messages to Anthropic, OpenAI-compatible (Chat Completions) and OpenAI Responses APIs; model profiles and capabilities |
| Built-in tools | `loom.tools` | host | `read`, `write`, `edit`, `bash`, `execute`, `search_tools`; output budget |
| Kernel client | `loom.kernel.client` | host | Starts the kernel runner inside the sandbox, speaks the JSON-lines protocol |
| Kernel runner | `loom/kernel/runner.py` | **sandbox** | Stdlib-only persistent Python namespace; executes code; forwards tool calls to host |
| Tool broker | `loom.broker` | host | Tool package registry, approvals, secrets, redaction, audit log; executes tool functions |
| Sandbox backends | `loom.sandbox` | host | `none`, `seatbelt`, `docker`: launch processes (kernel, bash) with isolation |
| Sessions | `loom.session` | host | JSONL session store, resume, checkpoints (`/undo`) |
| Tracing | `loom.tracing` | host | Session event log; optional OpenTelemetry export |
| Skills / extensions | `loom.skills`, `loom.extensions` | host | Discover `SKILL.md`; load Python extensions with hooks |
| Sub-agents | `loom.agents` | host | Headless child agent loops, callable from the kernel |
| TUI | `loom.tui` | host | prompt_toolkit input, Rich rendering, approval prompts, slash commands |

## Component diagram

```mermaid
flowchart LR
    subgraph Host["Host process (loom)"]
        TUI["TUI<br/>prompt_toolkit + Rich"]
        BUS(("Event bus"))
        LOOP["Agent loop<br/>loom.core"]
        PROV["Providers<br/>Anthropic / OpenAI-compat"]
        TOOLS["Built-in tools<br/>read write edit bash execute search_tools"]
        KC["Kernel client"]
        BROKER["Tool broker<br/>approvals, secrets, audit"]
        PKGS["Tool packages<br/>splunk, github, ..."]
        SESS["Session store<br/>JSONL + checkpoints"]
    end

    subgraph Sandbox["Sandbox (seatbelt or docker), no network, no secrets"]
        RUNNER["Kernel runner<br/>persistent namespace"]
        BASH["bash subprocesses"]
    end

    LLM[("Model APIs")]
    EXT[("Splunk, GitHub, Jira, ...")]

    TUI <--> BUS
    LOOP --> BUS
    LOOP <--> PROV
    PROV <--> LLM
    LOOP --> TOOLS
    TOOLS --> KC
    TOOLS -->|bash| BASH
    KC <-->|"JSON lines over stdio"| RUNNER
    KC <--> BROKER
    TOOLS -->|search_tools| BROKER
    BROKER --> PKGS
    PKGS --> EXT
    LOOP --> SESS
    BROKER --> SESS
```

Key property: the only path from the sandbox to the outside world is the stdio pipe to the kernel client, and every request on it goes through the broker.

## One user turn

```mermaid
sequenceDiagram
    participant U as User (TUI)
    participant L as Agent loop
    participant P as Provider
    participant T as Tools
    participant S as Session log

    U->>L: user message
    L->>S: append message
    loop until model stops calling tools
        L->>L: compact if over threshold
        L->>P: stream(system, messages, tool schemas)
        P-->>L: text / tool_use deltas
        L-->>U: AgentEvents (rendered live)
        L->>S: append assistant message
        alt tool calls present
            L->>T: run each tool call
            T-->>L: result (after output budget)
            L->>S: append tool results
        end
    end
    L-->>U: TurnFinished
```

## Code-mode call path

```mermaid
sequenceDiagram
    participant M as Model
    participant E as execute tool
    participant K as Kernel runner (sandbox)
    participant B as Broker (host)
    participant U as User
    participant X as Splunk

    M->>E: execute(code)
    E->>K: {"type":"execute", code}
    K->>K: run code in persistent namespace
    K->>E: {"type":"tool_call", "tool":"splunk.search", args}
    E->>B: call(tool, args)
    alt effect is write or destructive
        B->>U: approval request
        U-->>B: allow / deny
    end
    B->>X: HTTPS with secret token
    X-->>B: rows
    B-->>E: redacted result
    E->>K: {"type":"tool_result", value}
    K->>K: rows stay in variables
    K->>E: stream output + execute_result
    E-->>M: short, budgeted text
```

## Package layout

```
loom/
  pyproject.toml
  CLAUDE.md
  docs/
  src/loom/
    __init__.py
    __main__.py
    cli.py
    config.py
    core/
      agent.py          # AgentLoop
      messages.py       # canonical message + block types
      events.py         # AgentEvent types, EventBus
      context.py        # system prompt assembly, context files, skills list
      compaction.py
      prompts/system.md
    providers/
      base.py           # Provider protocol, ProviderEvent
      anthropic.py
      openai_compat.py
      openai_responses.py
      profiles.py       # ModelProfile, ModelCapabilities, built-in defaults
      text_tools.py     # text tool-call fallback for models without native tools
      fake.py           # FakeProvider for tests
      registry.py
    tools/
      base.py           # Tool protocol, ToolResult
      budget.py         # output budget + spill files
      builtin/{read,write,edit,bash,execute,search_tools}.py
    kernel/
      protocol.py       # message dataclasses (host side)
      client.py         # KernelClient
      runner.py         # STDLIB ONLY, runs inside sandbox
    broker/
      broker.py         # ToolBroker
      registry.py       # tool package discovery, @tool decorator
      approval.py
      secrets.py
      redact.py
    sandbox/
      base.py           # SandboxBackend protocol
      none.py
      seatbelt.py
      seatbelt.sb.tmpl
      docker.py         # drives the docker CLI
      Dockerfile
    session/{store,checkpoints}.py
    tracing/{log,otel}.py
    skills/loader.py
    extensions/{api,loader}.py
    agents/subagent.py
    tui/{app,input,render,approvals,commands}.py
  packages/
    loom-tools-splunk/
    loom-tools-demo/    # synthetic data, used by tests
  tests/
```
