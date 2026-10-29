# 09 — Configuration

TOML, validated with pydantic. Precedence (highest first): CLI flags → `<workspace>/.loom/config.toml` → `~/.loom/config.toml` → built-in defaults.

## Project config restrictions

A cloned repo controls `<workspace>/.loom/config.toml`, so it is treated as untrusted. Without that, a repo could set `[models.sonnet] base_url = "https://evil.example"` and receive your API key, or point `[packages.splunk] host` elsewhere and receive your Splunk token. Ignored keys produce one notice listing them.

| Category | Keys | Untrusted project config |
|---|---|---|
| Safe | `default_model` (must name an alias defined in user config or built-ins), `[agent]`, `[context]`, `[budget]`, `[kernel]`, `[skills]`, `[agents]` | applied |
| Tighten-only | `[sandbox] backend` (`none` < `seatbelt` < `docker`), `[sandbox] network` (`allowlist` < `none`), `[sandbox] deny_read` (additions only), `[approvals] mode` (`yolo` < `auto-write` < `ask` < `read-only`), `[approvals.overrides]` (only `deny` values; an `ask` override could loosen a `read-only` deny) | applied only if at least as strict as the user value |
| Restricted | `[models.*]`, `[providers.*]`, `[packages.*]`, `[sandbox] allowlist`, `env_passthrough`, `python_packages`, `[sandbox.docker]`, `[tracing]`, `[broker]` | ignored unless the workspace is trusted (`~/.loom/trust.json`, granted via the same prompt as project extensions, M8) |

CLI flags always apply (they come from the user).

## Example `~/.loom/config.toml`

```toml
default_model = "sonnet"

[models.sonnet]                    # also a built-in profile; shown here for illustration
provider = "anthropic"
model = "claude-sonnet-5-5"        # check provider docs for current IDs
context_window = 1000000           # built-in; lower it to compact earlier and cap per-turn cost
max_tokens = 64000
thinking = { effort = "medium" }   # omit to send no thinking setting (no thinking blocks were seen without it)
# pricing = { input = 2.0, output = 10.0, cache_read = 0.20, cache_write = 2.50 }  # USD per MTok (built-in)

[models.local]
provider = "openai_compat"
base_url = "http://localhost:11434/v1"   # Ollama
model = "qwen3:8b"                       # any tool-capable model you've pulled
context_window = 32768                    # must match num_ctx below
timeout_s = 900                           # slow local hardware
# no api_key_secret: local servers need no auth
[models.local.capabilities]
tools = true                              # set false to use the text tool-call fallback
images = false
parallel_tool_calls = false
[models.local.extra_body]
options = { num_ctx = 32768 }             # Ollama-specific, REQUIRED: Ollama's default context is small and
                                          # silently truncates the system prompt and tools

[models.gpt]
provider = "openai_responses"             # or "openai_compat" for Chat Completions
base_url = "https://api.openai.com/v1"
model = "<openai-model-id>"
api_key_secret = "openai_api_key"         # Keychain secret name
[models.gpt.capabilities]
images = true
thinking = true

[models.router]
provider = "openai_compat"
base_url = "https://openrouter.ai/api/v1"
model = "<vendor/model-id>"
api_key_secret = "openrouter_api_key"
[models.router.headers]
"HTTP-Referer" = "https://github.com/<you>/loom"
"X-Title" = "loom"

[models.internal]                          # self-hosted vLLM behind a corporate gateway
provider = "openai_compat"
base_url = "https://llm.internal.example.com/v1"
model = "<model-id>"
ca_bundle = "/etc/ssl/internal-ca.pem"
[models.internal.headers]
"X-Gateway-Key" = "secret:gateway_key"     # resolved from Keychain at request time

[providers.anthropic]
api_key_secret = "anthropic_api_key"      # falls back to ANTHROPIC_API_KEY env
[providers.openai_compat]
api_key_secret = "openai_api_key"         # default for models that don't set one; falls back to OPENAI_API_KEY

[agent]
max_steps = 50

[context]
compact_at = 0.75
keep_recent_turns = 3
max_context_file_chars = 40000

[budget]                           # output budget for tool results
max_chars = 12000
head_chars = 8000
tail_chars = 3000

[kernel]
eager = false
default_timeout_s = 300

[sandbox]
backend = "docker"                 # none | seatbelt | docker (default; needs a running Docker daemon)
network = "none"                   # none | allowlist (later)
allowlist = []                     # hosts, for allowlist mode
deny_read = []                     # extra paths
env_passthrough = []
python_packages = ["pandas", "numpy", "pyarrow"]

[sandbox.docker]
image = "loom-sandbox:latest"
memory = "4g"
cpus = 2

[approvals]
mode = "ask"                       # read-only | ask | auto-write | yolo
[approvals.overrides]
# "splunk.create_saved_search" = "allow"

[broker]
call_timeout_s = 120

[packages.splunk]                  # per-package config, passed to ctx.config
host = "splunk.example.com"
port = 8089
verify_tls = true
allowed_indexes = ["web", "app"]

[skills]
include_claude = false

[agents]
max_concurrency = 4

[tracing]
otlp_endpoint = ""                 # e.g. http://localhost:6006/v1/traces
log_tool_results = false
```

## Model entry reference (`[models.<alias>]`)

| Key | Type | Default | Notes |
|---|---|---|---|
| `provider` | str | required | `anthropic`, `openai_compat`, `openai_responses`, or an extension-registered name |
| `model` | str | required | Provider's model id |
| `base_url` | str | provider default | Required for local/self-hosted servers |
| `api_key_secret` | str | `[providers.<p>]` value | Keychain secret name; omit for no auth |
| `context_window` | int | built-in profile, else 32768 | Used for compaction thresholds |
| `max_tokens` | int | 8192 | Max output tokens |
| `timeout_s` | float | 600 | Per-request timeout |
| `verify_tls` | bool | true | |
| `ca_bundle` | path | none | Custom CA for self-hosted endpoints |
| `capabilities.*` | bool/str | see `02-…providers.md` | `tools`, `parallel_tool_calls`, `images`, `thinking`, `prompt_cache`, `streaming`, `system_role` |
| `thinking` | table | none (provider default) | `effort` (`low`/`medium`/`high`/`xhigh`/`max`) or, for older models only, `budget_tokens`; ignored with a notice if `capabilities.thinking` is false |
| `pricing` | table | built-in profile, else none | `input`, `output`, `cache_read`, `cache_write` in USD per MTok; none → cost shown as `—` |
| `headers` | table | {} | Values may be `"secret:<name>"` |
| `extra_body` | table | {} | Merged into the request body; may not set `model`, `messages`, `tools`, `stream` |

Unknown keys are a config error (pydantic `extra="forbid"`) so typos are caught early.

## CLI

```
loom                         start in cwd
loom -p "prompt"             one-shot, print final answer, exit (non-interactive; approvals → deny unless --mode allows)
loom -c | --resume <id>      resume latest / specific session
loom --model <alias>
loom --mode <mode>
loom --sandbox <backend>
loom sandbox setup [--docker]
loom secrets set|list|delete <name>
loom models list             configured models with resolved capabilities
loom models test <alias>     probe a model: reply, tool call, image; suggests capability settings
loom tools list              show catalog (installed `loom.tools` entry points)
loom trace <session-id>
loom sessions                list sessions for cwd
```
