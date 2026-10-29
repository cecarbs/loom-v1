# 07 — Sessions, context management, observability

## Storage layout

```
~/.loom/
  config.toml
  trust.json                      # trusted workspaces (extensions + restricted project config)
  history                         # prompt history
  sandbox-venv/                   # seatbelt kernel Python
  skills/  extensions/
  sessions/<project-hash>/<session-id>/
      session.jsonl               # append-only log (source of truth)
      out/0001.txt …              # spilled tool outputs (sandbox: read-only)
      agents/                     # sub-agent transcripts
      sandbox/                    # the ONLY sandbox-writable part
        runner.py                 # copy of the kernel runner
        home/                     # sandbox HOME
        tmp/                      # sandbox TMPDIR
  checkpoints/<project-hash>.git  # shadow git repo
```

`project-hash` = first 12 hex chars of sha256 of the resolved workspace path. `session-id` = sortable timestamp + 4 random chars (e.g. `20261028-1342-a9f2`).

## Session log (`loom.session.store`)

Append-only JSONL, written only by the host process (never writable from the sandbox). Every line: `{"ts": ..., "kind": ..., ...}`. All lines pass through the redactor before writing.

| kind | contents |
|---|---|
| `session_start` | workspace, model, sandbox, mode, loom version |
| `message` | a canonical `Message` (user/assistant) |
| `model_request` | model, token counts, cache stats, latency, stop reason (no full prompt; it's reconstructible) |
| `tool_call` / `tool_result` | name, args, result (full text or spill path), duration, is_error |
| `kernel_exec` | execute id, **full code**, status, duration, output size |
| `broker_call` | tool, args, effect, approval decision, duration, result size |
| `approval` | tool (built-in or broker), request, decision, by (user/policy/override/cancel) |
| `compaction` | summary text, tokens before/after, index of first kept message |
| `checkpoint` | commit sha, files changed |
| `error`, `notice` | text |

Resume (`loom -c` for latest, `loom --resume <id>`) rebuilds history from `message` and `compaction` entries. The kernel does **not** survive restarts; on resume the model is told the kernel is fresh.

## Context management (`loom.core.compaction`)

- Track context usage from provider-reported input tokens (fallback: estimate).
- **Auto-compact** when usage exceeds `[context] compact_at` (default 0.75 of the model's `context_window`).
- Procedure:
  1. Keep the last `keep_recent_turns` (default 3) user turns verbatim.
  2. Ask the current model to summarize everything older with a fixed compaction prompt: goal, decisions, files touched, findings, open questions, next steps.
  3. Fetch the **kernel inventory** and append it to the summary ("Variables available in the kernel: rows (list, len=48211), df (DataFrame, 48211x7) …"), so the model keeps using in-memory data instead of re-fetching.
  4. Replace older messages with one user message: `<summary>…</summary>`. Ensure tool-use/result pairs are never split. This prefix replacement is the only history rewrite loom performs (see "append-only" in `02-…`); messages after the cut stay byte-identical so their thinking blocks remain valid.
- `/compact <instructions>` runs it manually with extra focus instructions.

The kernel is effectively a second, larger memory: the prompt tells the model to store intermediate results in named variables.

## Checkpoints and undo (`loom.session.checkpoints`)

- A shadow git repo (`GIT_DIR=~/.loom/checkpoints/<hash>.git`, `GIT_WORK_TREE=<workspace>`) independent of the user's own git.
- Before the first file-mutating tool call in each turn (`write`, `edit`, `bash`, `execute`), commit a snapshot (`git add -A` respecting the workspace `.gitignore` plus defaults: `.venv`, `node_modules`, `__pycache__`, files > 5 MB).
- `/undo` lists recent checkpoints (turn number, first line of the user prompt, files changed) and restores the chosen one (`git checkout <sha> -- .` plus removal of files created since). Undo affects files only, not external systems or kernel state; say so in the UI.

## Observability (`loom.tracing`)

**Level 1 — built in:** the session JSONL is the trace. `loom trace <session-id>` prints a tree:

```
turn 3  "why did checkout errors spike"            14.2s  $0.06
├─ model  sonnet  in 12.4k (cache 10.1k) out 380   3.1s
├─ execute e4                                     2.4s
│  ├─ broker splunk.search [read] allow           2.1s  48211 rows
│  └─ result 412 chars
└─ model  sonnet  in 13.0k out 520                 3.8s
```

`/trace` in the TUI shows the same for the current session.

**Level 2 — OpenTelemetry (optional extra `loom[otel]`):** when `[tracing] otlp_endpoint` is set, emit spans: `agent.turn` → `llm.call`, `tool.call` → `kernel.execute` → `broker.call`. Use OpenInference semantic conventions for attribute names so traces render well in Arize Phoenix and other OTel backends. Include code text on `kernel.execute` spans; never include secrets (spans go through the redactor).
