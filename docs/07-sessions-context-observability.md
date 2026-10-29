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
      lock                        # pid of the loom process using the session
      checkpoint.index            # this session's index for the shadow repo
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
| `model_request` | model, settings (max_tokens, thinking, temperature, cache, tool and message counts), token counts, cache stats, latency, stop reason (no full prompt; it's reconstructible) |
| `tool_call` / `tool_result` | name, args, result (full text or spill path), duration, is_error |
| `kernel_exec` | execute id, **full code**, status, duration, output size |
| `broker_call` | tool, args, effect, approval decision, duration, result size |
| `approval` | tool (built-in or broker), request, decision, by (user/policy/override/cancel) |
| `compaction` | summary text, tokens before/after, index of first kept message |
| `compaction` (cont.) | also the summary `message` itself, so resume replays it exactly |
| `checkpoint` | commit sha, turn, prompt (first line), files changed since the previous checkpoint |
| `undo` | checkpoint sha restored, files restored, files removed |
| `note` | text queued for the model's next prompt (e.g. after `/undo`) |
| `session_resume` | like `session_start`, appended when a session is resumed |
| `error`, `notice` | text |

Resume (`loom -c` for latest, `loom --resume <id>` with an id or unique prefix, `/resume <id>` in the TUI) rebuilds history from `message` and `compaction` entries, plus turn count, token usage and cost (`model_request` entries carry `cost_usd`), checkpoints, and undelivered `note`s. It continues on the session's last model unless `--model` is given. The kernel does **not** survive restarts: if the session used it, the next prompt carries a note that the kernel is fresh. Not restored: "allow for this session" approval grants, the approval mode (comes from config/CLI), and the read set (files must be read again before `edit`). `model_request` entries also carry `purpose` (`turn` or `compaction`).

A session is used by one loom process at a time: `<session_dir>/lock` holds the pid, and resuming a session whose lock belongs to a live process fails with a clear error (a dead process's lock is taken over). `loom sessions` and `/sessions` list this workspace's sessions (id, last active, model, first prompt).

Notes to the model (resume, `/undo`) are extra text blocks in the next user message, never system-prompt changes, so the prompt cache stays valid.

## Context management (`loom.core.compaction`)

- Track context usage from provider-reported prompt tokens of the last request plus an estimate (~4 chars/token) of messages added since (fallback: estimate everything).
- **Auto-compact** when usage exceeds `[context] compact_at` (default 0.75 of the model's `context_window`).
- Procedure:
  1. Keep the last `keep_recent_turns` (default 3) user turns verbatim. If that leaves nothing older to summarize while a long turn runs (auto-compaction), cut before the latest assistant message instead, so only it and its tool results stay verbatim. A turn start is a user message with text and no tool results; an earlier summary isn't one.
  2. Ask the current model to summarize everything older with a fixed compaction prompt: goal, decisions, files touched, findings, open questions, next steps. The request is the unchanged history prefix plus one user message with the prompt (same system prompt and tools, so it reads from the prompt cache); its text isn't streamed to the UI and its usage counts toward the session.
  3. Fetch the **kernel inventory** and append it to the summary ("Variables available in the kernel: rows (list, len=48211), df (DataFrame, 48211x7) …"), so the model keeps using in-memory data instead of re-fetching.
  4. Replace older messages with one user message: `<summary>…</summary>`. Ensure tool-use/result pairs are never split. This prefix replacement is the only history rewrite loom performs (see "append-only" in `02-…`); messages after the cut stay byte-identical so their thinking blocks remain valid.
- The summary message may be followed by the first kept user message: two consecutive user messages, which the providers accept (Anthropic merges them).
- Auto-compaction runs before a model request, at most once per history growth (no loop when the kept part alone is over the threshold). A `context_too_long` error compacts (mid-turn rule) and retries once. A failed or cancelled compaction leaves history unchanged.
- The UI gets a `Compacted(before_tokens, after_tokens)` event (estimates).
- `/compact <instructions>` runs it manually with extra focus instructions.

The kernel is effectively a second, larger memory: the prompt tells the model to store intermediate results in named variables.

## Checkpoints and undo (`loom.session.checkpoints`)

- A shadow git repo (`GIT_DIR=~/.loom/checkpoints/<hash>.git`, `GIT_WORK_TREE=<workspace>`) independent of the user's own git (which it never touches; `.git` is never snapshotted). Each session has its own index (`<session_dir>/checkpoint.index`) and ref (`refs/loom/<session-id>`), so concurrent sessions don't interfere. Commits are made with plumbing (`write-tree`, `commit-tree`, `update-ref`) and the user's global/system git config is ignored. Nothing sandboxed can read or write the repo.
- Before the first file-mutating tool call in each turn (`write`, `edit`, `bash`, `execute`, after approval and only when the workspace is writable), commit a snapshot (`git add -A` respecting the workspace `.gitignore` plus defaults: `.venv`, `venv`, `node_modules`, `__pycache__`, `.DS_Store`, and files > 5 MB, which are never restored or removed by undo). A failed snapshot shows a notice and doesn't block the tool. Without `git` on the host, checkpoints are off.
- `/undo` lists the session's checkpoints whose turn changed something (turn number, first line of the user prompt, files that turn changed) and restores the chosen one: changed and deleted files are checked out from it, files created since are removed (empty directories too). The current state is checkpointed first ("before /undo"), so an undo can be undone. The model is told what was restored with its next prompt (a logged `note`). Undo affects files only, not external systems or kernel state; the UI says so.

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
