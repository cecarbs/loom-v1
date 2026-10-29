"""loom-tools-splunk: SPL search through the Splunk Python SDK (`splunk-sdk`).

Configure in ~/.loom/config.toml and store the token in the Keychain:

    [packages.splunk]
    host = "splunk.example.com"
    port = 8089
    verify_tls = true
    allowed_indexes = ["web", "app"]   # optional; when set, every search must name one

    $ loom secrets set splunk_token     # a Splunk authentication token (Bearer)
"""

from __future__ import annotations

import re
from typing import Any

from loom.broker import Secret, ToolPackage

package = ToolPackage(
    name="splunk",
    description="Search Splunk with SPL; list indexes; create saved searches.",
    secrets={"token": Secret("splunk_token", "Splunk authentication token")},
    config_keys=["host", "port", "scheme", "verify_tls", "allowed_indexes"],
)

MAX_ROWS = 50_000
INDEX_RE = re.compile(r"""\bindex\s*=\s*["']?([^\s"'|)]+)""", re.IGNORECASE)


@package.client("splunk")
def _connect(ctx) -> Any:
    import splunklib.client

    host = ctx.config.get("host")
    if not host:
        raise ValueError(
            "Splunk is not configured: set [packages.splunk] host in ~/.loom/config.toml"
        )
    return splunklib.client.connect(
        host=host,
        port=int(ctx.config.get("port", 8089)),
        scheme=ctx.config.get("scheme", "https"),
        splunkToken=ctx.secret("token"),
        verify=bool(ctx.config.get("verify_tls", True)),  # the SDK's default is False
        autologin=True,
    )


def normalize_spl(spl: str) -> str:
    """The search API needs a leading `search` command unless the SPL starts with a pipe."""
    spl = spl.strip()
    if spl.startswith("|") or spl.lower().startswith("search "):
        return spl
    return f"search {spl}"


def check_indexes(ctx, spl: str) -> None:
    allowed = ctx.config.get("allowed_indexes")
    if not allowed:
        return
    named = INDEX_RE.findall(spl)
    if not named:
        raise PermissionError(f"Name an index explicitly (allowed: {', '.join(allowed)}).")
    denied = sorted({i for i in named if i not in allowed})
    if denied:
        raise PermissionError(
            f"Index not allowed: {', '.join(denied)} (allowed: {', '.join(allowed)})."
        )


@package.tool(effect="read")
def search(
    ctx, spl: str, earliest: str = "-1h", latest: str = "now", max_rows: int = 1000
) -> list[dict]:
    """Run an SPL search and return result rows.

    Aggregate with stats/timechart in SPL where possible; keep max_rows small. Time bounds
    use Splunk syntax ("-24h", "@d", ISO timestamps)."""
    from splunklib import results

    if not 1 <= max_rows <= MAX_ROWS:
        raise ValueError(f"max_rows must be between 1 and {MAX_ROWS}")
    query = normalize_spl(spl)
    check_indexes(ctx, query)
    stream = ctx.client("splunk").jobs.oneshot(
        query, earliest_time=earliest, latest_time=latest, output_mode="json", count=max_rows
    )
    rows: list[dict] = []
    for item in results.JSONResultsReader(stream):
        if isinstance(item, dict):
            rows.append(item)
        elif getattr(item, "type", "") in ("FATAL", "ERROR"):
            raise RuntimeError(f"Splunk: {item.message}")
        if len(rows) >= max_rows:
            break
    return rows


@package.tool(effect="read")
def list_indexes(ctx) -> list[str]:
    """Index names visible to the token (only allowed ones, if allowed_indexes is set)."""
    allowed = ctx.config.get("allowed_indexes")
    names = sorted(ix.name for ix in ctx.client("splunk").indexes)
    return [n for n in names if not allowed or n in allowed]


@package.tool(effect="write")
def create_saved_search(ctx, name: str, spl: str) -> dict:
    """Create a saved search (e.g. to schedule an alert later)."""
    query = normalize_spl(spl)
    check_indexes(ctx, query)
    saved = ctx.client("splunk").saved_searches.create(name, query)
    return {"name": saved.name, "search": query}
