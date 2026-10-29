"""loom-tools-demo: synthetic web logs and support tickets, with no external dependencies.

Powers tests and first-run tutorials. The logs cover one day of traffic across a handful
of services, with a burst of 500s on `/checkout` from host web-3 around 02:00.
"""

from __future__ import annotations

import copy
import functools
import random
from datetime import UTC, datetime, timedelta
from typing import Any

from loom.broker import ToolPackage

package = ToolPackage(
    name="demo",
    description="Synthetic web logs and support tickets (demo data, no setup needed).",
)

DAY = datetime(2026, 1, 15, tzinfo=UTC)
SERVICES = ("web", "api", "auth", "payments", "search")
HOSTS = {s: [f"{s}-{i}" for i in range(1, 5)] for s in SERVICES}
PATHS = {
    "web": ["/", "/checkout", "/cart", "/product", "/login"],
    "api": ["/v1/orders", "/v1/users", "/v1/inventory"],
    "auth": ["/token", "/refresh", "/logout"],
    "payments": ["/charge", "/refund"],
    "search": ["/query", "/suggest"],
}
LEVELS = ("DEBUG", "INFO", "WARN", "ERROR")
TOTAL_ROWS = 48_000


@functools.cache
def _logs() -> tuple[dict[str, Any], ...]:
    rng = random.Random(42)
    rows = []
    for i in range(TOTAL_ROWS):
        ts = DAY + timedelta(seconds=rng.uniform(0, 86_400))
        service = rng.choice(SERVICES)
        host = rng.choice(HOSTS[service])
        path = rng.choice(PATHS[service])
        status = rng.choices([200, 201, 304, 404, 500], [80, 5, 8, 5, 2])[0]
        latency = int(rng.lognormvariate(4, 0.6))
        # The incident: web-3 fails checkouts between 01:50 and 02:30.
        if (ts.hour == 2 and ts.minute < 30) or (ts.hour == 1 and ts.minute >= 50):
            if service == "web" and rng.random() < 0.5:
                host, path, status = "web-3", "/checkout", 500
                latency = int(rng.uniform(2000, 9000))
        level = "ERROR" if status >= 500 else "WARN" if status == 404 else rng.choice(LEVELS[:2])
        message = {
            500: f"upstream error on {path}: payments timeout",
            404: f"not found: {path}",
        }.get(status, f"{path} served")
        rows.append(
            {
                "id": i,
                "ts": ts.isoformat(),
                "service": service,
                "host": host,
                "level": level,
                "status": status,
                "path": path,
                "latency_ms": latency,
                "message": message,
            }
        )
    rows.sort(key=lambda r: r["ts"])
    return tuple(rows)


@package.tool(effect="read")
def search_logs(
    ctx,
    service: str | None = None,
    level: str | None = None,
    contains: str | None = None,
    min_status: int | None = None,
    limit: int = 10_000,
) -> list[dict]:
    """Return log rows (ts, service, host, level, status, path, latency_ms, message).

    Filters are combined with AND; `contains` matches the message. Rows come oldest first.
    Keep the result in a variable and aggregate it in Python rather than printing it."""
    out = []
    for row in _logs():
        if service and row["service"] != service:
            continue
        if level and row["level"] != level.upper():
            continue
        if contains and contains.lower() not in row["message"].lower():
            continue
        if min_status is not None and row["status"] < min_status:
            continue
        out.append(dict(row))
        if len(out) >= limit:
            break
    return out


@package.tool(effect="read")
def list_services(ctx) -> list[str]:
    """Names of the services that appear in the logs."""
    return list(SERVICES)


SEED_TICKETS = [
    {
        "id": "T-101",
        "title": "Checkout failing for some users",
        "body": "Several customers report 500 errors at checkout overnight.",
        "status": "open",
        "priority": "P1",
        "service": "web",
    },
    {
        "id": "T-102",
        "title": "Slow search suggestions",
        "body": "Suggest endpoint feels sluggish in the afternoon.",
        "status": "open",
        "priority": "P3",
        "service": "search",
    },
    {
        "id": "T-103",
        "title": "Password reset emails delayed",
        "body": "Resolved by restarting the mail worker.",
        "status": "closed",
        "priority": "P2",
        "service": "auth",
    },
]


@package.client("tickets")
def _ticket_store(ctx) -> dict[str, dict[str, Any]]:
    """Per-session ticket store, so writes don't leak between sessions."""
    return {t["id"]: copy.deepcopy(t) for t in SEED_TICKETS}


@package.tool(effect="read")
def search_tickets(ctx, text: str = "", status: str | None = None) -> list[dict]:
    """Tickets whose title or body contains `text`, optionally filtered by status
    (open/closed)."""
    store = ctx.client("tickets")
    needle = text.lower()
    return [
        dict(t)
        for t in store.values()
        if (needle in t["title"].lower() or needle in t["body"].lower())
        and (status is None or t["status"] == status)
    ]


@package.tool(effect="read")
def get_ticket(ctx, ticket_id: str) -> dict:
    """One ticket by id (e.g. "T-101")."""
    store = ctx.client("tickets")
    if ticket_id not in store:
        raise KeyError(f"no ticket {ticket_id}")
    return dict(store[ticket_id])


@package.tool(effect="write")
def create_ticket(ctx, title: str, body: str = "", priority: str = "P3", service: str = "") -> dict:
    """Open a new ticket and return it."""
    store = ctx.client("tickets")
    ticket_id = f"T-{101 + len(store)}"
    ticket = {
        "id": ticket_id,
        "title": title,
        "body": body,
        "status": "open",
        "priority": priority,
        "service": service,
    }
    store[ticket_id] = ticket
    return dict(ticket)


@package.tool(effect="write")
def close_ticket(ctx, ticket_id: str, resolution: str) -> dict:
    """Close a ticket with a resolution note."""
    ticket = ctx.client("tickets")[ticket_id]
    ticket["status"] = "closed"
    ticket["body"] += f"\n\nResolution: {resolution}"
    return dict(ticket)


@package.tool(effect="destructive")
def delete_ticket(ctx, ticket_id: str) -> bool:
    """Delete a ticket permanently."""
    return ctx.client("tickets").pop(ticket_id, None) is not None
