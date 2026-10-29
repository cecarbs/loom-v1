"""loom-tools-splunk against a fake splunklib service (no network)."""

import io
import json
from types import SimpleNamespace

import pytest
from loom_tools_splunk import normalize_spl
from loom_tools_splunk import package as splunk

from loom.broker.broker import PackageContext
from loom.broker.secrets import SecretStore


class FakeService:
    def __init__(self, rows: list[dict], messages: list[dict] | None = None) -> None:
        self.rows = rows
        self.messages = messages or []
        self.queries: list[tuple[str, dict]] = []
        self.created: list[tuple[str, str]] = []
        self.jobs = SimpleNamespace(oneshot=self._oneshot)
        self.indexes = [SimpleNamespace(name=n) for n in ("web", "app", "_internal")]
        self.saved_searches = SimpleNamespace(create=self._create)

    def _oneshot(self, query: str, **kwargs):
        self.queries.append((query, kwargs))
        body = {"preview": False, "messages": self.messages, "results": self.rows}
        return io.BytesIO(json.dumps(body).encode())

    def _create(self, name: str, query: str):
        self.created.append((name, query))
        return SimpleNamespace(name=name)


def ctx(service: FakeService, **config) -> PackageContext:
    context = PackageContext(splunk, config, SecretStore({}))
    context._clients["splunk"] = service  # skip the real connect
    return context


def call(tool: str, context: PackageContext, /, **kwargs):
    return splunk.tools[tool].func(context, **kwargs)


def test_search_returns_rows_and_passes_time_bounds() -> None:
    service = FakeService([{"host": "web-3", "count": "4127"}, {"host": "web-1", "count": "12"}])
    rows = call("search", ctx(service), spl="index=web status>=500 | stats count by host",
                earliest="-6h", max_rows=1)  # fmt: skip
    assert rows == [{"host": "web-3", "count": "4127"}]
    query, kwargs = service.queries[0]
    assert query == "search index=web status>=500 | stats count by host"
    assert kwargs == {
        "earliest_time": "-6h",
        "latest_time": "now",
        "output_mode": "json",
        "count": 1,
    }


def test_search_raises_on_splunk_errors() -> None:
    service = FakeService([], messages=[{"type": "FATAL", "text": "Unknown search command"}])
    with pytest.raises(RuntimeError, match="Unknown search command"):
        call("search", ctx(service), spl="index=web | bogus")


def test_allowed_indexes() -> None:
    service = FakeService([])
    allowed = ctx(service, allowed_indexes=["web", "app"])
    call("search", allowed, spl='index="web" OR index=app')
    with pytest.raises(PermissionError, match="Index not allowed: _internal"):
        call("search", allowed, spl="index=web OR index=_internal")
    with pytest.raises(PermissionError, match="Name an index"):
        call("search", allowed, spl="error")
    assert call("list_indexes", allowed) == ["app", "web"]
    assert call("list_indexes", ctx(service)) == ["_internal", "app", "web"]


def test_create_saved_search_and_effects() -> None:
    service = FakeService([])
    result = call("create_saved_search", ctx(service), name="5xx", spl="index=web status>=500")
    assert result == {"name": "5xx", "search": "search index=web status>=500"}
    assert service.created == [("5xx", "search index=web status>=500")]
    effects = {name: spec.effect for name, spec in splunk.tools.items()}
    assert effects == {"search": "read", "list_indexes": "read", "create_saved_search": "write"}


def test_normalize_spl() -> None:
    assert normalize_spl("  | tstats count where index=web ") == "| tstats count where index=web"
    assert normalize_spl("Search index=web") == "Search index=web"
    assert normalize_spl("index=web") == "search index=web"


def test_connect_needs_host_and_token() -> None:
    no_host = PackageContext(splunk, {}, SecretStore({}))
    with pytest.raises(ValueError, match="host"):
        no_host.client("splunk")
