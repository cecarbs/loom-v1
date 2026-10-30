"""M9 acceptance: `network = "allowlist"` (docs/05). With one allowed host, only that host is
reachable from `bash`, on both sandboxing backends. No real network: the "internet" is a
local HTTP server, reached by two names of which only one is allowlisted."""

import ast
import asyncio
import http.server
import sys
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest
from pydantic import ValidationError
from test_sandboxes import IMAGE, Sandboxed, World, _image_exists, world  # noqa: F401

from loom.config import SandboxConfig
from loom.sandbox import proxy as proxy_module
from loom.sandbox.proxy import Allowlist, Proxy, parse_entry


class _Hello(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        body = b"hello from upstream\n"
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args) -> None:
        pass


@pytest.fixture
def upstream() -> Iterator[int]:
    """A local web server; yields its port."""
    server = http.server.ThreadingHTTPServer(("0.0.0.0", 0), _Hello)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server.server_address[1]
    server.shutdown()


# --- allowlist and proxy -------------------------------------------------------------


def test_proxy_imports_only_stdlib() -> None:
    """The docker sidecar runs this file with the image's Python (`python -I -c`)."""
    assert proxy_module.__file__ is not None
    tree = ast.parse(Path(proxy_module.__file__).read_text())
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0
            modules.add((node.module or "").split(".")[0])
    modules.discard("__future__")
    assert modules <= sys.stdlib_module_names, modules - sys.stdlib_module_names


def test_allowlist_entries() -> None:
    assert parse_entry("PyPI.org.") == ("pypi.org", None)
    assert parse_entry("api.github.com:443") == ("api.github.com", 443)
    assert parse_entry("[::1]:8080") == ("::1", 8080)
    for bad in ("https://pypi.org", "pypi.org/simple", "a.*.com", "*.", "x:0", "x:http", ""):
        with pytest.raises(ValueError):
            parse_entry(bad)
    allow = Allowlist(["pypi.org", "*.github.com", "internal:8443"])
    assert allow.allows("pypi.org", 443) and allow.allows("PYPI.ORG.", 80)
    assert allow.allows("api.github.com", 443) and not allow.allows("github.com", 443)
    assert not allow.allows("evilgithub.com", 443)
    assert allow.allows("internal", 8443) and not allow.allows("internal", 443)
    assert not allow.allows("files.pythonhosted.org", 443)


def test_config_rejects_urls_in_the_allowlist() -> None:
    with pytest.raises(ValidationError, match="not a host"):
        SandboxConfig.model_validate({"network": "allowlist", "allowlist": ["https://x.org"]})


async def _http(proxy_port: int, request: bytes) -> bytes:
    reader, writer = await asyncio.open_connection("127.0.0.1", proxy_port)
    writer.write(request)
    await writer.drain()
    data = await asyncio.wait_for(reader.read(), 5)
    writer.close()
    return data


async def test_proxy_forwards_only_allowlisted_hosts(upstream: int) -> None:
    events: list[dict] = []
    proxy = Proxy(Allowlist([f"localhost:{upstream}"]), events.append)
    port = await proxy.start()
    try:
        ok = await _http(
            port, f"GET http://localhost:{upstream}/x HTTP/1.0\r\nHost: localhost\r\n\r\n".encode()
        )
        assert ok.startswith(b"HTTP/1.0 200") and ok.endswith(b"hello from upstream\n")
        denied = await _http(port, f"GET http://127.0.0.1:{upstream}/ HTTP/1.0\r\n\r\n".encode())
        assert denied.startswith(b"HTTP/1.1 403") and b"not in the sandbox allowlist" in denied
        tunnel = await _http(port, f"CONNECT example.com:{upstream} HTTP/1.1\r\n\r\n".encode())
        assert tunnel.startswith(b"HTTP/1.1 403")
        assert (await _http(port, b"GET /relative HTTP/1.1\r\n\r\n")).startswith(b"HTTP/1.1 400")
    finally:
        await proxy.close()
    assert [(e["host"], e["allowed"]) for e in events] == [
        ("localhost", True),
        ("127.0.0.1", False),
        ("example.com", False),
    ]


async def test_proxy_connect_tunnels_bytes(upstream: int) -> None:
    proxy = Proxy(Allowlist([f"localhost:{upstream}"]))
    port = await proxy.start()
    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(f"CONNECT localhost:{upstream} HTTP/1.1\r\n\r\n".encode())
        assert (await reader.readuntil(b"\r\n\r\n")).startswith(b"HTTP/1.1 200")
        writer.write(b"GET / HTTP/1.0\r\n\r\n")  # what TLS would carry, in the clear
        await writer.drain()
        assert (await asyncio.wait_for(reader.read(), 5)).endswith(b"hello from upstream\n")
        writer.close()
    finally:
        await proxy.close()


# --- real sandboxes ------------------------------------------------------------------


def _curl(url: str, *extra: str) -> str:
    flags = " ".join(extra)
    return f"curl -sS -m 5 {flags} -o /dev/null -w '%{{http_code}}' {url} 2>&1; echo"


@pytest.mark.macos
async def test_seatbelt_allowlist_reaches_only_that_host(world: World, upstream: int) -> None:  # noqa: F811
    import sys

    from loom.sandbox.seatbelt import SeatbeltBackend

    backend = SeatbeltBackend(
        loom_home=world.loom_home,
        home=world.home,
        python=sys.executable,
        allowlist=[f"localhost:{upstream}"],
    )
    box = Sandboxed(world, backend)
    await box.start()
    try:
        assert backend.describe() == "seatbelt · net: 1 hosts"
        allowed = await box.bash(_curl(f"http://localhost:{upstream}/"))
        assert allowed.startswith("200"), allowed
        denied = await box.bash(_curl(f"http://127.0.0.1:{upstream}/"))
        assert "403" in denied, denied
        # Around the proxy: the profile blocks every other connection.
        direct = await box.bash(_curl(f"http://127.0.0.1:{upstream}/", "--noproxy '*'"))
        assert not direct.startswith("200") and "000" in direct, direct
        from_python = await box.py(
            "import urllib.request\n"
            f"urllib.request.urlopen('http://localhost:{upstream}/').read().decode()"
        )
        assert "hello from upstream" in from_python
    finally:
        await box.close()
    network = [r for r in box.agent.log.read() if r["kind"] == "network"]
    assert [(r["host"], r["allowed"]) for r in network] == [
        ("localhost", True),
        ("127.0.0.1", False),
        ("localhost", True),
    ]


@pytest.mark.docker
async def test_docker_allowlist_reaches_only_that_host(world: World, upstream: int) -> None:  # noqa: F811
    import subprocess

    from loom.sandbox.docker import DockerBackend

    if not _image_exists():
        pytest.skip(f"{IMAGE} missing; run `loom sandbox setup --docker`")
    host = "host.docker.internal"  # the test server, as seen from the proxy sidecar
    cfg = SandboxConfig.model_validate(
        {"network": "allowlist", "allowlist": [f"{host}:{upstream}"]}
    )
    backend = DockerBackend(cfg, loom_home=world.loom_home)
    box = Sandboxed(world, backend)
    await box.start()
    try:
        allowed = await box.bash(_curl(f"http://{host}:{upstream}/"))
        assert allowed.startswith("200"), allowed
        denied = await box.bash(_curl("https://example.com/"))
        assert "403" in denied or "CONNECT" in denied, denied
        direct = await box.bash(_curl(f"http://{host}:{upstream}/", "--noproxy '*'"))
        assert not direct.startswith("200"), direct  # internal network: no route, no DNS
        names = (backend.container, backend.proxy, backend.network)
        assert None not in names
    finally:
        await box.close()
    container, sidecar, network_name = (str(n) for n in names)
    for name in (container, sidecar):
        assert subprocess.run(["docker", "inspect", name], capture_output=True).returncode != 0
    gone = subprocess.run(["docker", "network", "inspect", network_name], capture_output=True)
    assert gone.returncode != 0
    network = [r for r in box.agent.log.read() if r["kind"] == "network"]
    assert [(r["host"], r["allowed"]) for r in network] == [(host, True), ("example.com", False)]


def test_prompt_names_the_allowlist_only_in_allowlist_mode(tmp_path, monkeypatch) -> None:
    from loom.app import App
    from loom.config import LoomConfig
    from loom.providers.fake import FakeProvider
    from loom.sandbox.docker import DockerBackend

    monkeypatch.setattr(
        "loom.app.create_backend", lambda cfg, **kw: DockerBackend(cfg, loom_home=tmp_path)
    )
    (tmp_path / "ws").mkdir()

    def system(**sandbox) -> str:
        app = App(
            tmp_path / "ws",
            LoomConfig.model_validate({"sandbox": sandbox}),
            provider_factory=lambda p: FakeProvider([]),
            home=tmp_path / "home",
            packages=[],
        )
        return app.agent.context.system

    allowlisted = system(network="allowlist", allowlist=["pypi.org", "*.github.com"])
    assert "# Network\n`bash` and `execute` can reach only these hosts" in allowlisted
    assert "pypi.org, *.github.com" in allowlisted
    assert "# Network" not in system()
