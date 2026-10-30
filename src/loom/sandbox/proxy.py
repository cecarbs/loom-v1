"""The network allowlist proxy (docs/05 "Network allowlist").

An HTTP proxy that only connects to allowlisted hosts: `CONNECT host:port` for HTTPS and
absolute-form requests (`GET http://host/path`) for plain HTTP. Anything else is refused.

Stdlib-only and runs on any Python >= 3.11: seatbelt runs it inside the loom process, docker
runs it in a sidecar container with `python -I -c <this file's source> --listen … --allow …`.
It never imports `loom`. Every connection attempt is reported as an event (one JSON line on
stdout when run standalone) so it ends up in the session log.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import sys
from collections.abc import Callable, Iterable
from typing import Any

HEAD_LIMIT = 64 * 1024
CONNECT_TIMEOUT_S = 10.0
CHUNK = 64 * 1024
DROP_HEADERS = (b"proxy-connection:", b"proxy-authorization:")

EventCallback = Callable[[dict[str, Any]], None]


def parse_entry(entry: str) -> tuple[str, int | None]:
    """`host`, `*.domain` or either with `:port`. Raises ValueError for anything else."""
    text = entry.strip().lower().rstrip(".")
    if not text or "/" in text or "@" in text or " " in text:
        raise ValueError(f"not a host: {entry!r}")
    host, port = text, None
    if text.startswith("["):  # [::1]:443
        end = text.find("]")
        if end < 0:
            raise ValueError(f"not a host: {entry!r}")
        host, rest = text[1:end], text[end + 1 :]
        if rest:
            if not rest.startswith(":"):
                raise ValueError(f"not a host: {entry!r}")
            port = _port(rest[1:], entry)
    elif text.count(":") == 1:
        host, _, port_text = text.partition(":")
        port = _port(port_text, entry)
    if host.startswith("*.") and ("*" in host[2:] or not host[2:]):
        raise ValueError(f"only a leading '*.' wildcard is supported: {entry!r}")
    if "*" in host and not host.startswith("*."):
        raise ValueError(f"only a leading '*.' wildcard is supported: {entry!r}")
    return host, port


def _port(text: str, entry: str) -> int:
    if not text.isdigit() or not 0 < int(text) < 65536:
        raise ValueError(f"invalid port in {entry!r}")
    return int(text)


class Allowlist:
    def __init__(self, entries: Iterable[str]) -> None:
        self.entries = [parse_entry(e) for e in entries]

    def allows(self, host: str, port: int) -> bool:
        host = host.lower().rstrip(".")
        for pattern, allowed_port in self.entries:
            if allowed_port is not None and allowed_port != port:
                continue
            if pattern.startswith("*."):
                if host.endswith(pattern[1:]) and len(host) > len(pattern) - 1:
                    return True
            elif host == pattern:
                return True
        return False


class Proxy:
    def __init__(self, allowlist: Allowlist, on_event: EventCallback | None = None) -> None:
        self.allowlist = allowlist
        self.on_event = on_event
        self._server: asyncio.Server | None = None
        self._tasks: set[asyncio.Task[None]] = set()

    async def start(self, host: str = "127.0.0.1", port: int = 0) -> int:
        """Listen; returns the bound port."""
        self._server = await asyncio.start_server(self._client, host, port)
        return self._server.sockets[0].getsockname()[1]

    async def close(self) -> None:
        if self._server is not None:
            self._server.close()
            for task in list(self._tasks):
                task.cancel()
            await asyncio.gather(*self._tasks, return_exceptions=True)
            with contextlib.suppress(Exception):
                await self._server.wait_closed()
            self._server = None

    async def serve_forever(self) -> None:
        assert self._server is not None
        await self._server.serve_forever()

    async def _client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task = asyncio.current_task()
        if task is not None:
            self._tasks.add(task)
        try:
            await self._handle(reader, writer)
        except (OSError, asyncio.IncompleteReadError, asyncio.LimitOverrunError, ValueError):
            pass
        finally:
            if task is not None:
                self._tasks.discard(task)
            writer.close()

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        head = await reader.readuntil(b"\r\n\r\n")
        if len(head) > HEAD_LIMIT:
            await _reply(writer, 431, "request head too large")
            return
        request_line, _, rest = head.partition(b"\r\n")
        try:
            method, target, version = request_line.decode("latin-1").split(" ")
        except ValueError:
            await _reply(writer, 400, "bad request line")
            return
        if method == "CONNECT":
            host, port = _split_host(target, default_port=443)
            forward = b""
        elif target.lower().startswith("http://"):
            authority, slash, path = target[7:].partition("/")
            host, port = _split_host(authority, default_port=80)
            line = f"{method} {slash}{path or ''} {version}\r\n".encode("latin-1")
            headers = b"".join(
                h + b"\r\n"
                for h in rest.split(b"\r\n")
                if h and not h.lower().startswith(DROP_HEADERS)
            )
            forward = line + headers + b"\r\n"
        else:
            await _reply(writer, 400, "loom proxy: only CONNECT and absolute http:// URLs")
            return
        allowed = bool(host) and self.allowlist.allows(host, port)
        self._event({"method": method, "host": host, "port": port, "allowed": allowed})
        if not allowed:
            await _reply(writer, 403, f"loom: {host}:{port} is not in the sandbox allowlist")
            return
        try:
            up_reader, up_writer = await asyncio.wait_for(
                asyncio.open_connection(host, port, limit=CHUNK), CONNECT_TIMEOUT_S
            )
        except (OSError, TimeoutError) as e:
            self._event({"method": method, "host": host, "port": port, "error": str(e)})
            await _reply(writer, 502, f"loom proxy: cannot connect to {host}:{port}")
            return
        try:
            if method == "CONNECT":
                writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
                await writer.drain()
            else:
                up_writer.write(forward)
                await up_writer.drain()
            await _pipe_both(reader, writer, up_reader, up_writer)
        finally:
            up_writer.close()

    def _event(self, event: dict[str, Any]) -> None:
        if self.on_event is not None:
            with contextlib.suppress(Exception):
                self.on_event(event)


def _split_host(authority: str, *, default_port: int) -> tuple[str, int]:
    authority = authority.rsplit("@", 1)[-1]  # userinfo is never a host
    if authority.startswith("["):
        host, _, rest = authority[1:].partition("]")
        port_text = rest[1:] if rest.startswith(":") else ""
    elif authority.count(":") == 1:
        host, _, port_text = authority.partition(":")
    else:
        host, port_text = authority, ""
    port = int(port_text) if port_text.isdigit() else default_port
    return host.lower().rstrip("."), port


async def _reply(writer: asyncio.StreamWriter, status: int, text: str) -> None:
    reasons = {400: "Bad Request", 403: "Forbidden", 431: "Too Large", 502: "Bad Gateway"}
    body = (text + "\n").encode()
    writer.write(
        f"HTTP/1.1 {status} {reasons.get(status, 'Error')}\r\n"
        f"Content-Type: text/plain\r\nContent-Length: {len(body)}\r\n"
        "Connection: close\r\n\r\n".encode()
        + body
    )
    with contextlib.suppress(OSError):
        await writer.drain()


async def _pipe_both(
    a_reader: asyncio.StreamReader,
    a_writer: asyncio.StreamWriter,
    b_reader: asyncio.StreamReader,
    b_writer: asyncio.StreamWriter,
) -> None:
    tasks = [
        asyncio.create_task(_pipe(a_reader, b_writer)),
        asyncio.create_task(_pipe(b_reader, a_writer)),
    ]
    try:
        await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            task.cancel()


async def _pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while data := await reader.read(CHUNK):
            writer.write(data)
            await writer.drain()
        if writer.can_write_eof():
            writer.write_eof()
    except OSError:
        writer.close()


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="loom-proxy")
    parser.add_argument("--listen", default="0.0.0.0:3128")
    parser.add_argument("--allow", default="[]", help="JSON list of allowlist entries")
    args = parser.parse_args(argv)
    host, _, port = args.listen.rpartition(":")

    def emit(event: dict[str, Any]) -> None:
        sys.stdout.write(json.dumps(event) + "\n")
        sys.stdout.flush()

    async def run() -> None:
        proxy = Proxy(Allowlist(json.loads(args.allow)), emit)
        await proxy.start(host, int(port))
        emit({"ready": True})
        await proxy.serve_forever()

    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(run())


if __name__ == "__main__":
    main(sys.argv[1:])
