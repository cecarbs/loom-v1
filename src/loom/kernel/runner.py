"""loom kernel runner (docs/04 "Runner behavior").

STDLIB ONLY. Runs inside the sandbox on any Python >= 3.11 and never imports loom; it is
copied into the session's sandbox dir and started as `python -I runner.py`.

Protocol: JSON lines on the process's original stdin/stdout. At startup those are moved to
private fds; fd 0 becomes /dev/null and fds 1/2 become pipes drained by capture threads,
so print(), C extensions and subprocesses can never corrupt the protocol stream.
"""

from __future__ import annotations

import ast
import asyncio
import base64
import codecs
import datetime
import inspect
import itertools
import json
import linecache
import os
import platform
import queue
import reprlib
import signal
import sys
import threading
import time
import traceback
import types
from typing import Any

# Written to a captured fd after each execution; the capture thread signals when it has
# read up to it, so all output of a cell is sent before its execute_result.
SYNC_MARKER = b"\x1e\x1floom-sync\x1f\x1e"
SYNC_TIMEOUT_S = 2.0
SUMMARY_CHARS = 60


def compose_text(output: str, value_repr: str | None, error: str | None) -> str:
    """The result text of one execution: captured output, then the last expression's repr,
    then the traceback. Shared with the host (loom.tools.builtin.execute)."""
    text = output
    for extra in (value_repr, error):
        if extra:
            if text and not text.endswith("\n"):
                text += "\n"
            text += extra
    return text


# --- tool calls (docs/04 "Protocol", "Runner behavior") --------------------------------


class ToolError(Exception):
    """A kernel tool call failed on the host."""


class PermissionDenied(ToolError):
    """The approval policy or the user denied the call."""


class ToolNotFound(ToolError):
    pass


class InvalidArguments(ToolError):
    pass


class ToolTimeout(ToolError):
    pass


class MissingSecret(ToolError):
    pass


TOOL_ERRORS: dict[str, type[ToolError]] = {
    cls.__name__: cls
    for cls in (
        ToolError,
        PermissionDenied,
        ToolNotFound,
        InvalidArguments,
        ToolTimeout,
        MissingSecret,
    )
}
for _cls in TOOL_ERRORS.values():
    _cls.__module__ = "tools"  # tracebacks read `tools.PermissionDenied: …`


def _encode_arg(value: Any) -> Any:
    """json.dumps fallback for tool arguments."""
    if isinstance(value, (bytes, bytearray)):
        return {"__bytes__": base64.b64encode(bytes(value)).decode("ascii")}
    if isinstance(value, (datetime.datetime, datetime.date, datetime.time)):
        return value.isoformat()
    if isinstance(value, (set, frozenset)):
        return list(value)
    raise TypeError(f"{type(value).__name__} is not JSON-serializable")


def _decode_value(value: Any) -> Any:
    if isinstance(value, dict):
        if len(value) == 1 and "__bytes__" in value and isinstance(value["__bytes__"], str):
            return base64.b64decode(value["__bytes__"])
        return {k: _decode_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_decode_value(v) for v in value]
    return value


class _Annotation:
    """Shows a catalog annotation string unquoted in signatures and help()."""

    def __init__(self, text: str) -> None:
        self.text = text

    def __repr__(self) -> str:
        return self.text


class ToolCalls:
    """Synchronous stubs: send `tool_call`, block until the matching `tool_result`."""

    def __init__(self, kernel: Kernel) -> None:
        self.kernel = kernel
        self._pending: dict[str, tuple[threading.Event, dict[str, Any]]] = {}
        self._lock = threading.Lock()
        self._ids = itertools.count(1)

    def call(self, tool: str, args: dict[str, Any]) -> Any:
        try:
            json.dumps(args, default=_encode_arg)
        except (TypeError, ValueError) as e:
            raise TypeError(f"{tool}: arguments must be JSON-serializable ({e})") from None
        with self._lock:
            call_id = f"c{next(self._ids)}"
            done = threading.Event()
            slot: dict[str, Any] = {}
            self._pending[call_id] = (done, slot)
        try:
            self.kernel.channel.send(
                {
                    "type": "tool_call",
                    "id": self.kernel.current_id,
                    "call_id": call_id,
                    "tool": tool,
                    "args": json.loads(json.dumps(args, default=_encode_arg)),
                }
            )
            done.wait()  # SIGINT still raises KeyboardInterrupt here
        finally:
            with self._lock:
                self._pending.pop(call_id, None)  # a late result is then dropped
        msg = slot["msg"]
        if msg.get("ok"):
            return _decode_value(msg.get("value"))
        error = msg.get("error") or {}
        cls = TOOL_ERRORS.get(str(error.get("type")), ToolError)
        raise cls(str(error.get("message", "tool call failed")))

    def resolve(self, msg: dict[str, Any]) -> None:
        with self._lock:
            entry = self._pending.get(str(msg.get("call_id")))
        if entry is not None:
            done, slot = entry
            slot["msg"] = msg
            done.set()


def make_stub(calls: ToolCalls, entry: dict[str, Any]) -> Any:
    package, name = str(entry["package"]), str(entry["name"])
    qualname = f"{package}.{name}"
    params = []
    for p in entry.get("params", []):
        params.append(
            inspect.Parameter(
                str(p["name"]),
                getattr(inspect.Parameter, str(p.get("kind", "POSITIONAL_OR_KEYWORD"))),
                default=p["default"] if "default" in p else inspect.Parameter.empty,
                annotation=_Annotation(p["annotation"])
                if "annotation" in p
                else inspect.Parameter.empty,
            )
        )
    returns = entry.get("returns")
    signature = inspect.Signature(
        params,
        return_annotation=_Annotation(returns) if returns else inspect.Signature.empty,
    )

    def stub(*args: Any, **kwargs: Any) -> Any:
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        return calls.call(qualname, dict(bound.arguments))

    stub.__name__ = stub.__qualname__ = name
    stub.__module__ = package
    doc = str(entry.get("doc") or "")
    stub.__doc__ = f"{doc}\n\nEffect: {entry.get('effect', '?')} (runs on the host via loom)."
    stub.__signature__ = signature  # type: ignore[attr-defined]
    return stub


class _Repr(reprlib.Repr):
    """reprlib, but dicts keep insertion order (reprlib sorts keys; print() doesn't)."""

    def repr_dict(self, x: dict[Any, Any], level: int) -> str:
        if not x:
            return "{}"
        if level <= 0:
            return "{...}"
        pieces = [
            f"{self.repr1(k, level - 1)}: {self.repr1(x[k], level - 1)}"
            for k in itertools.islice(x, self.maxdict)
        ]
        if len(x) > self.maxdict:
            pieces.append("...")
        return "{" + ", ".join(pieces) + "}"


class Channel:
    def __init__(self, fd: int) -> None:
        self._file = os.fdopen(fd, "wb")
        self._lock = threading.Lock()

    def send(self, msg: dict[str, Any]) -> None:
        data = (json.dumps(msg, default=repr) + "\n").encode("utf-8")
        with self._lock:
            try:
                self._file.write(data)
                self._file.flush()
            except (BrokenPipeError, ValueError):  # host is gone
                os._exit(0)


class Capture(threading.Thread):
    """Drains one redirected fd (1 or 2) and forwards it as `stream` messages."""

    def __init__(self, fd: int, name: str, kernel: Kernel) -> None:
        super().__init__(name=f"loom-capture-{name}", daemon=True)
        self.read_fd = fd
        self.stream_name = name
        self.kernel = kernel
        self.synced = threading.Event()
        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")

    def run(self) -> None:
        pending = b""
        while True:
            try:
                chunk = os.read(self.read_fd, 65536)
            except OSError:
                return
            if not chunk:
                return
            data = pending + chunk
            while (i := data.find(SYNC_MARKER)) >= 0:
                self._emit(data[:i])
                data = data[i + len(SYNC_MARKER) :]
                self.synced.set()
            keep = _partial_marker(data)
            self._emit(data[: len(data) - keep])
            pending = data[len(data) - keep :]

    def _emit(self, data: bytes) -> None:
        text = self._decoder.decode(data)
        if text:
            self.kernel.on_output(self.stream_name, text)


def _partial_marker(data: bytes) -> int:
    """Length of the longest suffix of `data` that could start a sync marker."""
    for n in range(min(len(data), len(SYNC_MARKER) - 1), 0, -1):
        if data.endswith(SYNC_MARKER[:n]):
            return n
    return 0


class Kernel:
    def __init__(self, channel: Channel) -> None:
        self.channel = channel
        self.module = types.ModuleType("__loom__")
        self.ns: dict[str, Any] = self.module.__dict__
        self.ns["_out"] = {}
        sys.modules["__loom__"] = self.module  # so classes defined in cells can be pickled
        self.reserved: set[str] = set(self.ns)
        self.repr_max = 2000
        self.out_max_chars = 12000
        self.executing = False
        self.current_id: str | None = None
        self.captures: list[Capture] = []
        self.loop = asyncio.new_event_loop()
        self.calls = ToolCalls(self)
        self._output: list[str] = []
        self._output_lock = threading.Lock()
        self._repr = _Repr()

    # --- setup -----------------------------------------------------------------------

    def init(self, msg: dict[str, Any]) -> None:
        settings = msg.get("settings") or {}
        self.repr_max = int(settings.get("repr_max", self.repr_max))
        self.out_max_chars = int(settings.get("out_max_chars", self.out_max_chars))
        r = self._repr
        r.maxlevel = 4
        r.maxlist = r.maxtuple = r.maxset = r.maxfrozenset = r.maxdeque = 50
        r.maxarray = r.maxdict = 50
        r.maxstring = r.maxlong = r.maxother = self.repr_max
        self._install_tools(msg.get("catalog") or [])
        self.reserved = set(self.ns)
        self.channel.send(
            {"type": "ready", "python": platform.python_version(), "pid": os.getpid()}
        )

    def _install_tools(self, catalog: list[dict[str, Any]]) -> None:
        """One module per package (`splunk.search(...)`), also reachable as `tools.splunk`."""
        tools = types.SimpleNamespace(**TOOL_ERRORS)
        for package in catalog:
            name = str(package.get("name"))
            module = types.ModuleType(name, str(package.get("description") or ""))
            for entry in package.get("tools", []):
                setattr(module, str(entry["name"]), make_stub(self.calls, entry))
            setattr(tools, name, module)
            self.ns[name] = module
            sys.modules.setdefault(name, module)  # `import splunk`, unless it shadows a module
        self.ns["tools"] = tools
        self.ns["ToolError"] = ToolError
        self.ns["PermissionDenied"] = PermissionDenied

    # --- output ----------------------------------------------------------------------

    def on_output(self, name: str, text: str) -> None:
        with self._output_lock:
            exec_id = self.current_id
            if exec_id is not None:
                self._output.append(text)
        self.channel.send({"type": "stream", "id": exec_id, "name": name, "text": text})

    def _sync_output(self) -> None:
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()
            except Exception:
                pass
        for capture in self.captures:
            capture.synced.clear()
            try:
                os.write(1 if capture.stream_name == "stdout" else 2, SYNC_MARKER)
            except OSError:
                continue
            capture.synced.wait(SYNC_TIMEOUT_S)

    # --- execution -------------------------------------------------------------------

    def execute(self, msg: dict[str, Any]) -> None:
        exec_id = str(msg.get("id"))
        code = str(msg.get("code", ""))
        with self._output_lock:
            self.current_id = exec_id
            self._output = []
        start = time.monotonic()
        value: Any = None
        has_value = False
        error: BaseException | None = None
        cell = f"<cell {exec_id}>"
        linecache.cache[cell] = (len(code), None, code.splitlines(True), cell)
        self.executing = True
        try:
            has_value, value = self._run_cell(code, cell)
        except BaseException as e:  # KeyboardInterrupt and SystemExit included
            error = e
        finally:
            self.executing = False

        value_repr = None
        if has_value and value is not None:
            self.ns["_"] = value
            value_repr = self.limited_repr(value)
        self._sync_output()
        with self._output_lock:
            output = "".join(self._output)
            self.current_id = None
            self._output = []

        result: dict[str, Any] = {
            "type": "execute_result",
            "id": exec_id,
            "duration_ms": int((time.monotonic() - start) * 1000),
        }
        error_text = None
        if error is None:
            result["status"] = "ok"
            if value_repr is not None:
                result["value_repr"] = value_repr
        else:
            error_text = format_error(error)
            result["status"] = "error"
            result["error"] = {
                "ename": type(error).__name__,
                "evalue": str(error),
                "traceback": error_text,
            }
        full = compose_text(output, value_repr, error_text)
        number = _exec_number(exec_id)
        if len(full) > self.out_max_chars and number is not None:
            self.ns["_out"][number] = full
            result["out_ref"] = number
        result["vars"] = len(self.variable_names())
        self.channel.send(result)

    def _run_cell(self, code: str, cell: str) -> tuple[bool, Any]:
        tree = ast.parse(code, filename=cell, mode="exec")
        last: ast.Expression | None = None
        tail = tree.body[-1] if tree.body else None
        if isinstance(tail, ast.Expr):
            tree.body.pop()
            last = ast.Expression(tail.value)
        flags = ast.PyCF_ALLOW_TOP_LEVEL_AWAIT
        self._run(compile(tree, cell, "exec", flags=flags, dont_inherit=True))
        if last is None:
            return False, None
        return True, self._run(compile(last, cell, "eval", flags=flags, dont_inherit=True))

    def _run(self, code: types.CodeType) -> Any:
        result = eval(code, self.ns)
        if code.co_flags & inspect.CO_COROUTINE:  # top-level await
            asyncio.set_event_loop(self.loop)
            result = self.loop.run_until_complete(result)
        return result

    def limited_repr(self, value: Any) -> str:
        try:
            text = self._repr.repr(value)
        except Exception as e:
            text = f"<repr failed: {type(e).__name__}: {e}>"
        if len(text) > self.repr_max:
            text = text[: self.repr_max] + f"… [{len(text) - self.repr_max:,} chars cut]"
        return text

    # --- inventory -------------------------------------------------------------------

    def variable_names(self) -> list[str]:
        for _ in range(5):  # the main thread may be mutating the namespace
            try:
                items = list(self.ns.items())
                break
            except RuntimeError:
                continue
        else:
            return []
        return [
            name
            for name, value in items
            if not name.startswith("_")
            and name not in self.reserved
            and not isinstance(value, types.ModuleType)
        ]

    def inventory(self, msg: dict[str, Any]) -> None:
        out = []
        for name in self.variable_names():
            value = self.ns.get(name)
            out.append({"name": name, "type": type(value).__name__, "summary": summarize(value)})
        self.channel.send({"type": "inventory_result", "id": msg.get("id"), "vars": out})


def summarize(value: Any) -> str:
    try:
        shape = getattr(value, "shape", None)
        if isinstance(shape, tuple) and all(isinstance(n, int) for n in shape):
            return "x".join(str(n) for n in shape)
        if isinstance(value, (bool, int, float, complex, type(None))):
            text = repr(value)
        elif isinstance(value, (types.FunctionType, type)):
            return ""
        elif hasattr(value, "__len__"):
            return f"len={len(value)}"
        else:
            text = repr(value)
    except Exception:
        return ""
    return text if len(text) <= SUMMARY_CHARS else text[: SUMMARY_CHARS - 1] + "…"


def format_error(error: BaseException) -> str:
    """Traceback from the first cell frame on, without runner frames (tool stubs): they
    are noise for the model."""
    tb = error.__traceback__
    while tb is not None and not tb.tb_frame.f_code.co_filename.startswith("<cell "):
        tb = tb.tb_next
    if isinstance(error, SyntaxError) or tb is None:
        return "".join(traceback.format_exception_only(type(error), error)).rstrip("\n")
    exc = traceback.TracebackException(type(error), error, tb)
    exc.stack = traceback.StackSummary.from_list(
        [frame for frame in exc.stack if frame.filename != __file__]
    )
    return "".join(exc.format()).rstrip("\n")


def _exec_number(exec_id: str) -> int | None:
    digits = exec_id[1:]
    return int(digits) if exec_id.startswith("e") and digits.isdigit() else None


def _take_over_stdio() -> tuple[int, int]:
    """Move the protocol to private fds and redirect fds 0/1/2. Returns (in_fd, out_fd)."""
    proto_in = os.dup(0)  # os.dup fds are non-inheritable: subprocesses never get them
    proto_out = os.dup(1)
    devnull = os.open(os.devnull, os.O_RDONLY)
    os.dup2(devnull, 0)
    os.close(devnull)
    sys.stdin = open(os.devnull, encoding="utf-8")
    return proto_in, proto_out


def _capture_fd(fd: int, name: str, kernel: Kernel) -> None:
    read_fd, write_fd = os.pipe()
    os.dup2(write_fd, fd)
    os.close(write_fd)
    stream = open(fd, "w", encoding="utf-8", errors="backslashreplace", buffering=1, closefd=False)
    setattr(sys, name, stream)
    capture = Capture(read_fd, name, kernel)
    kernel.captures.append(capture)
    capture.start()


def _read_protocol(fd: int, kernel: Kernel, work: queue.Queue[dict[str, Any]]) -> None:
    with os.fdopen(fd, "rb") as f:
        for line in f:
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            kind = msg.get("type")
            if kind == "inventory":
                kernel.inventory(msg)  # answered even while a cell runs
            elif kind == "tool_result":
                kernel.calls.resolve(msg)  # unknown or abandoned call ids are dropped
            elif kind in ("init", "execute", "shutdown"):
                work.put(msg)
    # EOF: the host is gone. Exit even if a cell is still running or a stub is waiting
    # for a tool result that will never come; nobody can read our output any more.
    os._exit(0)


def main() -> None:
    proto_in, proto_out = _take_over_stdio()
    kernel = Kernel(Channel(proto_out))
    _capture_fd(1, "stdout", kernel)
    _capture_fd(2, "stderr", kernel)
    sys.path.insert(0, os.getcwd())  # `import mymodule` from the workspace, like Jupyter

    def on_sigint(signum: int, frame: Any) -> None:
        if kernel.executing:
            raise KeyboardInterrupt

    signal.signal(signal.SIGINT, on_sigint)

    work: queue.Queue[dict[str, Any]] = queue.Queue()
    threading.Thread(
        target=_read_protocol, args=(proto_in, kernel, work), name="loom-protocol", daemon=True
    ).start()
    while True:
        msg = work.get()
        kind = msg.get("type")
        if kind == "shutdown":
            break
        if kind == "init":
            kernel.init(msg)
        elif kind == "execute":
            kernel.execute(msg)
    os._exit(0)  # don't wait for threads started by model code


if __name__ == "__main__":
    main()
