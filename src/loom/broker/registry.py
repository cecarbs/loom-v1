"""Tool packages (docs/04 "Tool packages"): `ToolPackage`, `@package.tool(effect=...)`,
catalog entries for the kernel's stubs, and entry-point discovery (group `loom.tools`)."""

from __future__ import annotations

import inspect
import json
import keyword
import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from typing import Any, Literal, TypeVar, get_args

from pydantic import BaseModel, ConfigDict, create_model

log = logging.getLogger(__name__)

Effect = Literal["read", "write", "destructive"]
ENTRY_POINT_GROUP = "loom.tools"
RESERVED_NAMES = frozenset({"tools", "agents", "loom"})  # kernel namespace objects

F = TypeVar("F", bound=Callable[..., Any])
ClientFactory = Callable[[Any], Any]  # (PackageContext) -> client


class PackageError(Exception):
    """A package or tool definition loom can't use."""


@dataclass(frozen=True)
class Secret:
    """A secret a package needs, by its `loom secrets` name."""

    name: str
    description: str = ""


@dataclass(frozen=True)
class ToolSpec:
    package: str
    name: str
    func: Callable[..., Any]
    effect: Effect
    signature: inspect.Signature  # without ctx
    doc: str
    args_model: type[BaseModel]

    @property
    def qualname(self) -> str:
        return f"{self.package}.{self.name}"

    @property
    def is_async(self) -> bool:
        return inspect.iscoroutinefunction(self.func)

    def display(self) -> str:
        """`splunk.search(spl: str, earliest: str = '-1h') -> list[dict]   [read]`"""
        return f"{self.qualname}{_format_signature(self.signature)}   [{self.effect}]"

    def catalog_entry(self) -> dict[str, Any]:
        """What the runner needs to build a stub with a real signature and docstring."""
        params = []
        for p in self.signature.parameters.values():
            entry: dict[str, Any] = {"name": p.name, "kind": p.kind.name}
            if p.annotation is not inspect.Parameter.empty:
                entry["annotation"] = _annotation_text(p.annotation)
            if p.default is not inspect.Parameter.empty:
                entry["default"] = p.default
            params.append(entry)
        entry = {"package": self.package, "name": self.name, "doc": self.doc, "params": params}
        entry["effect"] = self.effect
        if self.signature.return_annotation is not inspect.Signature.empty:
            entry["returns"] = _annotation_text(self.signature.return_annotation)
        return entry


@dataclass
class ToolPackage:
    name: str
    description: str
    secrets: dict[str, Secret] = field(default_factory=dict)
    config_keys: list[str] = field(default_factory=list)
    tools: dict[str, ToolSpec] = field(default_factory=dict)
    clients: dict[str, ClientFactory] = field(default_factory=dict)

    def __post_init__(self) -> None:
        name = self.name
        if not name.isidentifier() or name.startswith("_") or keyword.iskeyword(name):
            raise PackageError(f"package name {name!r} must be a public Python identifier")
        if name in RESERVED_NAMES:
            raise PackageError(f"package name {name!r} is reserved")

    def tool(self, *, effect: Effect) -> Callable[[F], F]:
        """Register a tool function. Its first parameter receives the broker's context."""
        if effect not in get_args(Effect):
            raise PackageError(f"effect must be one of {get_args(Effect)}, not {effect!r}")

        def register(func: F) -> F:
            spec = make_spec(self.name, func, effect)
            if spec.name in self.tools:
                raise PackageError(f"{spec.qualname} is defined twice")
            self.tools[spec.name] = spec
            return func

        return register

    def catalog_entry(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "tools": [spec.catalog_entry() for spec in self.tools.values()],
        }

    def client(self, name: str) -> Callable[[ClientFactory], ClientFactory]:
        """Register a factory for `ctx.client(name)` (built once per session, then cached)."""

        def register(factory: ClientFactory) -> ClientFactory:
            self.clients[name] = factory
            return factory

        return register


def make_spec(package: str, func: Callable[..., Any], effect: Effect) -> ToolSpec:
    name = func.__name__
    where = f"{package}.{name}"
    if name.startswith("_"):
        raise PackageError(f"{where}: tool names must not start with '_'")
    params = list(inspect.signature(func).parameters.values())
    if not params or params[0].kind not in (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    ):
        raise PackageError(f"{where}: the first parameter must be the context (ctx)")
    params = params[1:]
    hints = _resolved_hints(func)
    fields: dict[str, Any] = {}
    cleaned = []
    for p in params:
        if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            raise PackageError(f"{where}: *args/**kwargs are not allowed (args are sent by name)")
        if p.default is not inspect.Parameter.empty:
            try:
                json.dumps(p.default)
            except (TypeError, ValueError):
                raise PackageError(
                    f"{where}: default of {p.name!r} must be JSON-serializable"
                ) from None
        annotation = hints.get(p.name, p.annotation)
        fields[p.name] = (
            Any if annotation is inspect.Parameter.empty else annotation,
            ... if p.default is inspect.Parameter.empty else p.default,
        )
        cleaned.append(p.replace(annotation=annotation))
    returns = hints.get("return", inspect.signature(func).return_annotation)
    args_model = create_model(
        f"{package}_{name}_args",
        __config__=ConfigDict(extra="forbid", arbitrary_types_allowed=True),
        **fields,
    )
    return ToolSpec(
        package=package,
        name=name,
        func=func,
        effect=effect,
        signature=inspect.Signature(cleaned, return_annotation=returns),
        doc=inspect.cleandoc(func.__doc__ or ""),
        args_model=args_model,
    )


def _resolved_hints(func: Callable[..., Any]) -> dict[str, Any]:
    """Annotations with `from __future__ import annotations` strings resolved."""
    try:
        return inspect.get_annotations(func, eval_str=True)
    except Exception:
        return {}


def _annotation_text(annotation: Any) -> str:
    if isinstance(annotation, str):
        return annotation
    if isinstance(annotation, type) and not getattr(annotation, "__args__", None):
        return annotation.__qualname__
    return inspect.formatannotation(annotation).replace("typing.", "")


def _format_signature(signature: inspect.Signature) -> str:
    parts = []
    for p in signature.parameters.values():
        text = p.name
        if p.annotation is not inspect.Parameter.empty:
            text += f": {_annotation_text(p.annotation)}"
        if p.default is not inspect.Parameter.empty:
            text += f" = {p.default!r}"
        parts.append(text)
    out = f"({', '.join(parts)})"
    if signature.return_annotation is not inspect.Signature.empty:
        out += f" -> {_annotation_text(signature.return_annotation)}"
    return out


def discover_packages() -> tuple[list[ToolPackage], list[str]]:
    """Installed packages from the `loom.tools` entry points, plus a notice per failure."""
    packages: list[ToolPackage] = []
    notices: list[str] = []
    for ep in sorted(entry_points(group=ENTRY_POINT_GROUP), key=lambda e: e.name):
        try:
            package = ep.load()
        except Exception as e:
            log.exception("loading tool package %s failed", ep.name)
            notices.append(f"Tool package {ep.name!r} failed to load: {type(e).__name__}: {e}")
            continue
        if not isinstance(package, ToolPackage):
            notices.append(f"Tool package {ep.name!r} is not a loom ToolPackage; skipped.")
            continue
        packages.append(package)
    return dedupe(packages, notices), notices


def dedupe(packages: Iterable[ToolPackage], notices: list[str]) -> list[ToolPackage]:
    seen: dict[str, ToolPackage] = {}
    for package in packages:
        if package.name in seen:
            notices.append(f"Two tool packages are named {package.name!r}; using the first.")
            continue
        seen[package.name] = package
    return list(seen.values())
