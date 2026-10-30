"""Configuration: pydantic models, layer precedence, and project-config restrictions.

See docs/09-config.md. Precedence (highest first): CLI flags, project config, user config,
built-in defaults. Project config is untrusted: it may set safe keys, tighten security keys,
and set restricted keys only when the workspace is trusted.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

ApprovalMode = Literal["read-only", "ask", "auto-write", "yolo"]
SandboxBackendName = Literal["none", "seatbelt", "docker"]
NetworkMode = Literal["none", "allowlist"]
OverrideDecision = Literal["allow", "ask", "deny"]

# Built-in model aliases; the profiles live in loom.providers.profiles.
BUILTIN_MODEL_ALIASES = frozenset({"sonnet", "opus", "haiku"})

# Request-body keys loom sets itself in some adapter; extra_body may not override them.
RESERVED_BODY_KEYS = frozenset(
    {"model", "messages", "tools", "stream", "system", "input", "instructions"}
)

# Strictness ranks for tighten-only keys (higher = stricter).
_APPROVAL_RANK: dict[str, int] = {"yolo": 0, "auto-write": 1, "ask": 2, "read-only": 3}
_BACKEND_RANK: dict[str, int] = {"none": 0, "seatbelt": 1, "docker": 2}
_NETWORK_RANK: dict[str, int] = {"allowlist": 0, "none": 1}

# Top-level sections an untrusted project config may set freely.
_SAFE_PROJECT_SECTIONS = frozenset({"agent", "context", "budget", "kernel", "skills", "agents"})


def _secret_name(value: str | None) -> str | None:
    """`api_key_secret` and `secret:` headers name a secret; they never hold the value.
    The message must not echo the value: it may be a real key pasted by mistake."""
    from loom.broker.secrets import NAME_RE

    if value is not None and not NAME_RE.match(value):
        hint = " (this looks like a key itself)" if value.startswith(("sk-", "sk_")) else ""
        raise ValueError(
            f"must be a secret name like 'anthropic_api_key'{hint}; store the value with "
            "`loom secrets set <name>` and remove it from this file"
        )
    return value


class ConfigError(Exception):
    """Invalid or unreadable configuration. The message names the offending file."""


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CapabilitiesEntry(_Model):
    """Per-model capability overrides; None means "use the built-in profile default"."""

    tools: bool | None = None
    parallel_tool_calls: bool | None = None
    images: bool | None = None
    thinking: bool | None = None
    prompt_cache: bool | None = None
    streaming: bool | None = None
    system_role: Literal["system", "developer", "user"] | None = None


class ThinkingEntry(_Model):
    effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None
    budget_tokens: int | None = Field(default=None, gt=0)  # older models only


class PricingEntry(_Model):
    """USD per million tokens."""

    input: float = Field(ge=0)
    output: float = Field(ge=0)
    cache_read: float | None = Field(default=None, ge=0)
    cache_write: float | None = Field(default=None, ge=0)


class ModelEntry(_Model):
    provider: str
    model: str
    base_url: str | None = None
    api_key_secret: str | None = None
    context_window: int | None = Field(default=None, gt=0)
    max_tokens: int = Field(default=8192, gt=0)
    timeout_s: float = Field(default=600, gt=0)
    verify_tls: bool = True
    ca_bundle: Path | None = None
    capabilities: CapabilitiesEntry = Field(default_factory=CapabilitiesEntry)
    headers: dict[str, str] = Field(default_factory=dict)
    extra_body: dict[str, Any] = Field(default_factory=dict)
    thinking: ThinkingEntry | None = None
    pricing: PricingEntry | None = None

    @field_validator("api_key_secret")
    @classmethod
    def _api_key_secret_is_a_name(cls, value: str | None) -> str | None:
        return _secret_name(value)

    @field_validator("headers")
    @classmethod
    def _secret_headers_are_names(cls, value: dict[str, str]) -> dict[str, str]:
        for header in value.values():
            if header.startswith("secret:"):
                _secret_name(header.removeprefix("secret:"))
        return value

    @field_validator("extra_body")
    @classmethod
    def _no_reserved_keys(cls, value: dict[str, Any]) -> dict[str, Any]:
        reserved = sorted(RESERVED_BODY_KEYS & value.keys())
        if reserved:
            raise ValueError(f"may not set keys loom controls: {', '.join(reserved)}")
        return value


class ProviderEntry(_Model):
    api_key_secret: str | None = None

    @field_validator("api_key_secret")
    @classmethod
    def _api_key_secret_is_a_name(cls, value: str | None) -> str | None:
        return _secret_name(value)


class AgentConfig(_Model):
    max_steps: int = Field(default=50, gt=0)


class ContextConfig(_Model):
    compact_at: float = Field(default=0.75, gt=0, le=1)
    keep_recent_turns: int = Field(default=3, ge=0)
    max_context_file_chars: int = Field(default=40_000, ge=0)


class BudgetConfig(_Model):
    max_chars: int = Field(default=12_000, gt=0)
    head_chars: int = Field(default=8_000, ge=0)
    tail_chars: int = Field(default=3_000, ge=0)

    @model_validator(mode="after")
    def _head_tail_fit(self) -> BudgetConfig:
        if self.head_chars + self.tail_chars >= self.max_chars:
            raise ValueError("head_chars + tail_chars must be less than max_chars")
        return self


class KernelConfig(_Model):
    eager: bool = False
    default_timeout_s: int = Field(default=300, gt=0, le=1800)


class DockerConfig(_Model):
    image: str = "loom-sandbox:latest"
    memory: str = "4g"
    cpus: float = Field(default=2, gt=0)


class SandboxConfig(_Model):
    backend: SandboxBackendName = "docker"  # owner decision at M5 (docs/00)
    network: NetworkMode = "none"
    allowlist: list[str] = Field(default_factory=list)
    deny_read: list[str] = Field(default_factory=list)
    env_passthrough: list[str] = Field(default_factory=list)
    python_packages: list[str] = Field(default_factory=lambda: ["pandas", "numpy", "pyarrow"])
    docker: DockerConfig = Field(default_factory=DockerConfig)


class ApprovalsConfig(_Model):
    mode: ApprovalMode = "ask"
    overrides: dict[str, OverrideDecision] = Field(default_factory=dict)


class BrokerConfig(_Model):
    call_timeout_s: float = Field(default=120, gt=0)


class SkillsConfig(_Model):
    include_claude: bool = False


class AgentsConfig(_Model):
    max_concurrency: int = Field(default=4, ge=1)


class TracingConfig(_Model):
    otlp_endpoint: str = ""
    log_tool_results: bool = False


class LoomConfig(_Model):
    default_model: str = "sonnet"
    models: dict[str, ModelEntry] = Field(default_factory=dict)
    providers: dict[str, ProviderEntry] = Field(default_factory=dict)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    kernel: KernelConfig = Field(default_factory=KernelConfig)
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    approvals: ApprovalsConfig = Field(default_factory=ApprovalsConfig)
    broker: BrokerConfig = Field(default_factory=BrokerConfig)
    packages: dict[str, dict[str, Any]] = Field(default_factory=dict)
    skills: SkillsConfig = Field(default_factory=SkillsConfig)
    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    tracing: TracingConfig = Field(default_factory=TracingConfig)

    def model_aliases(self) -> frozenset[str]:
        return BUILTIN_MODEL_ALIASES | self.models.keys()


@dataclass(frozen=True)
class LoadedConfig:
    config: LoomConfig
    notices: tuple[str, ...] = ()


def user_config_path() -> Path:
    return Path.home() / ".loom" / "config.toml"


def project_config_path(workspace: Path) -> Path:
    return workspace / ".loom" / "config.toml"


def load_config(
    workspace: Path,
    *,
    user_path: Path | None = None,
    cli: dict[str, Any] | None = None,
    trusted: bool = False,
) -> LoadedConfig:
    """Load and merge config layers. `cli` is a nested dict of overrides from CLI flags."""
    user_path = user_path or user_config_path()
    project_path = project_config_path(workspace)

    user_raw = _read_toml(user_path)
    user = _validate(user_raw, user_path)

    notices: list[str] = []
    project_raw: dict[str, Any] = {}
    # When the workspace is $HOME, the project file is the user file; don't filter it as untrusted.
    if not _same_file(project_path, user_path):
        project_raw = _read_toml(project_path)
        _validate(project_raw, project_path)  # catch typos with the right file name
        project_raw, ignored = _filter_project(project_raw, user, trusted=trusted)
        if ignored:
            notices.append(f"Ignored settings in {project_path}: {'; '.join(ignored)}")

    merged = _deep_merge(_deep_merge(user_raw, project_raw), cli or {})
    config = _validate(merged, Path("<merged config>"))
    if config.default_model not in config.model_aliases():
        known = ", ".join(sorted(config.model_aliases()))
        raise ConfigError(f"unknown model alias {config.default_model!r} (defined: {known})")
    return LoadedConfig(config, tuple(notices))


def restricted_project_keys(workspace: Path, *, user_path: Path | None = None) -> list[str]:
    """Project-config settings that are ignored now but would apply if the workspace were
    trusted (what the trust prompt unlocks)."""
    user_path = user_path or user_config_path()
    project_path = project_config_path(workspace)
    if _same_file(project_path, user_path):
        return []
    project_raw = _read_toml(project_path)
    if not project_raw:
        return []
    user = _validate(_read_toml(user_path), user_path)
    _, untrusted = _filter_project(project_raw, user, trusted=False)
    _, trusted = _filter_project(project_raw, user, trusted=True)
    return [item.split(" (")[0] for item in untrusted if item not in trusted]


def _read_toml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as f:
            return tomllib.load(f)
    except FileNotFoundError:
        return {}
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"{path}: invalid TOML: {e}") from e
    except OSError as e:
        raise ConfigError(f"{path}: cannot read: {e}") from e


def _validate(raw: dict[str, Any], path: Path) -> LoomConfig:
    try:
        return LoomConfig.model_validate(raw)
    except ValidationError as e:
        lines = [
            f"  {'.'.join(str(p) for p in err['loc']) or '<root>'}: {err['msg']}"
            for err in e.errors()
        ]
        raise ConfigError(f"{path}: invalid config\n" + "\n".join(lines)) from e


def _same_file(a: Path, b: Path) -> bool:
    try:
        return a.resolve() == b.resolve()
    except OSError:
        return False


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Tables merge recursively; scalars and lists in `override` replace."""
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _filter_project(
    project: dict[str, Any], user: LoomConfig, *, trusted: bool
) -> tuple[dict[str, Any], list[str]]:
    """Drop project keys that would loosen security or (untrusted) route credentials/code."""
    kept: dict[str, Any] = {}
    ignored: list[str] = []
    restricted = "restricted; workspace not trusted"

    for key, value in project.items():
        if key in _SAFE_PROJECT_SECTIONS:
            kept[key] = value
        elif key == "default_model":
            aliases = set(user.model_aliases())
            if trusted:
                aliases |= project.get("models", {}).keys()
            if value in aliases:
                kept[key] = value
            else:
                ignored.append(f"default_model (unknown model alias {value!r})")
        elif key == "sandbox":
            section = _filter_sandbox(value, user.sandbox, trusted, ignored)
            if section:
                kept[key] = section
        elif key == "approvals":
            section = _filter_approvals(value, user.approvals, ignored)
            if section:
                kept[key] = section
        elif trusted:
            kept[key] = value
        else:
            ignored.append(f"{key} ({restricted})")
    return kept, ignored


def _filter_sandbox(
    section: dict[str, Any], user: SandboxConfig, trusted: bool, ignored: list[str]
) -> dict[str, Any]:
    kept: dict[str, Any] = {}
    for key, value in section.items():
        if key == "backend":
            _tighten(kept, ignored, "sandbox", key, value, user.backend, _BACKEND_RANK)
        elif key == "network":
            _tighten(kept, ignored, "sandbox", key, value, user.network, _NETWORK_RANK)
        elif key == "deny_read":
            kept[key] = [*user.deny_read, *(p for p in value if p not in user.deny_read)]
        elif trusted:
            kept[key] = value
        else:
            ignored.append(f"sandbox.{key} (restricted; workspace not trusted)")
    return kept


def _filter_approvals(
    section: dict[str, Any], user: ApprovalsConfig, ignored: list[str]
) -> dict[str, Any]:
    kept: dict[str, Any] = {}
    for key, value in section.items():
        if key == "mode":
            _tighten(kept, ignored, "approvals", key, value, user.mode, _APPROVAL_RANK)
        elif key == "overrides":
            # Only "deny" is stricter than every possible policy decision; an "ask" override
            # could loosen a read-only "deny", so it is not accepted from project config.
            overrides = {tool: d for tool, d in value.items() if d == "deny"}
            ignored.extend(
                f"approvals.overrides.{tool} (only 'deny' allowed from project config)"
                for tool in value
                if tool not in overrides
            )
            if overrides:
                kept[key] = {**user.overrides, **overrides}
    return kept


def _tighten(
    kept: dict[str, Any],
    ignored: list[str],
    section: str,
    key: str,
    value: str,
    user_value: str,
    rank: dict[str, int],
) -> None:
    if rank[value] >= rank[user_value]:
        kept[key] = value
    else:
        ignored.append(f"{section}.{key} (would loosen {user_value!r} to {value!r})")
