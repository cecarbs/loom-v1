"""`[packages.mcp.servers.<name>]`: a `command` (stdio) or a `url` (streamable HTTP)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ServerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    command: str | None = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)  # values may be "secret:<name>"
    cwd: str | None = None
    url: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)  # values may be "secret:<name>"
    description: str = ""
    startup_timeout_s: float = Field(default=30, gt=0)

    @model_validator(mode="after")
    def _one_transport(self) -> ServerConfig:
        if bool(self.command) == bool(self.url):
            raise ValueError("set either command (stdio) or url (streamable HTTP)")
        if self.command and self.headers:
            raise ValueError("headers need url")
        if self.url and (self.args or self.env or self.cwd):
            raise ValueError("args, env and cwd need command")
        return self
