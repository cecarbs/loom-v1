"""Host-side secret resolution. M1: environment only; the macOS Keychain arrives in M4."""

from __future__ import annotations

import os
from collections.abc import Mapping


def env_name(name: str) -> str:
    return "LOOM_SECRET_" + name.upper().replace("-", "_").replace(".", "_")


class SecretStore:
    def __init__(self, environ: Mapping[str, str] | None = None) -> None:
        self._environ = os.environ if environ is None else environ

    def get(self, name: str) -> str | None:
        return self._environ.get(env_name(name)) or None
