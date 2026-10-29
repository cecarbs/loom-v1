"""Host-side secrets (docs/04 "Secrets"): the macOS Keychain via `keyring` (service `loom`),
falling back to `LOOM_SECRET_<NAME>`. Consumers: the broker (tool packages) and provider
adapters. Every resolved value is registered with the redactor."""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol

from loom.broker.redact import Redactor

log = logging.getLogger(__name__)

SERVICE = "loom"
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]{0,63}$")


def env_name(name: str) -> str:
    return "LOOM_SECRET_" + name.upper().replace("-", "_").replace(".", "_")


class Keychain(Protocol):
    def get(self, name: str) -> str | None: ...
    def set(self, name: str, value: str) -> None: ...
    def delete(self, name: str) -> None: ...


class KeyringKeychain:
    """`keyring` with service `loom`. Backend errors read as "not set"."""

    def get(self, name: str) -> str | None:
        import keyring
        import keyring.errors

        try:
            return keyring.get_password(SERVICE, name)
        except keyring.errors.KeyringError as e:
            log.warning("keychain lookup for %s failed: %s", name, e)
            return None

    def set(self, name: str, value: str) -> None:
        import keyring

        keyring.set_password(SERVICE, name, value)

    def delete(self, name: str) -> None:
        import keyring
        import keyring.errors

        try:
            keyring.delete_password(SERVICE, name)
        except keyring.errors.PasswordDeleteError:
            pass


class SecretStore:
    def __init__(
        self,
        environ: Mapping[str, str] | None = None,
        *,
        keychain: Keychain | None = None,
        redactor: Redactor | None = None,
    ) -> None:
        self._environ = os.environ if environ is None else environ
        self._keychain = keychain
        self.redactor = redactor or Redactor()
        self._cache: dict[str, str | None] = {}

    def get(self, name: str) -> str | None:
        if name not in self._cache:
            value = self._keychain.get(name) if self._keychain is not None else None
            self._cache[name] = value or self._environ.get(env_name(name)) or None
        value = self._cache[name]
        if value:
            self.redactor.add(value)
        return value


class SecretIndex:
    """Names (never values) of secrets set with `loom secrets`, since the Keychain can't
    list them by service."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def names(self) -> list[str]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        return sorted(n for n in data if isinstance(n, str)) if isinstance(data, list) else []

    def add(self, name: str) -> None:
        self._write(sorted({*self.names(), name}))

    def remove(self, name: str) -> None:
        self._write([n for n in self.names() if n != name])

    def _write(self, names: list[str]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(names, indent=2) + "\n", encoding="utf-8")
