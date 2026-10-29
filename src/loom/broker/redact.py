"""Redaction of secret values and known token shapes (docs/04 "Broker call pipeline" step 6).

One `Redactor` per process. `SecretStore` registers every value it resolves, so anything
loom ever read from the Keychain or the environment is scrubbed from tool results, broker
results and the session log.
"""

from __future__ import annotations

import re
import threading
from typing import Any

REDACTED = "[REDACTED]"
MIN_SECRET_CHARS = 6  # shorter values would scrub ordinary text

TOKEN_PATTERNS = re.compile(
    "|".join(
        (
            r"sk-ant-[A-Za-z0-9_\-]{20,}",  # Anthropic
            r"sk-(?:proj-)?[A-Za-z0-9_\-]{20,}",  # OpenAI
            r"gh[pousr]_[A-Za-z0-9]{30,}",  # GitHub
            r"github_pat_[A-Za-z0-9_]{30,}",
            r"xox[abprs]-[A-Za-z0-9\-]{10,}",  # Slack
            r"AKIA[0-9A-Z]{16}",  # AWS access key id
            r"eyJ[A-Za-z0-9_\-]{10,}\.eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}",  # JWT
        )
    )
)


class Redactor:
    def __init__(self) -> None:
        self._values: set[str] = set()
        self._pattern: re.Pattern[str] | None = None
        self._lock = threading.Lock()  # sync tool functions run in threads

    def add(self, value: str) -> None:
        if len(value) < MIN_SECRET_CHARS or value in self._values:
            return
        with self._lock:
            self._values.add(value)
            # Longest first, so a secret containing another is replaced whole.
            alternatives = sorted(self._values, key=len, reverse=True)
            self._pattern = re.compile("|".join(re.escape(v) for v in alternatives))

    def text(self, text: str) -> str:
        pattern = self._pattern
        if pattern is not None:
            text = pattern.sub(REDACTED, text)
        return TOKEN_PATTERNS.sub(REDACTED, text)

    def value(self, value: Any) -> Any:
        """Redact strings anywhere inside dicts/lists/tuples (keys included)."""
        if isinstance(value, str):
            return self.text(value)
        if isinstance(value, dict):
            return {self.value(k): self.value(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self.value(v) for v in value]
        return value
