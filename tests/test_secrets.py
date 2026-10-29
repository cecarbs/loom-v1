from pathlib import Path

import pytest

from loom import cli
from loom.broker.redact import REDACTED, Redactor
from loom.broker.secrets import SecretIndex, SecretStore
from loom.session.store import SessionLog


class FakeKeychain:
    def __init__(self, **values: str) -> None:
        self.values = dict(values)
        self.reads: list[str] = []

    def get(self, name: str) -> str | None:
        self.reads.append(name)
        return self.values.get(name)

    def set(self, name: str, value: str) -> None:
        self.values[name] = value

    def delete(self, name: str) -> None:
        self.values.pop(name, None)


def test_store_prefers_keychain_then_env_and_caches() -> None:
    keychain = FakeKeychain(splunk_token="from-keychain")
    env = {"LOOM_SECRET_SPLUNK_TOKEN": "from-env", "LOOM_SECRET_OTHER_KEY": "env-only"}
    store = SecretStore(env, keychain=keychain)
    assert store.get("splunk_token") == "from-keychain"
    assert store.get("other-key") == "env-only"
    assert store.get("missing") is None
    store.get("splunk_token")
    assert keychain.reads.count("splunk_token") == 1


def test_resolved_secrets_are_redacted_everywhere(tmp_path: Path) -> None:
    store = SecretStore({"LOOM_SECRET_API": "abcdef123456"})
    assert store.redactor.text("key=abcdef123456") == "key=abcdef123456"  # not resolved yet
    store.get("api")
    assert store.redactor.text("key=abcdef123456!") == f"key={REDACTED}!"
    log = SessionLog(tmp_path / "s.jsonl", store.redactor)
    log.append("x", nested={"abcdef123456": ["Bearer abcdef123456"]})
    raw = (tmp_path / "s.jsonl").read_text()
    assert "abcdef123456" not in raw and raw.count(REDACTED) == 2


def test_redactor_patterns_and_short_values() -> None:
    r = Redactor()
    r.add("abc")  # too short: would scrub ordinary text
    r.add("longer-secret")
    r.add("longer-secret-with-suffix")
    assert r.text("abc longer-secret-with-suffix") == f"abc {REDACTED}"
    samples = [
        "sk-ant-api03-" + "a" * 40,
        "ghp_" + "b" * 36,
        "AKIAABCDEFGHIJKLMNOP",
        "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U",
    ]
    for token in samples:
        assert r.text(f"x {token} y") == f"x {REDACTED} y", token
    assert r.value({"k": ("ghp_" + "c" * 36, 5)}) == {"k": [REDACTED, 5]}


def test_secrets_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setattr(cli, "loom_home", lambda: tmp_path)
    monkeypatch.setenv("LOOM_SECRET_CI_TOKEN", "x")
    keychain = FakeKeychain()
    monkeypatch.setattr(cli.getpass, "getpass", lambda prompt: "hunter2-value")

    assert cli.secrets_command("set", "splunk_token", keychain) == 0
    assert keychain.values == {"splunk_token": "hunter2-value"}
    assert cli.secrets_command("list", None, keychain) == 0
    out = capsys.readouterr().out
    assert "splunk_token" in out and "LOOM_SECRET_CI_TOKEN  (environment)" in out
    assert "hunter2" not in out and "hunter2" not in (tmp_path / "secrets.json").read_text()

    assert cli.secrets_command("set", "bad name!", keychain) == 2
    assert cli.secrets_command("delete", "splunk_token", keychain) == 0
    assert keychain.values == {} and SecretIndex(tmp_path / "secrets.json").names() == []


def test_tools_list_cli(capsys) -> None:
    assert cli.main(["tools", "list"]) == 0
    out = capsys.readouterr().out
    assert "demo — Synthetic" in out and "splunk.search(spl: str" in out
