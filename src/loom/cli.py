"""Command-line entry point (`loom = "loom.cli:main"`). See docs/09-config.md."""

from __future__ import annotations

import argparse
import asyncio
import getpass
import os
import sys
from pathlib import Path
from typing import Any, get_args

from loom import __version__
from loom.config import ApprovalMode, ConfigError, LoomConfig, SandboxBackendName, load_config
from loom.sandbox.base import SandboxUnavailable
from loom.session.store import loom_home


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="loom",
        description="Terminal agent harness with a sandboxed, persistent code-mode kernel.",
    )
    parser.add_argument("--version", action="version", version=f"loom {__version__}")
    parser.add_argument("--model", metavar="ALIAS", help="model alias to use")
    parser.add_argument("--mode", choices=get_args(ApprovalMode), help="approval mode")
    parser.add_argument("--sandbox", choices=get_args(SandboxBackendName), help="sandbox backend")
    commands = parser.add_subparsers(dest="command", metavar="COMMAND")

    secrets = commands.add_parser("secrets", help="manage secrets in the macOS Keychain")
    secrets_cmd = secrets.add_subparsers(dest="action", required=True)
    for action, text in (("set", "store a secret (hidden input)"), ("delete", "remove a secret")):
        sub = secrets_cmd.add_parser(action, help=text)
        sub.add_argument("name")
    secrets_cmd.add_parser("list", help="list secret names (never values)")

    tools = commands.add_parser("tools", help="kernel tool packages")
    tools_cmd = tools.add_subparsers(dest="action", required=True)
    tools_cmd.add_parser("list", help="show the tool catalog")
    return parser


def cli_overrides(args: argparse.Namespace) -> dict[str, Any]:
    """Translate CLI flags into a nested config dict (the highest-precedence layer)."""
    overrides: dict[str, Any] = {}
    if args.model:
        overrides["default_model"] = args.model
    if args.mode:
        overrides["approvals"] = {"mode": args.mode}
    if args.sandbox:
        overrides["sandbox"] = {"backend": args.sandbox}
    return overrides


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "secrets":
        return secrets_command(args.action, getattr(args, "name", None))
    if args.command == "tools":
        return tools_command()
    try:
        loaded = load_config(Path.cwd().resolve(), cli=cli_overrides(args))
    except ConfigError as e:
        print(f"loom: {e}", file=sys.stderr)
        return 2
    for notice in loaded.notices:
        print(f"loom: {notice}", file=sys.stderr)
    try:
        asyncio.run(_interactive(loaded.config))
    except SandboxUnavailable as e:
        print(f"loom: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print(file=sys.stderr)
        return 130
    return 0


def secrets_command(action: str, name: str | None, keychain=None) -> int:
    from loom.broker.secrets import NAME_RE, KeyringKeychain, SecretIndex, env_name

    keychain = keychain or KeyringKeychain()
    index = SecretIndex(loom_home() / "secrets.json")
    if action == "list":
        names = index.names()
        env = sorted(k for k in os.environ if k.startswith("LOOM_SECRET_"))
        for n in names:
            print(n)
        for var in env:
            print(f"{var}  (environment)")
        if not names and not env:
            print("No secrets set. Add one with: loom secrets set <name>")
        return 0
    assert name is not None
    if not NAME_RE.match(name):
        print(f"loom: invalid secret name {name!r} (letters, digits, _ . -)", file=sys.stderr)
        return 2
    if action == "delete":
        keychain.delete(name)
        index.remove(name)
        print(f"Deleted {name}.")
        return 0
    try:
        value = getpass.getpass(f"Value for {name} (input hidden): ")
    except (EOFError, KeyboardInterrupt):
        print(file=sys.stderr)
        return 130
    if not value:
        print("loom: empty value; nothing stored.", file=sys.stderr)
        return 2
    keychain.set(name, value)
    index.add(name)
    print(f"Stored {name} in the Keychain (service 'loom'). CI can use {env_name(name)} instead.")
    return 0


def tools_command() -> int:
    from loom.broker.registry import discover_packages

    packages, notices = discover_packages()
    for notice in notices:
        print(f"loom: {notice}", file=sys.stderr)
    if not packages:
        print("No tool packages installed.")
    for package in packages:
        print(f"{package.name} — {package.description}")
        for spec in package.tools.values():
            print(f"  {spec.display()}")
    return 0


async def _interactive(config: LoomConfig) -> None:
    from loom.app import App
    from loom.tui.app import InlineUI

    app = App(Path.cwd().resolve(), config)
    await app.start()
    try:
        await InlineUI(app).run()
    finally:
        await app.aclose()
