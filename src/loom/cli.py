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
from loom.session.store import SessionError, loom_home


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="loom",
        description="Terminal agent harness with a sandboxed, persistent code-mode kernel.",
    )
    parser.add_argument("--version", action="version", version=f"loom {__version__}")
    parser.add_argument("--model", metavar="ALIAS", help="model alias to use")
    parser.add_argument("--mode", choices=get_args(ApprovalMode), help="approval mode")
    parser.add_argument("--sandbox", choices=get_args(SandboxBackendName), help="sandbox backend")
    resume = parser.add_mutually_exclusive_group()
    resume.add_argument(
        "-c", "--continue", dest="continue_", action="store_true", help="resume the latest session"
    )
    resume.add_argument("--resume", metavar="ID", help="resume a session (id or unique prefix)")
    commands = parser.add_subparsers(dest="command", metavar="COMMAND")

    commands.add_parser("sessions", help="list sessions for this directory")

    secrets = commands.add_parser("secrets", help="manage secrets in the macOS Keychain")
    secrets_cmd = secrets.add_subparsers(dest="action", required=True)
    for action, text in (("set", "store a secret (hidden input)"), ("delete", "remove a secret")):
        sub = secrets_cmd.add_parser(action, help=text)
        sub.add_argument("name")
    secrets_cmd.add_parser("list", help="list secret names (never values)")

    sandbox = commands.add_parser("sandbox", help="prepare sandbox backends")
    sandbox_cmd = sandbox.add_subparsers(dest="action", required=True)
    setup = sandbox_cmd.add_parser(
        "setup", help="create the seatbelt kernel venv (or, with --docker, build the image)"
    )
    setup.add_argument("--docker", action="store_true", help="build the docker sandbox image")

    models = commands.add_parser("models", help="configured models")
    models_cmd = models.add_subparsers(dest="action", required=True)
    models_cmd.add_parser("list", help="configured models with resolved capabilities")
    test = models_cmd.add_parser(
        "test", help="probe a model (reply, tool call, image) and suggest capability settings"
    )
    test.add_argument("alias")

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
    if args.command == "sessions":
        return sessions_command()
    workspace = Path.cwd().resolve()
    try:
        # Only the interactive run asks; subcommands use an existing grant.
        trusted = workspace_trust(workspace, ask=args.command is None and sys.stdin.isatty())
        loaded = load_config(workspace, cli=cli_overrides(args), trusted=trusted)
    except ConfigError as e:
        print(f"loom: {e}", file=sys.stderr)
        return 2
    for notice in loaded.notices:
        print(f"loom: {notice}", file=sys.stderr)
    if args.command == "models":
        if args.action == "list":
            return models_list(loaded.config)
        return asyncio.run(models_test(loaded.config, args.alias))
    if args.command == "sandbox":
        from loom.sandbox.setup import setup_docker, setup_seatbelt

        if args.docker:
            return setup_docker(loaded.config.sandbox)
        return setup_seatbelt(loaded.config.sandbox, loom_home())
    try:
        asyncio.run(
            _interactive(
                loaded.config,
                model=args.model,
                resume=args.resume,
                resume_latest=args.continue_,
                trusted=trusted,
            )
        )
    except (SandboxUnavailable, SessionError) as e:
        print(f"loom: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print(file=sys.stderr)
        return 130
    return 0


def workspace_trust(workspace: Path, *, ask: bool) -> bool:
    """Whether project extensions and restricted project settings apply (docs/08). With
    `ask`, prompts once when the workspace has something to trust and no current grant."""
    from loom.extensions.trust import grant, is_trusted, trust_request

    home = loom_home()
    if is_trusted(workspace, home):
        return True
    request = trust_request(workspace, home)
    if not request or not ask:
        return False
    print("loom: this workspace has", file=sys.stderr)
    for line in request.describe(workspace):
        print(f"  · {line}", file=sys.stderr)
    if request.extensions:
        print("  Extensions run as you on this machine, outside the sandbox.", file=sys.stderr)
    if request.config_keys:
        print(
            "  These settings can choose model endpoints, credentials and tool packages.",
            file=sys.stderr,
        )
    print(
        "  Only trust repositories you control. You are asked again if any of this changes.",
        file=sys.stderr,
    )
    try:
        answer = input(f"Trust {workspace}? [y/N] ")
    except (EOFError, KeyboardInterrupt):
        print(file=sys.stderr)
        answer = ""
    if answer.strip().lower() not in ("y", "yes"):
        return False
    grant(workspace, home)
    return True


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


def models_list(config: LoomConfig) -> int:
    from loom.providers.profiles import resolve_profile

    short = {"parallel_tool_calls": "parallel", "prompt_cache": "cache"}
    for alias in sorted(config.model_aliases()):
        p = resolve_profile(alias, config)
        caps = p.capabilities
        flags = [
            short.get(name, name)
            for name in ("tools", "parallel_tool_calls", "images", "thinking", "prompt_cache")
            if getattr(caps, name)
        ]
        if not caps.tools:
            flags.insert(0, "text tool calls")
        if not caps.streaming:
            flags.append("no streaming")
        if caps.system_role != "system":
            flags.append(f"system_role={caps.system_role}")
        price = (
            f"${p.pricing.input:g} / ${p.pricing.output:g} per MTok"
            if p.pricing
            else "cost unknown"
        )
        marker = "*" if alias == config.default_model else " "
        print(f"{marker} {alias:<10} {p.provider}: {p.model}")
        print(f"  {'':<10} {p.context_window:,} context · {p.max_tokens:,} max out · {price}")
        print(f"  {'':<10} {', '.join(flags)}")
    print("* default. Probe a model with: loom models test <alias>")
    return 0


async def models_test(config: LoomConfig, alias: str) -> int:
    from loom.broker.secrets import KeyringKeychain, SecretStore
    from loom.providers.probe import probe_model, suggestions
    from loom.providers.profiles import resolve_profile
    from loom.providers.registry import create_provider

    if alias not in config.model_aliases():
        known = ", ".join(sorted(config.model_aliases()))
        print(f"loom: unknown model alias {alias!r} (defined: {known})", file=sys.stderr)
        return 2
    profile = resolve_profile(alias, config)
    secrets = SecretStore(keychain=KeyringKeychain())
    print(f"Probing {alias} ({profile.provider}: {profile.model})…")
    results = await probe_model(profile, lambda p: create_provider(p, secrets, text_tools=False))
    for r in results:
        mark = "✓" if r.ok else "✗"
        timing = f"  ({r.seconds:.1f}s)" if r.seconds else ""
        print(f"  {mark} {r.name:<16} {r.detail}{timing}")
    if results[0].name == "reply" and not results[0].ok and len(results) == 1:
        return 1
    lines = suggestions(profile, results)
    if lines:
        print("\nSuggested config (~/.loom/config.toml):")
        for line in lines:
            print(f"  {line}")
    else:
        print("\nThe configured capabilities match what the model did.")
    return 0


def sessions_command() -> int:
    from rich.console import Console

    from loom.session.store import list_sessions
    from loom.tui.commands import session_table

    sessions = list_sessions(Path.cwd().resolve())
    if not sessions:
        print("No sessions in this directory.")
        return 0
    Console(highlight=False).print(session_table(sessions))
    print("Resume with: loom --resume <id>   (or loom -c for the latest)")
    return 0


async def _interactive(
    config: LoomConfig,
    *,
    model: str | None = None,
    resume: str | None = None,
    resume_latest: bool = False,
    trusted: bool = False,
) -> None:
    from loom.app import App
    from loom.tui.app import InlineUI

    app = App(
        Path.cwd().resolve(),
        config,
        model=model,
        resume=resume,
        resume_latest=resume_latest,
        trusted=trusted,
    )
    await app.start()
    try:
        await InlineUI(app).run()
    finally:
        await app.aclose()
