"""Command-line entry point (`loom = "loom.cli:main"`). See docs/09-config.md."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any, get_args

from loom import __version__
from loom.config import ApprovalMode, ConfigError, LoomConfig, SandboxBackendName, load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="loom",
        description="Terminal agent harness with a sandboxed, persistent code-mode kernel.",
    )
    parser.add_argument("--version", action="version", version=f"loom {__version__}")
    parser.add_argument("--model", metavar="ALIAS", help="model alias to use")
    parser.add_argument("--mode", choices=get_args(ApprovalMode), help="approval mode")
    parser.add_argument("--sandbox", choices=get_args(SandboxBackendName), help="sandbox backend")
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
    try:
        loaded = load_config(Path.cwd().resolve(), cli=cli_overrides(args))
    except ConfigError as e:
        print(f"loom: {e}", file=sys.stderr)
        return 2
    for notice in loaded.notices:
        print(f"loom: {notice}", file=sys.stderr)
    try:
        asyncio.run(_interactive(loaded.config))
    except KeyboardInterrupt:
        print(file=sys.stderr)
        return 130
    return 0


async def _interactive(config: LoomConfig) -> None:
    from loom.app import build_agent
    from loom.core.events import EventBus
    from loom.tui.basic import run_basic

    bus = EventBus()
    agent = build_agent(Path.cwd().resolve(), config, bus)
    await run_basic(agent, bus)
