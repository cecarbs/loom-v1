"""`loom sandbox setup [--docker]`: the seatbelt kernel venv or the docker image (docs/05)."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from loom.config import SandboxConfig

DOCKERFILE = Path(__file__).with_name("Dockerfile")


def _run(argv: list[str]) -> int:
    print("$ " + " ".join(argv), flush=True)
    return subprocess.run(argv, check=False).returncode


def setup_seatbelt(cfg: SandboxConfig, loom_home: Path) -> int:
    """Create ~/.loom/sandbox-venv (Python 3.12) with `[sandbox] python_packages`."""
    venv = loom_home / "sandbox-venv"
    python = venv / "bin" / "python"
    uv = shutil.which("uv")
    if uv:
        steps = [[uv, "venv", "--python", "3.12", "--allow-existing", str(venv)]]
        if cfg.python_packages:
            steps.append([uv, "pip", "install", "--python", str(python), *cfg.python_packages])
    else:
        steps = [[sys.executable, "-m", "venv", str(venv)]]
        if cfg.python_packages:
            steps.append([str(python), "-m", "pip", "install", *cfg.python_packages])
    for step in steps:
        if code := _run(step):
            print(f"loom: sandbox setup failed (exit {code})", file=sys.stderr)
            return code
    print(f"Seatbelt kernel Python ready: {python}")
    return 0


def setup_docker(cfg: SandboxConfig) -> int:
    """Build `[sandbox.docker] image` from loom's Dockerfile."""
    if shutil.which("docker") is None:
        print("loom: the docker CLI is not installed", file=sys.stderr)
        return 2
    if subprocess.run(["docker", "info"], capture_output=True, check=False).returncode:
        print("loom: Docker isn't running; start Docker Desktop first", file=sys.stderr)
        return 2
    code = _run(
        [
            "docker",
            "build",
            "-t",
            cfg.docker.image,
            "--build-arg",
            "PYTHON_PACKAGES=" + " ".join(cfg.python_packages),
            "-f",
            str(DOCKERFILE),
            str(DOCKERFILE.parent),
        ]
    )
    if code == 0:
        print(f"Docker sandbox image ready: {cfg.docker.image}")
    return code
