import shutil
import subprocess
import sys

import pytest


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        result = subprocess.run(["docker", "info"], capture_output=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    has_macos = any(item.get_closest_marker("macos") for item in items)
    has_docker = any(item.get_closest_marker("docker") for item in items)
    macos_ok = not has_macos or (sys.platform == "darwin" and shutil.which("sandbox-exec"))
    docker_ok = not has_docker or _docker_available()
    for item in items:
        if item.get_closest_marker("macos") and not macos_ok:
            item.add_marker(pytest.mark.skip(reason="needs macOS sandbox-exec"))
        if item.get_closest_marker("docker") and not docker_ok:
            item.add_marker(pytest.mark.skip(reason="needs a running Docker daemon"))


@pytest.fixture(autouse=True)
def _no_real_sandbox_by_default(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch):
    """The built-in default backend is docker. Unit tests run on `none` unless they are
    marked `docker`/`macos` (those exercise the real backends and are skipped when
    unavailable)."""
    if request.node.get_closest_marker("docker") or request.node.get_closest_marker("macos"):
        return
    from loom.sandbox.none import NoneBackend

    monkeypatch.setattr("loom.app.create_backend", lambda cfg, **kw: NoneBackend())
