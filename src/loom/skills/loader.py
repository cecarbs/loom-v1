"""Skill discovery (docs/08 "Skills"): directories with a `SKILL.md` whose YAML frontmatter
names and describes the skill. Only `name — description (path)` goes into the system prompt;
the model reads the file with `read` when a skill is relevant."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

MAX_DESCRIPTION_CHARS = 300
_FRONTMATTER_RE = re.compile(r"\A---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|\Z)", re.DOTALL)


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    path: Path  # the SKILL.md
    source: str  # "user", "workspace" or "claude"

    @property
    def directory(self) -> Path:
        return self.path.parent

    def prompt_line(self) -> str:
        return f"- {self.name} — {self.description} ({self.path})"

    def body(self) -> str:
        """SKILL.md without its frontmatter."""
        text = self.path.read_text(encoding="utf-8")
        match = _FRONTMATTER_RE.match(text)
        return text[match.end() :].strip() if match else text.strip()


def skill_roots(
    workspace: Path, home: Path, *, include_claude: bool = False, user_home: Path | None = None
) -> list[tuple[str, Path]]:
    """Search order. Earlier roots win on a name clash: the user's own skills can't be
    shadowed by a cloned repo's. `home` is loom's home (~/.loom)."""
    roots = [("user", home / "skills"), ("workspace", workspace / ".loom" / "skills")]
    if include_claude:
        roots.append(("claude", (user_home or Path.home()) / ".claude" / "skills"))
    return roots


def discover_skills(roots: list[tuple[str, Path]]) -> tuple[list[Skill], list[str]]:
    """Returns the skills (sorted by name) and notices about skipped ones."""
    found: dict[str, Skill] = {}
    notices: list[str] = []
    for source, root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.glob("*/SKILL.md")):
            try:
                skill = parse_skill(path, source)
            except (OSError, UnicodeDecodeError, ValueError) as e:
                notices.append(f"Skipped skill {path}: {e}")
                continue
            if skill.name in found:
                notices.append(
                    f"Skipped skill {path}: {skill.name!r} is already defined at "
                    f"{found[skill.name].path}"
                )
                continue
            found[skill.name] = skill
    return sorted(found.values(), key=lambda s: s.name), notices


def parse_skill(path: Path, source: str) -> Skill:
    text = path.read_text(encoding="utf-8")
    match = _FRONTMATTER_RE.match(text)
    if match is None:
        raise ValueError("no YAML frontmatter (--- name/description ---)")
    try:
        meta = yaml.safe_load(match.group(1))
    except yaml.YAMLError as e:
        raise ValueError(f"invalid frontmatter: {e}") from None
    if not isinstance(meta, dict):
        raise ValueError("frontmatter must be a mapping")
    name, description = meta.get("name"), meta.get("description")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("frontmatter needs a 'name'")
    if not isinstance(description, str) or not description.strip():
        raise ValueError("frontmatter needs a 'description'")
    description = " ".join(description.split())  # one line in the system prompt
    if len(description) > MAX_DESCRIPTION_CHARS:
        description = description[: MAX_DESCRIPTION_CHARS - 1] + "…"
    return Skill(" ".join(name.split()), description, path.resolve(), source)
