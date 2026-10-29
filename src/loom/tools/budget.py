"""Output budget applied to every tool result before it reaches the model (docs/03)."""

from __future__ import annotations

from pathlib import Path

from loom.config import BudgetConfig


class OutputBudget:
    def __init__(self, cfg: BudgetConfig, spill_dir: Path) -> None:
        self.cfg = cfg
        self.spill_dir = spill_dir
        spill_dir.mkdir(parents=True, exist_ok=True)
        self._count = len(list(spill_dir.glob("*.txt")))

    def apply(self, text: str, *, kernel_ref: int | None = None) -> tuple[str, Path | None]:
        """Return (text for the model, spill path or None)."""
        if len(text) <= self.cfg.max_chars:
            return text, None
        self._count += 1
        path = self.spill_dir / f"{self._count:04d}.txt"
        path.write_text(text, encoding="utf-8")
        head, tail = self.cfg.head_chars, self.cfg.tail_chars
        omitted = len(text) - head - tail
        kernel = f" In kernel: _out[{kernel_ref}]" if kernel_ref is not None else ""
        marker = (
            f"\n[… {omitted:,} chars omitted. Full output: {path} "
            f"(use read with offset).{kernel}]\n"
        )
        return text[:head] + marker + (text[-tail:] if tail else ""), path
