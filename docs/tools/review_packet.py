"""Build a ~30k-token review packet for the Orchestrator.

Usage:
    python tools/review_packet.py --phase 2 --files src/cutoff.py tests/test_no_leakage.py
    python tools/review_packet.py --phase 4 --files src/evaluate.py src/models.py --run-pytest --out reports/review_p4.md
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

MAX_CHARS = 120_000  # ~30k tokens at ~4 chars/token


def _read_text(path: Path) -> str:
    if not path.exists():
        return f"[MISSING: {path.as_posix()}]"
    return path.read_text(encoding="utf-8", errors="replace")


def _run_cmd(cmd: list[str]) -> str:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        out = (proc.stdout or "") + (proc.stderr or "")
        return out.strip()
    except Exception as exc:
        return f"[Command failed: {' '.join(cmd)} -> {exc}]"


def _truncate(text: str, limit: int, label: str) -> str:
    if len(text) <= limit:
        return text
    keep = max(0, limit - 120)
    return text[:keep] + f"\n\n... [TRUNCATED {label}: {len(text) - keep} chars omitted] ..."


def build_packet(
    phase: int,
    critical_files: list[str],
    run_pytest: bool = False,
) -> str:
    task_path = Path(f"docs/tasks/phase_{phase:02d}.md")
    status_path = Path(f"reports/status/phase_{phase}.md")

    sections: list[tuple[str, str, int]] = []
    # Priority sections (never aggressively truncated unless huge)
    sections.append((f"Task File: {task_path.as_posix()}", _read_text(task_path), 20_000))
    sections.append((f"Status Note: {status_path.as_posix()}", _read_text(status_path), 15_000))

    if run_pytest:
        pytest_out = _run_cmd([sys.executable, "-m", "pytest", "-q"])
        sections.append(("Live Pytest Output", pytest_out or "[no output]", 10_000))

    for f in critical_files[:4]:
        p = Path(f)
        sections.append((f"Critical File: {p.as_posix()}", _read_text(p), 20_000))

    diff_unstaged = _run_cmd(["git", "diff"])
    diff_staged = _run_cmd(["git", "diff", "--cached"])
    git_diff = "\n".join(part for part in (diff_staged, diff_unstaged) if part)
    if not git_diff:
        git_diff = _run_cmd(["git", "diff", "HEAD~1..HEAD"])
    sections.append(("Git Diff", git_diff or "[no git diff]", 25_000))

    # Assemble within MAX_CHARS budget
    rendered: list[str] = [f"# Review Packet — Phase {phase:02d}\n"]
    remaining = MAX_CHARS - len(rendered[0])

    for title, body, cap in sections:
        header = f"\n## {title}\n\n"
        budget = min(cap, max(500, remaining - len(header) - 50))
        clipped = _truncate(body, budget, title)
        block = f"{header}```text\n{clipped}\n```\n" if title in ("Git Diff", "Live Pytest Output") else f"{header}{clipped}\n"
        if len(block) > remaining and remaining > 300:
            block = _truncate(block, remaining, title)
        rendered.append(block)
        remaining -= len(block)
        if remaining <= 300:
            rendered.append("\n[Budget reached — remaining sections omitted]\n")
            break

    return "".join(rendered)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ~30k-token review packet.")
    parser.add_argument("--phase", type=int, required=True, help="Phase number (0-10).")
    parser.add_argument(
        "--files",
        nargs="*",
        default=[],
        help="2 to 4 critical files to include.",
    )
    parser.add_argument(
        "--run-pytest",
        action="store_true",
        help="Run pytest and include live output.",
    )
    parser.add_argument("--out", type=str, default="", help="Optional output file path.")
    args = parser.parse_args()

    packet = build_packet(args.phase, args.files, run_pytest=args.run_pytest)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(packet, encoding="utf-8")
        print(f"Wrote review packet ({len(packet)} chars) to {out_path}")
    else:
        print(packet)


if __name__ == "__main__":
    main()
