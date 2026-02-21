#!/usr/bin/env python3
"""Create GitHub issues from markdown drafts in github_issues/ using gh CLI.

Expected draft format:
  Line 1: # Issue: <title>
  Remaining lines: issue body (markdown)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create GitHub issues from markdown drafts via gh CLI.",
    )
    parser.add_argument(
        "--draft-dir",
        default="github_issues",
        help="Directory containing issue draft .md files (default: github_issues)",
    )
    parser.add_argument(
        "--repo",
        default="",
        help="Target GitHub repo (optional; uses current gh default if omitted)",
    )
    parser.add_argument(
        "--labels",
        default="",
        help='Extra labels CSV (e.g. "enhancement,backend")',
    )
    parser.add_argument(
        "--assignees",
        default="",
        help='Assignees CSV (e.g. "alice,bob")',
    )
    parser.add_argument(
        "--milestone",
        default="",
        help="Milestone name",
    )
    parser.add_argument(
        "--state-label",
        default="planning",
        help="Planning/status label added to every issue (default: planning)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be created without calling gh",
    )
    return parser.parse_args()


def parse_title(first_line: str) -> str | None:
    line = first_line.strip("\r\n")
    if line.startswith("# Issue: "):
        return line.removeprefix("# Issue: ").strip()
    if line.startswith("# "):
        return line.removeprefix("# ").strip()
    return None


def split_csv(value: str) -> list[str]:
    if not value.strip():
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def main() -> int:
    args = parse_args()

    draft_dir = Path(args.draft_dir)
    if not draft_dir.is_dir():
        print(f"Draft directory not found: {draft_dir}", file=sys.stderr)
        return 1

    drafts = sorted(draft_dir.glob("*.md"))
    if not drafts:
        print(f"No markdown drafts found in: {draft_dir}", file=sys.stderr)
        return 1

    if not args.dry_run:
        if subprocess.run(["gh", "--version"], capture_output=True).returncode != 0:
            print("gh CLI not found in PATH.", file=sys.stderr)
            return 1
        auth_check = subprocess.run(["gh", "auth", "status"], capture_output=True)
        if auth_check.returncode != 0:
            print(auth_check.stderr.decode("utf-8", errors="replace"), file=sys.stderr)
            return 1

    state_labels = split_csv(args.state_label)
    extra_labels = split_csv(args.labels)
    all_labels = state_labels + [l for l in extra_labels if l not in state_labels]

    assignees = split_csv(args.assignees)
    created = 0

    for draft in drafts:
        lines = draft.read_text(encoding="utf-8").splitlines()
        if not lines:
            print(f"Skipping {draft}: empty file", file=sys.stderr)
            continue

        title = parse_title(lines[0])
        if not title:
            print(
                f"Skipping {draft}: first line must start with '# Issue:' or '# '",
                file=sys.stderr,
            )
            continue

        body = "\n".join(lines[1:]).lstrip("\n")

        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as tmp:
            tmp.write(body)
            body_path = Path(tmp.name)

        try:
            cmd = ["gh", "issue", "create", "--title", title, "--body-file", str(body_path)]
            if args.repo:
                cmd += ["--repo", args.repo]
            for label in all_labels:
                cmd += ["--label", label]
            for assignee in assignees:
                cmd += ["--assignee", assignee]
            if args.milestone:
                cmd += ["--milestone", args.milestone]

            if args.dry_run:
                labels_str = ",".join(all_labels)
                print(f"[DRY-RUN] {draft}")
                print(f"  title: {title}")
                print(f"  labels: {labels_str}")
                print(f"  command: {' '.join(cmd)}")
            else:
                print(f"Creating issue from {draft}")
                subprocess.run(cmd, check=True)
                created += 1
        finally:
            body_path.unlink(missing_ok=True)

    if args.dry_run:
        print(f"Dry run complete. Drafts processed: {len(drafts)}")
    else:
        print(f"Issue creation complete. Created: {created}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
