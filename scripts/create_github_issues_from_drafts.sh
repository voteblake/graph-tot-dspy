#!/usr/bin/env bash
set -euo pipefail

# Create GitHub issues from markdown drafts in github_issues/ using gh CLI.
#
# Expected draft format:
#   Line 1: # Issue: <title>
#   Remaining lines: issue body (markdown)

DRAFT_DIR="github_issues"
REPO=""
LABELS=""
ASSIGNEES=""
MILESTONE=""
STATE_LABEL="planning"
DRY_RUN=0

usage() {
  cat <<USAGE
Usage: $(basename "$0") [options]

Options:
  --draft-dir <dir>      Directory containing issue draft .md files (default: github_issues)
  --repo <owner/name>    Target GitHub repo (optional; uses current gh default if omitted)
  --labels <csv>         Extra labels to apply (e.g. "enhancement,backend")
  --assignees <csv>      Assignees (e.g. "alice,bob")
  --milestone <name>     Milestone name
  --state-label <label>  Planning/status label added to every issue (default: planning)
  --dry-run              Print what would be created without calling gh
  -h, --help             Show this help

Example:
  $(basename "$0") --repo org/repo --labels "enhancement" --state-label "planning"
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --draft-dir)
      DRAFT_DIR="$2"; shift 2 ;;
    --repo)
      REPO="$2"; shift 2 ;;
    --labels)
      LABELS="$2"; shift 2 ;;
    --assignees)
      ASSIGNEES="$2"; shift 2 ;;
    --milestone)
      MILESTONE="$2"; shift 2 ;;
    --state-label)
      STATE_LABEL="$2"; shift 2 ;;
    --dry-run)
      DRY_RUN=1; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1 ;;
  esac
done

if [[ ! -d "$DRAFT_DIR" ]]; then
  echo "Draft directory not found: $DRAFT_DIR" >&2
  exit 1
fi

mapfile -t drafts < <(find "$DRAFT_DIR" -maxdepth 1 -type f -name '*.md' | sort)
if [[ ${#drafts[@]} -eq 0 ]]; then
  echo "No markdown drafts found in: $DRAFT_DIR" >&2
  exit 1
fi

if [[ $DRY_RUN -eq 0 ]]; then
  if ! command -v gh >/dev/null 2>&1; then
    echo "gh CLI not found in PATH." >&2
    exit 1
  fi
  gh auth status >/dev/null
fi

created=0
for draft in "${drafts[@]}"; do
  title_line="$(head -n 1 "$draft" | tr -d '\r')"
  if [[ "$title_line" =~ ^#\ Issue:\ (.+)$ ]]; then
    title="${BASH_REMATCH[1]}"
  elif [[ "$title_line" =~ ^#\ (.+)$ ]]; then
    title="${BASH_REMATCH[1]}"
  else
    echo "Skipping $draft: first line must start with '# Issue:' or '# '" >&2
    continue
  fi

  body_file="$(mktemp)"
  tail -n +2 "$draft" > "$body_file"

  cmd=(gh issue create --title "$title" --body-file "$body_file")
  [[ -n "$REPO" ]] && cmd+=(--repo "$REPO")

  combined_labels="$STATE_LABEL"
  if [[ -n "$LABELS" ]]; then
    combined_labels+="${combined_labels:+,}$LABELS"
  fi
  [[ -n "$combined_labels" ]] && cmd+=(--label "$combined_labels")
  [[ -n "$ASSIGNEES" ]] && cmd+=(--assignee "$ASSIGNEES")
  [[ -n "$MILESTONE" ]] && cmd+=(--milestone "$MILESTONE")

  if [[ $DRY_RUN -eq 1 ]]; then
    echo "[DRY-RUN] $draft"
    echo "  title: $title"
    echo "  labels: $combined_labels"
    echo "  command: ${cmd[*]}"
  else
    echo "Creating issue from $draft"
    "${cmd[@]}"
    created=$((created + 1))
  fi

  rm -f "$body_file"
done

if [[ $DRY_RUN -eq 1 ]]; then
  echo "Dry run complete. Drafts processed: ${#drafts[@]}"
else
  echo "Issue creation complete. Created: $created"
fi
