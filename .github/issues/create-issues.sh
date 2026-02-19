#!/usr/bin/env bash
#
# Creates GitHub issues from the review feedback markdown files.
# Prerequisites: gh CLI authenticated (run `gh auth login` first).
#
# Usage:
#   cd <repo-root>
#   bash .github/issues/create-issues.sh
#
set -euo pipefail

REPO="voteblake/graph-tot-dspy"
DIR="$(cd "$(dirname "$0")" && pwd)"

declare -A TITLES
TITLES[01]="Implement thought-level ToT branching to match paper's Algorithm 1"
TITLES[02]="Fix evaluation fidelity: score vote semantics and selection vote truncation"
TITLES[03]="Remove unnecessary synthesis LLM call and fix double-scoring on single-round runs"
TITLES[04]="Fix _dicts_to_branches answer-text matching causing silent branch collisions"
TITLES[05]="Add DSPy compilation and optimization support"
TITLES[06]="Adopt DSPy idioms: declarative signatures, typed forward(), and assertions"
TITLES[07]="Add parallel branch generation for k independent agent traces"

declare -A LABELS
LABELS[01]="algorithm"
LABELS[02]="algorithm"
LABELS[03]="bug,algorithm"
LABELS[04]="bug"
LABELS[05]="enhancement,dspy"
LABELS[06]="enhancement,dspy"
LABELS[07]="performance"

for num in 01 02 03 04 05 06 07; do
    file="$DIR/${num}-*.md"
    # shellcheck disable=SC2086
    file=$(ls $file 2>/dev/null | head -1)
    if [[ ! -f "$file" ]]; then
        echo "SKIP: no file for issue $num"
        continue
    fi

    title="${TITLES[$num]}"
    body=$(cat "$file")

    echo "Creating issue: $title"

    label_flag=""
    if [[ -n "${LABELS[$num]:-}" ]]; then
        IFS=',' read -ra LABEL_ARR <<< "${LABELS[$num]}"
        for lbl in "${LABEL_ARR[@]}"; do
            label_flag="$label_flag --label $lbl"
        done
    fi

    # Create issue; ignore label errors (labels may not exist yet)
    # shellcheck disable=SC2086
    gh issue create \
        --repo "$REPO" \
        --title "$title" \
        --body "$body" \
        $label_flag 2>&1 || \
    gh issue create \
        --repo "$REPO" \
        --title "$title" \
        --body "$body" 2>&1

    echo "  -> done"
    echo
done

echo "All issues created."
