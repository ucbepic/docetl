#!/usr/bin/env bash
# UserPromptSubmit hook: surface the model-cascade milestone state each turn and
# remind Claude to end its response with a progress report against the plan.
# stdout from a UserPromptSubmit hook is injected into the model's context.
set -euo pipefail

progress_file="${CLAUDE_PROJECT_DIR:-.}/.claude/cascade-progress.md"

echo "=== Model Cascade progress (auto-injected; plan: docs/design/model-cascade.md) ==="
if [[ -f "$progress_file" ]]; then
  cat "$progress_file"
else
  echo "(no .claude/cascade-progress.md found)"
fi
echo
echo "REMINDER: End your response with a short 'Progress' section reporting status"
echo "against the milestone plan above — which milestone is in flight, what moved"
echo "this turn, and what's next. Keep the table in .claude/cascade-progress.md in"
echo "sync when a milestone's status changes."
