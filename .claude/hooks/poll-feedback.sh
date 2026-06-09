#!/bin/bash
# PostToolUse hook: after "docetl run" completes, check the feedback server
# for human feedback and inject it into Claude's context.

PORT_FILE=".docetl_server_port"

# Check if feedback server is running
if [ ! -f "$PORT_FILE" ]; then
  exit 0
fi

PORT=$(cat "$PORT_FILE")

# Verify server is alive
if ! curl -s --connect-timeout 1 "http://localhost:$PORT/health" > /dev/null 2>&1; then
  exit 0
fi

# Poll for any existing feedback
FEEDBACK=$(curl -s --connect-timeout 2 "http://localhost:$PORT/feedback/poll" 2>/dev/null)

DOC_COUNT=$(echo "$FEEDBACK" | jq -r '.doc_feedback | length' 2>/dev/null)
PIPE_COUNT=$(echo "$FEEDBACK" | jq -r '.pipeline_feedback | length' 2>/dev/null)
KILLED=$(echo "$FEEDBACK" | jq -r '.killed' 2>/dev/null)

if [ "$KILLED" = "true" ]; then
  jq -n --arg fb "$FEEDBACK" '{
    "additionalContext": ("The human KILLED the pipeline. Feedback data:\n" + $fb + "\n\nRead the kill_reason and feedback, then adjust the pipeline accordingly.")
  }'
  exit 0
fi

TOTAL=$(( ${DOC_COUNT:-0} + ${PIPE_COUNT:-0} ))

if [ "$TOTAL" -gt 0 ]; then
  jq -n --arg fb "$FEEDBACK" '{
    "additionalContext": ("Human feedback from the web UI:\n" + $fb + "\n\nRead this feedback carefully. Adjust the pipeline prompts/schema based on it, then re-run.")
  }'
else
  jq -n --arg port "$PORT" '{
    "additionalContext": ("The pipeline finished but the human has not submitted feedback yet. They are reviewing outputs in the browser.\n\nYou MUST wait for their feedback before proceeding. Run:\n  curl -s http://localhost:" + $port + "/feedback/wait\n\nThis blocks until the human clicks Done reviewing, then returns their feedback as JSON. Do NOT skip this step.")
  }'
fi

exit 0
