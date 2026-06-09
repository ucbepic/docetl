#!/bin/bash
# Hook: automatically check the DocETL feedback server for new human feedback.
# Fires after Bash commands and after each agent turn (Stop event).
# Deduplicates so already-seen feedback is not re-injected.

PORT_FILE=".docetl_server_port"
SEEN_FILE="/tmp/.docetl_feedback_seen_$$"

# No server running? Exit silently.
if [ ! -f "$PORT_FILE" ]; then
  exit 0
fi

PORT=$(cat "$PORT_FILE" 2>/dev/null)
if [ -z "$PORT" ]; then
  exit 0
fi

# Verify server is alive (fast timeout)
if ! curl -s --connect-timeout 1 "http://localhost:$PORT/health" > /dev/null 2>&1; then
  exit 0
fi

# Poll current feedback
FEEDBACK=$(curl -s --connect-timeout 2 "http://localhost:$PORT/feedback/poll" 2>/dev/null)
if [ -z "$FEEDBACK" ]; then
  exit 0
fi

DOC_COUNT=$(echo "$FEEDBACK" | jq -r '.doc_feedback | length' 2>/dev/null || echo 0)
PIPE_COUNT=$(echo "$FEEDBACK" | jq -r '.pipeline_feedback | length' 2>/dev/null || echo 0)
KILLED=$(echo "$FEEDBACK" | jq -r '.killed' 2>/dev/null || echo false)
TOTAL=$(( ${DOC_COUNT:-0} + ${PIPE_COUNT:-0} ))

# Read last-seen count for deduplication
LAST_SEEN=0
if [ -f "$SEEN_FILE" ]; then
  LAST_SEEN=$(cat "$SEEN_FILE" 2>/dev/null || echo 0)
fi

# Pipeline was killed — always report
if [ "$KILLED" = "true" ] && [ "$LAST_SEEN" != "killed" ]; then
  echo "killed" > "$SEEN_FILE"
  jq -n --arg fb "$FEEDBACK" '{
    "additionalContext": ("URGENT: The human KILLED the pipeline via the web UI.\n\nFeedback data:\n" + $fb + "\n\nRead the kill_reason and any feedback, then adjust the pipeline accordingly.")
  }'
  exit 0
fi

# New feedback since last check?
if [ "$TOTAL" -gt "$LAST_SEEN" ]; then
  echo "$TOTAL" > "$SEEN_FILE"
  NEW_COUNT=$(( TOTAL - LAST_SEEN ))
  jq -n --arg fb "$FEEDBACK" --arg n "$NEW_COUNT" '{
    "additionalContext": ($n + " new human feedback item(s) from the DocETL web UI:\n\n" + $fb + "\n\nRead this feedback carefully. Adjust the pipeline prompts/schema based on it, then re-run with `docetl run`.")
  }'
  exit 0
fi

# No new feedback — exit silently
exit 0
