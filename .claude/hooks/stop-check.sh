#!/bin/bash
# WIM-Z Stop Hook — Lightweight task completion check
# Prevents Claude from stopping when there are obvious incomplete items
# Location: .claude/hooks/stop-check.sh

# Read the hook input from stdin
INPUT=$(cat)

# Check if this is already a re-check (stop_hook_active = true means we already ran once)
STOP_HOOK_ACTIVE=$(echo "$INPUT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('stop_hook_active', False))" 2>/dev/null)

if [ "$STOP_HOOK_ACTIVE" = "True" ]; then
    # Already ran once, allow the stop
    exit 0
fi

# Check the transcript for signs of incomplete work
TRANSCRIPT_FILE="${CLAUDE_PROJECT_DIR:-.}/.claude/transcript.jsonl"

if [ -f "$TRANSCRIPT_FILE" ]; then
    # Look for incomplete markers in the last 30 lines
    TAIL=$(tail -30 "$TRANSCRIPT_FILE" 2>/dev/null)
    
    # Check for TODO/FIXME left in recent output
    if echo "$TAIL" | grep -qi "TODO\|FIXME\|not yet implemented\|will implement later"; then
        echo "⚠️ Found TODO/FIXME markers in recent work. Please verify all tasks are complete before stopping." >&2
        exit 2
    fi
    
    # Check for test failures
    if echo "$TAIL" | grep -qi "FAILED\|Error\|Traceback" | grep -qvi "fixed\|resolved\|handled"; then
        echo "⚠️ Found potential errors in recent output. Please verify all issues are resolved." >&2
        exit 2
    fi
fi

# All clear, allow stop
exit 0
