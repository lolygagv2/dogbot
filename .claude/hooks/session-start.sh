#!/bin/bash
# WIM-Z Session Start Hook
# Loads git status, recent commits, and development TODOs into Claude Code context
# Location: .claude/hooks/session-start.sh

# Get the project directory from environment or default
PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$(pwd)}"

echo "=== WIM-Z SESSION CONTEXT ==="
echo ""

# Current branch and git status
echo "--- Git Status ---"
cd "$PROJECT_DIR" 2>/dev/null && {
    BRANCH=$(git branch --show-current 2>/dev/null)
    echo "Branch: ${BRANCH:-unknown}"
    echo ""
    
    # Show uncommitted changes (abbreviated)
    CHANGES=$(git status --short 2>/dev/null)
    if [ -n "$CHANGES" ]; then
        echo "Uncommitted changes:"
        echo "$CHANGES" | head -20
        TOTAL=$(echo "$CHANGES" | wc -l)
        if [ "$TOTAL" -gt 20 ]; then
            echo "... and $((TOTAL - 20)) more files"
        fi
    else
        echo "Working tree clean"
    fi
    echo ""
    
    # Recent commits
    echo "--- Recent Commits ---"
    git log --oneline -5 2>/dev/null || echo "No git history available"
    echo ""
} || echo "Not in a git repository"

# Load development TODOs summary
echo "--- Development Status ---"
TODO_FILE="$PROJECT_DIR/.claude/development_todos.md"
ISSUES_FILE="$PROJECT_DIR/.claude/resume_chat.md"

if [ -f "$ISSUES_FILE" ]; then
    # Pull the summary stats from the issues tracker
    echo "From Outstanding Issues Tracker:"
    grep -A 5 "## 📊 Summary" "$ISSUES_FILE" 2>/dev/null | head -8
    echo ""
    # Show not-yet-developed items
    echo "Capital-dependent items still pending:"
    grep "Not developed\|Not done yet" "$ISSUES_FILE" 2>/dev/null | head -5
elif [ -f "$TODO_FILE" ]; then
    # Fallback to the older TODO file
    echo "From development_todos.md:"
    # Show in-progress items
    grep -A 1 "IN PROGRESS\|PRIORITY 1" "$TODO_FILE" 2>/dev/null | head -10
fi

echo ""
echo "--- Quick Reference ---"
echo "Entry point: main_treatbot.py"
echo "AI pipeline: core/ai_controller_3stage_fixed.py"
echo "Modes: modes/silent_guardian.py | orchestrators/coaching_engine.py"
echo "API: api/server.py"
echo "App: Flutter project (separate repo)"
echo ""
echo "=== END SESSION CONTEXT ==="
