#!/bin/bash
# WIM-Z PreToolUse Hook — Protect critical files from accidental edits
# Blocks writes to model files, database, and production configs
# Location: .claude/hooks/protect-files.sh

# Read the hook input from stdin
INPUT=$(cat)

# Extract the file path being edited
FILE_PATH=$(echo "$INPUT" | python3 -c "
import sys, json
d = json.load(sys.stdin)
params = d.get('tool_input', {})
# Check various tool parameter names for file paths
path = params.get('file_path', params.get('path', params.get('filename', '')))
print(path)
" 2>/dev/null)

# Protected patterns — block writes to these
PROTECTED_PATTERNS=(
    "*.hef"                          # Compiled Hailo model files
    "data/treatbot.db"               # Live database
    "*.ts"                           # TorchScript model (behavior_14.ts)
    ".env"                           # Environment secrets
    "/etc/systemd/*"                 # System service files
)

for PATTERN in "${PROTECTED_PATTERNS[@]}"; do
    if [[ "$FILE_PATH" == $PATTERN ]]; then
        echo "🛑 BLOCKED: Cannot edit protected file: $FILE_PATH" >&2
        echo "These files require manual deployment. Use the pi-deploy skill for safe deployment." >&2
        exit 2
    fi
done

# Allow the operation
exit 0
