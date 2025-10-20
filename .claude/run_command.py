#!/usr/bin/env python3
"""
Command runner for TreatBot project commands
Usage: python3 .claude/run_command.py [command]
"""

import sys
import os
from pathlib import Path

COMMANDS = {
    'session-start': '.claude/commands/session_start.md',
    'session-end': '.claude/commands/session_end.md',
    'safe-cleanup': '.claude/commands/safe-cleanup.md'
}

def show_command(command_file):
    """Display the command documentation"""
    if os.path.exists(command_file):
        with open(command_file, 'r') as f:
            print(f.read())
    else:
        print(f"Command file not found: {command_file}")

def main():
    if len(sys.argv) < 2:
        print("Available commands:")
        for cmd in COMMANDS:
            print(f"  - {cmd}")
        print("\nUsage: python3 .claude/run_command.py [command]")
        return

    command = sys.argv[1]
    if command in COMMANDS:
        print(f"📋 Showing protocol for: {command}")
        print("=" * 50)
        show_command(COMMANDS[command])
        print("=" * 50)
        print("\n✅ Ask Claude to execute this protocol")
    else:
        print(f"Unknown command: {command}")
        print("Available commands:", ', '.join(COMMANDS.keys()))

if __name__ == "__main__":
    main()