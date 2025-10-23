# TreatBot Project - Development Rules

## NEVER DELETE OR MODIFY
- notes.txt (user's personal notes - IGNORE)
- /docs/ folder (reference documentation only)
- config/robot_config.yaml (requires explicit permission)
- Any file with "KEEPME" or "NOTES" in filename

## Project Structure - MUST MAINTAIN
Follow the structure defined in  `.claude/TreatBot_Project_Directory_Structure.md` keep this reference to files in mind to not recreate, duplicate or misunderstand progress

/hardware/ - Hardware control modules ONLY
/ai/ - AI/computer vision ONLY  
/tests/ - ALL test files go here, organized by component
/utils/ - Shared utilities only

## Project Documentation
- .claude/product_roadmap.md - Development timeline and phases
- .claude/development_todos.md - Priority-sorted tasks
- .claude/hardware_specs.md - Hardware configuration (AUTHORITATIVE)
- docs/IR_DOCKING_SYSTEM.md - IR beacon docking guide

## Session Protocol
ALWAYS use /project:session-start when opening
ALWAYS use /project:session-end before closing
NEVER skip these commands

## Test File Rules
- ALL test/debug files go in /tests/ subdirectories
- Name test files: test_<component>_<description>.py
- Before creating ANY new test file, check if similar test exists
- After completing testing phase, ASK before deleting old test files
- NEVER leave test files scattered in main project directories

## Cleanup Protocol
1. Before any cleanup, show me list of files to be removed
2. Wait for explicit approval
3. Move to /tests/archive/ rather than deleting
4. Keep maximum 3 test files per component unless told otherwise

## Development Workflow
1. Plan changes in Plan Mode (Shift+Tab twice)
2. Get approval before file operations
3. Commit incrementally
4. Ask before creating >5 new files

## When Refactoring
- Preserve existing working code structure
- Don't create new test folders outside /tests/
- Clean up as you go, but ASK first