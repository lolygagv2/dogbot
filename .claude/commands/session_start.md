# TreatBot Session Start Command

## Initialization Protocol

### 1. Load Project Context
- Read `.claude/CLAUDE.md` for development rules
- **Check `.claude/resume_chat.md` for previous session history**
- Search project knowledge: "TreatBot Project Directory Structure"
- Review `.claude/PRODUCT_ROADMAP.md` for current phase
- Check `.claude/DEVELOPMENT_TODOS.md` for active tasks

### 2. Verify Protected Files
Check that these files exist and are unmodified:
- `notes.txt` (user notes - READ ONLY)
- `config.yaml` (system configuration)
- `/docs/` directory contents
- `HARDWARE_SPECS.md`
- `PRODUCT_ROADMAP.md`

### 3. Review Previous Session
**Previous Session Summary:**
- Read last 50-100 lines from `.claude/resume_chat.md`
- Note any unfinished tasks or issues
- Check for any warnings or critical fixes made

### 4. System Status Check
Display the following:

**Git Status:**
```bash
git status --short
git branch --show-current
git log --oneline -3
```

**Uncommitted Changes:**
- Count modified files
- List new untracked files
- Identify any files outside proper directories

**Project Health:**
- Battery level (if hardware connected)
- Last successful test run timestamp
- Any error logs from last session
- Check for saved detection results or test outputs

### 5. Hardware Connection Check
If hardware is accessible:
- Test Pi 5 connection
- Verify Hailo-8 HAT status
- Check camera availability
- Test servo controller connection

### 6. Ask User for Session Goal
Present options:
```
What are we working on today?

A. Hardware Testing (servos, motors, camera)
B. AI Development (pose detection, tracking)
C. Mission System (training sequences, rules)
D. Navigation (obstacle avoidance, docking)
E. API/Dashboard (web interface, control)
F. General Development (specify task)
G. Code Review/Refactoring

Enter letter or describe custom task:
```

### 7. Set Session Constraints
Based on user goal, establish boundaries:

**Example - Hardware Testing (A):**
- Working directory: `/tests/` only
- May create new test scripts in `/tests/hardware/`
- NO modifications to production code in `/hardware/` without explicit approval
- All changes tracked in git before proceeding

**Example - Code Review (G):**
- READ ONLY mode for all files
- No file creation or modification
- Only analysis and recommendations

### 8. Final Confirmation
```
✅ Session initialized for: [USER GOAL]
✅ Protected files verified
✅ Git status: [CLEAN / X uncommitted files]
✅ Current branch: [BRANCH NAME]

Ready to begin. Proceed? (yes/no)
```

---

## Usage
Call this command at the start of every Claude Code session:
```bash
/project:session-start
```

Or in VS Code extension: Type `/session-start` in chat

---

## CRITICAL RULES
- NEVER skip protected file verification
- NEVER assume session goal - always ask
- NEVER make changes before user confirmation
- ALWAYS show git status first