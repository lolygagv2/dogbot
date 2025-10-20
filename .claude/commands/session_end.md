# TreatBot Session End Command

## Safe Shutdown Protocol

### PHASE 0: Save Chat History
**Update Resume Chat Log:**
- Save last 1000 lines of conversation to `.claude/resume_chat.md`
- Include:
  - Problems solved this session
  - Key code changes made
  - Any unresolved issues or warnings
  - Next steps identified
- Format as markdown with timestamps
- Keep most recent entries at top

### PHASE 1: Inventory (Show Only, Don't Execute)

#### New Files Created This Session
List all files created with timestamps:
```bash
# Files created in last session (since session start)
find . -type f -newer /tmp/session_start_marker -not -path './.git/*'
```

Group by directory:
- `/hardware/` - Production code
- `/tests/` - Test scripts
- `/missions/` - Training sequences
- `/docs/` - Documentation
- **ROOT or other** - ⚠️ Files in wrong location!

#### Modified Files
```bash
git status --short
git diff --stat
```

Show:
- Lines added/removed per file
- Which functions/classes changed
- Any TODO comments added

#### Files Outside Proper Structure
Check for violations of project structure:
- Test files in `/hardware/` or `/ai/` → Should be in `/tests/`
- Temporary files in root → Should be in `/tests/temp/`
- Debug scripts not in `/tests/debug/`

---

### PHASE 2: Verification

#### Protected Files Integrity Check
Verify these files are unchanged:
```bash
git diff HEAD -- notes.txt
git diff HEAD -- config/robot_config.yaml
git diff HEAD -- docs/
```

If modified without permission:
```
⚠️  WARNING: Protected file modified!
File: notes.txt
Status: Modified (12 lines changed)

This file should NOT be modified. 
Options:
1. Revert changes (git checkout -- notes.txt)
2. Review changes first (show diff)
3. Keep changes (explain why)

What should I do?
```

#### Test File Location Check
List any test/debug files NOT in `/tests/`:
```
⚠️  Files in wrong location:
- ./motor_test.py → Should be ./tests/hardware/motor_test.py
- ./ai/test_pose.py → Should be ./tests/ai/test_pose.py

Move these files? (yes/no)
```

---

### PHASE 3: Commit Planning

#### If Uncommitted Changes Exist

**Option A: Stage All Changes**
```bash
git add -A
git status
```

Show proposed commit message:
```
Proposed commit message:

feat: Add camera servo tracking system

Changes:
- Created hardware/servo_control.py with CameraTracker class
- Added PID control for smooth tracking
- Integrated with YOLOv8 detection pipeline
- Added tests in tests/hardware/test_servo_tracking.py

Files changed: 4
Lines added: 287
Lines removed: 12

Approve this commit? (yes/no/edit)
```

**Option B: Selective Staging**
```
Which files should be committed?

[ ] hardware/servo_control.py (new feature)
[ ] tests/hardware/test_servo_tracking.py (tests)
[x] notes.txt (skip - user notes)
[ ] config/robot_config.yaml (skip - config file)

Or: 
- 'all' to stage everything
- 'none' to skip commit
- 'review' to see diffs
```

#### If No Changes
```
✅ No uncommitted changes
Last commit: feat: Add camera tracking (2 hours ago)
```

---

### PHASE 4: Documentation Update

**Check if major functionality changed:**
- New modules created?
- API endpoints added?
- Hardware connections modified?
- Mission system updated?

If yes, ask:
```
Should I update documentation?

Changed components:
- hardware/servo_control.py (new)
- HARDWARE_SPECS.md might need servo details
- README.md could mention camera tracking

Options:
1. Update HARDWARE_SPECS.md (add servo specs)
2. Update README.md (add tracking feature)
3. Create new doc: docs/camera_tracking.md
4. Skip for now (manual update later)

What should I update? (1/2/3/4/all/none)
```

---

### PHASE 5: Cleanup Recommendations

#### Suggest Cleanup Actions (Don't Execute)
```
📋 Cleanup Recommendations:

1. Old test files detected:
   - tests/debug/test_old_1.py (14 days old, not used)
   - tests/debug/test_old_2.py (14 days old, not used)
   → Archive to tests/archive/2025-10/ ? (yes/no)

2. Duplicate test files:
   - tests/test_leds.py
   - tests/hardware/test_leds.py (newer, more complete)
   → Remove old test_leds.py? (yes/no)

3. Temporary debug files:
   - tests/temp/*.log (45MB total)
   → Delete all .log files in tests/temp/? (yes/no)

4. Scattered test files (not in /tests/):
   - None found ✅

Execute cleanup? (yes/no/review)
```

---

### PHASE 6: Session Summary

#### Show Work Completed
```
📊 Session Summary
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Goal: Implement camera servo tracking
Duration: 2 hours 15 minutes

✅ Completed:
  - Created CameraTracker class (hardware/servo_control.py)
  - Added PID control for smooth servo movement
  - Integrated with YOLOv8 detection
  - Wrote 8 unit tests (all passing)

📝 Files Modified:
  - 2 new files
  - 2 existing files modified
  - 287 lines added, 12 removed

🔬 Tests:
  - 8/8 tests passing ✅
  - No hardware tests run (device not connected)

📦 Git Status:
  - Ready to commit: 4 files
  - Protected files: Unchanged ✅

🚧 Next Steps:
  1. Test tracking with real hardware
  2. Tune PID parameters for smoothness
  3. Add lost-target recovery behavior

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Everything look good? (yes/no)
```

---

### PHASE 7: Final Actions

#### Commit Changes (If Approved)
```bash
git commit -m "feat: Add camera servo tracking system"
git log --oneline -1  # Show commit
```

#### Update Session Log
Update `.claude/resume_chat.md` with session summary:
```markdown
## Session: 2025-10-17 15:30
**Goal:** Camera servo tracking
**Status:** ✅ Complete

### Work Completed:
- Created hardware/servo_control.py with CameraTracker class
- Added PID control for smooth tracking
- Integrated with YOLOv8 detection pipeline
- Added tests in tests/hardware/test_servo_tracking.py

### Key Solutions:
- Fixed 21,504 false detection issue by [specific fix]
- Implemented camera rotation correction (90° CCW)
- Optimized inference speed from 5.3s to [new time]

### Files Modified:
- hardware/servo_control.py (new)
- tests/hardware/test_servo_tracking.py (new)
- [other files...]

### Commit: a1b2c3d - feat: Add camera servo tracking system

### Next Session:
- Test with real hardware
- Tune PID parameters
- Add lost-target recovery behavior

### Important Notes/Warnings:
- [Any critical information for next session]
```

Optionally append brief note to `notes.txt` if user wants personal tracking

#### Final Checklist
```
✅ All changes committed (or intentionally skipped)
✅ Protected files unchanged
✅ No files in wrong locations
✅ Documentation updated (or deferred)
✅ Session logged

Safe to close Claude Code. Goodbye! 👋
```

---

## Usage
Call this command at the end of every Claude Code session:
```bash
/project:session-end
```

Or in VS Code extension: Type `/session-end` in chat

---

## CRITICAL RULES
- NEVER commit without showing user what will be committed
- NEVER delete files without explicit approval
- NEVER skip verification of protected files
- ALWAYS show session summary before closing
- ALWAYS offer to update documentation if needed

## Emergency Recovery
If something went wrong this session:
```bash
# See what changed
git diff HEAD

# Undo all changes
git checkout HEAD -- .

# Or undo specific file
git checkout HEAD -- path/to/file.py
```