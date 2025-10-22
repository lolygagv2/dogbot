# CRITICAL SESSION NOTES FOR AI ASSISTANTS

## ⚠️ VISUAL FEEDBACK LIMITATION

**PROBLEM:** When AI assistants modify detection/vision systems, humans cannot visually verify if changes are improvements or regressions through chat logs alone.

**IMPACT:**
- Humans lose confidence when AI "improves" systems they can't see
- Log analysis doesn't show if detection quality actually improved
- No way to verify if pose detection, behavior classification, or visual accuracy got better/worse

**SOLUTION:**
- ALWAYS use existing working scripts as base - copy/paste and make minimal changes
- NEVER rewrite working vision systems from scratch
- When improving thresholds, change ONLY the specific values, not the entire architecture
- Acknowledge that humans need visual tools (GUI scripts) to verify detection quality

## EXAMPLE FROM 2025-10-20 SESSION:

**MISTAKE:** AI rewrote `test_mission_with_controls.py` from scratch, introduced multiple changes:
- Removed threading, servo controller, class structure
- Changed variable names, import statements
- Modified camera initialization
- Result: Broke detection completely (0 dogs detected vs working script)

**CORRECT APPROACH:** `cp test_mission_with_controls.py test_white_pom_fixed.py` + change ONE line (confidence threshold)

**LOG ANALYSIS RESULTS:**
- ✅ Detection: 70% success rate (21/30 frames)
- ✅ Pose processing: Running (9 tensors parsed correctly)
- ⚠️ Pose detection: Still failing ("No valid pose detection found" in most frames)
- ✅ ONE SUCCESS: Frame 17 showed pose confidence 0.381 > 0.15 threshold - "Found pose keypoints with confidence 0.381"

**CONCLUSION:** The lowered threshold (0.3 → 0.15) IS working - we got one pose detection. Need more testing to see if behavior analysis triggers with enough pose data.

## KEY REMINDER:
Trust working code. Make minimal changes. Humans need visual confirmation of "improvements."