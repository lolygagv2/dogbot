# WIM-Z Claude Code Skills & Hooks

## Installation

Copy these files into your dogbot project's `.claude/` directory:

```bash
# From the dogbot project root:

# 1. Create the directory structure
mkdir -p .claude/skills/pi-deploy
mkdir -p .claude/skills/hailo-pipeline
mkdir -p .claude/skills/lightsail-server
mkdir -p .claude/skills/flutter-app
mkdir -p .claude/skills/hardware-safety
mkdir -p .claude/hooks

# 2. Copy skill files into their directories
cp skills/pi-deploy/SKILL.md        .claude/skills/pi-deploy/SKILL.md
cp skills/hailo-pipeline/SKILL.md   .claude/skills/hailo-pipeline/SKILL.md
cp skills/lightsail-server/SKILL.md .claude/skills/lightsail-server/SKILL.md
cp skills/flutter-app/SKILL.md      .claude/skills/flutter-app/SKILL.md
cp skills/hardware-safety/SKILL.md  .claude/skills/hardware-safety/SKILL.md

# 3. Copy hook scripts
cp hooks/session-start.sh   .claude/hooks/session-start.sh
cp hooks/stop-check.sh      .claude/hooks/stop-check.sh
cp hooks/protect-files.sh   .claude/hooks/protect-files.sh

# 4. Make hooks executable
chmod +x .claude/hooks/*.sh

# 5. Merge settings.json into your existing .claude/settings.json
# If you don't have one yet, just copy it:
cp settings.json .claude/settings.json
# If you already have settings.json, merge the "hooks" section manually
```

## What You Get

### Skills (auto-loaded by Claude Code when relevant)
| Skill | Triggers When You... |
|-------|---------------------|
| `pi-deploy` | Deploy code, push to robot, restart services |
| `hailo-pipeline` | Work on AI detection, models, vision pipeline |
| `lightsail-server` | Manage TURN server, backend, SSL, WebRTC infra |
| `flutter-app` | Build/modify the iOS/Android app |
| `hardware-safety` | Touch motor, servo, GPIO, or power code |

### Hooks (run automatically)
| Hook | Event | What It Does |
|------|-------|-------------|
| `session-start.sh` | SessionStart | Loads git status, recent commits, and TODO status |
| `stop-check.sh` | Stop | Prevents Claude from stopping with incomplete work |
| `protect-files.sh` | PreToolUse (Edit/Write) | Blocks edits to .hef models, database, .env files |

## Configuration Status

All placeholders have been configured (Feb 21, 2026):
1. ✅ `pi-deploy`: Using hostname `treatbot1`
2. ✅ `lightsail-server`: Using `api.wimzai.com`
3. ✅ `protect-files.sh`: Patterns configured for .hef, .db, .env, .ts files
4. ✅ `flutter-app`: Note added (Flutter app in separate repo)
5. ✅ `session-start.sh`: Paths point to `.claude/` directory

## Updating

These skills should evolve as the project does. When you add new features
or change architecture, ask Claude Code to update the relevant skill files.
Skills are just markdown — easy to edit.
