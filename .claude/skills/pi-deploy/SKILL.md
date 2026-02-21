---
name: pi-deploy
description: Deploy updated Python code to the WIM-Z Raspberry Pi robot. Use when deploying code, pushing updates to the robot, syncing changes, restarting services, or checking robot status.
---

# WIM-Z Raspberry Pi Deployment

## Connection
- **Host:** morgan@treatbot1 (check local network or use hostname `wimz.local`)
- **Project path:** `/home/morgan/dogbot/`
- **Service name:** `treatbot`

## Pre-Deploy Checklist
1. Run local tests: `python -m pytest tests/ --tb=short`
2. Verify no syntax errors: `python -m py_compile <changed_file>`
3. Confirm target files are correct (no accidental config overwrites)

## Deploy Commands
```bash
# Sync changed files (dry-run first)
rsync -avz --dry-run --exclude='__pycache__' --exclude='*.pyc' --exclude='data/' --exclude='.git/' ./ morgan@treatbot1:/home/morgan/dogbot/

# Actual deploy (remove --dry-run after confirming)
rsync -avz --exclude='__pycache__' --exclude='*.pyc' --exclude='data/' --exclude='.git/' ./ morgan@treatbot1:/home/morgan/dogbot/
```

## Post-Deploy
```bash
# Restart the service
ssh morgan@treatbot1 'sudo systemctl restart treatbot'

# Watch logs for errors
ssh morgan@treatbot1 'journalctl -u treatbot -f --no-pager -n 50'

# Quick health check
ssh morgan@treatbot1 'curl -s http://localhost:8000/health'
```

## NEVER Deploy
- Without running pytest first
- The `data/treatbot.db` database file (contains live session data)
- `.hef` model files without explicit version suffix
- Direct edits to production configs without backup
- While the robot is mid-mission (check `/api/status` first)

## Emergency Rollback
```bash
# If deploy breaks something, restore from git
ssh morgan@treatbot1 'cd /home/morgan/dogbot && git checkout -- <broken_file>'

# Or restart with last known good state
ssh morgan@treatbot1 'sudo systemctl restart treatbot'
```

## Key Service Files
- **Systemd unit:** `/etc/systemd/system/treatbot.service`
- **Entry point:** `main_treatbot.py`
- **Config:** `configs/rules/silent_guardian_rules.yaml`
