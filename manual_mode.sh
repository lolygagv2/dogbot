#!/bin/bash
# Quick script to activate manual mode
echo "Activating manual mode for 2 minutes..."
curl -s -X POST http://localhost:8000/mode/set -H "Content-Type: application/json" -d '{"mode":"manual","duration":120}' | grep -q '"success":true' && echo "✅ Manual mode activated" || echo "❌ Failed to activate manual mode"