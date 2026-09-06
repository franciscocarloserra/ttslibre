#!/bin/bash
# Syntax-check the JS the panel serves (a broken string silently kills the whole page). Usage: ./check_panel.sh [port]
curl -s "http://localhost:${1:-7807}/" | sed -n '/<script>/,/<\/script>/p' | sed '1d;$d' > /tmp/panel_check.js && node --check /tmp/panel_check.js && echo "panel JS ok"
