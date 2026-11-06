#!/bin/bash
# Runs as many deployment sessions as fit in ~8.5 minutes (resumable; finished sessions are skipped).
cd "$(dirname "$0")"
curl -s localhost:8333/ >/dev/null || (nohup python3 abr_server.py 8333 >> abr.log 2>&1 &)
sleep 3
TR=$(python3 -c "import json;print(' '.join(json.load(open('trace_list.json'))))")
NODE_PATH=$(npm root -g) timeout -k 5 510 node run_sessions.js runs_dep nesy,mpc,bba,dashjs $TR > batch.log 2>&1
pkill -f 'seg_server[.]py'
pkill -x headless_shell
pkill -x chrome
grep -v '^$' batch.log | tail -10
echo "completed: $(ls runs_dep/*.json 2>/dev/null | grep -v seglog | wc -l) / 120"
exit 0
