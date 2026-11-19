#!/bin/bash
# pause-watcher. arg1 = training PID. SIGSTOP while Song Forge (127.0.0.1:8767) has jobs_running>0
# or system free memory <15%; SIGCONT when jobs_running=0 and free >=20%. State -> watch.state
P=$1; D=$(dirname "$0"); paused=0
while kill -0 $P 2>/dev/null; do
  jr=$(curl -s -m 3 127.0.0.1:8767/api/status | /usr/bin/python3 -c "import sys,json;print(json.load(sys.stdin).get('jobs_running',0))" 2>/dev/null || echo 0)
  free=$(memory_pressure 2>/dev/null | awk -F': ' '/free percentage/{gsub("%","",$2);print $2}'); free=${free:-100}
  why=""
  [ "${jr:-0}" -gt 0 ] && why="songforge jobs_running=$jr"
  if [ "$free" -lt 15 ] || { [ $paused = 1 ] && [ "$free" -lt 20 ] && [ -z "$why" ] && grep -q memory "$D/watch.state" 2>/dev/null; }; then why="${why:+$why, }memory free ${free}%"; fi
  if [ -n "$why" ] && [ $paused = 0 ]; then kill -STOP $P; paused=1; echo "$(date '+%F %T') PAUSE $why" >> "$D/watch.log"; fi
  if [ -z "$why" ] && [ $paused = 1 ]; then kill -CONT $P; paused=0; echo "$(date '+%F %T') RESUME (free ${free}%)" >> "$D/watch.log"; fi
  if [ $paused = 1 ]; then echo "paused: $why" > "$D/watch.state"; else echo "running (free ${free}%)" > "$D/watch.state"; fi
  sleep 5
done
echo "training process gone" > "$D/watch.state"
