#!/bin/bash
# launchd entry: m5-bigjob start -> train (auto-resume, up to 3 attempts on crash) + pause-watcher -> m5-bigjob end
# Differences from run 1: waits for m5-bigjob start for up to 24 h (another job's render may hold the M5), and at the end
# never kills another job's render: if one is running, Song Forge is brought back without m5-bigjob's pkill list.
D=$(cd "$(dirname "$0")" && pwd); cd "$D"
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
exec >> "$D/run.log" 2>&1
echo "$(date '+%F %T') wrapper start pid $$"; echo $$ > wrapper.pid
for i in $(seq 1 1440); do m5-bigjob start && break; echo "$(date '+%T') bigjob start rc=$?, retry in 60s ($i/1440)"; [ -f STOP ] && exit 0; sleep 60; done
bigjob_end() {
  echo "$(date '+%F %T') m5-bigjob end"
  if pgrep -f "krea_batch.py|ltx-2-mlx|make-ltx|make-k6|make-wa2|character_voice.py|latentsync|musetalk|longcat|film_qc.py" >/dev/null; then
    echo "another job's render is running: restoring Song Forge without m5-bigjob's pkill"
    localone-flip off >/dev/null; rm -f /tmp/songforge_paused_by
    launchctl kickstart -k gui/$(id -u)/com.nicedreamz.localone-overflow
    (cd ~/.local/localone && /usr/bin/python3 -c "import sys;sys.argv=['x'];sys.path.insert(0,'.');import overflow as o;ok,s=o.forge_start();print('M5 Song Forge WARM again after %ds'%s if ok else 'M5 Song Forge NOT warm after %ds'%s)")
  else
    m5-bigjob end
  fi
}
trap 'bigjob_end' EXIT
trap 'touch "$D/STOP"; [ -n "$TP" ] && kill -CONT $TP' TERM INT
for attempt in 1 2 3; do
  ~/.local/mlx-server/bin/python "$D/train_full.py" >> "$D/train.log" 2>&1 &
  TP=$!; echo $TP > train.pid; echo "$(date '+%F %T') train attempt $attempt pid $TP"
  "$D/watch.sh" $TP &
  rc=0; while kill -0 $TP 2>/dev/null; do wait $TP; rc=$?; done
  echo "$(date '+%F %T') train exit rc=$rc"; echo "$rc $(date '+%F %T')" > train.exit
  [ $rc = 0 ] && break; [ -f STOP ] && break
  grep -q nonfinite train.jsonl 2>/dev/null && [ "$(tail -1 train.jsonl | grep -c nonfinite)" = 1 ] && break
  sleep 20
done
