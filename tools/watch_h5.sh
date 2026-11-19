P=/public/home/zengzhicun/user/huanghaijun/projects/course_ctr_fcn_20261118
H5="$P/data/Criteo_x4_10_h5"
LOG="$P/data/watch_h5.log"
START=$(date +%s)
while true; do
  if ! kill -0 7620 2>/dev/null; then
    echo "[$(date -Is)] PID 7620 EXITED" >> "$LOG"; break
  fi
  now=$(date +%s); el=$(( (now-START)/60 ))
  rss=$(awk '/VmRSS/{print $2}' /proc/7620/status 2>/dev/null)
  hwm=$(awk '/VmHWM/{print $2}' /proc/7620/status 2>/dev/null)
  sz=$(du -sb --apparent-size "$H5" 2>/dev/null | awk '{print $1}'); [ -z "$sz" ] && sz=0
  cur=$(ls "$H5" 2>/dev/null | tr '\n' ' ')
  echo "[$(date -Is)] +${el}min rss=$((rss/1024))MB hwm=$((hwm/1024))MB h5=$sz [$cur]" >> "$LOG"
  [ $el -ge 210 ] && { echo "timeout" >> "$LOG"; break; }
  sleep 180
done
echo "=== FINAL $(date -Is) ===" >> "$LOG"
tail -30 "$P/data/build_h5.log" >> "$LOG"
ls -l "$H5" >> "$LOG"