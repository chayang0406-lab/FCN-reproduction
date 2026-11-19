P=/public/home/zengzhicun/user/huanghaijun/projects/course_ctr_fcn_20261118
H5="$P/data/Criteo_x4_10_h5"
# block until build_h5 finishes (max 240 min), then report
for i in $(seq 1 480); do
  if ! kill -0 7620 2>/dev/null; then echo "EXITED after ~$((i*30/60))min"; break; fi
  sleep 30
done
echo "=== FINAL STATE $(date -Is) ==="
echo "--- build_h5.log (full) ---"
cat "$P/data/build_h5.log"
echo "--- h5 dir ---"
ls -l "$H5" 2>&1
echo "--- build_summary.json ---"
cat "$H5/build_summary.json" 2>/dev/null || echo "(no build_summary.json)"