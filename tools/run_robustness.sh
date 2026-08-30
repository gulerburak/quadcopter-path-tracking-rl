#!/usr/bin/env bash
# Multi-seed robustness sweep on the circle: the piece run_matrix.sh did NOT do.
# For each authority config x seed, sweep a persistent single-rotor thrust loss and
# find the largest loss the policy still recovers from (completes, no crash); plus a
# one-shot wind gust.
#
# Usage:  bash tools/run_robustness.sh        (resumes: skips CSV rows already present)
# Watch:  tail -f results_matrix/robustness_progress.log
set -uo pipefail

# Repo root is derived from this script's location, so the scripts work from
# any checkout. Override PY to point at the interpreter of your conda env.
ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PY=${PY:-python}
OUT=${OUT:-$ROOT/results_matrix}
SEEDS=${SEEDS:-"0 1 2"}
LOSSES=${LOSSES:-"5 7 9 11 13 15 17 20"}   # percent single-rotor thrust loss, ascending
SHAPE=circle
GEOM="--size 2.0 --z 1.0"
FAULT_ROTOR=0
FAULT_TIME=4
WIND_IMPULSE=1.5
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2

# authority sweep: name | rl_action_scale (must match training). randomized = the
# nominal +-5% "standard RL"; capstone = +-15%; authority25 = +-25%.
CONFIGS=( "randomized|0.05" "capstone|0.15" "authority25|0.25" )

ROTOR_CSV="$OUT/robustness_rotor.csv"
WIND_CSV="$OUT/robustness_wind.csv"
PROGRESS="$OUT/robustness_progress.log"
mkdir -p "$OUT/logs"
log() { echo "[$(date '+%F %T')] $*" >> "$PROGRESS"; }

[ -f "$ROTOR_CSV" ] || echo "config,seed,loss_pct,fault_factor,completed_pct,crashed,pre_rms,peak,post_rms" > "$ROTOR_CSV"
[ -f "$WIND_CSV" ]  || echo "config,seed,completed_pct,crashed,pre_rms,peak,post_rms" > "$WIND_CSV"

cd "$ROOT" || exit 1

# parse the disturbance summary line into "completed,crashed,pre,peak,post"
parse_dist() {
  $PY - "$1" <<'EOF'
import re, sys
t = open(sys.argv[1], errors="replace").read()
m = re.search(r"completed=([\d.]+)%\s+crashed=(\w+)\s+pre_rms=([\d.]+)\s+peak=([\d.]+)\s+post_rms=([\d.]+)", t)
if m:
    print(",".join(m.groups()))
else:  # crashed before the fault step -> no peak/pre/post line
    m2 = re.search(r"completed=([\d.]+)%\s+crashed=(\w+)", t)
    print(",".join([m2.group(1), m2.group(2), "nan", "nan", "nan"]) if m2 else "PARSE_ERROR,,,,")
EOF
}

model_for() { ls -dt "$OUT/models/$1"/ppo_*/best_model.zip 2>/dev/null | head -1; }

run_dist() {  # name model scale shape dist_args logf
  local name=$1 model=$2 scale=$3 dist=$4 logf=$5
  local ctrl="--controller rl --model $model --rl_action_scale $scale"
  [ "$name" = pid ] && ctrl="--controller pid"
  # shellcheck disable=SC2086
  $PY -m src.rl.evaluate_shapes --shape $SHAPE $ctrl $GEOM $dist \
      --out "$ROOT/results/robust_eval" > "$logf" 2>&1
}

# ---- rotor sweep with early-break at the recovery threshold -----------------
sweep_rotor() {  # name seed model scale
  local name=$1 seed=$2 model=$3 scale=$4 loss ff res crashed comp
  for loss in $LOSSES; do
    if grep -q "^${name},${seed},${loss}," "$ROTOR_CSV"; then
      res=$(grep -m1 "^${name},${seed},${loss}," "$ROTOR_CSV" | cut -d, -f5,6)
    else
      ff=$(awk "BEGIN{printf \"%.4f\",(100-$loss)/100}")
      log "ROTOR ${name}_s${seed} loss=${loss}% (f=${ff})"
      run_dist "$name" "$model" "$scale" \
        "--disturbance rotor --fault_rotor $FAULT_ROTOR --fault_time $FAULT_TIME --fault_factor $ff" \
        "$OUT/logs/rotor_${name}_s${seed}_L${loss}.log"
      res=$(parse_dist "$OUT/logs/rotor_${name}_s${seed}_L${loss}.log")
      echo "${name},${seed},${loss},${ff},${res}" >> "$ROTOR_CSV"
      res=$(echo "$res" | cut -d, -f1,2)
    fi
    comp=$(echo "$res" | cut -d, -f1); crashed=$(echo "$res" | cut -d, -f2)
    # recovery is monotonic in loss: stop at the first crash / incomplete run
    if [ "$crashed" = True ] || awk "BEGIN{exit !(${comp:-0}<99)}"; then
      log "  ${name}_s${seed} threshold: recovers < ${loss}% (stop)"
      return 0
    fi
  done
  log "  ${name}_s${seed} recovers the full grid (>= max loss)"
}

for entry in "${CONFIGS[@]}"; do
  IFS='|' read -r name scale <<< "$entry"
  for seed in $SEEDS; do
    model=$(model_for "${name}_s${seed}")
    [ -z "$model" ] && { log "SKIP ${name}_s${seed} (no checkpoint)"; continue; }
    sweep_rotor "$name" "$seed" "$model" "$scale"
    if ! grep -q "^${name},${seed}," "$WIND_CSV"; then
      log "WIND ${name}_s${seed}"
      run_dist "$name" "$model" "$scale" \
        "--disturbance wind --fault_time $FAULT_TIME --wind_impulse $WIND_IMPULSE" \
        "$OUT/logs/wind_${name}_s${seed}.log"
      echo "${name},${seed},$(parse_dist "$OUT/logs/wind_${name}_s${seed}.log")" >> "$WIND_CSV"
    fi
  done
done

# ---- PID baseline (deterministic, one pass) ---------------------------------
sweep_rotor pid 0 "" ""
if ! grep -q "^pid,0," "$WIND_CSV"; then
  log "WIND pid"
  run_dist pid "" "" "--disturbance wind --fault_time $FAULT_TIME --wind_impulse $WIND_IMPULSE" \
    "$OUT/logs/wind_pid.log"
  echo "pid,0,$(parse_dist "$OUT/logs/wind_pid.log")" >> "$WIND_CSV"
fi

log "=== ROBUSTNESS DONE ==="
touch "$OUT/ROBUSTNESS_DONE"
