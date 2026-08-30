#!/usr/bin/env bash
# Overnight training matrix: 5 configs x 3 seeds, then shape-tracking eval of every
# resulting checkpoint (+ the PID baseline) into a single CSV.
#
# Results land in $OUT, which is tracked, rather than in a scratch directory: an
# earlier reclone destroyed every model/log/result because they sat in ignored dirs.
# Only the summary CSVs are committed; the checkpoints themselves are not.
#
# Usage:  bash tools/run_matrix.sh            (resumes: skips runs already finished)
# Watch:  tail -f results_matrix/progress.log
set -uo pipefail

# Repo root is derived from this script's location, so the scripts work from
# any checkout. Override PY to point at the interpreter of your conda env.
ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PY=${PY:-python}
OUT=${OUT:-$ROOT/results_matrix}
CONCURRENCY=${CONCURRENCY:-3}
SEEDS=${SEEDS:-"0 1 2"}
TIMESTEPS=${TIMESTEPS:-4000000}   # lowered for smoke-validating this script
ONLY=${ONLY:-}                    # optional: run a single config by name

mkdir -p "$OUT/logs" "$OUT/models"
PROGRESS="$OUT/progress.log"
CSV="$OUT/shape_eval.csv"

log() { echo "[$(date '+%F %T')] $*" >> "$PROGRESS"; }

# Torch would otherwise grab every core per process and thrash under concurrency.
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

# --- config matrix -----------------------------------------------------------
# name | training flags | action_scale (must be mirrored at eval time)
CONFIGS=(
  "baseline|--action_scale 0.05|0.05"
  "control|--action_scale 0.05 --crash_terminates|0.05"
  "randomized|--action_scale 0.05 --crash_terminates --init_randomization 1.0|0.05"
  "capstone|--action_scale 0.15 --rotor_randomization 0.12 --crash_terminates --init_randomization 1.0|0.15"
  "authority25|--action_scale 0.25 --rotor_randomization 0.12 --crash_terminates --init_randomization 1.0|0.25"
)

COMMON="--algo ppo --timesteps $TIMESTEPS --lr 1e-4 --target_kl 0.03 \
--eval_init_randomization 0.0 \
--stage_fractions 0.15,0.30,0.55 --stage_target_bounds_xy 2.0,4.0,6.0 \
--stage_target_bounds_z 0.75:1.5,0.6:2.0,0.5:2.5 \
--stage_episode_len_sec 6,10,16 --stage_num_segments 1,2,4"

# Vertical shapes take --z as the CENTRE altitude and span +-size/2, so the
# defaults (size 2.0, z 1.0) put the bottom edge at z=0 and the drone spawns
# inside the floor. Per-shape geometry is therefore explicit.
SHAPE_ARGS_FLAT="--size 2.0 --z 1.0"
SHAPE_ARGS_VERT="--size 1.5 --z 1.5"

cd "$ROOT" || exit 1

train_one() {
  local name="$1" flags="$2" seed="$3"
  local tag="${name}_s${seed}"
  local dir="$OUT/models/$tag"
  if [ -f "$dir/DONE" ]; then log "SKIP train $tag (already done)"; return 0; fi
  mkdir -p "$dir"
  echo "$PY -m src.rl.train $COMMON $flags --seed $seed --output_folder $dir" > "$dir/command.txt"
  log "START train $tag"
  # shellcheck disable=SC2086
  $PY -m src.rl.train $COMMON $flags --seed "$seed" --output_folder "$dir" \
      > "$OUT/logs/train_${tag}.log" 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then touch "$dir/DONE"; log "OK    train $tag"; else log "FAIL  train $tag (rc=$rc)"; fi
  return $rc
}

# --- phase 1: training, CONCURRENCY at a time --------------------------------
log "=== PHASE 1: training (${CONCURRENCY} concurrent) ==="
for entry in "${CONFIGS[@]}"; do
  IFS='|' read -r name flags _scale <<< "$entry"
  [ -n "$ONLY" ] && [ "$ONLY" != "$name" ] && continue
  for seed in $SEEDS; do
    while [ "$(jobs -rp | wc -l)" -ge "$CONCURRENCY" ]; do wait -n; done
    train_one "$name" "$flags" "$seed" &
  done
done
wait
log "=== PHASE 1 complete ==="

# --- phase 2: shape evaluation ----------------------------------------------
log "=== PHASE 2: shape evaluation ==="
[ -f "$CSV" ] || echo "config,seed,shape,completion_pct,crashed,dev_mean_m,dev_rms_m,dev_max_m" > "$CSV"

eval_one() {  # config seed model_path action_scale shape
  local name="$1" seed="$2" model="$3" scale="$4" shape="$5"
  local geom="$SHAPE_ARGS_FLAT"
  case "$shape" in vsquare|cube) geom="$SHAPE_ARGS_VERT" ;; esac
  local logf="$OUT/logs/eval_${name}_s${seed}_${shape}.log"
  local ctrl_args="--controller rl --model $model --rl_action_scale $scale"
  [ "$name" = "pid" ] && ctrl_args="--controller pid"
  # shellcheck disable=SC2086
  $PY -m src.rl.evaluate_shapes --shape "$shape" $ctrl_args $geom \
      --out "$ROOT/results/matrix_eval" > "$logf" 2>&1
  $PY - "$logf" "$name" "$seed" "$shape" >> "$CSV" <<'EOF'
import re, sys
logf, name, seed, shape = sys.argv[1:5]
txt = open(logf, errors="replace").read()
m = re.search(r"completed=([\d.]+)%\s+crashed=(\w+)\s+deviation:\s+mean=([\d.]+)\s+rms=([\d.]+)\s+max=([\d.]+)", txt)
print(",".join([name, seed, shape, *m.groups()]) if m else f"{name},{seed},{shape},PARSE_ERROR,,,,")
EOF
}

for entry in "${CONFIGS[@]}"; do
  IFS='|' read -r name _flags scale <<< "$entry"
  [ -n "$ONLY" ] && [ "$ONLY" != "$name" ] && continue
  for seed in $SEEDS; do
    dir="$OUT/models/${name}_s${seed}"
    model=$(ls -dt "$dir"/ppo_*/best_model.zip 2>/dev/null | head -1)
    if [ -z "$model" ]; then log "SKIP eval ${name}_s${seed} (no checkpoint)"; continue; fi
    for shape in hsquare triangle circle vsquare cube; do
      grep -q "^${name},${seed},${shape}," "$CSV" && continue
      log "EVAL ${name}_s${seed} $shape"
      eval_one "$name" "$seed" "$model" "$scale" "$shape"
    done
  done
done

# PID is deterministic -> one pass, no seeds.
for shape in hsquare triangle circle vsquare cube; do
  grep -q "^pid,0,${shape}," "$CSV" && continue
  log "EVAL pid $shape"
  eval_one "pid" "0" "" "" "$shape"
done

log "=== ALL DONE ==="
touch "$OUT/MATRIX_DONE"
