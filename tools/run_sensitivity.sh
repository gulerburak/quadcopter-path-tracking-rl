#!/usr/bin/env bash
# Learning-parameter + reward-parameter sensitivity sweep.
#
# The 2026-07 matrix (tools/run_matrix.sh) swept ENVIRONMENT/design parameters only:
# action_scale, crash_terminates, init_randomization, rotor_randomization. The course
# assignment additionally requires a sensitivity analysis over the LEARNING parameters,
# and the approved proposal promised the main learning and reward parameters. Neither the
# PPO hyperparameters nor the reward weights had ever been varied -- all 15 matrix runs
# used the defaults. This script closes that gap.
#
# Design: one-factor-at-a-time around the `baseline` config of run_matrix.sh
# (action_scale 0.05, no crash_terminates, no init_randomization). That centre point already
# exists at seeds 0/1/2 in results_matrix/models/baseline_s*, so it is re-used, not retrained.
# 4 parameters x 2 off-centre values x 2 seeds = 16 new runs.
#
# Usage:  bash tools/run_sensitivity.sh          (resumes: skips runs already finished)
# Watch:  tail -f results_sensitivity/progress.log
set -uo pipefail

# Repo root is derived from this script's location, so the scripts work from
# any checkout. Override PY to point at the interpreter of your conda env.
ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PY=${PY:-python}
OUT=${OUT:-$ROOT/results_sensitivity}
MATRIX=${MATRIX:-$ROOT/results_matrix}   # centre-point checkpoints live here
CONCURRENCY=${CONCURRENCY:-3}            # 3 was validated by run_matrix.sh on this box
SEEDS=${SEEDS:-"0 1"}
TIMESTEPS=${TIMESTEPS:-4000000}
ONLY=${ONLY:-}

mkdir -p "$OUT/logs" "$OUT/models"
PROGRESS="$OUT/progress.log"
CSV="$OUT/sensitivity_eval.csv"

log() { echo "[$(date '+%F %T')] $*" >> "$PROGRESS"; }

# Torch would otherwise grab every core per process and thrash under concurrency.
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

# --- sweep matrix -------------------------------------------------------------
# name | override flags (appended after COMMON, so argparse last-wins)
# Centre values, held by every run below unless overridden:
#   lr 1e-4 | gamma 0.99 | net_width 256 | alive_bonus 1.0
CONFIGS=(
  "lr_3e5|--lr 3e-5"
  "lr_3e4|--lr 3e-4"
  "gamma_095|--gamma 0.95"
  "gamma_0999|--gamma 0.999"
  "width_64|--net_width 64"
  "width_512|--net_width 512"
  "alive_025|--alive_bonus 0.25"
  "alive_20|--alive_bonus 2.0"
)

# Identical to run_matrix.sh's COMMON, plus the baseline action scale. Keeping this
# byte-for-byte compatible is what makes results_matrix's `baseline` a valid centre point.
COMMON="--algo ppo --timesteps $TIMESTEPS --lr 1e-4 --target_kl 0.03 \
--eval_init_randomization 0.0 \
--stage_fractions 0.15,0.30,0.55 --stage_target_bounds_xy 2.0,4.0,6.0 \
--stage_target_bounds_z 0.75:1.5,0.6:2.0,0.5:2.5 \
--stage_episode_len_sec 6,10,16 --stage_num_segments 1,2,4 \
--action_scale 0.05"

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
log "=== PHASE 1: training (${CONCURRENCY} concurrent, ${TIMESTEPS} steps, seeds: ${SEEDS}) ==="
for entry in "${CONFIGS[@]}"; do
  IFS='|' read -r name flags <<< "$entry"
  [ -n "$ONLY" ] && [ "$ONLY" != "$name" ] && continue
  for seed in $SEEDS; do
    while [ "$(jobs -rp | wc -l)" -ge "$CONCURRENCY" ]; do wait -n; done
    train_one "$name" "$flags" "$seed" &
  done
done
wait
log "=== PHASE 1 complete ==="

# --- phase 2: shape evaluation ------------------------------------------------
log "=== PHASE 2: shape evaluation ==="
[ -f "$CSV" ] || echo "config,seed,shape,completion_pct,crashed,dev_mean_m,dev_rms_m,dev_max_m" > "$CSV"

eval_one() {  # config seed model_path shape
  local name="$1" seed="$2" model="$3" shape="$4"
  local geom="$SHAPE_ARGS_FLAT"
  case "$shape" in vsquare|cube) geom="$SHAPE_ARGS_VERT" ;; esac
  local logf="$OUT/logs/eval_${name}_s${seed}_${shape}.log"
  # shellcheck disable=SC2086
  $PY -m src.rl.evaluate_shapes --shape "$shape" \
      --controller rl --model "$model" --rl_action_scale 0.05 $geom \
      --out "$ROOT/results/sensitivity_eval" > "$logf" 2>&1
  $PY - "$logf" "$name" "$seed" "$shape" >> "$CSV" <<'EOF'
import re, sys
logf, name, seed, shape = sys.argv[1:5]
txt = open(logf, errors="replace").read()
m = re.search(r"completed=([\d.]+)%\s+crashed=(\w+)\s+deviation:\s+mean=([\d.]+)\s+rms=([\d.]+)\s+max=([\d.]+)", txt)
print(",".join([name, seed, shape, *m.groups()]) if m else f"{name},{seed},{shape},PARSE_ERROR,,,,")
EOF
}

eval_config() {  # name seed model_dir_root
  local name="$1" seed="$2" root="$3"
  local model
  model=$(ls -dt "$root"/ppo_*/best_model.zip 2>/dev/null | head -1)
  if [ -z "$model" ]; then log "SKIP eval ${name}_s${seed} (no checkpoint)"; return; fi
  for shape in hsquare triangle circle vsquare cube; do
    grep -q "^${name},${seed},${shape}," "$CSV" && continue
    log "EVAL ${name}_s${seed} $shape"
    eval_one "$name" "$seed" "$model" "$shape"
  done
}

for entry in "${CONFIGS[@]}"; do
  IFS='|' read -r name _flags <<< "$entry"
  [ -n "$ONLY" ] && [ "$ONLY" != "$name" ] && continue
  for seed in $SEEDS; do
    eval_config "$name" "$seed" "$OUT/models/${name}_s${seed}"
  done
done

# Centre point: re-use the existing baseline checkpoints instead of retraining them,
# so the sweep CSV is self-contained and every point is scored by the same code path.
for seed in $SEEDS; do
  eval_config "centre" "$seed" "$MATRIX/models/baseline_s${seed}"
done

log "=== ALL DONE ==="
touch "$OUT/SENSITIVITY_DONE"
