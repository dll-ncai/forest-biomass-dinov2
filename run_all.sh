#!/usr/bin/env bash
# Full analysis pipeline. Each step is independent and logged;
# a failure in one step does NOT abort the rest (so the screen session always finishes).
# Launch detached:  screen -dmS biomass bash run_all.sh
set -u
cd "$(dirname "$0")"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate biomass

# limit BLAS threads per process so the 4 parallel pipeline runs don't oversubscribe
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4

LOG=results/logs
mkdir -p "$LOG"
MASTER="$LOG/run_$(date +%Y%m%d_%H%M%S).log"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$MASTER"; }

say "START full pipeline (rich hyper-parameter tuning)"
say "python: $(which python)  |  cores: $(nproc)"

# ── Step 1: main pipeline, 4 site/backbone combos IN PARALLEL ────────────────
say "STEP 1: main pipeline (pakistan/german x dino/resnet50) in parallel"
pids=()
for combo in "pakistan dino" "pakistan resnet50" "german dino" "german resnet50"; do
  set -- $combo; site=$1; feat=$2
  ( python pipeline.py --site "$site" --features "$feat" \
      > "$LOG/pipeline_${site}_${feat}.log" 2>&1 ; \
    echo "exit=$? ${site}/${feat}" >> "$MASTER" ) &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
say "STEP 1 done"

# ── Step 2: ablations + literature baselines in parallel ─────────────────────
say "STEP 2: german ablations + literature baselines"
( python ablations.py  > "$LOG/ablations.log" 2>&1 ; echo "exit=$? ablations" >> "$MASTER" ) &
a=$!
( python literature_baselines.py > "$LOG/literature.log" 2>&1 ; echo "exit=$? literature" >> "$MASTER" ) &
b=$!
wait $a; wait $b
say "STEP 2 done"

# ── Step 3: German robustness over 20 stratified splits (light grid) ─────────
say "STEP 3: robustness (20 splits, light grid)"
python robustness_german.py --seeds 20 --profile light \
    > "$LOG/robustness.log" 2>&1
echo "exit=$? robustness" >> "$MASTER"
say "STEP 3 done"

# ── Step 4: regenerate all figures from the result JSONs ─────────────────────
say "STEP 4: figures"
python make_figures.py > "$LOG/figures.log" 2>&1
echo "exit=$? figures" >> "$MASTER"
say "STEP 4 done"

# ── Step 5: refresh the human-readable results summary ───────────────────────
say "STEP 5: results summary"
python summarize_results.py > "$LOG/summary.log" 2>&1
echo "exit=$? summary" >> "$MASTER"

say "ALL DONE"
touch results/.RUN_COMPLETE
