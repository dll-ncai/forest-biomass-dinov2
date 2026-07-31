#!/usr/bin/env bash
# Re-run the Pakistan combos (with the NNLS small-sample simplex guard) + refresh
# figures and the results summary. Detached:  screen -dmS pak bash run_pak.sh
cd "$(dirname "$0")"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate biomass
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6
mkdir -p results/logs
( python pipeline.py --site pakistan --features dino \
    > results/logs/pipeline_pakistan_dino.log 2>&1 ) &
( python pipeline.py --site pakistan --features resnet50 \
    > results/logs/pipeline_pakistan_resnet50.log 2>&1 ) &
wait
python make_figures.py > results/logs/figures.log 2>&1
python summarize_results.py     > results/logs/summary.log 2>&1
touch results/.PAK_DONE
