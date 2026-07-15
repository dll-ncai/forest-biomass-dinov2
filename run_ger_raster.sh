#!/usr/bin/env bash
cd "$(dirname "$0")"
source ~/anaconda3/etc/profile.d/conda.sh; conda activate AI_ForestWatch
python make_biomass_rasters.py --site german > results_corrected/logs/raster_german.log 2>&1
touch results_corrected/.RASTER_GER_DONE
