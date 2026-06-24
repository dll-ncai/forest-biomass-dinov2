#!/bin/bash
# Run feature-biomass correlation analysis and UMAP 2D/3D visualization
# on the Pakistani dataset using FULL features (dino_features_with_labels_and_split.csv).

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
INPUT_CSV="${1:-../dino_features_with_labels_and_split.csv}"

echo "=============================================="
echo "Pakistani dataset analysis (same methodology as German data)"
echo "=============================================="
echo "Input CSV: $INPUT_CSV"
echo "Output directory: $SCRIPT_DIR"
echo ""

echo "Step 1: Feature-biomass correlation and ranking..."
python3 feature_biomass_correlation_analysis.py \
  --input_csv "$INPUT_CSV" \
  --output_dir "$SCRIPT_DIR" \
  --top_n 20

echo ""
echo "Step 2: UMAP 2D and 3D visualization..."
python3 visualize_with_umap.py \
  --input_csv "$INPUT_CSV" \
  --output_dir "$SCRIPT_DIR" \
  --n_neighbors 15 \
  --min_dist 0.1

echo ""
echo "Step 3: Best models by top-ranked features (20-250, step 10)..."
python3 find_best_models_top_features.py --output_dir "$SCRIPT_DIR"

echo ""
echo "Done. Outputs in: $SCRIPT_DIR"
echo "  - feature_correlations.csv, correlation_summary.json"
echo "  - top_features_correlation_analysis.png, top_10_features_detailed.png, correlation_distribution.png"
echo "  - umap_visualization.png, umap_biomass.png, umap_train_test.png (if split present)"
echo "  - umap_3d_biomass.png, umap_3d_train_test.png, umap_3d_multiple_views.png"
echo "  - results_top_features.csv, best_configurations.csv, performance_vs_n_features.png, heatmaps, METHODOLOGY_REPORT.md"
