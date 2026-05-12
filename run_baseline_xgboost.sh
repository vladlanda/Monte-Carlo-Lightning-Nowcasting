#!/usr/bin/env bash
# =============================================================================
# run_baseline.sh
#
# Runs the XGBoost lightning nowcasting baseline.
#
# Train : ENTLN 2022-2023 season.xlsx
# Test  : ENTLN 2024-2025 season.xlsx
# Leads : 60 120 180 240 300 360 min  (1h – 6h, 1h cadence)
#
# Usage:
#   bash run_baseline.sh                        # defaults
#   bash run_baseline.sh --neg_ratio 0.1        # override any argument
#
# Outputs are written to ./results/
# Each lead time gets its own subdirectory:
#   results/60min/   predictions.parquet  metrics.json  model.json  feature_importance.json
#   results/120min/  ...
#   ...
#   results/360min/  ...
#   results/summary_metrics.json
# =============================================================================

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN="${SCRIPT_DIR}/ENTLN 2022-2023 season.xlsx"
TEST="${SCRIPT_DIR}/ENTLN 2023-2024 season.xlsx"
OUTPUT="${SCRIPT_DIR}/baseline_results_entln"
PYTHON="${SCRIPT_DIR}/xgboostalgo.py"
PYTHON_PLOT="${SCRIPT_DIR}/plot_baseline.py"
PYTHON_PLOT_CASESTUDY="${SCRIPT_DIR}/plot_case_study.py"




# ── Validate inputs ───────────────────────────────────────────────────────────
if [[ ! -f "$TRAIN" ]]; then
    echo "ERROR: training file not found: $TRAIN" >&2
    exit 1
fi

if [[ ! -f "$TEST" ]]; then
    echo "ERROR: test file not found: $TEST" >&2
    exit 1
fi

if [[ ! -f "$PYTHON" ]]; then
    echo "ERROR: xgboostalgo.py not found: $PYTHON" >&2
    exit 1
fi

# ── Parameters ────────────────────────────────────────────────────────────────
#
# --grid       0.1 degrees  (~11 km over Israel)
# --windows    10 20 40 120 minutes  (history accumulation windows)
# --leadtimes  60 120 180 240 300 360 minutes  (1 h – 6 h)
# --depth      8   (XGBoost max tree depth)
# --trees      300 (number of boosting rounds)
# --lr         0.05 (learning rate)
# --neg_ratio  0.05 (keep 5 % of negative training samples)
#
# Any of these can be overridden by passing extra arguments to this script:
#   bash run_baseline.sh --neg_ratio 0.10 --trees 500

echo "============================================================"
echo "  XGBoost lightning nowcasting baseline"
echo "  Train : $TRAIN"
echo "  Test  : $TEST"
echo "  Output: $OUTPUT"
echo "  Start : $(date)"
echo "============================================================"

# python3 "$PYTHON" \
#     --train    "ENTLN 2022-2023 season.xlsx" \
#     --test     "ENTLN 2023-2024 season.xlsx" \
#     --output   "$OUTPUT" \
#     --grid     0.1 \
#     --windows  10 20 40 120 \
#     --leadtimes 60 120 180 240 300 360 \
#     --depth    8 \
#     --trees    300 \
#     --lr       0.05 \
#     --neg_ratio 0.05 \
#     "$@"


# Remove previous results directory -------------------------------------------

rm -rf "$OUTPUT"

# -- Train Eval Plot ----------------------------------------------------------
GRID=0.16

python3 "$PYTHON" \
    --train    "$TRAIN" \
    --test     "$TEST" \
    --output   "$OUTPUT" \
    --grid     "$GRID" \
    --windows 10 20 40 120 240 360 \
    --leadtimes 60 120 180 240 300 360 \
    --depth    12 \
    --trees    700 \
    --lr       0.01 \
    --neg_ratio 0.15 \
    "$@"

python3 "$PYTHON_PLOT" \
    --results "$OUTPUT"


# python3 "$PYTHON_PLOT_CASESTUDY" \
#     --results "$OUTPUT" --seed 42 --grid "$GRID" --test "$TEST"

python3 "$PYTHON_PLOT_CASESTUDY" \
    --results "$OUTPUT" --time "2024-01-31 09:00" --grid "$GRID" --test "$TEST" --windows 10 20 40 120 240 360

echo "============================================================"
echo "  Done : $(date)"
echo "  Results written to: $OUTPUT"
echo "============================================================"
