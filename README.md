Lightning Nowcasting: XGBoost Baseline

This repository contains the baseline classical machine learning pipeline for short-term lightning forecasting (nowcasting). It uses XGBoost trained on gridded historical lightning strike data to predict the probability of future lightning occurrences at various lead times (1 to 6 hours).

Note on Methodology: This is a tree-based ensemble approach, serving as a classical Machine Learning baseline to be compared against more complex Deep Learning (e.g., CNN, ConvLSTM) spatial-temporal models.

🧠 Methodology & Feature Engineering

The pipeline converts raw, sparse lightning strike data into dense spatio-temporal cubes (default 5-minute bins). Because XGBoost cannot natively process spatial or sequential data, the script engineers explicit tabular features for each grid cell:

Temporal History: Strike counts aggregated over configurable rolling windows (e.g., 10, 20, 40, 120 minutes).

Spatial Context: Neighbourhood sums (radius 1 and 2) of recent strikes to detect incoming/expanding storms.

Decay Mechanics: An exponentially decaying sum of past strikes to capture storm dissipation.

Cyclic Time: Sine/Cosine encodings of the hour of the day and month of the year to capture diurnal and seasonal climatology.

Static Spatial Proxies: Normalized latitude, longitude, distance to the coast, a binary land/sea mask, and a linear-regression proxy for topography.

Calibration: The model applies Isotonic Calibration post-training using a temporal holdout set to ensure predicted probabilities match empirical observation frequencies.

📦 Requirements

pip install numpy pandas xgboost scikit-learn openpyxl pyarrow


📂 Data Format

The input data must be provided as Excel files (.xlsx). Both training and testing files must contain the following columns:

UTC: A parseable datetime string/object.

lat: Latitude of the strike (float).

lon: Longitude of the strike (float).

Data Constraints enforced by the pipeline:

Strict Temporal Separation: The script will halt if the maximum date in the training set overlaps with the minimum date in the testing set.

Wet Season Only: Data is automatically filtered to keep only the active lightning season (October – April) to prevent zero-inflation from summer months.

ROI Filtering: Data is strictly filtered to a hardcoded bounding box: w: 31.0, e: 37.0, s: 29.0, n: 35.0 (Eastern Mediterranean).

🚀 Usage

Option 1: Automated Bash Script (Recommended)

The repository includes an orchestration script that handles file paths, hyperparameter configuration, and sequential execution of training and plotting.

# Run with default parameters
bash run_baseline_xgboost.sh

# Override parameters (e.g., lower the negative downsampling ratio)
bash run_baseline_xgboost.sh --neg_ratio 0.05


Option 2: Manual Python Execution

You can run the Python script directly for finer control over specific lead times or grid sizes.

python xgboostalgo.py \
    --train "ENTLN 2022-2023 season.xlsx" \
    --test "ENTLN 2023-2024 season.xlsx" \
    --grid 0.16 \
    --windows 10 20 40 120 240 360 \
    --leadtimes 60 120 180 240 300 360 \
    --depth 12 \
    --trees 700 \
    --lr 0.01 \
    --neg_ratio 0.15 \
    --output baseline_results_entln


📊 Outputs

Results are saved in the directory specified by --output. The pipeline generates a separate directory for each targeted lead time (e.g., 60min/, 120min/).

Inside each lead time directory, you will find:

predictions.parquet: A dataframe containing the test set time, iy, ix, y_true, and y_prob.

metrics.json: Threshold-independent (AUC-ROC, AUC-PR, Brier) and threshold-optimal (Precision, Recall, F1) metrics.

model.json: The raw serialized XGBoost model.

feature_importance.json: Gain-based feature importance extracted from the trees.

A global summary_metrics.json is generated at the root of the output directory comparing performance across all lead times.

⚠️ Critical Limitations & Assumptions

When evaluating this baseline against other models, note the following built-in assumptions:

Topographic Oversimplification: The "elevation" feature is a linear proxy derived mathematically from coordinates (-400 × (lat - 31.5) + 600 × (lon - 34.8)). It is not a true Digital Elevation Model (DEM). This assumes storm behavior scales linearly with coordinate gradients, which ignores complex local orography.

Geographic Hardcoding: The script contains hardcoded coordinates for Israel/Lebanon coastlines and bounding boxes. It will require code refactoring to apply to other global regions.

Loss of Spatial Topology: Flattening grids into tabular features (even with neighborhood_sum) is inherently lossy. The model cannot learn complex spatial advection patterns (e.g., storm cell rotation, frontal boundaries) the way a Deep Learning vision model would.

Temporal Independence Assumption: XGBoost treats every timestep and grid cell as an independent IID observation. While the script implements a gap_threshold_minutes to prevent feature leakage across calendar gaps, it cannot learn continuous temporal dependencies (like an LSTM).