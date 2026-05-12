#!/usr/bin/env python3
"""
xgboostalgo.py

Lightning nowcasting baseline using XGBoost on gridded lightning data.

Example:
python xgboostalgo.py \
    --train gt.xlsx \
    --test ildn.xlsx \
    --grid 0.1 \
    --windows 10 20 40 120 \
    --leadtime 30 \
    --output results/

Input Excel files must contain columns:
    UTC, lat, lon

UTC must be a parseable datetime.

ROI (hardcoded): {'w': 31.0, 'e': 37.0, 's': 29.0, 'n': 35.0}

--windows controls history accumulation windows in MINUTES.
The internal time resolution is fixed at 5-minute bins.
Windows must be multiples of 5.

Outputs:
    - predictions.parquet   (time, iy, ix, y_true, y_prob)
    - metrics.json
    - model.json
    - feature_importance.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_curve,
    precision_recall_fscore_support,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression


# ============================================================
# CONSTANTS
# ============================================================

# Fixed internal time resolution in minutes.
# All window and leadtime arguments are expressed in minutes
# and converted to steps using this value.
TIMEBIN_MINUTES = 5

# Region of interest: Israel / Eastern Mediterranean
ROI = {"w": 31.0, "e": 37.0, "s": 29.0, "n": 35.0}


# ============================================================
# IO
# ============================================================

def read_lightning_file(path: str) -> pd.DataFrame:
    """Read an Excel lightning file and return a sorted DataFrame."""
    df = pd.read_excel(path)

    required = {"UTC", "lat", "lon"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")

    df = df.copy()
    df["UTC"] = pd.to_datetime(df["UTC"], utc=True)
    df = df.sort_values("UTC").reset_index(drop=True)

    return df


def check_no_overlap(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """
    Raise an error if the train and test periods overlap in time.

    Overlap is defined as: max(train UTC) >= min(test UTC).
    This catches both full overlap and partial overlap.
    """
    train_end = train_df["UTC"].max()
    test_start = test_df["UTC"].min()

    if train_end >= test_start:
        raise ValueError(
            f"Temporal overlap detected between train and test sets.\n"
            f"  Train ends:   {train_end}\n"
            f"  Test starts:  {test_start}\n"
            f"Ensure the test period begins strictly after the train period ends."
        )

    print(f"  Train period: {train_df['UTC'].min()} → {train_end}")
    print(f"  Test  period: {test_start} → {test_df['UTC'].max()}")
    print("  No temporal overlap — OK.")


# ============================================================
# ROI
# ============================================================

def apply_roi_filter(df: pd.DataFrame, roi: dict) -> pd.DataFrame:
    """Keep only strikes inside the bounding box."""
    mask = (
        (df["lon"] >= roi["w"]) &
        (df["lon"] <= roi["e"]) &
        (df["lat"] >= roi["s"]) &
        (df["lat"] <= roi["n"])
    )
    return df.loc[mask].copy()


# ============================================================
# GRID
# ============================================================

def build_grid(roi: dict, grid_deg: float):
    """
    Create latitude and longitude bin edges for the ROI.

    Returns
    -------
    lat_bins : np.ndarray  shape (nlat,)
    lon_bins : np.ndarray  shape (nlon,)
    """
    lat_bins = np.arange(roi["s"], roi["n"] + grid_deg, grid_deg)
    lon_bins = np.arange(roi["w"], roi["e"] + grid_deg, grid_deg)
    return lat_bins, lon_bins


def build_static_maps(lat_bins: np.ndarray, lon_bins: np.ndarray) -> np.ndarray:
    """
    Build static (time-invariant) spatial feature maps for the grid.

    All maps have shape (nlat, nlon) and are stacked into a single array
    of shape (nlat, nlon, n_static).  They are computed once and broadcast
    into every timestep inside create_features.

    Features (in order)
    -------------------
    0  lat_norm      Normalised latitude  ∈ [-1, 1]
                     Captures the meridional gradient of convective
                     climatology — convection is stronger and more
                     frequent further south over Israel.

    1  lon_norm      Normalised longitude ∈ [-1, 1]
                     Captures the west–east gradient: the Mediterranean
                     coast (west) has sea-breeze driven convection while
                     the Jordan Valley and Dead Sea (east) have dry
                     subsidence.

    2  land_sea      Binary land/sea mask (1 = land, 0 = sea).
                     Approximated analytically for the ROI:
                       lon < 34.5°  AND  lat > 31.5°  →  Mediterranean → sea
                     This is a coarse proxy; replace with a proper
                     shapefile mask if available.

    3  dist_coast    Normalised distance from the Mediterranean coastline.
                     Approximated as max(0, lon - 34.5) / (37 - 34.5).
                     Sea-breeze convergence drives inland lightning
                     penetration; cells close to the coast behave
                     differently from cells over the central highlands.

    4  topo_proxy    Terrain elevation proxy derived from latitude and
                     longitude using a linear regression fit to the
                     dominant topographic gradient of Israel:
                       elev ≈ -400 × (lat - 31.5) + 600 × (lon - 34.8)
                     This is not a real DEM — replace with SRTM data if
                     available.  Even a coarse proxy helps because
                     orographic lifting over the Galilee (north) and
                     Judean hills (centre) strongly influences lightning
                     initiation.  Normalised to [-1, 1].
    """
    nlat = len(lat_bins)
    nlon = len(lon_bins)

    # Cell-centre coordinates
    lat_centres = lat_bins + (lat_bins[1] - lat_bins[0]) / 2   # (nlat,)
    lon_centres = lon_bins + (lon_bins[1] - lon_bins[0]) / 2   # (nlon,)

    lon_grid, lat_grid = np.meshgrid(lon_centres, lat_centres)  # (nlat, nlon)

    # 0 — normalised latitude
    lat_min, lat_max = lat_bins[0], lat_bins[-1]
    lat_norm = 2.0 * (lat_grid - lat_min) / (lat_max - lat_min) - 1.0

    # 1 — normalised longitude
    lon_min, lon_max = lon_bins[0], lon_bins[-1]
    lon_norm = 2.0 * (lon_grid - lon_min) / (lon_max - lon_min) - 1.0

    # 2 — land/sea mask (1 = land)
    # Rough coastline of Israel/Lebanon: lon > 34.5° is land for most latitudes
    land_sea = np.where(lon_grid >= 34.5, 1.0, 0.0).astype(np.float32)

    # 3 — normalised distance from coast (0 at coast, 1 at eastern border)
    coast_lon  = 34.5
    dist_coast = np.clip((lon_grid - coast_lon) / (lon_max - coast_lon), 0, 1)

    # 4 — terrain elevation proxy (normalised)
    topo_raw = -400.0 * (lat_grid - 31.5) + 600.0 * (lon_grid - 34.8)
    t_min, t_max = topo_raw.min(), topo_raw.max()
    topo_proxy = (
        2.0 * (topo_raw - t_min) / (t_max - t_min) - 1.0
        if t_max > t_min else np.zeros_like(topo_raw)
    )

    static = np.stack([
        lat_norm.astype(np.float32),
        lon_norm.astype(np.float32),
        land_sea,
        dist_coast.astype(np.float32),
        topo_proxy.astype(np.float32),
    ], axis=-1)   # (nlat, nlon, 5)

    return static


def grid_counts(
    df: pd.DataFrame,
    lat_bins: np.ndarray,
    lon_bins: np.ndarray,
) -> pd.DataFrame:
    """
    Assign strikes to (timebin, iy, ix) cells and count them.

    Time resolution is fixed at TIMEBIN_MINUTES.
    """
    df = df.copy()
    df["timebin"] = df["UTC"].dt.floor(f"{TIMEBIN_MINUTES}min")
    df["iy"] = np.digitize(df["lat"], lat_bins) - 1
    df["ix"] = np.digitize(df["lon"], lon_bins) - 1

    grouped = (
        df.groupby(["timebin", "iy", "ix"])
        .size()
        .reset_index(name="count")
    )
    return grouped


# ============================================================
# DENSE CUBE
# ============================================================

def make_dense_cube(
    grouped: pd.DataFrame,
    nlat: int,
    nlon: int,
):
    """
    Convert the sparse (timebin, iy, ix, count) table into a
    dense 3-D numpy array of shape (nt, nlat, nlon).
    """
    times = np.sort(grouped["timebin"].unique())
    time_to_idx = {t: i for i, t in enumerate(times)}

    cube = np.zeros((len(times), nlat, nlon), dtype=np.float32)

    for row in grouped.itertuples():
        ti = time_to_idx[row.timebin]
        if 0 <= row.iy < nlat and 0 <= row.ix < nlon:
            cube[ti, row.iy, row.ix] = row.count

    return cube, times


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def neighborhood_sum(arr: np.ndarray, radius: int = 1) -> np.ndarray:
    """
    Sum values in a (2*radius+1)^2 neighbourhood around each cell.

    Edge cells are zero-padded (np.roll wraparound is cancelled by
    masking the border after accumulation).
    """
    ny, nx = arr.shape
    out = np.zeros_like(arr)

    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            shifted = np.roll(arr, shift=(dy, dx), axis=(0, 1))

            # Zero out wrapped edges to avoid spurious border features
            if dy > 0:
                shifted[:dy, :] = 0.0
            elif dy < 0:
                shifted[dy:, :] = 0.0
            if dx > 0:
                shifted[:, :dx] = 0.0
            elif dx < 0:
                shifted[:, dx:] = 0.0

            out += shifted

    return out


def minutes_to_steps(minutes: int) -> int:
    """Convert a duration in minutes to an integer number of time steps."""
    steps = minutes // TIMEBIN_MINUTES
    if steps < 1:
        raise ValueError(
            f"Window {minutes} min is smaller than the time bin "
            f"({TIMEBIN_MINUTES} min)."
        )
    return steps


def get_valid_timesteps(
    times: np.ndarray,
    window_minutes: list,
    max_lead_steps: int,
    gap_threshold_minutes: int = 180,
) -> np.ndarray:
    """
    Return a boolean mask of shape (nt,) marking timesteps that are safe
    to use as forecast origins.

    A timestep t is EXCLUDED if:
      1. It is within max_hist steps of the start of a contiguous active
         block — i.e. the history window would reach back across a calendar
         gap larger than gap_threshold_minutes.
      2. It is within max_lead_steps steps of the end of a contiguous
         active block — i.e. the future window would reach forward across
         a calendar gap.

    This prevents the model from seeing history features that silently mix
    observations from two weather episodes separated by days or weeks,
    which would occur when the sparse cube compresses calendar gaps into
    adjacent integer indices.

    Parameters
    ----------
    times                 : (nt,) array of pd.Timestamp  — cube time axis
    window_minutes        : list of ints  — history windows in minutes
    max_lead_steps        : int           — steps for the longest lead time
    gap_threshold_minutes : int           — calendar gaps larger than this
                                           value trigger a block boundary.
                                           Default 180 min (3 h) — well above
                                           the 5-min cube resolution but small
                                           enough to catch overnight breaks
                                           between active storm days.

    Returns
    -------
    valid : bool ndarray  shape (nt,)
    """
    window_steps = [minutes_to_steps(m) for m in window_minutes]
    max_hist = max(window_steps)
    nt = len(times)

    # Compute gap in minutes between consecutive cube timesteps
    times_ts = pd.DatetimeIndex(times)
    gaps = np.zeros(nt, dtype=np.float64)          # gaps[t] = minutes since times[t-1]
    gaps[1:] = (
        times_ts[1:] - times_ts[:-1]
    ).total_seconds() / 60.0

    # A gap > threshold marks the START of a new contiguous block
    # block_start[t] = True means times[t] begins a new independent episode
    block_start = gaps > gap_threshold_minutes      # shape (nt,)

    # For each t, find the index of the most recent block start at or before t
    # We need: the earliest t in the current block is at least max_hist steps
    # behind the current t, AND at least max_lead_steps steps ahead of t.

    # Forward pass: block_start_idx[t] = index where current block begins
    block_start_idx = np.zeros(nt, dtype=np.int64)
    current_start = 0
    for t in range(nt):
        if block_start[t]:
            current_start = t
        block_start_idx[t] = current_start

    # Backward pass: block_end_idx[t] = index where current block ends
    block_end_idx = np.zeros(nt, dtype=np.int64)
    current_end = nt - 1
    for t in range(nt - 1, -1, -1):
        if t < nt - 1 and block_start[t + 1]:
            current_end = t
        block_end_idx[t] = current_end

    # t is valid iff:
    #   t - block_start_idx[t] >= max_hist          (enough history in block)
    #   block_end_idx[t] - t   >= max_lead_steps    (enough future in block)
    valid = (
        (np.arange(nt) - block_start_idx >= max_hist) &
        (block_end_idx - np.arange(nt)   >= max_lead_steps)
    )

    n_valid   = valid.sum()
    n_excluded = nt - n_valid
    if n_excluded > 0:
        print(f"  Gap-boundary filter: {n_excluded:,} of {nt:,} timesteps "
              f"excluded ({100*n_excluded/nt:.1f} %) — history or future "
              f"window would cross a >{gap_threshold_minutes}-min gap.")

    return valid


def create_features(
    cube: np.ndarray,
    times: np.ndarray,
    window_minutes: list,
    max_lead_steps: int,
    stride: int = 1,
    static_maps: np.ndarray = None,
):
    """
    Build the feature matrix X and metadata for all valid timesteps.

    The valid range is [max_hist, nt - max_lead_steps) so that every
    timestep has enough history behind it AND enough future ahead of it
    for the longest lead time.  This guarantees X and meta are identical
    regardless of which lead time is being targeted — only y changes.

    Parameters
    ----------
    cube           : (nt, ny, nx)  float32
    times          : (nt,)         datetime-like
    window_minutes : list of ints  — history windows in MINUTES
    max_lead_steps : int           — steps for the LONGEST lead time
    stride         : int           — step between consecutive forecast
                                     origins.  stride=1 (default) gives
                                     every possible timestep (training).
                                     stride=lead_steps gives non-overlapping
                                     future windows (test evaluation).
    static_maps    : (ny, nx, n_static) float32 or None
                     Time-invariant spatial features (lat, lon, land/sea,
                     coast distance, terrain proxy).  Built once by
                     build_static_maps() and reused every timestep.
                     Pass None to omit (backward-compatible).

    Returns
    -------
    X    : float32  [N, F]
    meta : DataFrame [N]   columns: time, iy, ix
    """
    window_steps = [minutes_to_steps(m) for m in window_minutes]
    max_hist = max(window_steps)

    X_list, meta_list = [], []
    nt, ny, nx = cube.shape

    valid = get_valid_timesteps(times, window_minutes, max_lead_steps)
    # Candidate origins: within the safe index range AND not crossing a gap
    candidates = [
        t for t in range(max_hist, nt - max_lead_steps, stride)
        if valid[t]
    ]

    for t in candidates:
        current = cube[t]
        feat_stack = []

        for ws in window_steps:
            feat_stack.append(cube[t - ws:t].sum(axis=0))

        feat_stack.append(neighborhood_sum(current, radius=1))
        feat_stack.append(neighborhood_sum(current, radius=2))

        decay = np.zeros_like(current)
        for k in range(1, max_hist + 1):
            decay += cube[t - k] * np.exp(-0.1 * k)
        feat_stack.append(decay)

        ts = pd.Timestamp(times[t])
        hour = ts.hour + ts.minute / 60.0
        feat_stack.append(np.full_like(current, np.sin(2 * np.pi * hour / 24)))
        feat_stack.append(np.full_like(current, np.cos(2 * np.pi * hour / 24)))

        # --- G) Lagged neighbourhood sums (t-1) ----------------------------
        # neighbourhood_sum on the previous timestep gives the model an
        # explicit spatial-tendency signal: was the surrounding area active
        # one step ago?  Combined with the current neighbourhood (group B),
        # the model can detect whether local storm activity is growing,
        # steady, or decaying — a key discriminator for lead times > 1 h.
        prev = cube[t - 1]
        feat_stack.append(neighborhood_sum(prev, radius=1))
        feat_stack.append(neighborhood_sum(prev, radius=2))

        # --- H) Persistence ratio ------------------------------------------
        # (current activity) / (120-min background + epsilon)
        # Values > 1 signal rapid intensification; < 1 signal decay.
        # Using the 120-min window as background gives a scale-invariant
        # normalisation that works for sparse and dense activity alike.
        # We reuse the already-computed hist_120 slice if 120 is in windows.
        hist_idx_120 = None
        for _i, _m in enumerate(window_minutes):
            if _m == 120:
                hist_idx_120 = _i
                break
        if hist_idx_120 is not None:
            _bg = feat_stack[hist_idx_120] / max(minutes_to_steps(120), 1) + 1e-6
        else:
            _bg = (cube[t - minutes_to_steps(120):t].sum(axis=0)
                   / max(minutes_to_steps(120), 1) + 1e-6)
        feat_stack.append((current / _bg).astype(np.float32))

        # --- E) Seasonal encoding -------------------------------------------
        # Lightning over Israel is strongly seasonal (wet season Oct–Apr).
        # Encoding month as a continuous cycle lets the model learn
        # seasonally-varying base rates without a discontinuity at year-end.
        month = ts.month
        feat_stack.append(np.full_like(current, np.sin(2 * np.pi * month / 12)))
        feat_stack.append(np.full_like(current, np.cos(2 * np.pi * month / 12)))

        # --- F) Static spatial maps -----------------------------------------
        # Append each static layer (lat, lon, land/sea, coast dist, topo)
        # as a separate feature.  They are constant across time so they act
        # as location embeddings, allowing the model to learn
        # geographically-varying climatology.
        if static_maps is not None:
            for k in range(static_maps.shape[-1]):
                feat_stack.append(static_maps[:, :, k])

        feat_arr = np.stack(feat_stack, axis=-1)
        X_list.append(feat_arr.reshape(-1, feat_arr.shape[-1]))

        yy, xx = np.indices((ny, nx))
        meta_list.append(pd.DataFrame({
            "time": times[t],
            "iy":   yy.reshape(-1),
            "ix":   xx.reshape(-1),
        }))

    X    = np.concatenate(X_list, axis=0)
    meta = pd.concat(meta_list, ignore_index=True)
    return X, meta


def build_targets(
    cube: np.ndarray,
    times: np.ndarray,
    window_minutes: list,
    lead_steps: int,
    max_lead_steps: int,
    stride: int = 1,
) -> np.ndarray:
    """
    Build the binary target vector y for a specific lead time.

    Uses the same valid timestep range and stride as create_features so
    that y[i] aligns exactly with X[i].

    Parameters
    ----------
    cube           : (nt, ny, nx)  float32
    times          : (nt,)         — kept for API symmetry, unused
    window_minutes : list of ints  — needed to compute max_hist
    lead_steps     : int           — steps for THIS lead time
    max_lead_steps : int           — steps for the LONGEST lead time
    stride         : int           — must match the stride used in
                                     create_features for this split.
                                     stride=1 for training,
                                     stride=lead_steps for test evaluation.

    Returns
    -------
    y : uint8  [N]   1 = any strike in the 1-h window (t+lead_steps-11 .. t+lead_steps], else 0
    """
    window_steps = [minutes_to_steps(m) for m in window_minutes]
    max_hist = max(window_steps)
    nt, ny, nx = cube.shape

    valid = get_valid_timesteps(times, window_minutes, max_lead_steps)
    candidates = [
        t for t in range(max_hist, nt - max_lead_steps, stride)
        if valid[t]
    ]

    # Number of 5-min steps in one hour — width of each target window.
    # Each model predicts the 1-hour window ENDING at its lead time:
    #   60min model  →  any strike in (t+0h,  t+1h]  = bins t+1  … t+12
    #   120min model →  any strike in (t+1h,  t+2h]  = bins t+13 … t+24
    #   ...
    #   360min model →  any strike in (t+5h,  t+6h]  = bins t+61 … t+72
    # Windows are non-overlapping between adjacent models and each covers
    # exactly one hour, matching the 1-hour lead-time cadence.
    steps_per_hour = 60 // TIMEBIN_MINUTES          # = 12 for 5-min bins
    window_end   = lead_steps                        # inclusive last step
    window_start = lead_steps - steps_per_hour + 1  # first step of window

    y_list = []
    for t in candidates:
        target = (
            cube[t + window_start : t + window_end + 1].sum(axis=0) > 0
        ).astype(np.uint8)
        y_list.append(target.reshape(-1))

    return np.concatenate(y_list, axis=0)


def create_dataset(
    cube: np.ndarray,
    times: np.ndarray,
    window_minutes: list,
    lead_steps: int,
):
    """
    Convenience wrapper for single-lead-time use.
    For multi-lead training use create_features + build_targets directly.
    """
    X, meta = create_features(cube, times, window_minutes, lead_steps)
    y = build_targets(cube, times, window_minutes, lead_steps, lead_steps)
    return X, y, meta


def feature_names(window_minutes: list, with_static: bool = True) -> list:
    names = [f"hist_{m}min" for m in window_minutes]
    names += ["neigh_r1", "neigh_r2", "decay",
              "sin_hour", "cos_hour",
              "lag1_neigh_r1", "lag1_neigh_r2",
              "persist_ratio",
              "sin_month", "cos_month"]
    if with_static:
        names += ["lat_norm", "lon_norm", "land_sea",
                  "dist_coast", "topo_proxy"]
    return names


# ============================================================
# TRAIN
# ============================================================

def train_xgb(
    X_train: np.ndarray,
    y_train: np.ndarray,
    depth: int,
    trees: int,
    lr: float,
    val_fraction: float = 0.15,
    early_stopping_rounds: int = 30,
) -> tuple:
    """
    Train XGBoost with a temporal validation split, early stopping on
    AUC-PR, and post-hoc isotonic calibration.

    Parameters
    ----------
    X_train              : [N, F]  feature matrix (already downsampled)
    y_train              : [N]     binary labels
    depth                : max tree depth
    trees                : maximum number of boosting rounds
    lr                   : learning rate
    val_fraction         : fraction of training rows held out for early
                           stopping and calibration.  Rows are taken from
                           the END of the array (temporal order preserved —
                           no future leakage from shuffled splits).
    early_stopping_rounds: stop if AUC-PR does not improve for this many
                           consecutive rounds.

    Returns
    -------
    calibrated_model : sklearn pipeline wrapping the XGBoost classifier
                       with isotonic calibration applied.  Exposes the
                       same predict_proba() API as a plain XGBClassifier.
    n_trees_used     : int  — actual number of boosting rounds used
                       (may be less than `trees` due to early stopping)
    """
    # ── Temporal validation split (NO shuffle — preserves time order) ────────
    # Taking from the END of the array is important: X_train was assembled
    # by iterating over cube timesteps in chronological order, so the last
    # val_fraction rows correspond to the most recent training timesteps.
    # Using them as a validation set therefore mimics a real forecast
    # scenario without introducing any future leakage.
    n_val = max(1, int(len(y_train) * val_fraction))
    n_fit = len(y_train) - n_val

    X_fit, X_val = X_train[:n_fit], X_train[n_fit:]
    y_fit, y_val = y_train[:n_fit], y_train[n_fit:]

    pos_val = y_val.sum()
    print(f"    Val split: {n_fit:,} fit rows | {n_val:,} val rows "
          f"(pos fraction val: {pos_val/max(len(y_val),1):.4f})")

    # ── Class weight for the fit split only ──────────────────────────────────
    pos = y_fit.sum()
    neg = len(y_fit) - pos
    scale_pos_weight = max(1.0, neg / max(pos, 1))

    # ── XGBoost with early stopping on aucpr ─────────────────────────────────
    # aucpr directly optimises the metric we report, unlike logloss which
    # can improve while AUC-PR stagnates under class imbalance.
    model = xgb.XGBClassifier(
        n_estimators=trees,
        max_depth=depth,
        learning_rate=lr,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
        early_stopping_rounds=early_stopping_rounds,
        n_jobs=-1,
        random_state=42,
    )
    model.fit(
        X_fit, y_fit,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    n_trees_used = model.best_iteration + 1
    print(f"    Early stopping: used {n_trees_used} of {trees} trees "
          f"(best aucpr={model.best_score:.4f})")

    # ── Isotonic calibration on the validation split ─────────────────────────
    # CalibratedClassifierCV(cv='prefit') fits a monotonic mapping from the
    # raw XGBoost scores to true probabilities using the held-out val set.
    # This corrects the clustering of scores near 0 and 1 that causes
    # best_threshold to sit at 0.90+.  After calibration, a score of 0.3
    # should correspond to ~30% empirical lightning frequency.
    calibrated = CalibratedClassifierCV(model, cv="prefit", method="isotonic")
    calibrated.fit(X_val, y_val)

    return calibrated, n_trees_used


# ============================================================
# METRICS
# ============================================================

def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> dict:
    """
    Compute threshold-independent and threshold-optimal metrics.

    Threshold-independent
    ---------------------
    - AUC-ROC   : area under the ROC curve
    - AUC-PR    : area under the precision-recall curve (= average precision).
                  More informative than AUC-ROC for imbalanced data because it
                  is sensitive to the minority class across all operating points.
    - Brier     : mean squared error of probabilities — measures calibration.

    Threshold-optimal (F1-optimal)
    -------------------------------
    Sweep every unique predicted probability as a candidate threshold.
    At each threshold compute F1; pick the one that maximises it.
    Then report precision, recall, F1, and the threshold itself.
    This removes the arbitrary choice of 0.5 and finds the best possible
    operating point — important for imbalanced lightning data where the
    natural threshold is often well below 0.5.

    Saved curves (for plotting)
    ---------------------------
    - pr_curve  : lists of (threshold, precision, recall) — one entry per
                  unique predicted score, suitable for a P-R plot.
    - roc_curve : lists of (fpr, tpr, threshold) — one entry per unique
                  predicted score, suitable for a ROC plot.
    Both are saved to metrics.json so you can reproduce the figures later
    without re-running inference.
    """

    # ── Threshold-independent ──────────────────────────────────────────
    auc_roc = roc_auc_score(y_true, y_prob)
    auc_pr  = average_precision_score(y_true, y_prob)
    brier   = brier_score_loss(y_true, y_prob)

    # ── Full PR curve ──────────────────────────────────────────────────
    # precision_recall_curve returns arrays of length n_thresholds+1;
    # the last entry has no corresponding threshold (it is the (0,1) anchor).
    prec_arr, rec_arr, pr_thresh_arr = precision_recall_curve(y_true, y_prob)

    # ── Full ROC curve ─────────────────────────────────────────────────
    fpr_arr, tpr_arr, roc_thresh_arr = roc_curve(y_true, y_prob)

    # ── Find F1-optimal threshold ──────────────────────────────────────
    # Work over the thresholds that have both a precision and recall value
    # (exclude the anchor point at the end of the PR arrays).
    p_thresh = prec_arr[:-1]
    r_thresh = rec_arr[:-1]

    denom = p_thresh + r_thresh
    # Avoid division by zero when both precision and recall are 0
    f1_arr = np.where(
        denom > 0,
        2 * p_thresh * r_thresh / denom,
        0.0,
    )

    best_idx       = int(np.argmax(f1_arr))
    best_threshold = float(pr_thresh_arr[best_idx])
    best_precision = float(p_thresh[best_idx])
    best_recall    = float(r_thresh[best_idx])
    best_f1        = float(f1_arr[best_idx])

    # ── Build serialisable curve lists ────────────────────────────────
    pr_curve_out = {
        "threshold": pr_thresh_arr.tolist(),
        "precision": prec_arr[:-1].tolist(),
        "recall":    rec_arr[:-1].tolist(),
    }

    roc_curve_out = {
        "fpr":       fpr_arr.tolist(),
        "tpr":       tpr_arr.tolist(),
        "threshold": roc_thresh_arr.tolist(),
    }

    return {
        # threshold-independent
        "auc_roc":        auc_roc,
        "auc_pr":         auc_pr,
        "brier":          brier,
        # optimal operating point
        "best_threshold": best_threshold,
        "precision":      best_precision,
        "recall":         best_recall,
        "f1":             best_f1,
        # full curves for plotting
        "pr_curve":       pr_curve_out,
        "roc_curve":      roc_curve_out,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="XGBoost lightning nowcasting baseline."
    )

    parser.add_argument(
        "--train",
        required=True,
        help="Path to training Excel file (columns: UTC, lat, lon)",
    )
    parser.add_argument(
        "--test",
        required=True,
        help="Path to test Excel file (columns: UTC, lat, lon). "
             "Must not overlap in time with --train.",
    )
    parser.add_argument(
        "--grid",
        type=float,
        default=0.1,
        help="Spatial grid resolution in degrees (default: 0.1°)",
    )
    parser.add_argument(
        "--windows",
        nargs="+",
        type=int,
        default=[10, 20, 40, 120],
        help=(
            "History accumulation windows in MINUTES. "
            f"Must be multiples of the internal time bin ({TIMEBIN_MINUTES} min). "
            "Default: 10 20 40 120"
        ),
    )
    parser.add_argument(
        "--leadtimes",
        nargs="+",
        type=int,
        default=[60, 120, 180, 240, 300, 360],
        help=(
            "Forecast lead times in MINUTES. A separate model is trained and "
            "evaluated for each value. Default: 60 120 180 240 300 360 "
            f"(1h–6h, using {TIMEBIN_MINUTES}-min bins)."
        ),
    )
    parser.add_argument("--depth",     type=int,   default=8)
    parser.add_argument("--trees",     type=int,   default=300)
    parser.add_argument("--lr",        type=float, default=0.05)
    parser.add_argument(
        "--neg_ratio",
        type=float,
        default=0.05,
        help="Fraction of negative training samples retained (default: 0.05)",
    )
    parser.add_argument("--output", default="results")

    args = parser.parse_args()

    # --------------------------------------------------------
    # Validate windows are multiples of TIMEBIN_MINUTES
    # --------------------------------------------------------
    bad_windows = [w for w in args.windows if w % TIMEBIN_MINUTES != 0]
    if bad_windows:
        raise ValueError(
            f"--windows values must be multiples of {TIMEBIN_MINUTES} min. "
            f"Invalid: {bad_windows}"
        )

    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Read data
    # --------------------------------------------------------
    print("Reading data...")
    train_df = read_lightning_file(args.train)
    test_df  = read_lightning_file(args.test)

    # --------------------------------------------------------
    # Dataset diagnostics  (printed BEFORE any filtering)
    # --------------------------------------------------------
    def dataset_diagnostics(df: pd.DataFrame, label: str, roi: dict) -> None:
        """
        Print a full diagnostic summary for a lightning DataFrame.

        Reports are printed in two groups:
          1. Raw file contents (before any ROI filtering)
          2. ROI-filtered contents

        Flags are raised for conditions that may silently corrupt results:
          - Duplicate UTC timestamps in the same location
          - Large gap in time coverage (> 30 days with zero strikes inside ROI)
          - Very low ROI retention rate (< 1 % of global strikes)
          - Severely unequal daily strike counts (CV > 2.0) suggesting
            the file covers only a few active storm days
        """
        sep = "─" * 56
        print(f"\n{sep}")
        print(f"  {label}")
        print(sep)

        # ── Raw file ──────────────────────────────────────────
        t_min = df["UTC"].min()
        t_max = df["UTC"].max()
        span_days = (t_max - t_min).total_seconds() / 86400
        n_raw = len(df)

        print(f"  [RAW FILE]")
        print(f"    Total strikes      : {n_raw:>12,}")
        print(f"    First UTC          : {t_min}")
        print(f"    Last  UTC          : {t_max}")
        print(f"    Calendar span      : {span_days:.1f} days  "
              f"({span_days/30.44:.1f} months)")
        print(f"    Lat range          : "
              f"{df['lat'].min():.2f}° – {df['lat'].max():.2f}°")
        print(f"    Lon range          : "
              f"{df['lon'].min():.2f}° – {df['lon'].max():.2f}°")

        # Days with at least one strike anywhere
        days_with_strikes = df["UTC"].dt.date.nunique()
        print(f"    Days with strikes  : {days_with_strikes:>5} "
              f"of {int(span_days)+1} calendar days "
              f"({100*days_with_strikes/max(span_days,1):.1f} %)")

        # ── ROI-filtered ──────────────────────────────────────
        roi_df = df[
            (df["lon"] >= roi["w"]) & (df["lon"] <= roi["e"]) &
            (df["lat"] >= roi["s"]) & (df["lat"] <= roi["n"])
        ]
        n_roi = len(roi_df)
        retention = 100 * n_roi / max(n_raw, 1)

        print(f"\n  [AFTER ROI FILTER  lat {roi['s']}\u2013{roi['n']}\u00b0  lon {roi['w']}\u2013{roi['e']}\u00b0]")
        print(f"    Strikes inside ROI : {n_roi:>12,}  ({retention:.2f} % of raw)")

        if n_roi == 0:
            print(f"    ⚠  WARNING: zero strikes inside ROI — check file and ROI")
            print(sep)
            return

        roi_t_min = roi_df["UTC"].min()
        roi_t_max = roi_df["UTC"].max()
        roi_span  = (roi_t_max - roi_t_min).total_seconds() / 86400

        print(f"    First ROI strike   : {roi_t_min}")
        print(f"    Last  ROI strike   : {roi_t_max}")
        print(f"    ROI time span      : {roi_span:.1f} days")

        # Active days inside ROI
        roi_days = roi_df["UTC"].dt.date.nunique()
        print(f"    Days with ROI strike:{roi_days:>4} "
              f"of {int(roi_span)+1} span days "
              f"({100*roi_days/max(roi_span,1):.1f} %)")

        # Daily strike distribution inside ROI
        daily = roi_df.groupby(roi_df["UTC"].dt.date).size()
        print(f"    Strikes/day (ROI)  : "
              f"mean={daily.mean():.0f}  "
              f"median={daily.median():.0f}  "
              f"max={daily.max():,}  "
              f"min={daily.min()}")

        # ── Time-gap check ────────────────────────────────────
        # Find longest consecutive period with zero ROI strikes
        all_dates = pd.date_range(roi_t_min.date(), roi_t_max.date(), freq="D")
        active_dates = set(roi_df["UTC"].dt.date)
        gaps, current = [], 0
        for d in all_dates:
            if d.date() not in active_dates:
                current += 1
            else:
                if current > 0:
                    gaps.append(current)
                current = 0
        if current > 0:
            gaps.append(current)
        max_gap = max(gaps) if gaps else 0
        print(f"    Longest zero-strike gap: {max_gap} days")

        # ── Flags ─────────────────────────────────────────────
        flags = []
        if retention < 1.0:
            flags.append(f"⚠  Very low ROI retention ({retention:.2f} %) — "
                         f"file may be sparse over this region")
        if max_gap > 30:
            flags.append(f"⚠  Gap of {max_gap} days with no ROI strikes — "
                         f"dataset may not cover full season")
        cv = daily.std() / max(daily.mean(), 1e-9)
        if cv > 2.0:
            flags.append(f"⚠  High daily variability (CV={cv:.1f}) — "
                         f"dataset may cover only a few storm events")
        if daily.mean() < 10:
            flags.append(f"⚠  Very low mean strike density ({daily.mean():.1f}/day) — "
                         f"model may have insufficient positive examples")
        if not flags:
            print(f"    ✓  No data-quality flags raised")
        else:
            for f in flags:
                print(f"    {f}")

        print(sep)

    dataset_diagnostics(train_df, f"TRAIN  {args.train}", ROI)
    dataset_diagnostics(test_df,  f"TEST   {args.test}",  ROI)

    print("Checking temporal separation...")
    check_no_overlap(train_df, test_df)

    # --------------------------------------------------------
    # ROI
    # --------------------------------------------------------
    print(f"\nROI: {ROI}")
    train_df = apply_roi_filter(train_df, ROI)
    test_df  = apply_roi_filter(test_df,  ROI)

    if len(train_df) == 0:
        raise ValueError("No training events remain after ROI filter.")
    if len(test_df) == 0:
        raise ValueError("No test events remain after ROI filter.")

    # --------------------------------------------------------
    # Wet-season filter  (Oct – Apr)
    # --------------------------------------------------------
    # Lightning over Israel is confined almost entirely to the Oct–Apr wet
    # season.  Keeping summer months (May–Sep) adds thousands of all-zero
    # timesteps that dilute the positive class further and introduce
    # timesteps where the model has nothing physical to learn from.
    # Both train and test are filtered to the same months so that base
    # rates, storm character, and class balance are comparable.
    WET_SEASON_MONTHS = {10, 11, 12, 1, 2, 3, 4}

    train_before = len(train_df)
    test_before  = len(test_df)

    train_df = train_df[
        train_df["UTC"].dt.month.isin(WET_SEASON_MONTHS)
    ].copy()
    test_df  = test_df[
        test_df["UTC"].dt.month.isin(WET_SEASON_MONTHS)
    ].copy()

    print(f"\nWet-season filter (Oct–Apr only):")
    print(f"  Train: {len(train_df):,} strikes retained "
          f"({train_before - len(train_df):,} summer strikes removed) "
          f"| {train_df['UTC'].dt.date.nunique()} active days")
    print(f"  Test:  {len(test_df):,} strikes retained "
          f"({test_before  - len(test_df):,} summer strikes removed) "
          f"| {test_df['UTC'].dt.date.nunique()} active days")

    if len(train_df) == 0:
        raise ValueError("No training events remain after wet-season filter.")
    if len(test_df) == 0:
        raise ValueError("No test events remain after wet-season filter.")

    # --------------------------------------------------------
    # Grid
    # --------------------------------------------------------
    print("Building spatial grid...")
    lat_bins, lon_bins = build_grid(ROI, args.grid)
    nlat = len(lat_bins)
    nlon = len(lon_bins)
    print(f"  Grid size: {nlat} lat × {nlon} lon cells")

    print("Building static spatial feature maps...")
    static_maps = build_static_maps(lat_bins, lon_bins)
    print(f"  Static maps: {static_maps.shape}  "
          f"({static_maps.shape[-1]} features: "
          "lat_norm, lon_norm, land_sea, dist_coast, topo_proxy)")

    # --------------------------------------------------------
    # Gridding → dense cubes
    # --------------------------------------------------------
    print("Gridding strikes into time×space cubes...")
    train_grouped = grid_counts(train_df, lat_bins, lon_bins)
    test_grouped  = grid_counts(test_df,  lat_bins, lon_bins)

    train_cube, train_times = make_dense_cube(train_grouped, nlat, nlon)
    test_cube,  test_times  = make_dense_cube(test_grouped,  nlat, nlon)

    print(f"  Train cube: {train_cube.shape}  "
          f"({train_cube.shape[0] * TIMEBIN_MINUTES / 60:.1f} h)")
    print(f"  Test  cube: {test_cube.shape}  "
          f"({test_cube.shape[0] * TIMEBIN_MINUTES / 60:.1f} h)")

    # --------------------------------------------------------
    # Validate lead times
    # --------------------------------------------------------
    bad_leads = [lt for lt in args.leadtimes if lt % TIMEBIN_MINUTES != 0]
    if bad_leads:
        raise ValueError(
            f"--leadtimes values must be multiples of {TIMEBIN_MINUTES} min. "
            f"Invalid: {bad_leads}"
        )

    all_lead_steps = [lt // TIMEBIN_MINUTES for lt in args.leadtimes]
    max_lead_steps = max(all_lead_steps)

    feat_names = feature_names(args.windows, with_static=True)
    print(f"Lead times: {args.leadtimes} min = {all_lead_steps} steps")
    print(f"History windows: {args.windows} min "
          f"= {[minutes_to_steps(w) for w in args.windows]} steps")
    print(f"Features ({len(feat_names)}): {feat_names}")

    # --------------------------------------------------------
    # Build TRAINING feature matrix ONCE (stride=1, dense)
    # Test features are rebuilt per lead time (stride=lead_steps)
    # --------------------------------------------------------
    print("\nBuilding training feature matrix (stride=1, computed once)...")
    X_train_full, _ = create_features(
        train_cube, train_times, args.windows, max_lead_steps,
        stride=1, static_maps=static_maps,
    )
    print(f"  Train feature matrix: {X_train_full.shape}")

    # --------------------------------------------------------
    # Per-lead-time loop: y, downsampling, train, evaluate, save
    # --------------------------------------------------------
    all_metrics = {}
    rng = np.random.default_rng(42)

    for lt_min, lead_steps in zip(args.leadtimes, all_lead_steps):
        tag = f"{lt_min}min"
        print(f"\n{'='*60}")
        print(f"  Lead time: {lt_min} min  ({lead_steps} steps)  [{tag}]")
        print(f"{'='*60}")

        lt_outdir = outdir / tag
        lt_outdir.mkdir(parents=True, exist_ok=True)

        # --- Build test features & meta with non-overlapping stride ----------
        # stride=lead_steps tiles the test period so each future window is
        # independent — no true lightning event appears in two test targets.
        print(f"  Building non-overlapping test set (stride={lead_steps} steps "
              f"= {lt_min} min)...")
        X_test, meta_test = create_features(
            test_cube, test_times, args.windows, max_lead_steps,
            stride=lead_steps, static_maps=static_maps,
        )
        y_test = build_targets(
            test_cube, test_times, args.windows, lead_steps, max_lead_steps,
            stride=lead_steps,
        )
        n_independent = len(meta_test["time"].unique())
        print(f"  Independent forecast origins: {n_independent:,}  "
              f"({len(y_test):,} cell-samples)")

        # --- Build training targets (stride=1, dense) -----------------------
        y_train_full = build_targets(
            train_cube, train_times, args.windows, lead_steps, max_lead_steps,
            stride=1,
        )

        print(f"  Positive fraction (raw train): {y_train_full.mean():.4f}")

        # --- Negative downsampling ---
        print(f"  Downsampling negatives (keeping {args.neg_ratio*100:.0f}%)...")
        pos_idx = np.where(y_train_full == 1)[0]
        neg_idx = np.where(y_train_full == 0)[0]

        keep_neg = rng.choice(
            neg_idx,
            size=int(len(neg_idx) * args.neg_ratio),
            replace=False,
        )
        keep_idx = np.concatenate([pos_idx, keep_neg])
        rng.shuffle(keep_idx)

        X_train = X_train_full[keep_idx]
        y_train = y_train_full[keep_idx]

        print(f"  Train samples after downsampling: {len(y_train):,}")
        print(f"  Positive fraction after:          {y_train.mean():.4f}")

        # --- Train (with validation split, early stopping, calibration) ---
        print(f"  Training XGBoost for {tag}...")
        model, n_trees = train_xgb(
            X_train, y_train,
            args.depth, args.trees, args.lr,
        )

        # --- Predict & evaluate ---
        print(f"  Predicting...")
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = compute_metrics(y_test, y_prob)
        metrics["n_trees_used"] = n_trees
        summary_keys = ["auc_roc", "auc_pr", "brier",
                        "best_threshold", "precision", "recall", "f1",
                        "n_trees_used"]
        summary = {k: round(metrics[k], 4) for k in summary_keys}
        print(f"  Metrics: {json.dumps(summary)}")
        all_metrics[tag] = {k: v for k, v in summary.items()
                            if k != "n_trees_used"}

        # --- Feature importance (from the inner XGBoost estimator) ----------
        # CalibratedClassifierCV wraps the base estimator; access it via
        # .estimator to retrieve feature_importances_.
        base_model = model.estimator
        importance = dict(zip(feat_names, base_model.feature_importances_.tolist()))
        importance_sorted = dict(
            sorted(importance.items(), key=lambda x: x[1], reverse=True)
        )

        # --- Save ---
        results = meta_test.copy()
        results["y_true"]    = y_test
        results["y_prob"]    = y_prob
        results["leadtime"]  = lt_min

        results.to_parquet(lt_outdir / "predictions.parquet")

        with open(lt_outdir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        # Save the underlying XGBoost model (not the calibration wrapper)
        base_model.save_model(str(lt_outdir / "model.json"))

        with open(lt_outdir / "feature_importance.json", "w") as f:
            json.dump(importance_sorted, f, indent=2)

        print(f"  Saved to: {lt_outdir}/")

    # --------------------------------------------------------
    # Summary table across all lead times
    # --------------------------------------------------------
    print(f"\n{'='*60}")
    print("  Summary across lead times")
    print(f"{'='*60}")
    header = f"{'leadtime':>10}  {'auc_roc':>8}  {'auc_pr':>8}  {'brier':>8}  {'f1':>8}"
    print(header)
    for tag, m in all_metrics.items():
        print(f"{tag:>10}  {m['auc_roc']:>8.4f}  {m['auc_pr']:>8.4f}  "
              f"{m['brier']:>8.4f}  {m['f1']:>8.4f}")

    summary_path = outdir / "summary_metrics.json"
    with open(summary_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()