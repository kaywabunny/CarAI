
# price_helper.py
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
# --- price_helper.py (append) ---
from io import BytesIO
import matplotlib.pyplot as plt

def render_price_chart(bands: dict, title: str | None = None) -> bytes:
    """Render a simple bar chart (green/yellow/red) and return PNG bytes."""
    labels = ["Green (sell fast)", "Yellow (median)", "Red (hold out)"]
    vals = [bands["green_median"], bands["yellow"], bands["red_median"]]

    fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=140)
    bars = ax.bar(labels, vals)  # no explicit colors, default palette
    ax.set_title(title or "Price Bands (THB)")
    ax.set_ylabel("THB")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    # Value labels on top of bars
    for b in bars:
        y = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, y, f"{y:,.0f}",
                ha="center", va="bottom", fontsize=9)

    # Confidence (if present)
    conf = bands.get("confidence")
    if conf is not None:
        ax.text(0.99, 0.02, f"Confidence: {conf:.2f}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=9)

    fig.tight_layout()

    # Save as PNG bytes
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()

# --- where the promoted model lives ---
MODELS_DIR = Path("models/price_quantiles_v3")

# --- globals loaded at import (safe no-op if missing) ---
_q20 = _q50 = _q80 = None
_gmed_df: pd.DataFrame | None = None
_FEATURES: list[str] = []
_CAT: list[str] = []
_NUM: list[str] = []
_BLEND_W: float = 0.30  # default if not present in feature_config
_SCALE: float = 1.0     # reserved (not used)

def _std_cat(x: str) -> str:
    """Normalize categorical text for inference."""
    if x is None:
        return "UNKNOWN"
    x = str(x).strip()
    return "UNKNOWN" if x == "" else x.upper()

def _load_feature_config(dirpath: Path):
    global _FEATURES, _CAT, _NUM, _BLEND_W
    cfg_path = dirpath / "feature_config"
    # allow .json without extension marker in Windows view
    if cfg_path.is_file():
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    elif (dirpath / "feature_config.json").is_file():
        with open(dirpath / "feature_config.json", "r", encoding="utf-8") as f:
            cfg = json.load(f)
    else:
        cfg = {}
    _FEATURES = cfg.get("features", _FEATURES or ["brand", "model", "year", "mileage_km_num"])
    _CAT = cfg.get("categorical", _CAT or ["brand", "model"])
    _NUM = cfg.get("numeric", _NUM or ["year", "mileage_km_num"])
    _BLEND_W = float(cfg.get("blend_weight", _BLEND_W))

def load_artifacts(dirpath: Path | str = MODELS_DIR):
    """Load model artifacts from disk into globals."""
    global _q20, _q50, _q80, _gmed_df
    d = Path(dirpath)
    # models
    _q20 = joblib.load(d / "q20_lgbm.pkl")
    _q50 = joblib.load(d / "q50_lgbm.pkl")
    _q80 = joblib.load(d / "q80_lgbm.pkl")
    # feature config
    _load_feature_config(d)
    # group medians (optional)
    gmed_path_csv = d / "group_medians"
    if not gmed_path_csv.suffix:
        # add .csv if it's missing from filename display
        gmed_path_csv = gmed_path_csv.with_suffix(".csv")
    if gmed_path_csv.exists():
        _gmed_df = pd.read_csv(gmed_path_csv)
        # normalize keys
        for col in _gmed_df.columns:
            if col.lower() in ("brand", "model"):
                _gmed_df[col] = _gmed_df[col].map(_std_cat)
    else:
        _gmed_df = None
    return True

def is_ready() -> bool:
    return all(m is not None for m in (_q20, _q50, _q80))

def _add_features(req: dict) -> pd.DataFrame:
    """Create a single-row dataframe with the exact feature set the model expects."""
    # base fields (tolerant to keys)
    brand = _std_cat(req.get("make") or req.get("brand"))
    model = _std_cat(req.get("model"))
    submodel = _std_cat(req.get("trim") or req.get("submodel") or "")
    gear = _std_cat(req.get("gear") or "")
    color = _std_cat(req.get("color") or "")
    engine = _std_cat(req.get("engine") or "")
    year = req.get("year")
    mileage = req.get("mileage_km_num") or req.get("mileage_km") or req.get("mileage")
    try:
        year = int(year) if year is not None else 0
    except Exception:
        year = 0
    try:
        mileage = float(mileage) if mileage is not None else 0.0
    except Exception:
        mileage = 0.0

    # Calculate current year for age calculation
    from datetime import datetime
    current_year = datetime.now().year
    
    # Calculate age
    age = max(0, current_year - year) if year > 0 else 0
    
    # Normalize mileage (use mileage_km_num if available, otherwise use mileage)
    mileage_value = mileage if mileage else 0.0

    row = {
        "brand": brand,
        "model": model,
        "submodel": submodel,
        "gear": gear,
        "color": color,
        "engine": engine,
        "year": year,
        "mileage_km_num": mileage_value,
    }
    df = pd.DataFrame([row])

    # Construct engineered features the model expects
    if "age" in _FEATURES:
        df["age"] = age
    if "log_mileage" in _FEATURES:
        df["log_mileage"] = np.log1p(mileage_value)
    if "sqrt_mileage" in _FEATURES:
        df["sqrt_mileage"] = np.sqrt(mileage_value)
    if "mileage_per_year" in _FEATURES:
        df["mileage_per_year"] = mileage_value / max(age, 1) if age > 0 else 0.0
    if "age_x_mileage" in _FEATURES:
        df["age_x_mileage"] = age * mileage_value
    if "mileage_per_age" in _FEATURES:
        df["mileage_per_age"] = mileage_value / max(age, 1) if age > 0 else 0.0
    # Handle "mileage" as alias for mileage_km_num
    if "mileage" in _FEATURES and "mileage" not in df.columns:
        df["mileage"] = mileage_value

    # Ensure all categorical fields normalized
    for c in _CAT:
        if c in df.columns:
            df[c] = df[c].map(_std_cat)
        else:
            df[c] = "UNKNOWN"

    # Ensure all numeric fields exist
    for n in _NUM:
        if n not in df.columns:
            df[n] = 0.0
        df[n] = pd.to_numeric(df[n], errors="coerce").fillna(0.0)

    # Final column order strictly matches training features
    for f in _FEATURES:
        if f not in df.columns:
            # backfill missing feature by type guess
            df[f] = "UNKNOWN" if f in _CAT else 0.0
    df = df[_FEATURES]
    
    # CRITICAL: Set categorical columns to category dtype for LightGBM
    for c in _CAT:
        if c in df.columns:
            df[c] = df[c].astype('category')
    
    return df

def _lookup_group_median(brand: str, model: str, year: int) -> float | None:
    if _gmed_df is None or _gmed_df.empty:
        return None
    # prefer brand+model+year, then brand+model
    gm = None
    df = _gmed_df
    if {"brand","model","year"}.issubset(df.columns):
        hit = df[(df["brand"] == brand) & (df["model"] == model) & (df["year"] == year)]
        if not hit.empty and "median_price" in hit.columns:
            gm = float(hit.iloc[0]["median_price"])
    if gm is None and {"brand","model"}.issubset(df.columns):
        hit = df[(df["brand"] == brand) & (df["model"] == model)]
        if not hit.empty and "median_price" in hit.columns:
            gm = float(hit.iloc[0]["median_price"])
    return gm


def _lookup_sample_size(brand: str, model: str, year: int) -> int:
    """Return the number of comparable rows used for the group median (if available).

    This is for transparency only. If the artifact doesn't include counts, returns 0.
    """
    if _gmed_df is None or _gmed_df.empty:
        return 0

    # Common column names for counts in exported artifacts
    count_cols = [c for c in _gmed_df.columns if str(c).lower() in {"n", "count", "rows", "sample_size", "num_rows"}]
    if not count_cols:
        return 0
    count_col = count_cols[0]

    df = _gmed_df
    b_col = next((c for c in df.columns if str(c).lower() in {"brand", "make"}), None)
    m_col = next((c for c in df.columns if str(c).lower() in {"model"}), None)
    y_col = next((c for c in df.columns if str(c).lower() in {"year"}), None)
    if not (b_col and m_col and y_col):
        return 0

    mask = (df[b_col].astype(str).str.lower() == str(brand).lower()) &                (df[m_col].astype(str).str.lower() == str(model).lower()) &                (df[y_col].astype(int, errors="ignore") == int(year))
    sub = df.loc[mask]
    if sub.empty:
        return 0
    try:
        val = int(sub.iloc[0][count_col])
        return max(val, 0)
    except Exception:
        return 0

def _round_price(price: float) -> int:
    """Round price according to config rules: 1000 THB if < 1M, 5000 THB if >= 1M."""
    if price < 1_000_000:
        return int(round(price / 1000) * 1000)
    else:
        return int(round(price / 5000) * 5000)

def predict_price(req: dict) -> dict:
    """Return green/yellow/red prices with post-prediction sanity caps."""
    if not is_ready():
        raise RuntimeError("Price model not loaded")

    # build features and also keep raw brand/model/year for anchoring
    brand = _std_cat(req.get("make") or req.get("brand"))
    model = _std_cat(req.get("model"))
    try:
        year = int(req.get("year"))
    except Exception:
        year = 0

    X = _add_features(req)

    # model quantiles - these are in log space (log(price))
    log_q20 = float(_q20.predict(X)[0])
    log_q50 = float(_q50.predict(X)[0])
    log_q80 = float(_q80.predict(X)[0])

    # Convert from log space to actual price space
    q20 = np.exp(log_q20)
    q50 = np.exp(log_q50)
    q80 = np.exp(log_q80)

    # optional blend toward group median for stability (light, 30% default)
    # Note: group median is in regular price space, not log space
    gmed = _lookup_group_median(brand, model, year)
    if gmed is not None and np.isfinite(gmed) and gmed > 0:
        blend = float(_BLEND_W)
        q50 = (1.0 - blend) * q50 + blend * gmed

    # --- sanity caps around q50 (median) ---
    LOW_RATIO  = 0.65   # q20 not lower than 65% of q50
    HIGH_RATIO = 1.75   # q80 not higher than 175% of q50

    q20 = max(q20, q50 * LOW_RATIO)
    q80 = min(q80, q50 * HIGH_RATIO)

    # --- optional: re-anchor to group median window and re-cap ---
    if gmed is not None and np.isfinite(gmed) and gmed > 0:
        # keep q50 within ±35% of group median
        q50 = float(np.clip(q50, 0.65 * gmed, 1.35 * gmed))
        # re-apply band caps around the (possibly adjusted) q50
        q20 = max(q20, q50 * LOW_RATIO)
        q80 = min(q80, q50 * HIGH_RATIO)

    # Calculate price ranges relative to market median (q50)
    # Green range: -12% to -8% below market median
    green_low = q50 * 0.88   # 12% below median
    green_median = q50 * 0.90  # 10% below median (middle of -8% to -12% range)
    green_high = q50 * 0.92  # 8% below median
    
    # Yellow: market median
    yellow = q50
    
    # Red range: +10% to +18% above market median
    red_low = q50 * 1.10     # 10% above median
    red_median = q50 * 1.14  # 14% above median (middle of +10% to +18% range)
    red_high = q50 * 1.18    # 18% above median
    
    # Round all prices

    # Band-width clamp (prevents extreme/unrealistic spreads from leaking to UI)
    bandwidth_clamped = False
    if (red_high > yellow * 1.35) or (green_low < yellow * 0.65):
        bandwidth_clamped = True
        # tighten around yellow (keeps the centre price unchanged)
        green_low = yellow * 0.88
        green_median = yellow * 0.92
        green_high = yellow * 0.96
        red_low = yellow * 1.04
        red_median = yellow * 1.14
        red_high = yellow * 1.18
    # --- Transparency / explanation fields (do not affect the model itself) ---
    sample_size = _lookup_sample_size(brand, model, year)

    # If we have enough comparable listings for that exact make+model+year, we can say it's based on comps.
    if sample_size >= 5:
        estimate_basis = "based_on_comparable_listings"
    else:
        estimate_basis = "market_trends"

    # Confidence is a simple, explainable heuristic (not model probability).
    # It only depends on how much supporting data we have (sample size) and whether we had to clamp the band width.
    if sample_size >= 20:
        confidence = 0.85
    elif sample_size >= 10:
        confidence = 0.75
    elif sample_size >= 5:
        confidence = 0.60
    elif sample_size >= 2:
        confidence = 0.45
    elif sample_size >= 1:
        confidence = 0.30
    else:
        confidence = 0.20

    if estimate_basis == "market_trends":
        confidence = min(confidence, 0.45)

    if bandwidth_clamped:
        confidence = max(0.20, confidence - 0.10)
    out = {
        "green_low": _round_price(green_low),
        "green_median": _round_price(green_median),
        "green_high": _round_price(green_high),
        "yellow": _round_price(yellow),
        "red_low": _round_price(red_low),
        "red_median": _round_price(red_median),
        "red_high": _round_price(red_high),
        # Transparency metadata
        "estimate_basis": "market_trends",
        "confidence": 0.0,  # placeholder until uncertainty calibration is added
        "estimate_basis": estimate_basis,
        "confidence": round(float(confidence), 2),
        "sample_size": int(sample_size),
        "bandwidth_clamped": bool(bandwidth_clamped),
    }
    return out