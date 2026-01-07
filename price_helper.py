# price_helper.py
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd


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

# training-time category lists (used to fix LightGBM categorical mismatch)
_PANDAS_CAT_MAP: dict[str, list[str]] = {}

# stored from group-median lookup (transparency)
_LAST_YEAR_WINDOW_USED = 0
_LAST_SAMPLE_SIZE = 0


def _std_cat(x: str) -> str:
    """Normalize categorical text for inference."""
    if x is None:
        return "UNKNOWN"
    x = str(x).strip()
    return "UNKNOWN" if x == "" else x.upper()


def _extract_pandas_categorical(model) -> list[list[str]] | None:
    """Best-effort: pull training-time pandas categorical lists from a LightGBM model."""
    try:
        booster = getattr(model, "booster_", None)
        if booster is not None:
            pc = getattr(booster, "pandas_categorical", None)
            if pc:
                return pc
        pc = getattr(model, "pandas_categorical", None)
        if pc:
            return pc
    except Exception:
        pass
    return None


def _load_feature_config(dirpath: Path):
    global _FEATURES, _CAT, _NUM, _BLEND_W
    cfg_path = dirpath / "feature_config"
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


def load_artifacts(dirpath: Path | str = MODELS_DIR, *args, **kwargs):
    """Load model artifacts from disk into globals.

    Compatibility: accepts extra args so admin reload endpoints that pass a path won't break.
    """
    global _q20, _q50, _q80, _gmed_df, _PANDAS_CAT_MAP

    # If caller passed a path positionally, prefer it.
    if args and (dirpath == MODELS_DIR):
        try:
            dirpath = args[0]
        except Exception:
            pass

    d = Path(dirpath)

    # models
    _q20 = joblib.load(d / "q20_lgbm.pkl")
    _q50 = joblib.load(d / "q50_lgbm.pkl")
    _q80 = joblib.load(d / "q80_lgbm.pkl")

    # feature config
    _load_feature_config(d)

    # capture training-time categorical lists (fix for categorical_feature mismatch)
    _PANDAS_CAT_MAP = {}
    cats = _extract_pandas_categorical(_q50) or _extract_pandas_categorical(_q20) or _extract_pandas_categorical(_q80)
    if cats and _CAT:
        for i, col in enumerate(_CAT):
            if i < len(cats):
                _PANDAS_CAT_MAP[col] = list(map(str, cats[i]))

    # group medians (optional)
    gmed_path_csv = d / "group_medians"
    if not gmed_path_csv.suffix:
        gmed_path_csv = gmed_path_csv.with_suffix(".csv")

    if gmed_path_csv.exists():
        _gmed_df = pd.read_csv(gmed_path_csv)
        # normalize keys
        for col in _gmed_df.columns:
            if col.lower() in ("brand", "model", "make"):
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

    # Calculate age (only if model expects it)
    from datetime import datetime
    current_year = datetime.now().year
    age = max(0, current_year - year) if year > 0 else 0

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

    # Engineered features the model may expect
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
    if "mileage" in _FEATURES and "mileage" not in df.columns:
        df["mileage"] = mileage_value

    # Ensure all categorical fields exist + normalize
    for c in _CAT:
        if c in df.columns:
            df[c] = df[c].map(_std_cat)
        else:
            df[c] = "UNKNOWN"

        # CRITICAL FIX:
        # If the model was trained with pandas categorical lists, force SAME categories.
        cats = _PANDAS_CAT_MAP.get(c)
        if cats:
            v = str(df[c].iloc[0]) if len(df) else "UNKNOWN"
            if v not in cats:
                fallback = "UNKNOWN" if "UNKNOWN" in cats else cats[0]
                df[c] = fallback
            df[c] = pd.Categorical(df[c], categories=cats)
        else:
            # fallback: at least make dtype categorical
            df[c] = df[c].astype("category")

    # Ensure all numeric fields exist
    for n in _NUM:
        if n not in df.columns:
            df[n] = 0.0
        df[n] = pd.to_numeric(df[n], errors="coerce").fillna(0.0)

    # Final column order strictly matches training features
    for f in _FEATURES:
        if f not in df.columns:
            df[f] = "UNKNOWN" if f in _CAT else 0.0

    df = df[_FEATURES]
    return df


def _lookup_group_median(brand: str, model: str, year: int) -> float | None:
    """Lookup a group median price for (brand, model, year)."""
    global _LAST_YEAR_WINDOW_USED, _LAST_SAMPLE_SIZE
    _LAST_YEAR_WINDOW_USED = 0
    _LAST_SAMPLE_SIZE = 0

    if _gmed_df is None or _gmed_df.empty:
        return None

    df = _gmed_df

    b_col = next((c for c in df.columns if str(c).lower() in {"brand", "make"}), None)
    m_col = next((c for c in df.columns if str(c).lower() == "model"), None)
    y_col = next((c for c in df.columns if str(c).lower() == "year"), None)
    p_col = next((c for c in df.columns if str(c).lower() in {"median_price", "price_median", "median"}), None)
    if not (b_col and m_col and y_col and p_col):
        return None

    count_col = next((c for c in df.columns if str(c).lower() in {"n", "count", "rows", "sample_size", "num_rows"}), None)

    brand_s = str(brand).strip().lower()
    model_s = str(model).strip().lower()

    MIN_COMPS_FOR_STABILITY = 5
    MAX_YEAR_WINDOW = 4

    sub_all = df[
        (df[b_col].astype(str).str.strip().str.lower() == brand_s) &
        (df[m_col].astype(str).str.strip().str.lower() == model_s)
    ].copy()

    if sub_all.empty:
        return None

    sub_all["_year_int"] = pd.to_numeric(sub_all[y_col], errors="coerce").astype("Int64")
    sub_all = sub_all[sub_all["_year_int"].notna()]
    if sub_all.empty:
        return None

    best_sub = None
    best_w = 0
    best_n = 0

    for w in range(0, MAX_YEAR_WINDOW + 1):
        if w == 0:
            sub = sub_all[sub_all["_year_int"].astype(int) == int(year)]
        else:
            lo, hi = int(year) - w, int(year) + w
            sub = sub_all[(sub_all["_year_int"].astype(int) >= lo) & (sub_all["_year_int"].astype(int) <= hi)]

        if sub.empty:
            continue

        if count_col:
            try:
                n = int(pd.to_numeric(sub[count_col], errors="coerce").fillna(0).sum())
            except Exception:
                n = int(len(sub))
        else:
            n = int(len(sub))

        if best_sub is None:
            best_sub, best_w, best_n = sub, w, n
        elif best_n < MIN_COMPS_FOR_STABILITY and n > best_n:
            best_sub, best_w, best_n = sub, w, n

        if n >= MIN_COMPS_FOR_STABILITY:
            best_sub, best_w, best_n = sub, w, n
            break

    if best_sub is None or best_sub.empty:
        return None

    prices = pd.to_numeric(best_sub[p_col], errors="coerce")
    years = pd.to_numeric(best_sub["_year_int"], errors="coerce")
    mask = prices.notna() & years.notna()
    prices = prices[mask]
    years = years[mask]
    if prices.empty:
        return None

    if best_w == 0:
        gmed = float(prices.median())
    else:
        dy = (years.astype(int) - int(year)).abs()
        wts = 1.0 / (1.0 + dy.astype(float))  # closer years weigh more heavily
        gmed = float(np.average(prices.astype(float), weights=wts))

    _LAST_YEAR_WINDOW_USED = int(best_w)
    _LAST_SAMPLE_SIZE = int(best_n)
    return gmed


def _lookup_sample_size(brand: str, model: str, year: int) -> int:
    """Return comparable sample size (best-effort)."""
    if _gmed_df is None or _gmed_df.empty:
        return 0

    # if group median was already computed in this request, reuse the last size
    if _LAST_SAMPLE_SIZE:
        return int(_LAST_SAMPLE_SIZE)

    df = _gmed_df
    b_col = next((c for c in df.columns if str(c).lower() in {"brand", "make"}), None)
    m_col = next((c for c in df.columns if str(c).lower() == "model"), None)
    y_col = next((c for c in df.columns if str(c).lower() == "year"), None)
    if not (b_col and m_col and y_col):
        return 0

    count_col = next((c for c in df.columns if str(c).lower() in {"n", "count", "rows", "sample_size", "num_rows"}), None)

    brand_s = str(brand).strip().lower()
    model_s = str(model).strip().lower()

    sub = df[
        (df[b_col].astype(str).str.strip().str.lower() == brand_s) &
        (df[m_col].astype(str).str.strip().str.lower() == model_s) &
        (pd.to_numeric(df[y_col], errors="coerce").fillna(-1).astype(int) == int(year))
    ]

    if sub.empty:
        return 0

    if count_col:
        try:
            return int(pd.to_numeric(sub[count_col], errors="coerce").fillna(0).sum())
        except Exception:
            return int(len(sub))

    return int(len(sub))


def _round_price(price: float) -> int:
    """Round price: 1000 THB if < 1M, else 5000 THB."""
    if price < 1_000_000:
        return int(round(price / 1000) * 1000)
    else:
        return int(round(price / 5000) * 5000)


def predict_price(req: dict) -> dict:
    """Return green/yellow/red prices with post-prediction sanity caps + transparency."""
    if not is_ready():
        raise RuntimeError("Price model not loaded")

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
    gmed = _lookup_group_median(brand, model, year)
    if gmed is not None and np.isfinite(gmed) and gmed > 0:
        blend = float(_BLEND_W)
        q50 = (1.0 - blend) * q50 + blend * gmed

    # --- sanity caps around q50 (median) ---
    LOW_RATIO = 0.65   # q20 not lower than 65% of q50
    HIGH_RATIO = 1.75  # q80 not higher than 175% of q50

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
    green_low = q50 * 0.88
    green_median = q50 * 0.90
    green_high = q50 * 0.92

    yellow = q50

    red_low = q50 * 1.10
    red_median = q50 * 1.14
    red_high = q50 * 1.18

    # Band-width clamp (prevents extreme spreads)
    bandwidth_clamped = False
    if (red_high > yellow * 1.35) or (green_low < yellow * 0.65):
        bandwidth_clamped = True
        green_low = yellow * 0.88
        green_median = yellow * 0.92
        green_high = yellow * 0.96
        red_low = yellow * 1.04
        red_median = yellow * 1.14
        red_high = yellow * 1.18

    # Transparency
    sample_size = _lookup_sample_size(brand, model, year)

    if sample_size >= 5:
        estimate_basis = "based_on_comparable_listings"
    else:
        estimate_basis = "market_trends"

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

        "estimate_basis": estimate_basis,
        "confidence": round(float(confidence), 2),
        "sample_size": int(sample_size),
        "bandwidth_clamped": bool(bandwidth_clamped),
    }
    return out

def render_price_chart(*args, **kwargs):
    """
    Compatibility stub.
    This project previously exposed render_price_chart but the API
    no longer requires it. Keeping this avoids import errors.
    """
    return None
