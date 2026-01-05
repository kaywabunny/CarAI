# train_model_v3.2.py
import argparse, json
from pathlib import Path
import joblib, lightgbm as lgb
import numpy as np, pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import train_test_split

def mape(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    mask = y_true > 0
    return float(mean_absolute_percentage_error(y_true[mask], y_pred[mask]))

def wape(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    return float(np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)))

def rmse_log(y_true_log, y_pred_log):
    from sklearn.metrics import mean_squared_error
    return float(np.sqrt(mean_squared_error(y_true_log, y_pred_log)))

def thb_round(v):
    if pd.isna(v): return v
    return int(round(v / (1000 if v < 1_000_000 else 5000)) * (1000 if v < 1_000_000 else 5000))

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    use = df.copy()
    for c in ["brand","model","submodel","gear","color"]:
        if c in use.columns:
            use[c] = (use[c].astype(str).str.strip().str.upper()
                      .replace({"NAN":"UNKNOWN"}).fillna("UNKNOWN").astype("category"))
    for c in ["year","age","mileage","mileage_per_year"]:
        if c in use.columns:
            use[c] = pd.to_numeric(use[c], errors="coerce")
    use["log_mileage"]   = np.log1p(use["mileage"].clip(lower=0))
    use["sqrt_mileage"]  = np.sqrt(use["mileage"].clip(lower=0))
    use["age_x_mileage"] = use["age"].fillna(0) * use["mileage"].fillna(0)
    use["mileage_per_age"] = (use["mileage"] / use["age"].replace({0: np.nan})).fillna(use["mileage"])
    return use

def train_quantile_models(X_train, y_train, X_val, y_val, cat_idx):
    base = dict(boosting_type="gbdt", num_leaves=96, learning_rate=0.05,
                min_data_in_leaf=80, feature_fraction=0.9, bagging_fraction=0.9,
                bagging_freq=1, max_depth=-1, lambda_l2=2.0, verbose=-1, force_col_wise=True)
    models = {}
    for name, alpha in {"q20":0.20,"q50":0.50,"q80":0.80}.items():
        params = base | {"objective":"quantile","alpha":alpha,"metric":"quantile"}
        dtr = lgb.Dataset(X_train, y_train, categorical_feature=cat_idx, free_raw_data=False)
        dvl = lgb.Dataset(X_val, y_val, categorical_feature=cat_idx, reference=dtr, free_raw_data=False)
        models[name] = lgb.train(params, dtr, num_boost_round=6000,
                                 valid_sets=[dtr,dvl], valid_names=["train","val"],
                                 callbacks=[lgb.early_stopping(300, verbose=False),
                                            lgb.log_evaluation(200)])
    return models

def compute_group_medians(df: pd.DataFrame):
    g = df.groupby(["brand","model","year"], dropna=False)["price"].agg(["median","count"]).reset_index()
    g.rename(columns={"median":"median_price"}, inplace=True)
    g["median_price"] = g["median_price"].astype(float)
    return g

def blend_with_median(pred, key_df, gmed, weight=0.30):
    merged = key_df.copy()
    med_my = gmed.set_index(["brand","model","year"])["median_price"]
    merged["median_price"] = merged.set_index(["brand","model","year"]).index.map(med_my)
    by_model = gmed.groupby(["brand","model"])["median_price"].median()
    mask = merged["median_price"].isna()
    merged.loc[mask,"median_price"] = merged.loc[mask].set_index(["brand","model"]).index.map(by_model)
    by_brand = gmed.groupby(["brand"])["median_price"].median()
    mask = merged["median_price"].isna()
    merged.loc[mask,"median_price"] = merged.loc[mask]["brand"].map(by_brand)
    global_med = float(gmed["median_price"].median()) if len(gmed) else float(np.nanmedian(pred))
    merged["median_price"] = merged["median_price"].astype(float).fillna(global_med)
    med = merged["median_price"].to_numpy(dtype=float)
    pred = np.asarray(pred, dtype=float); w = float(weight)
    return (1.0 - w) * pred + w * med

def widen_to_target(q20, q50, q80, y_true, target=0.68, cap=1.6, tol=0.01):
    # binary/linear search over scale s to reach target coverage
    def cov_for(s):
        q20s = q50 - s*(q50 - q20)
        q80s = q50 + s*(q80 - q50)
        return float(np.mean((y_true >= q20s) & (y_true <= q80s))), q20s, q80s
    lo, hi = 1.0, cap
    best_s, best_cov, best_q20, best_q80 = lo, *cov_for(lo)
    if abs(best_cov - target) <= tol:
        return best_q20, best_q80, best_s, best_cov
    for _ in range(30):
        mid = (lo + hi) / 2
        cov, q20m, q80m = cov_for(mid)
        # track best
        if abs(cov - target) < abs(best_cov - target):
            best_s, best_cov, best_q20, best_q80 = mid, cov, q20m, q80m
        if cov < target:
            lo = mid
        else:
            hi = mid
    return best_q20, best_q80, best_s, best_cov

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--blend", type=float, default=0.30)
    ap.add_argument("--cov_target", type=float, default=0.68)
    ap.add_argument("--cov_cap", type=float, default=1.6)
    args = ap.parse_args()

    prefix = Path(args.inp); outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    if Path(str(prefix)+".parquet").exists(): df = pd.read_parquet(str(prefix)+".parquet")
    elif Path(str(prefix)+".csv").exists():   df = pd.read_csv(str(prefix)+".csv")
    else: raise FileNotFoundError("Cleaned data not found")

    needed = ["price","brand","model","year","age","mileage","mileage_per_year"]
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    y_log = np.log(df["price"].astype(float).values)
    feat = add_features(df)
    cat_cols = [c for c in ["brand","model","submodel","gear","color"] if c in feat.columns]
    num_cols = [c for c in ["year","age","mileage","mileage_per_year","log_mileage","sqrt_mileage","age_x_mileage","mileage_per_age"] if c in feat.columns]
    X = feat[cat_cols + num_cols].copy()
    id_col = "id" if "id" in feat.columns else None

    X_tr, X_val, y_tr, y_val = train_test_split(X, y_log, test_size=0.2, random_state=42)
    key_val = df.loc[X_val.index, ["brand","model","year"]].copy()
    id_val  = df.loc[X_val.index, id_col] if id_col else pd.Series(np.arange(len(y_val)))
    cat_idx = [X.columns.get_loc(c) for c in cat_cols]

    models = train_quantile_models(X_tr, y_tr, X_val, y_val, cat_idx)
    q20 = np.exp(models["q20"].predict(X_val, num_iteration=models["q20"].best_iteration))
    q50 = np.exp(models["q50"].predict(X_val, num_iteration=models["q50"].best_iteration))
    q80 = np.exp(models["q80"].predict(X_val, num_iteration=models["q80"].best_iteration))
    y_true = np.exp(y_val)

    gmed = compute_group_medians(df[["brand","model","year","price"]].copy())
    q20_b = blend_with_median(q20, key_val, gmed, weight=args.blend)
    q50_b = blend_with_median(q50, key_val, gmed, weight=args.blend)
    q80_b = blend_with_median(q80, key_val, gmed, weight=args.blend)

    # auto-tune coverage
    q20_w, q80_w, scale_used, cov_after = widen_to_target(q20_b, q50_b, q80_b, y_true,
                                                          target=args.cov_target, cap=args.cov_cap, tol=0.01)

    metrics = {
        "split_type":"random",
        "val_rows": int(len(y_true)),
        "WAPE@q50_before": wape(y_true, q50_b),
        "MAPE@q50_before": mape(y_true, q50_b),
        "coverage_before": float(np.mean((y_true >= q20_b) & (y_true <= q80_b))),
        "WAPE@q50_after": wape(y_true, q50_b),
        "MAPE@q50_after": mape(y_true, q50_b),
        "coverage_after": cov_after,
        "blend_weight": float(args.blend),
        "coverage_target": float(args.cov_target),
        "coverage_scale_used": float(scale_used),
        "RMSE_log@q50": rmse_log(y_val, np.log(np.maximum(q50_b, 1))),
    }
    with open(outdir/"metrics.json","w",encoding="utf-8") as f: json.dump(metrics,f,indent=2)

    def confidence_from_iqr(q20v, q50v, q80v):
        width = (q80v - q20v) / np.maximum(q50v, 1)
        c = 1 - np.clip(width, 0, 1.5) / 1.5
        return np.clip(c, 0, 1)

    conf   = confidence_from_iqr(q20_w, q50_b, q80_w)
    green  = np.array([thb_round(v) for v in q20_w])
    yellow = np.array([thb_round(v) for v in q50_b])
    red    = np.array([thb_round(v) for v in q80_w])

    pd.DataFrame({
        "id": id_val.values,
        "actual_price": y_true,
        "q20": q20_w, "q50": q50_b, "q80": q80_w,
        "green_quick_sale": green, "yellow_median": yellow, "red_max_profit": red,
        "confidence": conf,
    }).to_csv(outdir/"val_preds.csv", index=False)

    for n,m in models.items(): joblib.dump(m, outdir/f"{n}_lgbm.pkl")
    gmed.to_csv(outdir/"group_medians.csv", index=False)
    cfg = {
        "features": list(X.columns), "categorical": cat_cols, "numeric": num_cols,
        "target":"log(price)", "rounding":{"<1M":"1000 THB","≥1M":"5000 THB"},
        "blend_weight": float(args.blend),
        "coverage_target": float(args.cov_target),
        "coverage_scale_used": float(scale_used),
        "version":"v3.2"
    }
    with open(outdir/"feature_config.json","w",encoding="utf-8") as f: json.dump(cfg,f,indent=2)
    print("\nSaved models & artifacts to:", outdir)
    print("Metrics:", json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
