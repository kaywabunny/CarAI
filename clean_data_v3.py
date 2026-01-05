"""
clean_data_v3.py

Goals:
- Keep as many rows as possible (loose validity rules)
- Drop only truly broken data
- Impute mileage sanely (per brand-model-year, else global)
- Exact-duplicate drop only
- SAFE per-(brand, model, year) price outlier trimming using aligned bounds
- Export CSV + Parquet

Usage:
  python clean_data_v3.py --table car_listings_master --out data/cleaned_listings_v3

Requires .env with:
  MYSQL_HOST=...
  MYSQL_PORT=3306
  MYSQL_USER=...
  MYSQL_PASSWORD=...
  MYSQL_DB=...
"""

import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd
import mysql.connector
from dotenv import load_dotenv


# ---------- DB ----------
def get_conn():
    load_dotenv()
    return mysql.connector.connect(
        host=os.getenv("MYSQL_HOST"),
        port=int(os.getenv("MYSQL_PORT", "3306")),
        user=os.getenv("MYSQL_USER"),
        password=os.getenv("MYSQL_PASSWORD"),
        database=os.getenv("MYSQL_DB"),
        autocommit=False,
    )


def read_table(table: str) -> pd.DataFrame:
    conn = get_conn()
    try:
        return pd.read_sql(f"SELECT * FROM {table};", conn)
    finally:
        conn.close()


# ---------- helpers ----------
def std_text(x: str) -> str:
    if pd.isna(x):
        return "UNKNOWN"
    x = str(x).strip()
    x = " ".join(x.split())
    return x.upper() if x else "UNKNOWN"


def coerce_int(x):
    try:
        if pd.isna(x):
            return None
        return int(float(x))
    except Exception:
        return None


def clean_dataframe(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df.columns = [c.strip() for c in df.columns]

    # keep only relevant columns (if present)
    keep = [
        "id",
        "brand",
        "model",
        "submodel",
        "year",
        "gear",
        "engine",
        "color",
        "mileage",
        "price",
        "priceType",
        "sourceName",
        "sourceType",
        "sourceUrl",
        "externalId",
        "dateRecorded",
        "created_at",
        "updated_at",
    ]
    df = df[[c for c in keep if c in df.columns]].copy()

    # text normalize (engine/gear kept but never filtered on)
    for c in ["brand", "model", "submodel", "gear", "engine", "color", "priceType", "sourceName", "sourceType"]:
        if c in df.columns:
            df[c] = df[c].apply(std_text)

    # numerics
    df["price"] = pd.to_numeric(df.get("price"), errors="coerce")
    df["mileage"] = pd.to_numeric(df.get("mileage"), errors="coerce")
    if "year" in df.columns:
        df["year"] = df["year"].apply(coerce_int)

    # dates
    for dc in ["dateRecorded", "created_at", "updated_at"]:
        if dc in df.columns:
            df[dc] = pd.to_datetime(df[dc], errors="coerce")

    now_year = datetime.now().year

    # ---------- loose validity ----------
    # price 5k–15M THB
    df = df[df["price"].notna()]
    df = df[df["price"].between(5_000, 15_000_000, inclusive="both")]

    # year 1990–(now+1)
    if "year" in df.columns:
        df = df[df["year"].notna()]
        df = df[df["year"].between(1990, now_year + 1, inclusive="both")]

    # brand/model: drop only if BOTH unknown
    both_unknown = (df["brand"] == "UNKNOWN") & (df["model"] == "UNKNOWN")
    df = df[~both_unknown].copy()

    # mileage: allow 0/NaN; negative -> NaN
    df.loc[df["mileage"] < 0, "mileage"] = np.nan

    # ---------- impute mileage ----------
    # model-year median -> global median -> fallback 100k
    global_med_mileage = df["mileage"].median()
    if np.isnan(global_med_mileage):
        global_med_mileage = 100_000

    if {"brand", "model", "year"}.issubset(df.columns):
        grp_med = df.groupby(["brand", "model", "year"])["mileage"].transform("median")
    else:
        grp_med = pd.Series(index=df.index, dtype=float)

    df["mileage_missing"] = df["mileage"].isna().astype("int8")
    df["mileage"] = df["mileage"].fillna(grp_med).fillna(global_med_mileage)
    df["mileage"] = df["mileage"].clip(lower=0, upper=800_000)

    # ---------- derived features ----------
    df["age"] = (now_year - df["year"]).clip(lower=0)
    df["mileage_per_year"] = (df["mileage"] / df["age"].replace(0, np.nan)).fillna(df["mileage"])

    # final gentle price clip
    df["price"] = df["price"].clip(lower=5_000, upper=15_000_000)

    # ---------- SAFE per-(brand, model, year) price outlier trimming ----------
    # Only for groups with >= 20 rows; others are kept intact
    if {"brand", "model", "year"}.issubset(df.columns):
        df["_group"] = df["brand"].astype(str) + "|" + df["model"].astype(str) + "|" + df["year"].astype(str)

        grp_sizes = df["_group"].value_counts()
        big_groups = grp_sizes[grp_sizes >= 20].index

        # aligned lower/upper bounds
        lower_bounds = pd.Series(index=df.index, dtype=float)
        upper_bounds = pd.Series(index=df.index, dtype=float)

        for grp in big_groups:
            idx = df.index[df["_group"] == grp]
            prices = df.loc[idx, "price"]
            q1 = prices.quantile(0.25)
            q3 = prices.quantile(0.75)
            iqr = q3 - q1
            lb = q1 - 2.5 * iqr
            ub = q3 + 2.5 * iqr
            lower_bounds.loc[idx] = lb
            upper_bounds.loc[idx] = ub

        trimmed_mask = (
            (df["_group"].isin(big_groups) & df["price"].between(lower_bounds, upper_bounds, inclusive="both"))
            | (~df["_group"].isin(big_groups))
        )
        df = df[trimmed_mask].copy()
        df.drop(columns=["_group"], inplace=True)

    # exact-duplicate removal only
    df = df.drop_duplicates(keep="first")

    # final column order
    final_cols = [
        "brand", "model", "submodel", "year", "age",
        "mileage", "mileage_per_year", "mileage_missing",
        "gear", "engine", "color",
        "price", "priceType", "sourceName", "sourceType", "sourceUrl",
        "externalId", "dateRecorded", "created_at", "updated_at", "id",
    ]
    df = df[[c for c in final_cols if c in df.columns]]
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", required=True, help="Source table (e.g., car_listings_master)")
    parser.add_argument("--out", required=True, help="Output prefix (no extension), e.g., data/cleaned_listings_v3")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    raw = read_table(args.table)
    cleaned = clean_dataframe(raw)

    cleaned.to_parquet(f"{args.out}.parquet", index=False)
    cleaned.to_csv(f"{args.out}.csv", index=False)

    print(f"Raw rows:     {len(raw):,}")
    print(f"Cleaned rows: {len(cleaned):,}")
    print(f"Saved: {args.out}.parquet")
    print(f"Saved: {args.out}.csv")


if __name__ == "__main__":
    main()
