"""
clean_data_v2.py

Looser cleaning to keep more rows for ML:

- Read from MySQL using .env (MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DB)
- Keep most rows; only drop truly broken data
- Allow duplicates except exact row clones
- Ignore engine/gear for filtering
- Impute missing mileage using model-year medians
- Create age + mileage_per_year features
- Save to <out>.parquet and <out>.csv

Usage:
  python clean_data_v2.py --table car_listings_master --out data/cleaned_listings
"""

import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd
import mysql.connector
from dotenv import load_dotenv


# -------- DB utils --------
def get_conn():
    load_dotenv()
    conn = mysql.connector.connect(
        host=os.getenv("MYSQL_HOST"),
        port=int(os.getenv("MYSQL_PORT", "3306")),
        user=os.getenv("MYSQL_USER"),
        password=os.getenv("MYSQL_PASSWORD"),
        database=os.getenv("MYSQL_DB"),
        autocommit=False,
    )
    return conn


def read_table(table: str) -> pd.DataFrame:
    conn = get_conn()
    try:
        df = pd.read_sql(f"SELECT * FROM {table};", conn)
    finally:
        conn.close()
    return df


# -------- cleaning helpers --------
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

    # Keep the important columns (drop random extras if any)
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

    # Standardize main text fields
    for c in ["brand", "model", "submodel", "gear", "engine", "color", "priceType", "sourceName", "sourceType"]:
        if c in df.columns:
            df[c] = df[c].apply(std_text)

    # Coerce numerics
    df["price"] = pd.to_numeric(df.get("price"), errors="coerce")
    df["mileage"] = pd.to_numeric(df.get("mileage"), errors="coerce")
    if "year" in df.columns:
        df["year"] = df["year"].apply(coerce_int)

    # Dates
    for dc in ["dateRecorded", "created_at", "updated_at"]:
        if dc in df.columns:
            df[dc] = pd.to_datetime(df[dc], errors="coerce")

    current_year = datetime.now().year

    # ---- BASIC FILTERS (very loose) ----
    # Price: allow 5k–15M THB
    df = df[df["price"].notna()]
    df = df[df["price"].between(5_000, 15_000_000, inclusive="both")]

    # Year: allow 1990–next year
    if "year" in df.columns:
        df = df[df["year"].notna()]
        df = df[df["year"].between(1990, current_year + 1, inclusive="both")]

    # Brand/model: only drop if BOTH are unknown
    mask_unknown_both = (df["brand"] == "UNKNOWN") & (df["model"] == "UNKNOWN")
    df = df[~mask_unknown_both].copy()

    # Mileage: allow NaN and 0; only treat negative as invalid
    df.loc[df["mileage"] < 0, "mileage"] = np.nan

    # ---- IMPUTE MILEAGE ----
    # use median per (brand, model, year) when possible, else global median
    global_median_mileage = df["mileage"].median()
    if np.isnan(global_median_mileage):
        global_median_mileage = 100_000  # fallback

    if {"brand", "model", "year"}.issubset(df.columns):
        group_medians = df.groupby(["brand", "model", "year"])["mileage"].transform("median")
    else:
        group_medians = pd.Series(index=df.index, dtype=float)

    df["mileage_missing"] = df["mileage"].isna().astype("int8")
    df["mileage"] = df["mileage"].fillna(group_medians).fillna(global_median_mileage)

    # Clip mileage a bit but keep wide range
    df["mileage"] = df["mileage"].clip(lower=0, upper=800_000)

    # ---- AGE + MILEAGE PER YEAR ----
    df["age"] = current_year - df["year"]
    df.loc[df["age"] < 0, "age"] = 0

    df["mileage_per_year"] = df["mileage"] / df["age"].replace(0, np.nan)
    df["mileage_per_year"] = df["mileage_per_year"].fillna(df["mileage"])

    # Final price clipping (just in case)
    df["price"] = df["price"].clip(lower=5_000, upper=15_000_000)

    # ---- DEDUP: only drop exact row clones ----
    # This keeps real duplicate listings (useful signal) and only removes true duplicates.
    df = df.drop_duplicates(keep="first")

    # Arrange columns nicely
    final_cols = [
        "brand",
        "model",
        "submodel",
        "year",
        "age",
        "mileage",
        "mileage_per_year",
        "mileage_missing",
        "gear",
        "engine",
        "color",
        "price",
        "priceType",
        "sourceName",
        "sourceType",
        "sourceUrl",
        "externalId",
        "dateRecorded",
        "created_at",
        "updated_at",
        "id",
    ]
    df = df[[c for c in final_cols if c in df.columns]]

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", required=True, help="Source table name (e.g., car_listings_master)")
    parser.add_argument(
        "--out",
        required=True,
        help="Output path prefix (no extension). Example: data/cleaned_listings",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    raw = read_table(args.table)
    cleaned = clean_dataframe(raw)

    parquet_path = f"{args.out}.parquet"
    csv_path = f"{args.out}.csv"

    cleaned.to_parquet(parquet_path, index=False)
    cleaned.to_csv(csv_path, index=False)

    print(f"Raw rows:     {len(raw):,}")
    print(f"Cleaned rows: {len(cleaned):,}")
    print(f"Saved: {parquet_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
