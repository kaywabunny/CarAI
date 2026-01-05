# -*- coding: utf-8 -*-
"""
depreciation_single.py
----------------------
Single-car depreciation estimator.

Connects to MySQL table `car_listings_master`, learns a robust statistical model
for a brand/model (optionally submodel), and returns:
- fair price today with an uncertainty band
- projected prices for the next N years (with uncertainty)

Model:
  log(price) ~ age_years + log(mileage+1)
where age_years = current_year - model_year

DB config via env (preferred):
  MYSQL_HOST, MYSQL_PORT, MYSQL_DB, MYSQL_USER, MYSQL_PASSWORD
or pass a full SQLAlchemy URL to DepreciationEstimator(..., db_url="...").

Requires: pandas, numpy, SQLAlchemy, PyMySQL, scikit-learn
"""

from __future__ import annotations
import os, math, warnings
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from datetime import date

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sklearn.linear_model import HuberRegressor
from sklearn.exceptions import ConvergenceWarning

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, skip .env loading


# ---------------------------
# Utilities
# ---------------------------

def _now_year() -> int:
    return date.today().year

def _normalize_brand(s: str) -> str:
    return (s or "").strip().upper()

def _normalize_model(s: str) -> str:
    return (s or "").strip().title()

def _engine_from_env_or_url(db_url: Optional[str]):
    if db_url:
        return create_engine(db_url, pool_pre_ping=True, future=True)
    host = os.getenv("MYSQL_HOST", "localhost")
    port = int(os.getenv("MYSQL_PORT", "3306"))
    db   = os.getenv("MYSQL_DB", "carpricing")
    user = os.getenv("MYSQL_USER", "car_user")
    pwd  = os.getenv("MYSQL_PASSWORD", "StrongPassword123")
    url  = f"mysql+pymysql://{user}:{pwd}@{host}:{port}/{db}?charset=utf8mb4"
    return create_engine(url, pool_pre_ping=True, future=True)


@dataclass
class DepreciationResult:
    brand: str
    model: str
    submodel: Optional[str]
    year: int
    mileage: Optional[int]
    sample_size: int
    predicted_price_now: float
    lower_now: float
    upper_now: float
    annual_projection: List[Dict[str, Any]]  # [{calendar_year, age, mileage, price, lower, upper, depreciation_from_now_pct}]
    km_per_year_assumed: int
    notes: str


# ---------------------------
# Estimator
# ---------------------------

class DepreciationEstimator:
    def __init__(self, db_url: Optional[str] = None, table: str = "car_listings_master"):
        self.engine = _engine_from_env_or_url(db_url)
        self.table = table

    def _fetch_cohort(self, brand: str, model: str, submodel: Optional[str]) -> pd.DataFrame:
        brand_n = _normalize_brand(brand)
        model_n = _normalize_model(model)
        params = {"brand": brand_n, "model": model_n, "maxy": _now_year() + 1}

        if submodel:
            sub_n = (submodel or "").strip().title()
            sql = f"""
                SELECT brand, model, submodel, year, mileage, price, dateRecorded
                FROM {self.table}
                WHERE brand = :brand
                  AND model = :model
                  AND submodel = :submodel
                  AND price IS NOT NULL
                  AND year IS NOT NULL
                  AND year BETWEEN 1990 AND :maxy
            """
            params["submodel"] = sub_n
        else:
            sql = f"""
                SELECT brand, model, submodel, year, mileage, price, dateRecorded
                FROM {self.table}
                WHERE brand = :brand
                  AND model = :model
                  AND price IS NOT NULL
                  AND year IS NOT NULL
                  AND year BETWEEN 1990 AND :maxy
            """

        with self.engine.begin() as conn:
            conn.exec_driver_sql("SET NAMES utf8mb4;")
            df = pd.read_sql(text(sql), conn, params=params)

        # Basic cleaning
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        df["year"] = pd.to_numeric(df["year"], errors="coerce", downcast="integer")
        if "mileage" in df.columns:
            df["mileage"] = pd.to_numeric(df["mileage"], errors="coerce")
        df = df.dropna(subset=["price", "year"])
        df = df[(df["price"] > 0) & (df["year"] >= 1990)]
        df["age"] = _now_year() - df["year"]
        df = df[df["age"] >= 0]

        # Winsorize price (5–95%) to stabilize fit
        if len(df) >= 20:
            lo, hi = np.percentile(df["price"], [5, 95])
            df = df[(df["price"] >= lo) & (df["price"] <= hi)]

        return df

    def _fit_log_model(self, df: pd.DataFrame):
        """
        Robust log-linear regression:
            log(price) ~ age + log(mileage+1)
        Returns (model, sigma_log, median_km_per_year).
        """
        if df.empty:
            raise ValueError("No data to fit depreciation model.")

        X_age = df["age"].to_numpy()
        log_km = np.log1p(np.clip(df["mileage"].fillna(0).astype(float).to_numpy(), 0, None))
        X = np.column_stack([X_age, log_km])
        y = np.log(np.clip(df["price"].astype(float).to_numpy(), 1.0, None))

        # km/year from cohort (median of mileage/age where age>0)
        valid = (df["age"] > 0) & df["mileage"].notna() & (df["mileage"] > 0)
        if valid.any():
            km_per_year = int(np.median((df.loc[valid, "mileage"] / df.loc[valid, "age"]).clip(1000, 50000)))
        else:
            km_per_year = 12000

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            model = HuberRegressor(epsilon=1.35, alpha=1e-4, max_iter=500)
            model.fit(X, y)

        y_hat = model.predict(X)
        sigma_log = float(np.std(y - y_hat)) if len(y) > 1 else 0.25  # fallback ~±28% band

        return model, sigma_log, km_per_year

    def estimate(
        self,
        brand: str,
        model: str,
        year: int,
        mileage: Optional[int] = None,
        submodel: Optional[str] = None,
        horizon_years: int = 5,
    ) -> DepreciationResult:
        """
        Compute fair price now and future projections for a single car.
        """
        df = self._fetch_cohort(brand, model, submodel)
        if len(df) < 30 and submodel:  # relax submodel if too small
            df = self._fetch_cohort(brand, model, None)
        if len(df) < 30:
            raise ValueError(f"Not enough cohort data for {brand} {model} (need >=30, have {len(df)}).")

        model_fit, sigma_log, km_py = self._fit_log_model(df)

        brand_n = _normalize_brand(brand)
        model_n = _normalize_model(model)
        sub_n = (submodel or "").strip().title() if submodel else None

        target_age = max(0, _now_year() - int(year))
        if mileage is None:
            mileage = int(km_py * max(1, target_age))  # impute cohort-typical km

        def _predict(age, km):
            X = np.array([[age, math.log1p(max(0, km))]], dtype=float)
            lp = float(model_fit.predict(X)[0])
            p  = math.exp(lp)
            lo = math.exp(lp - sigma_log)
            hi = math.exp(lp + sigma_log)
            return p, lo, hi

        price_now, lower_now, upper_now = _predict(target_age, mileage)

        # Project forward
        projections = []
        running_km = int(mileage)
        for t in range(1, int(horizon_years) + 1):
            age_t = target_age + t
            running_km += km_py
            p, lo, hi = _predict(age_t, running_km)
            projections.append({
                "calendar_year": _now_year() + t,
                "age": age_t,
                "mileage": running_km,
                "price": round(p, 2),
                "lower": round(lo, 2),
                "upper": round(hi, 2),
                "depreciation_from_now_pct": round(100.0 * (price_now - p) / price_now, 2) if price_now > 0 else None
            })

        return DepreciationResult(
            brand=brand_n,
            model=model_n,
            submodel=sub_n if submodel else None,
            year=int(year),
            mileage=int(mileage) if mileage is not None else None,
            sample_size=len(df),
            predicted_price_now=round(price_now, 2),
            lower_now=round(lower_now, 2),
            upper_now=round(upper_now, 2),
            annual_projection=projections,
            km_per_year_assumed=int(km_py),
            notes=(
                "Robust log-linear model: log(price) ~ age + log(mileage+1). "
                "Age = current_year - model_year. "
                "Uncertainty = residual std.dev. in log space. "
                "Annual km from cohort median; fallback 12,000 km/yr."
            ),
        )


# ---------------------------
# Minimal CLI (optional)
# ---------------------------

def _print_result(res: DepreciationResult):
    print(f"\n=== Depreciation Estimate ===")
    print(f"Cohort: {res.brand} {res.model}" + (f" {res.submodel}" if res.submodel else ""))
    print(f"Target: year={res.year}, mileage={res.mileage:,} km")
    print(f"Sample size used: {res.sample_size}")
    print(f"Assumed km/year: {res.km_per_year_assumed:,}")
    print(f"\nFair price now: {res.predicted_price_now:,.0f} THB "
          f"(~ {res.lower_now:,.0f} … {res.upper_now:,.0f})")
    print("\nProjection:")
    for row in res.annual_projection:
        print(f"  {row['calendar_year']}: {row['price']:,.0f} THB "
              f"(~ {row['lower']:,.0f} … {row['upper']:,.0f})  "
              f"[dep from now: {row['depreciation_from_now_pct']}%]")
    print(f"\nNotes: {res.notes}\n")


# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser(description="Single-car depreciation estimator")
#     parser.add_argument("--brand", required=True, help="Brand, e.g., TOYOTA")
#     parser.add_argument("--model", required=True, help="Model, e.g., Yaris")
#     parser.add_argument("--year", type=int, required=True, help="Model year, e.g., 2019")
#     parser.add_argument("--mileage", type=int, default=None, help="Current mileage (km)")
#     parser.add_argument("--submodel", type=str, default=None, help="Optional submodel/series")
#     parser.add_argument("--horizon", type=int, default=5, help="Projection horizon (years)")
#     parser.add_argument("--db_url", type=str, default=None, help="Optional SQLAlchemy DB URL")

#     args = parser.parse_args()
#     est = DepreciationEstimator(db_url=args.db_url)
#     res = est.estimate(
#         brand=args.brand, model=args.model, year=args.year,
#         mileage=args.mileage, submodel=args.submodel, horizon_years=args.horizon
#     )
#     _print_result(res)
