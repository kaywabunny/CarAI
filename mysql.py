#!/usr/bin/env python3
# Auto-generated from mysql.ipynb
import sys

def main():
    # --- 0) Setup ---------------------------------------------------------------
    from datetime import date
    import json, math, re
    import pandas as pd
    from sqlalchemy import create_engine, text
    
    # <-- update if your creds/DB URL live elsewhere -->
    DB_URL = "mysql+pymysql://car_user:StrongPassword123@localhost:3306/carpricing?charset=utf8mb4"
    engine = create_engine(DB_URL, pool_pre_ping=True, future=True)
    
    # --- 1) Load your AI-parsed JSON -------------------------------------------
    # cars_detailed.json should be a list of dicts with keys you showed:
    # brand, model, submodel, year, gear, engine, color, mileage, price,
    # priceType, sourceName, sourceType, sourceUrl, externalId, dateRecorded
    with open("taladrodpar.json", "r", encoding="utf-8") as f:
        raw = json.load(f)
    
    df = pd.DataFrame(raw)
    
    # --- 2) Normalizers: fix mojibake / Thai romanizations ----------------------
    def clean_str(x):
        if x is None: return None
        x = str(x).strip()
        if x in ("Not Applicable", "N/A", "", "None", "null", "NaN"): return None
        return x
    
    # frequent romanized/garbled Thai → English colors
    COLOR_MAP = {
        # Thai romanizations that often show up
        "khaaw": "White",      # ขาว
        "khaw": "White",
        "khǎo": "White",
        "dam": "Black",        # ดำ
        "black": "Black",
        "แดง": "Red",
        "aedng": "Red",        # แดง (garbled)
        "daeng": "Red",
        "namtaan": "Brown",    # น้ำตาล
        "namtaal": "Brown",
        "thao": "Gray",        # เทา
        "etha": "Gray",
        "ethaa": "Gray",
        "gray": "Gray",
        "silver": "Silver",
        "เงิน": "Silver",
        "น้ำเงิน": "Blue",
        "blue": "Blue",
        "เขียว": "Green",
        "green": "Green",
        "เหลือง": "Yellow",
        "yellow": "Yellow",
    }
    
    def norm_color(s):
        s = clean_str(s)
        if not s: return None
        key = re.sub(r"[^A-Za-zก-๙]", "", s).lower()
        return COLOR_MAP.get(key, s.title())
    
    # gear: map Thai words / mojibake to Auto/Manual
    def norm_gear(s):
        s = clean_str(s)
        if not s: return None
        t = s.lower()
        # common Thai / broken Thai hits for automatic
        if any(k in t for k in ["auto", "อัตโนมัติ", "ekiiyr", "ออโต้"]):
            return "Auto"
        if any(k in t for k in ["manual", "ธรรมดา", "กระปุก"]):
            return "Manual"
        return s.title()
    
    def to_int(x):
        if x is None:
            return None
        if pd.isna(x):
            return None
        x_str = str(x).strip()
        if x_str in ("", "Not Applicable", "None", "NaN", "nan"):
            return None
        try:
            # Handle float inputs (e.g., 2023.0 -> 2023)
            if isinstance(x, float):
                if math.isnan(x):
                    return None
                x = int(x)
            # remove commas and non-digits
            n = re.sub(r"[^\d\-]", "", x_str)
            return int(n) if n not in ("", "-",) else None
        except:
            return None
    
    def to_date_or_today(x):
        if not clean_str(x):
            return date.today()
        try:
            return pd.to_datetime(x).date()
        except:
            return date.today()
    
    # --- 3) Apply normalizations ------------------------------------------------
    df["brand"]       = df["brand"].map(clean_str)
    df["model"]       = df["model"].map(clean_str)
    df["submodel"]    = df.get("submodel", None)
    df["submodel"]    = df["submodel"].map(clean_str) if "submodel" in df else None
    df["gear"]        = df["gear"].map(norm_gear)
    df["engine"]      = df["engine"].map(clean_str)
    df["color"]       = df["color"].map(norm_color)
    df["year"]        = df["year"].map(to_int).astype("Int64")  # Nullable integer type
    df["mileage"]     = df["mileage"].map(to_int).astype("Int64")
    df["price"]       = df["price"].map(to_int).astype("Int64")
    df["priceType"]   = df.get("priceType", None)
    df["priceType"]   = df["priceType"].map(clean_str) if "priceType" in df else None
    df["sourceName"]  = df.get("sourceName", None)
    df["sourceName"]  = df["sourceName"].map(clean_str) if "sourceName" in df else None
    df["sourceType"]  = df.get("sourceType", None)
    df["sourceType"]  = df["sourceType"].map(clean_str) if "sourceType" in df else None
    df["sourceUrl"]   = df.get("sourceUrl", None)
    df["sourceUrl"]   = df["sourceUrl"].map(clean_str) if "sourceUrl" in df else None
    df["externalId"]  = df.get("externalId", None)
    df["externalId"]  = df["externalId"].map(clean_str) if "externalId" in df else None
    if "dateRecorded" in df.columns:
        df["dateRecorded"] = df["dateRecorded"].map(to_date_or_today)
    else:
        df["dateRecorded"] = date.today()
    
    # (optional) keep only columns that exist in the table, in the right order
    COLS = [
        "brand","model","submodel","year","gear","engine","color",
        "mileage","price","priceType","sourceName","sourceType",
        "sourceUrl","externalId","dateRecorded"
    ]
    df = df[COLS]
    
    print("Rows to upsert:", len(df))
    print(df.head(3))
    
    # --- 4) Upsert (ON DUPLICATE KEY) using the unique key (sourceName, externalId)
    # NOTE: rows with NULL externalId will NOT trigger the unique constraint.
    placeholders = ",".join(["%s"]*len(COLS))
    col_list     = ", ".join(COLS)
    update_set   = ", ".join([f"{c}=VALUES({c})" for c in COLS])
    
    sql = f"""
    INSERT INTO car_listings_master ({col_list})
    VALUES ({placeholders})
    ON DUPLICATE KEY UPDATE {update_set}
    """
    
    # Convert rows to tuples, ensuring proper types (int not float, date objects, etc.)
    rows = []
    for i in range(len(df)):
        row = []
        for c in COLS:
            val = df[c].iloc[i]
            # Convert pandas Int64 nullable integers to Python int (not float)
            if pd.api.types.is_integer_dtype(df[c].dtype) and pd.notna(val):
                val = int(val)
            # Ensure None for NaN values
            elif pd.isna(val):
                val = None
            row.append(val)
        rows.append(tuple(row))
    
    # Use raw connection for executemany (bulk insert with proper parameter binding)
    raw_conn = engine.raw_connection()
    try:
        cursor = raw_conn.cursor()
        try:
            # Set charset for Thai text
            cursor.execute("SET NAMES utf8mb4;")
            # Execute bulk insert
            cursor.executemany(sql, rows)
            raw_conn.commit()
        except Exception as e:
            raw_conn.rollback()
            raise
        finally:
            cursor.close()
    finally:
        raw_conn.close()
    
    print("✅ Upsert complete.")

if __name__ == '__main__':
    # allow filename override via CLI: python mysql.py <filename>
    if len(sys.argv) > 1:
        globals()['INPUT_FILENAME'] = sys.argv[1]
    main()
