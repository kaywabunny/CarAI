#!/usr/bin/env python3
import argparse
import io
from fastapi.responses import StreamingResponse
import json
import urllib.request
import urllib.error
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")

def to_price_bands(d):
    return {
        "green_price": float(d["green_price"]),
        "yellow_price": float(d["yellow_price"]),
        "red_price": float(d["red_price"]),
        "confidence": float(d.get("confidence")) if d.get("confidence") is not None else None,
    }


def fetch_from_api(api_url: str, make: str, model: str, year: int, mileage_km: int):
    payload = {
        "make": make,
        "model": model,
        "year": int(year),
        "mileage_km_num": int(mileage_km),
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(api_url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            out = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        text = e.read().decode("utf-8", "ignore")
        raise SystemExit(f"API error {e.code}: {text}")
    except urllib.error.URLError as e:
        raise SystemExit(f"Failed to reach API: {e.reason}")
    return to_price_bands(out)


def plot_price_bands(bands, title=None, save_path=None):
    labels = ["Green (sell fast)", "Yellow (median)", "Red (hold out)"]
    vals = [bands["green_price"], bands["yellow_price"], bands["red_price"]]

    fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=140)
    bars = ax.bar(labels, vals)
    ax.set_title(title or "Price Bands (THB)")
    ax.set_ylabel("THB")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    for b in bars:
        y = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, y, f"{y:,.0f}", ha="center", va="bottom", fontsize=9)

    if bands.get("confidence") is not None:
        ax.text(0.99, 0.02, f"Confidence: {bands['confidence']:.2f}", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=9)

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


def main():
    p = argparse.ArgumentParser(description="Plot price bands (green/yellow/red) from values or API.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--green", type=float, help="Green (sell fast) price")
    src.add_argument("--api", type=str, help="POST endpoint (e.g., http://localhost:8000/price)")

    p.add_argument("--yellow", type=float, help="Yellow (median) price")
    p.add_argument("--red", type=float, help="Red (hold out) price")

    p.add_argument("--make", type=str, help="Car make")
    p.add_argument("--model", type=str, help="Car model")
    p.add_argument("--year", type=int, help="Year")
    p.add_argument("--mileage", type=int, help="Mileage (km)")

    p.add_argument("--title", type=str, default=None, help="Chart title")
    p.add_argument("--out", type=str, default=None, help="Save figure to this path (e.g., bands.png)")

    args = p.parse_args()

    if args.api:
        missing = [k for k in ("make", "model", "year", "mileage") if getattr(args, k) is None]
        if missing:
            raise SystemExit(f"--api mode requires: --make, --model, --year, --mileage (missing: {', '.join(missing)})")
        bands = fetch_from_api(args.api, args.make, args.model, args.year, args.mileage)
    else:
        if args.green is None or args.yellow is None or args.red is None:
            raise SystemExit("Direct mode requires --green, --yellow, --red")
        bands = {
            "green_price": args.green,
            "yellow_price": args.yellow,
            "red_price": args.red,
            "confidence": None,
        }

    return plot_price_bands(bands, title=args.title, save_path=args.out)


if __name__ == "__main__":
    main()
