"""
generate_skewed.py
------------------
Produces two versions of the same payment event dataset:

    events_uniform.parquet  — 500 merchants, evenly distributed (~0.2% each)
    events_skewed.parquet   — 500 merchants, but one merchant owns 60% of rows

Same schema as Project 01 so the datasets are familiar.
Saved as Parquet — the format we established as the production standard.

Usage:
    python generate_skewed.py
    python generate_skewed.py --rows 
"""

import argparse
import random
from pathlib import Path

import pandas as pd


def generate(n_rows: int, skewed: bool, seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)

    merchants = [f"mer-{i:05d}" for i in range(1, 501)]
    users     = [f"usr-{i:05d}" for i in range(1, 10_001)]

    if skewed:
        # One merchant gets 60% of all rows — the remaining 499 share 40%
        # This simulates a dominant merchant in a real payments dataset,
        # e.g. a large marketplace like Amazon or Shopify on your platform
        hot_merchant = merchants[0]
        weights = [60] + [40 / 499] * 499
        merchant_id = rng.choices(merchants, weights=weights, k=n_rows)
    else:
        # Uniform: every merchant gets roughly 1/500 of rows
        merchant_id = rng.choices(merchants, k=n_rows)

    currencies = ["USD", "CAD", "GBP", "EUR", "NGN"]
    statuses   = ["approved", "declined", "pending"]
    sw         = [85, 10, 5]

    import time
    now_ms  = int(time.time() * 1000)
    span_ms = 90 * 24 * 60 * 60 * 1000

    data = {
        "event_id":    [f"evt-{i:08d}" for i in range(n_rows)],
        "user_id":     rng.choices(users, k=n_rows),
        "merchant_id": merchant_id,
        "amount":      [round(rng.uniform(0.01, 9999.99), 2) for _ in range(n_rows)],
        "currency":    rng.choices(currencies, k=n_rows),
        "status":      rng.choices(statuses, weights=sw, k=n_rows),
        "event_ts":    [now_ms - rng.randint(0, span_ms) for _ in range(n_rows)],
        "is_flagged":  rng.choices([True, False], weights=[3, 97], k=n_rows),
    }

    return pd.DataFrame(data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=5_000_000)
    args = parser.parse_args()

    Path("data").mkdir(exist_ok=True)

    for skewed in [False, True]:
        label = "skewed" if skewed else "uniform"
        print(f"Generating {args.rows:,} rows — {label} ...", end="", flush=True)
        df = generate(n_rows=args.rows, skewed=skewed)
        out = f"data/events_{label}.parquet"
        df.to_parquet(out, index=False)
        size_mb = Path(out).stat().st_size / 1_048_576

        if skewed:
            top = df["merchant_id"].value_counts().iloc[0]
            pct = top / len(df) * 100
            print(f" done  |  {size_mb:.1f} MB  |  top merchant owns {pct:.1f}% of rows")
        else:
            print(f" done  |  {size_mb:.1f} MB")


if __name__ == "__main__":
    main()