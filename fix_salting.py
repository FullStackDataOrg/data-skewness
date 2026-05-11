"""
fix_salting.py
--------------
Fixes skew using the salting technique.

The problem: all rows for the hot merchant hash to the same partition.
The fix: append a random salt suffix (0-9) to the merchant_id before the
shuffle. This spreads the hot merchant's rows across 10 partitions instead
of 1. After the first aggregation, strip the salt and re-aggregate to get
the correct per-merchant totals.

Cost of salting: two aggregation passes instead of one.
When to use it: when a small number of keys dominate and you cannot
change the upstream data distribution.

Usage:
    python fix_salting.py
"""

import time
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


SALT_BUCKETS = 10  # number of partitions to spread the hot key across


def build_spark() -> SparkSession:
    return (
        SparkSession.builder
        .master("local[*]")
        .appName("02-fix-salting")
        .config("spark.sql.adaptive.enabled", "false")
        .config("spark.sql.shuffle.partitions", "50")
        .getOrCreate()
    )


def run_salted(spark: SparkSession) -> float:
    df = spark.read.parquet("data/events_skewed.parquet")

    t0 = time.perf_counter()

    # Step 1 — Add a random salt to the merchant_id
    # floor(rand() * SALT_BUCKETS) gives an integer in [0, SALT_BUCKETS)
    # concat creates keys like "mer-00001_3", "mer-00001_7" etc.
    # The hot merchant's rows now hash to 10 different keys → 10 partitions
    salted = df.withColumn(
        "merchant_id_salted",
        F.concat(
            F.col("merchant_id"),
            F.lit("_"),
            (F.floor(F.rand() * SALT_BUCKETS)).cast("string")
        )
    )

    # Step 2 — First aggregation on the salted key
    # Work is now distributed across SALT_BUCKETS partitions per merchant
    partial = (
        salted.groupBy("merchant_id_salted")
              .agg(
                  F.sum("amount").alias("partial_revenue"),
                  F.count("*").alias("partial_count"),
              )
    )

    # Step 3 — Strip the salt suffix to recover the original merchant_id
    # substring_index splits on "_" and takes everything before the last "_"
    partial = partial.withColumn(
        "merchant_id",
        F.expr("substring_index(merchant_id_salted, '_', 1)")
    )

    # Step 4 — Second aggregation to combine the partial results
    # This shuffle is cheap — only 500 distinct merchant keys, no hot key
    result = (
        partial.groupBy("merchant_id")
               .agg(
                   F.sum("partial_revenue").alias("total_revenue"),
                   F.sum("partial_count").alias("tx_count"),
               )
               .orderBy(F.desc("total_revenue"))
    )

    result.write.mode("overwrite").parquet("data/result_salted.parquet")
    elapsed = time.perf_counter() - t0

    print(f"\n  [salted]  Job completed in {elapsed:.2f}s  (salt buckets: {SALT_BUCKETS})")
    print("  Two-pass aggregation: partial on salted key → combine on original key")
    return elapsed


def main():
    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")
    run_salted(spark)
    spark.stop()


if __name__ == "__main__":
    main()