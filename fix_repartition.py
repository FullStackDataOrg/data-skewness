"""
fix_repartition.py
------------------
Fixes skew using explicit repartitioning before the shuffle.

repartition(n) redistributes rows across n partitions using round-robin
assignment — it does not respect key boundaries. This means the hot
merchant's rows are spread across many partitions, but each partition
now contains rows from multiple merchants. The subsequent groupBy must
do a second shuffle to bring matching keys together.

repartition() is simpler than salting but adds a full extra shuffle pass.
Use it when:
  - Your dataset is heavily skewed across many keys, not just one hot key
  - You want to control parallelism for downstream operations
  - The extra shuffle cost is acceptable given the skew severity

repartitionByRange() is an alternative that creates roughly equal-sized
partitions while keeping similar key values together — better for
range-based queries and sorted output.

Usage:
    python fix_repartition.py
"""

import time
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


def build_spark() -> SparkSession:
    return (
        SparkSession.builder
        .master("local[*]")
        .appName("02-fix-repartition")
        .config("spark.sql.adaptive.enabled", "false")
        .config("spark.sql.shuffle.partitions", "50")
        .getOrCreate()
    )


def run_repartitioned(spark: SparkSession) -> float:
    df = spark.read.parquet("data/events_skewed.parquet")

    t0 = time.perf_counter()

    # Repartition into 200 partitions using round-robin assignment.
    # This breaks up the hot merchant's rows before the groupBy shuffle.
    # The trade-off: this itself is a shuffle operation — you are adding
    # one shuffle to avoid a worse shuffle downstream.
    result = (
        df.repartition(200)
          .groupBy("merchant_id")
          .agg(
              F.sum("amount").alias("total_revenue"),
              F.count("*").alias("tx_count"),
              F.avg("amount").alias("avg_amount"),
          )
          .orderBy(F.desc("total_revenue"))
    )

    result.write.mode("overwrite").parquet("data/result_repartitioned.parquet")
    elapsed = time.perf_counter() - t0

    print(f"\n  [repartition]  Job completed in {elapsed:.2f}s  (200 partitions)")

    # Also demonstrate repartitionByRange for comparison
    t1 = time.perf_counter()
    result_range = (
        df.repartitionByRange(200, "merchant_id")
          .groupBy("merchant_id")
          .agg(
              F.sum("amount").alias("total_revenue"),
              F.count("*").alias("tx_count"),
          )
          .orderBy(F.desc("total_revenue"))
    )
    result_range.write.mode("overwrite").parquet("data/result_repartitioned_range.parquet")
    elapsed_range = time.perf_counter() - t1

    print(f"  [repartitionByRange]  Job completed in {elapsed_range:.2f}s  (200 partitions)")
    print("\n  repartition()        = round-robin, even partition sizes, extra shuffle")
    print("  repartitionByRange() = range-based, sorted output, better for range queries")

    return elapsed


def main():
    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")
    run_repartitioned(spark)
    spark.stop()


if __name__ == "__main__":
    main()