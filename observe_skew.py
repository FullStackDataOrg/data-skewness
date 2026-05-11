"""
observe_skew.py
---------------
Runs the same groupBy aggregation on both the uniform and skewed datasets.
Prints job duration for each and the top 5 merchant row counts so you can
see the distribution imbalance in the data before looking at the Spark UI.

Open localhost:4040 in your browser WHILE this script is running to observe
the task timeline. On the skewed job you will see one task bar that is
dramatically longer than all others — that is the straggler.

Usage:
    python observe_skew.py
"""

import time
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


def build_spark() -> SparkSession:
    return (
        SparkSession.builder
        .master("local[*]")
        .appName("02-observe-skew")
        # Disable adaptive query execution so skew is visible unmitigated.
        # AQE would automatically apply skew hints in Spark 3+, masking the problem.
        .config("spark.sql.adaptive.enabled", "false")
        .config("spark.sql.shuffle.partitions", "50")
        .config("spark.ui.enabled", "true")
        .getOrCreate()
    )


def run_groupby(spark: SparkSession, path: str, label: str) -> float:
    df = spark.read.parquet(path)

    # Inspect distribution before running the job
    print(f"\n  [{label}] Top 5 merchants by row count:")
    df.groupBy("merchant_id") \
      .count() \
      .orderBy(F.desc("count")) \
      .limit(5) \
      .show(truncate=False)

    # The actual aggregation — total revenue per merchant
    # This is the operation that suffers under skew because all rows for
    # the hot merchant must land on one partition during the shuffle
    t0 = time.perf_counter()
    result = (
        df.groupBy("merchant_id")
          .agg(
              F.sum("amount").alias("total_revenue"),
              F.count("*").alias("tx_count"),
              F.avg("amount").alias("avg_amount"),
          )
          .orderBy(F.desc("total_revenue"))
    )
    result.write.mode("overwrite").parquet(f"data/result_{label}.parquet")
    elapsed = time.perf_counter() - t0

    print(f"  [{label}] Job completed in {elapsed:.2f}s")
    return elapsed


def main():
    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")

    print("\n  Spark UI available at http://localhost:4040 while jobs are running\n")

    uniform_s = run_groupby(spark, "data/events_uniform.parquet", "uniform")
    input("\n  Uniform done. Open localhost:4040 → Stages, then press Enter to run the skewed job...")
    skewed_s = run_groupby(spark, "data/events_skewed.parquet", "skewed")

    slowdown = skewed_s / uniform_s
    print(f"\n  Skewed job was {slowdown:.1f}x slower than uniform")
    print("  Check localhost:4040 → Stages → Task timeline to see the straggler\n")

    spark.stop()


if __name__ == "__main__":
    main()