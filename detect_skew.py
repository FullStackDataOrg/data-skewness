"""
detect_skew.py
--------------
Measures partition size distribution before and after each technique.
This is the locally observable proxy for what the Spark UI task timeline
shows in a cluster — uneven partition sizes = uneven task durations.

Usage:
    python detect_skew.py
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from tabulate import tabulate


SALT_BUCKETS = 10


def build_spark() -> SparkSession:
    return (
        SparkSession.builder
        .master("local[*]")
        .appName("02-detect-skew")
        .config("spark.sql.adaptive.enabled", "false")
        .config("spark.sql.shuffle.partitions", "50")
        .getOrCreate()
    )


def partition_stats(df, label: str) -> dict:
    """
    Collects rows-per-partition distribution and returns summary stats.
    The gap between max and median is the skew signal —
    in a balanced dataset max ≈ median.
    In a skewed dataset max >> median.
    """
    counts = (
        df.withColumn("pid", F.spark_partition_id())
          .groupBy("pid")
          .count()
          .agg(
              F.count("pid").alias("num_partitions"),
              F.sum("count").alias("total_rows"),
              F.max("count").alias("max_rows"),
              F.min("count").alias("min_rows"),
              F.expr("percentile(count, 0.5)").alias("median_rows"),
              F.stddev("count").alias("stddev_rows"),
          )
          .collect()[0]
    )

    skew_ratio = counts["max_rows"] / counts["median_rows"] if counts["median_rows"] else 0

    return {
        "scenario":       label,
        "partitions":     counts["num_partitions"],
        "max_rows":       f"{int(counts['max_rows']):,}",
        "min_rows":       f"{int(counts['min_rows']):,}",
        "median_rows":    f"{int(counts['median_rows']):,}",
        "skew_ratio":     f"{skew_ratio:.1f}x",
    }


def main():
    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")

    df_uniform = spark.read.parquet("data/events_uniform.parquet")
    df_skewed  = spark.read.parquet("data/events_skewed.parquet")

    results = []

    # ── Scenario 1: Uniform, no fix ─────────────────────────────────────────
    after_shuffle_uniform = (
        df_uniform
        .repartition(50, "merchant_id")  # simulate the groupBy shuffle
    )
    results.append(partition_stats(after_shuffle_uniform, "Uniform — no fix"))

    # ── Scenario 2: Skewed, no fix ───────────────────────────────────────────
    # repartition by merchant_id simulates what groupBy does internally —
    # all rows for the same key land on the same partition
    after_shuffle_skewed = (
        df_skewed
        .repartition(50, "merchant_id")
    )
    results.append(partition_stats(after_shuffle_skewed, "Skewed — no fix"))

    # ── Scenario 3: Skewed + salting ─────────────────────────────────────────
    salted = df_skewed.withColumn(
        "merchant_id_salted",
        F.concat(
            F.col("merchant_id"),
            F.lit("_"),
            (F.floor(F.rand(seed=42) * SALT_BUCKETS)).cast("string")
        )
    )
    after_shuffle_salted = salted.repartition(50, "merchant_id_salted")
    results.append(partition_stats(after_shuffle_salted, "Skewed + salting"))

    # ── Scenario 4: Skewed + repartition ─────────────────────────────────────
    after_repartition = df_skewed.repartition(50)  # round-robin, ignores keys
    results.append(partition_stats(after_repartition, "Skewed + repartition"))

    # ── Print results ────────────────────────────────────────────────────────
    headers = ["Scenario", "Partitions", "Max rows", "Min rows", "Median rows", "Skew ratio"]
    rows = [[r["scenario"], r["partitions"], r["max_rows"],
             r["min_rows"], r["median_rows"], r["skew_ratio"]] for r in results]

    print(f"\n{'=' * 72}")
    print("  Partition distribution — skew ratio = max rows / median rows")
    print("  A ratio close to 1.0x means balanced partitions.")
    print("  A high ratio means one partition is doing most of the work.")
    print(f"{'=' * 72}")
    print(tabulate(rows, headers=headers, tablefmt="rounded_outline"))

    print("\n  What this means in a cluster:")
    print("  Max rows partition = the slowest task = the straggler.")
    print("  Skew ratio ≈ how many times longer that task runs vs the median task.\n")

    spark.stop()


if __name__ == "__main__":
    main()