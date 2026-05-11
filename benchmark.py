"""
benchmark.py
------------
Runs all four scenarios back to back and prints a comparison table:

    1. Uniform distribution  — baseline, no skew
    2. Skewed, no fix        — demonstrates the straggler problem
    3. Skewed + salting      — two-pass aggregation fix
    4. Skewed + repartition  — explicit shuffle fix

Usage:
    python generate_skewed.py   # run once
    python benchmark.py
"""

import time
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from tabulate import tabulate


SALT_BUCKETS = 10


def build_spark() -> SparkSession:
    return (
        SparkSession.builder
        .master("local[*]")
        .appName("02-benchmark")
        .config("spark.sql.adaptive.enabled", "false")
        .config("spark.sql.shuffle.partitions", "50")
        .getOrCreate()
    )


def scenario_uniform(spark: SparkSession) -> float:
    df = spark.read.parquet("data/events_uniform.parquet")
    t0 = time.perf_counter()
    df.groupBy("merchant_id") \
      .agg(F.sum("amount"), F.count("*")) \
      .write.mode("overwrite").parquet("data/bench_uniform.parquet")
    return time.perf_counter() - t0


def scenario_skewed_raw(spark: SparkSession) -> float:
    df = spark.read.parquet("data/events_skewed.parquet")
    t0 = time.perf_counter()
    df.groupBy("merchant_id") \
      .agg(F.sum("amount"), F.count("*")) \
      .write.mode("overwrite").parquet("data/bench_skewed_raw.parquet")
    return time.perf_counter() - t0


def scenario_salted(spark: SparkSession) -> float:
    df = spark.read.parquet("data/events_skewed.parquet")
    t0 = time.perf_counter()
    salted = df.withColumn(
        "merchant_id_salted",
        F.concat(
            F.col("merchant_id"),
            F.lit("_"),
            (F.floor(F.rand() * SALT_BUCKETS)).cast("string")
        )
    )
    partial = salted.groupBy("merchant_id_salted") \
                    .agg(F.sum("amount").alias("pr"), F.count("*").alias("pc"))
    partial = partial.withColumn(
        "merchant_id",
        F.expr("substring_index(merchant_id_salted, '_', 1)")
    )
    partial.groupBy("merchant_id") \
           .agg(F.sum("pr"), F.sum("pc")) \
           .write.mode("overwrite").parquet("data/bench_salted.parquet")
    return time.perf_counter() - t0


def scenario_repartitioned(spark: SparkSession) -> float:
    df = spark.read.parquet("data/events_skewed.parquet")
    t0 = time.perf_counter()
    df.repartition(200) \
      .groupBy("merchant_id") \
      .agg(F.sum("amount"), F.count("*")) \
      .write.mode("overwrite").parquet("data/bench_repartitioned.parquet")
    return time.perf_counter() - t0


def main():
    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")

    print("\n  Running all four scenarios ...\n")

    scenarios = [
        ("Uniform — no skew",      scenario_uniform),
        ("Skewed — no fix",        scenario_skewed_raw),
        ("Skewed + salting",       scenario_salted),
        ("Skewed + repartition",   scenario_repartitioned),
    ]

    results = []
    baseline = None

    for label, fn in scenarios:
        print(f"  {label} ...", end="", flush=True)
        elapsed = fn(spark)
        if baseline is None:
            baseline = elapsed
        ratio = elapsed / baseline
        print(f" {elapsed:.2f}s")
        results.append([label, f"{elapsed:.2f}s", f"{ratio:.1f}x"])

    print(f"\n{'=' * 56}")
    print("  Benchmark results")
    print(f"{'=' * 56}")
    print(tabulate(
        results,
        headers=["Scenario", "Duration", "vs Uniform"],
        tablefmt="rounded_outline"
    ))
    print()
    print("  Ratio vs uniform baseline.")
    print("  Skewed raw shows the straggler cost.")
    print("  Salting and repartition show the recovery.\n")

    spark.stop()


if __name__ == "__main__":
    main()