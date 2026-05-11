# Teardown — Project 02: Data Skewness

## What I built

A PySpark experiment that generates a 5M-row synthetic payment event
dataset in two versions — uniform distribution and skewed distribution
(one merchant owning 60% of rows) — then measures the impact of skew
and the effectiveness of two mitigation techniques using partition
distribution statistics as the locally observable proxy for cluster
task duration.

## Scripts

| Script | Purpose |
|---|---|
| `generate_skewed.py` | Produces uniform and skewed Parquet datasets |
| `observe_skew.py` | Runs groupBy on both datasets, exposes Spark UI |
| `fix_salting.py` | Two-pass aggregation using salted keys |
| `fix_repartition.py` | Round-robin redistribution before groupBy |
| `detect_skew.py` | Measures partition size distribution across all four scenarios |
| `benchmark.py` | End-to-end job duration comparison |

## Key results

### Partition distribution (detect_skew.py)

| Scenario | Partitions | Max rows | Min rows | Median rows | Skew ratio |
|---|---|---|---|---|---|
| Uniform — no fix | 50 | 189,958 | 40,236 | 99,800 | 1.9x |
| Skewed — no fix | 50 | 3,072,668 | 16,003 | 40,003 | 76.8x |
| Skewed + salting | 50 | 643,014 | 31,678 | 41,266 | 15.6x |
| Skewed + repartition | 50 | 100,002 | 99,998 | 100,000 | 1.0x |

### Job duration (benchmark.py)

| Scenario | Duration | vs Uniform |
|---|---|---|
| Uniform — no skew | 2.84s | 1.0x |
| Skewed — no fix | 0.84s | 0.3x |
| Skewed + salting | 1.64s | 0.6x |
| Skewed + repartition | 3.59s | 1.3x |

## What the partition distribution proves

The skew ratio is the key metric. A ratio of 76.8x on the raw skewed
dataset means one partition received 3,072,668 rows while the median
partition received 40,003. In a real cluster that single partition
would become the straggler task — running 76× longer than every other
task while all other executors sit idle waiting for it to finish.

Salting reduced the ratio from 76.8x to 15.6x — a 5× improvement but
not elimination. The residual skew comes from random salt distribution
being imperfect — 10 buckets do not split 3M rows into exactly 300k
each. Increasing SALT_BUCKETS would reduce the ratio further at the
cost of more aggregation overhead.

Repartition achieved a perfect 1.0x ratio — max 100,002 vs min 99,998
across 5M rows. Round-robin assignment is mechanically even. The cost
is two full shuffle passes instead of one.

## The counterintuitive benchmark result

The benchmark.py job duration numbers appear to contradict the theory —
the skewed raw job was *faster* than uniform, and repartition was the
*slowest* scenario. This is not wrong data. It is a local mode
limitation.

In Spark local mode everything runs in a single JVM on one machine.
The straggler task runs on the same CPU as all other tasks — there is
no network shuffle between machines, no executor waiting overhead, and
Spark's task scheduler fills idle cores with other work while the hot
partition processes. The penalty that destroys cluster performance is
invisible locally.

The benchmark.py numbers measure JVM and shuffle coordination overhead,
not distributed execution cost. The detect_skew.py partition ratios are
the correct locally observable metric for skew severity.

**Reference detect_skew.py results to understand skew impact.
Reference benchmark.py results to understand the overhead of each fix.**

## Decisions made during the build

**Disabling Adaptive Query Execution.**
AQE in Spark 3+ automatically detects and mitigates skew at runtime.
Setting `spark.sql.adaptive.enabled=false` was essential — with AQE
enabled Spark would have applied its own skew hints automatically,
masking the problem entirely and making the experiment meaningless.

**5M rows over 500k.**
500k rows completed too fast on a local i7-11850H to observe the Spark
UI task timeline. 5M rows gave enough volume for the partition
distribution numbers to be meaningful and for the UI jobs to stay
active long enough to navigate to the Stages view.

**Partition stats over task timeline.**
The Spark UI task timeline is the canonical way to observe skew but it
only exists while a job is active. `detect_skew.py` produces a
persistent, reproducible, shareable table that tells the same story
without requiring real-time UI observation.

## What I would do differently

Run on a multi-node Docker Compose Spark cluster (master + 2 workers)
to make the straggler task penalty visible in job duration, not just
partition statistics. The partition distribution numbers prove the skew
is real, a cluster would prove the performance cost is real.

Test adaptive salting, dynamically determining SALT_BUCKETS based on
the actual skew ratio of the hot key rather than hardcoding 10. A
ratio of 76.8x suggests ~77 buckets would theoretically bring the
ratio to 1.0x, matching repartition's balance without the double
shuffle cost.