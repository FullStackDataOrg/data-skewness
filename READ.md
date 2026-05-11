# Data Skewness

A hands-on PySpark experiment that generates, observes, measures, and
fixes data skewness using partition distribution statistics as the
locally observable proxy for cluster straggler task behaviour.

No cluster required. Spark runs in local mode using all available cores.

## What this project answers

- What is data skewness and why does it destroy distributed job performance?
- How do you detect skew before running a job?
- What is salting and when should you use it?
- What is repartitioning and what does it cost?
- Why does the benchmark show counterintuitive results on local mode?

## Stack

| Tool | Role |
|---|---|
| PySpark 3.5.1 | Distributed processing engine (local mode) |
| Python 3.12 | Runtime |
| pandas | Dataset generation |
| pyarrow | Parquet read/write |
| tabulate | Results formatting |

## Project structure
02-data-skewness/
├── generate_skewed.py    # Uniform and skewed dataset generator
├── observe_skew.py       # GroupBy on both datasets, Spark UI observation
├── fix_salting.py        # Two-pass aggregation with salted keys
├── fix_repartition.py    # Round-robin redistribution before groupBy
├── detect_skew.py        # Partition distribution statistics
├── benchmark.py          # End-to-end job duration comparison
└── data/                 # Generated Parquet files — gitignored

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python generate_skewed.py           # generates uniform + skewed parquet
python observe_skew.py              # open localhost:4040 while running
python detect_skew.py               # partition distribution — the key metric
python benchmark.py                 # job duration comparison
```

## Results

### Partition distribution (the skew signal)

| Scenario | Max rows | Median rows | Skew ratio |
|---|---|---|---|
| Uniform — no fix | 189,958 | 99,800 | 1.9x |
| Skewed — no fix | 3,072,668 | 40,003 | 76.8x |
| Skewed + salting | 643,014 | 41,266 | 15.6x |
| Skewed + repartition | 100,002 | 100,000 | 1.0x |

The skew ratio = max partition rows ÷ median partition rows.
In a real cluster this ratio approximates the straggler task
duration multiplier — a 76.8x ratio means your job runs 76×
slower than it should.

### Job duration (local mode — read with context)

| Scenario | Duration | vs Uniform |
|---|---|---|
| Uniform — no skew | 2.84s | 1.0x |
| Skewed — no fix | 0.84s | 0.3x |
| Skewed + salting | 1.64s | 0.6x |
| Skewed + repartition | 3.59s | 1.3x |

> The skewed raw job appears *faster* than uniform on local.
> This is a local mode artefact — see teardown.md for the full
> explanation. Use detect_skew.py partition ratios to understand
> skew severity. Use benchmark.py to understand fix overhead.

## Key findings

**Salting** reduces the skew ratio from 76.8x to 15.6x — a 5×
improvement. It does not eliminate skew because random salt
distribution across 10 buckets is imperfect. More buckets lower
the ratio at the cost of more aggregation overhead.

**Repartition** achieves a perfect 1.0x ratio through round-robin
assignment. The cost is two full shuffle passes. It is the blunt
instrument — effective for skew spread across many keys, expensive
for skew concentrated in one.

**Local mode cannot show job duration penalties from skew.**
A multi-node cluster is required to observe the straggler task
duration impact. The partition distribution metric is the correct
locally observable signal.

## When to use each fix

| Fix | Use when |
|---|---|
| Salting | A small number of known hot keys dominate |
| Repartition | Skew is spread across many keys |
| AQE (Spark 3+) | You want Spark to handle it automatically in production |