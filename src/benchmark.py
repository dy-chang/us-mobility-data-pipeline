import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
"""
High-Performance Benchmark Suite: Pandas vs Polars vs DuckDB.
Measures execution latency, peak memory consumption (RSS), and I/O efficiency
across large-scale mobility datasets.
"""

import argparse
import gc
import json
import os
from pathlib import Path
import time
from typing import Any, Dict, List
import duckdb
import pandas as pd
import polars as pl
import psutil

from src.config import CORRIDORS
from src.etl import MobilityETLPipeline
from src.generator import ProbeTrajectoryGenerator


def get_current_memory_mb() -> float:
    """Return current process Resident Set Size (RSS) in Megabytes."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


class MobilityBenchmarkRunner:
    """
    Executes systematic benchmarks comparing Pandas, Polars, and DuckDB
    on identical analytical aggregation workloads.
    """

    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.glob_path = str(data_path / "**/*.parquet")

    def benchmark_pandas(self) -> Dict[str, Any]:
        """Benchmark Vanilla Pandas loading and in-memory aggregation."""
        gc.collect()
        mem_before = get_current_memory_mb()
        t0 = time.time()

        # Read parquet
        df = pd.read_parquet(self.data_path)
        
        # Aggregation: group by segment_id and hour, compute count, mean, median, p85
        grouped = df.groupby(["segment_id", "hour"])["speed_mph"].agg(
            count="count",
            mean="mean",
            median="median",
            p85=lambda x: x.quantile(0.85)
        ).reset_index()

        elapsed = time.time() - t0
        mem_peak = get_current_memory_mb()
        del df, grouped
        gc.collect()

        return {
            "engine": "Pandas (In-Memory)",
            "latency_sec": round(elapsed, 3),
            "peak_memory_mb": round(mem_peak - mem_before, 1),
            "status": "Completed",
        }

    def benchmark_polars(self) -> Dict[str, Any]:
        """Benchmark Polars Lazy multi-threaded scan and aggregation."""
        gc.collect()
        mem_before = get_current_memory_mb()
        t0 = time.time()

        lf = pl.scan_parquet(self.glob_path)
        res = (
            lf.group_by(["segment_id", "hour"])
            .agg([
                pl.col("speed_mph").count().alias("count"),
                pl.col("speed_mph").mean().alias("mean"),
                pl.col("speed_mph").median().alias("median"),
                pl.col("speed_mph").quantile(0.85).alias("p85"),
            ])
            .collect()
        )

        elapsed = time.time() - t0
        mem_peak = get_current_memory_mb()
        del lf, res
        gc.collect()

        return {
            "engine": "Polars (Lazy Vectorized)",
            "latency_sec": round(elapsed, 3),
            "peak_memory_mb": round(max(mem_peak - mem_before, 1.0), 1),
            "status": "Completed",
        }

    def benchmark_duckdb(self) -> Dict[str, Any]:
        """Benchmark DuckDB out-of-core columnar SQL query."""
        gc.collect()
        mem_before = get_current_memory_mb()
        t0 = time.time()

        conn = duckdb.connect(database=":memory:")
        conn.execute("SET max_memory = '2GB'")
        conn.execute("SET threads = 4")

        query = f"""
        SELECT 
            segment_id,
            hour,
            COUNT(*) AS count,
            AVG(speed_mph) AS mean,
            MEDIAN(speed_mph) AS median,
            QUANTILE_CONT(speed_mph, 0.85) AS p85
        FROM read_parquet('{self.glob_path}', hive_partitioning = true)
        GROUP BY segment_id, hour;
        """
        res = conn.execute(query).fetchdf()

        elapsed = time.time() - t0
        mem_peak = get_current_memory_mb()
        del res, conn
        gc.collect()

        return {
            "engine": "DuckDB (Out-of-Core Columnar)",
            "latency_sec": round(elapsed, 3),
            "peak_memory_mb": round(max(mem_peak - mem_before, 1.0), 1),
            "status": "Completed",
        }

    def run_suite(self, skip_pandas_threshold: int = 15_000_000, total_rows: int = 1_000_000) -> List[Dict[str, Any]]:
        """Run full comparative suite."""
        results = []

        # 1. DuckDB
        print("[Benchmark] Running DuckDB...")
        d_res = self.benchmark_duckdb()
        results.append(d_res)

        # 2. Polars
        print("[Benchmark] Running Polars...")
        p_res = self.benchmark_polars()
        results.append(p_res)

        # 3. Pandas (skip if row count exceeds threshold to avoid OOM)
        if total_rows <= skip_pandas_threshold:
            print("[Benchmark] Running Pandas...")
            try:
                pd_res = self.benchmark_pandas()
                results.append(pd_res)
            except Exception as e:
                results.append({
                    "engine": "Pandas (In-Memory)",
                    "latency_sec": float("nan"),
                    "peak_memory_mb": float("nan"),
                    "status": f"OOM / Error ({str(e)})",
                })
        else:
            print(f"[Benchmark] Skipping Pandas ({total_rows:,} rows > {skip_pandas_threshold:,} limit) due to OOM risk.")
            results.append({
                "engine": "Pandas (In-Memory)",
                "latency_sec": float("nan"),
                "peak_memory_mb": float("nan"),
                "status": "Skipped (OOM Risk)",
            })

        return results


def main():
    parser = argparse.ArgumentParser(description="Mobility Data Engineering Benchmark Suite")
    parser.add_argument("--rows", type=int, default=1_000_000, help="Number of probe records to benchmark")
    parser.add_argument("--corridor", type=str, default="I-95_MIA", help="Corridor ID")
    parser.add_argument("--output-dir", type=str, default="data/benchmark_data", help="Directory for benchmark data")
    args = parser.parse_args()

    out_path = Path(args.output_dir)
    print(f"=== US-Scale Mobility Benchmark ===")
    print(f"Target Scale: {args.rows:,} records")
    print(f"Corridor: {args.corridor}")
    print(f"Data Directory: {out_path}")

    if not (out_path / f"corridor_id={args.corridor}").exists():
        print("\n[1/3] Generating calibrated probe stream & executing ETL...")
        gen = ProbeTrajectoryGenerator()
        etl = MobilityETLPipeline()
        stream = gen.stream_dataset(total_records=args.rows, corridor_id=args.corridor, chunk_size=500_000)
        stats = etl.process_and_persist(stream, corridor_id=args.corridor, output_dir=out_path)
        print(f"Ingestion completed: {stats['total_valid_records']:,} valid records in {stats['storage_size_mb']} MB Parquet.")
    else:
        print("\n[1/3] Existing dataset detected, reusing data.")

    print("\n[2/3] Executing Comparative Performance Tests...")
    runner = MobilityBenchmarkRunner(out_path)
    results = runner.run_suite(total_rows=args.rows)

    print("\n[3/3] Benchmark Results:")
    print(f"{'Engine':<30} | {'Latency (sec)':<15} | {'Peak RAM (MB)':<15} | {'Status':<15}")
    print("-" * 80)
    for r in results:
        lat = f"{r['latency_sec']:.3f}s" if not pd.isna(r['latency_sec']) else "N/A"
        mem = f"{r['peak_memory_mb']:.1f} MB" if not pd.isna(r['peak_memory_mb']) else "N/A"
        print(f"{r['engine']:<30} | {lat:<15} | {mem:<15} | {r['status']:<15}")

    bench_dir = Path("benchmarks")
    bench_dir.mkdir(exist_ok=True)
    summary_file = bench_dir / "results_summary.md"
    with open(summary_file, "w") as f:
        f.write(f"# Benchmark Results: Large-Scale Mobility Analytics\n\n")
        f.write(f"**Test Scale**: {args.rows:,} records  \n")
        f.write(f"**Corridor**: {args.corridor}  \n\n")
        f.write("| Engine | Latency (sec) | Peak Memory (MB) | Relative Speedup | Status |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        
        baseline_lat = next((r["latency_sec"] for r in results if "Pandas" in r["engine"] and not pd.isna(r["latency_sec"])), None)
        
        for r in results:
            lat_str = f"{r['latency_sec']:.3f} s" if not pd.isna(r['latency_sec']) else "N/A"
            mem_str = f"{r['peak_memory_mb']:.1f} MB" if not pd.isna(r['peak_memory_mb']) else "N/A"
            if baseline_lat and not pd.isna(r["latency_sec"]) and r["latency_sec"] > 0:
                speedup = f"{baseline_lat / r['latency_sec']:.1f}x"
            else:
                speedup = "-"
            f.write(f"| **{r['engine']}** | {lat_str} | {mem_str} | {speedup} | {r['status']} |\n")
    
    print(f"\nSummary saved to {summary_file}")


if __name__ == "__main__":
    main()
