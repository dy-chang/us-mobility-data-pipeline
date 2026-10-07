# Benchmark Results: Large-Scale Mobility Analytics

**Test Scale**: 10,000,000 records  
**Corridor**: I-95_MIA  

| Engine | Latency (sec) | Peak Memory (MB) | Relative Speedup | Status |
| :--- | :--- | :--- | :--- | :--- |
| **DuckDB (Out-of-Core Columnar)** | 0.318 s | 102.5 MB | 3.8x | Completed |
| **Polars (Lazy Vectorized)** | 0.303 s | 110.2 MB | 4.0x | Completed |
| **Pandas (In-Memory)** | 1.213 s | 1319.7 MB | 1.0x | Completed |
