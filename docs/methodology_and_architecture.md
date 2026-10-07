# Engineering Specification & Theoretical Methodology

## 1. Executive Overview & Problem Context
State Departments of Transportation (DOTs) and Metropolitan Planning Organizations (MPOs) increasingly ingest terabyte-scale vehicle probe trajectories (from providers like INRIX, HERE, TomTom, and connected vehicle telematics) to evaluate network reliability. Traditional transportation modeling workflows rely on in-memory DataFrame libraries (`pandas`, `geopandas`) or heavy relational spatial databases (`PostGIS`). When scaling past $10^7$ rows, these approaches encounter severe limitations:
- **Out-of-Memory (OOM) Exceptions**: In-memory Python representations expand raw data size by $4\times$ to $8\times$.
- **High CPU Overhead in Spatial Joins**: Continuous geometry polygon-in-polygon R-Tree searches scale with $O(N \log M)$ computational complexity.
- **Cluster Complexity**: Distributed engines (e.g., Apache Spark) introduce substantial operational overhead, serialization latency, and cloud infrastructure costs.

This repository demonstrates that a **modern vectorized columnar pipeline**—combining **DuckDB out-of-core SQL**, **Hive-partitioned Apache Parquet (ZSTD)**, **Polars multi-threaded execution**, and **Uber H3 Discrete Global Grid Systems (DGGS)**—processes over **100,000,000 trajectory points on commodity single-node hardware** with sub-second analytical response times and strictly bounded memory limits.

---

## 2. Columnar Data Architecture & Storage Layout

### 2.1 Storage Encodings & Compression
Probe datasets are stored in Hive-partitioned Apache Parquet files structured as follows:
```text
data/
└── corridor_id={CORRIDOR_ID}/
    └── date={YYYY-MM-DD}/
        └── chunk_{CHUNK_ID}_{PART_ID}.parquet
```

- **Dictionary Encoding**: Applied to repeated categorical attributes (`corridor_id`, `segment_id`, `vehicle_type`). Reduces string memory footprint by up to 85%.
- **Delta Encodings & Bit-Packing**: Applied to high-frequency timestamps (`timestamp[us]`) and sequence IDs.
- **Compression**: **Zstandard (ZSTD) Level 7**, offering a superior Pareto frontier between read decompression speed ($> 1.2 \text{ GB/s}$) and disk storage ratio ($4.5\times$ to $6\times$ compression relative to raw CSV/JSON).

### 2.2 Discrete Global Grid System (Uber H3)
Rather than executing compute-heavy geometric point-in-polygon queries against highway centerlines, spatial points are indexed into Uber H3 discrete hexagons:
- **Macro Resolution 7** ($R_7$):
  - Average Hexagon Area: $\approx 5.16 \text{ km}^2$
  - Edge Length: $\approx 1.22 \text{ km}$
  - Role: Primary spatial partitioning key and macro corridor binning.
- **Micro Resolution 8** ($R_8$):
  - Average Hexagon Area: $\approx 0.74 \text{ km}^2$
  - Edge Length: $\approx 461 \text{ m}$
  - Role: Bottleneck queue detection, shockwave spatial resolution, and segment boundary discretization.

---

## 3. Mathematical Formulation of Transportation Metrics

### 3.1 FHWA MAP-21 / PM3 System Performance Measures (23 CFR Part 490)

#### 1) Level of Travel Time Reliability (LOTTR) - 23 CFR § 490.511
For a highway segment $i$ and federal time period $p$, travel time $TT$ (minutes) is derived from segment length $L_i$ (miles) and observed speed $v$ (mph):
$$TT = \frac{L_i}{v} \times 60$$

LOTTR for period $p$ is defined as the ratio of the 80th percentile travel time to the 50th percentile (median) travel time:
$$\text{LOTTR}_{i,p} = \frac{TT_{i,p, 80\text{th}}}{TT_{i,p, 50\text{th}}}$$

Federal evaluation occurs across four distinct time periods:
1. **Weekday Morning Peak**: Monday–Friday 06:00 – 10:00
2. **Weekday Midday**: Monday–Friday 10:00 – 16:00
3. **Weekday Evening Peak**: Monday–Friday 16:00 – 20:00
4. **Weekend**: Saturday–Sunday 06:00 – 20:00

**Federal Compliance Standard**: Segment $i$ is classified as **Reliable** if and only if:
$$\max_{p \in \{1, 2, 3, 4\}} \left( \text{LOTTR}_{i,p} \right) < 1.50$$

Corridor-wide reliability is computed as the percentage of total segment mileage meeting the compliance threshold:
$$\text{Reliability}_{\text{corridor}} = \frac{\sum_{i \in \text{Reliable}} L_i}{\sum_{i=1}^N L_i} \times 100\%$$

#### 2) Truck Travel Time Reliability (TTTR) - 23 CFR § 490.611
Freight reliability focuses on extreme tail risk (95th percentile) for commercial motor carriers across five federal periods:
$$\text{TTTR}_{i,p} = \frac{TT_{i,p, 95\text{th}}}{TT_{i,p, 50\text{th}}}$$

Periods:
1. Weekday Morning (06:00 – 10:00)
2. Weekday Midday (10:00 – 16:00)
3. Weekday Afternoon (16:00 – 20:00)
4. Weekday Overnight (20:00 – 06:00)
5. Weekend (06:00 – 20:00)

The system-level freight index is the length-weighted sum of maximum TTTR scores:
$$\text{TTTR}_{\text{system}} = \frac{\sum_{i=1}^N \left( L_i \times \max_{p} \text{TTTR}_{i,p} \right)}{\sum_{i=1}^N L_i}$$

#### 3) Planning Time Index (PTI) & Buffer Time Index (BTI)
- **Planning Time Index (PTI)**:
  $$\text{PTI}_i = \frac{TT_{i, 95\text{th}}}{TT_{i, \text{free\_flow}}}$$
- **Buffer Time Index (BTI)**:
  $$\text{BTI}_i = \frac{TT_{i, 95\text{th}} - \overline{TT}_i}{\overline{TT}_i} \times 100\%$$

---

## 4. Query Engine Architecture & Memory Management

### 4.1 DuckDB Out-of-Core Execution
DuckDB manages memory via an internal buffer manager. By configuring:
```sql
SET max_memory = '2GB';
SET threads = 4;
```
DuckDB executes vector chunks directly over Parquet files. Predicates on `corridor_id` and `date` prune Parquet row groups and directory partitions before disk reads, eliminating unnecessary I/O.

### 4.2 Zero-Copy Memory Hand-Off via Apache Arrow
When analytical results are transferred from DuckDB to Python visualization libraries (Plotly/PyDeck) or FastAPI endpoints, DuckDB exports Arrow tables (`conn.execute(...).arrow()`). This avoids redundant JSON/Pandas object serialization, operating at memory-bandwidth speeds.
