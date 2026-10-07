"""
DuckDB Analytical Streaming Engine.
Executes out-of-core SQL scans, predicate pushdown, spatial aggregations,
and rolling percentiles over partitioned Apache Parquet datasets.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import duckdb
import pyarrow as pa


class DuckDBMobilityEngine:
    """
    Columnar out-of-core query engine built on DuckDB.
    Scans Hive-partitioned Parquet files with zero-copy Arrow memory integration
    and strictly bounded memory footprints.
    """

    def __init__(self, max_memory: str = "2GB", threads: int = 4):
        self.conn = duckdb.connect(database=":memory:")
        self.conn.execute(f"SET max_memory = '{max_memory}'")
        self.conn.execute(f"SET threads = {threads}")
        self.conn.execute("SET preserve_insertion_order = false")

    def register_dataset(self, data_path: Union[str, Path], view_name: str = "mobility_probe"):
        """
        Register Hive-partitioned Parquet directory as a virtual DuckDB view.
        """
        path_str = str(data_path)
        if Path(path_str).is_dir():
            target_glob = f"{path_str}/**/*.parquet"
        else:
            target_glob = path_str

        query = f"""
        CREATE OR REPLACE VIEW {view_name} AS 
        SELECT * FROM read_parquet('{target_glob}', hive_partitioning = true);
        """
        self.conn.execute(query)

    def get_corridor_summary(self, view_name: str = "mobility_probe") -> List[Dict[str, Any]]:
        """
        Return high-level volume, date ranges, and average speed per corridor.
        """
        query = f"""
        SELECT 
            corridor_id,
            COUNT(*) AS record_count,
            MIN(timestamp) AS min_timestamp,
            MAX(timestamp) AS max_timestamp,
            ROUND(AVG(speed_mph), 2) AS mean_speed_mph,
            ROUND(MEDIAN(speed_mph), 2) AS median_speed_mph,
            ROUND(QUANTILE_CONT(speed_mph, 0.85), 2) AS p85_speed_mph,
            COUNT(DISTINCT device_id) AS unique_devices,
            COUNT(DISTINCT h3_res7) AS unique_h3_hexagons
        FROM {view_name}
        GROUP BY corridor_id
        ORDER BY record_count DESC;
        """
        return self.conn.execute(query).fetchdf().to_dict(orient="records")

    def compute_segment_speed_profiles(
        self,
        corridor_id: str,
        view_name: str = "mobility_probe",
    ) -> pa.Table:
        """
        Compute hourly speed statistics per corridor segment using vector quantiles.
        """
        query = f"""
        SELECT 
            segment_id,
            EXTRACT(DOW FROM timestamp) AS day_of_week,
            hour,
            COUNT(*) AS observation_count,
            ROUND(AVG(speed_mph), 2) AS mean_speed_mph,
            ROUND(MEDIAN(speed_mph), 2) AS median_speed_mph,
            ROUND(QUANTILE_CONT(speed_mph, 0.15), 2) AS p15_speed_mph,
            ROUND(QUANTILE_CONT(speed_mph, 0.85), 2) AS p85_speed_mph,
            ROUND(STDDEV_SAMP(speed_mph), 2) AS std_speed_mph
        FROM {view_name}
        WHERE corridor_id = '{corridor_id}'
        GROUP BY segment_id, EXTRACT(DOW FROM timestamp), hour
        ORDER BY segment_id, day_of_week, hour;
        """
        return self.conn.execute(query).arrow()

    def compute_h3_spatial_grid(
        self,
        corridor_id: str,
        resolution: str = "res7",
        view_name: str = "mobility_probe",
    ) -> List[Dict[str, Any]]:
        """
        Aggregate spatial speeds and density at Uber H3 hexagon resolution.
        """
        h3_col = "h3_res7" if resolution == "res7" else "h3_res8"
        query = f"""
        SELECT 
            {h3_col} AS h3_index,
            COUNT(*) AS point_density,
            ROUND(AVG(latitude), 5) AS center_lat,
            ROUND(AVG(longitude), 5) AS center_lon,
            ROUND(AVG(speed_mph), 2) AS mean_speed,
            ROUND(QUANTILE_CONT(speed_mph, 0.15), 2) AS congested_speed_p15,
            ROUND(COUNT(CASE WHEN speed_mph < 35.0 THEN 1 END) * 100.0 / COUNT(*), 2) AS congestion_pct
        FROM {view_name}
        WHERE corridor_id = '{corridor_id}'
        GROUP BY {h3_col}
        ORDER BY point_density DESC;
        """
        return self.conn.execute(query).fetchdf().to_dict(orient="records")

    def execute_custom_query(self, sql: str) -> pa.Table:
        """Execute arbitrary SQL query returning PyArrow Table for zero-copy memory transfer."""
        return self.conn.execute(sql).arrow()
