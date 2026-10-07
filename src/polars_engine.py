"""
Polars Vectorized Mobility Analytics Engine.
Performs multi-threaded expressions, lazy scans, rolling quantiles,
and space-time matrices for highway corridors.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import polars as pl
from src.config import CORRIDORS, CorridorConfig


class PolarsMobilityEngine:
    """
    Vectorized multi-threaded in-memory analytics engine powered by Polars.
    Optimized for complex group-by transforms, temporal bucketing, and space-time heatmaps.
    """

    def __init__(self, data_path: Union[str, Path]):
        self.data_path = Path(data_path)
        path_str = str(self.data_path)
        if self.data_path.is_dir():
            self.glob_path = f"{path_str}/**/*.parquet"
        else:
            self.glob_path = path_str

    def scan(self) -> pl.LazyFrame:
        """Return a lazy frame scanning the partitioned parquet dataset."""
        target = str(self.data_path) if self.data_path.is_dir() else self.glob_path
        return pl.scan_parquet(target, hive_partitioning=True)

    def compute_spacetime_speed_matrix(
        self,
        corridor_id: str,
        time_bucket_minutes: int = 15,
        target_day_of_week: Optional[int] = 2,  # Wednesday default (0=Mon, 6=Sun)
    ) -> pl.DataFrame:
        """
        Build a Space-Time speed matrix (Segment x Time of Day).
        Crucial for visualizing shockwave bottleneck propagation.
        """
        lf = self.scan().filter(pl.col("corridor_id") == corridor_id)
        
        # Day of week filter (Monday=1 in Polars weekday)
        if target_day_of_week is not None:
            # Note: Polars dt.weekday() returns 1 (Mon) to 7 (Sun)
            polars_weekday = target_day_of_week + 1
            lf = lf.filter(pl.col("timestamp").dt.weekday() == polars_weekday)

        # 15-minute or custom interval binning
        lf = lf.with_columns([
            pl.col("timestamp").dt.truncate(f"{time_bucket_minutes}m").alias("time_bin"),
            pl.col("timestamp").dt.hour().alias("hour"),
            pl.col("timestamp").dt.minute().alias("minute"),
        ])

        # Aggregate median speed per segment and time bin
        agg = (
            lf.group_by(["segment_id", "hour", "minute"])
            .agg([
                pl.col("speed_mph").median().round(1).alias("median_speed_mph"),
                pl.col("speed_mph").quantile(0.15).round(1).alias("p15_speed_mph"),
                pl.col("speed_mph").count().alias("sample_size"),
            ])
            .sort(["segment_id", "hour", "minute"])
            .collect()
        )
        return agg

    def compute_vehicle_class_disparity(self, corridor_id: str) -> pl.DataFrame:
        """
        Compare commercial freight trucks vs passenger cars across peak and off-peak periods.
        """
        lf = self.scan().filter(pl.col("corridor_id") == corridor_id)
        
        summary = (
            lf.group_by(["vehicle_type", "hour"])
            .agg([
                pl.col("speed_mph").count().alias("observations"),
                pl.col("speed_mph").mean().round(2).alias("mean_speed"),
                pl.col("speed_mph").median().round(2).alias("median_speed"),
                pl.col("speed_mph").quantile(0.85).round(2).alias("p85_speed"),
            ])
            .sort(["hour", "vehicle_type"])
            .collect()
        )
        return summary
