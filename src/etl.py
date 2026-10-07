"""
High-Performance Mobility ETL Pipeline.
Performs data validation contracts, spatial bounding filtering, Uber H3 hexagonal indexing,
and Hive-partitioned Apache Parquet persistence with ZSTD compression.
"""

import datetime
from pathlib import Path
import time
from typing import Dict, Generator, Optional, Tuple, Union
import h3
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from src.config import (
    CORRIDORS,
    CorridorConfig,
    DEFAULT_COMPRESSION,
    DEFAULT_COMPRESSION_LEVEL,
    H3_MACRO_RES,
    H3_MICRO_RES,
)


class MobilityETLPipeline:
    """
    Vectorized ETL processor designed for high-throughput probe streams.
    Transforms raw trajectory points into clean, spatially-indexed, partitioned Parquet datasets.
    """

    def __init__(
        self,
        macro_res: int = H3_MACRO_RES,
        micro_res: int = H3_MICRO_RES,
        compression: str = DEFAULT_COMPRESSION,
        compression_level: int = DEFAULT_COMPRESSION_LEVEL,
    ):
        self.macro_res = macro_res
        self.micro_res = micro_res
        self.compression = compression
        self.compression_level = compression_level

    def clean_and_index_table(
        self,
        table: pa.Table,
        corridor: CorridorConfig,
    ) -> Tuple[pa.Table, int]:
        """
        Filter GPS outliers, apply coordinate bounding box criteria,
        and assign discrete hexagonal H3 spatial cells.
        """
        initial_count = len(table)

        # 1. Vectorized Boolean Masking via PyArrow Compute
        lat = table["latitude"]
        lon = table["longitude"]
        speed = table["speed_mph"]

        # Valid coordinate bounding box
        valid_lat = pc.and_(
            pc.greater_equal(lat, corridor.min_lat),
            pc.less_equal(lat, corridor.max_lat)
        )
        valid_lon = pc.and_(
            pc.greater_equal(lon, corridor.min_lon),
            pc.less_equal(lon, corridor.max_lon)
        )
        valid_coords = pc.and_(valid_lat, valid_lon)

        # Valid physical speeds (0 mph to 105 mph)
        valid_speed = pc.and_(
            pc.greater_equal(speed, 0.0),
            pc.less_equal(speed, 105.0)
        )

        valid_mask = pc.and_(valid_coords, valid_speed)
        clean_table = table.filter(valid_mask)
        clean_count = len(clean_table)
        dropped_count = initial_count - clean_count

        if clean_count == 0:
            return clean_table, dropped_count

        # 2. Extract Lat/Lon numpy arrays for H3 spatial assignment
        clean_lats = clean_table["latitude"].to_numpy()
        clean_lons = clean_table["longitude"].to_numpy()

        # Vectorized H3 Cell Computation
        h3_macro = [h3.latlng_to_cell(lat, lon, self.macro_res) for lat, lon in zip(clean_lats, clean_lons)]
        h3_micro = [h3.latlng_to_cell(lat, lon, self.micro_res) for lat, lon in zip(clean_lats, clean_lons)]

        # Extract Date string for Hive partitioning
        # Timestamp is in timestamp[us]
        ts_py = clean_table["timestamp"].to_pylist()
        dates = [dt.strftime("%Y-%m-%d") for dt in ts_py]
        hours = [dt.hour for dt in ts_py]

        # 3. Append Columns to PyArrow Table
        indexed_table = clean_table.append_column("h3_res7", pa.array(h3_macro, type=pa.string()))
        indexed_table = indexed_table.append_column("h3_res8", pa.array(h3_micro, type=pa.string()))
        indexed_table = indexed_table.append_column("date", pa.array(dates, type=pa.string()))
        indexed_table = indexed_table.append_column("hour", pa.array(hours, type=pa.int8()))

        return indexed_table, dropped_count

    def process_and_persist(
        self,
        table_generator: Generator[pa.Table, None, None],
        corridor_id: str,
        output_dir: Union[str, Path],
    ) -> Dict[str, Union[int, float]]:
        """
        Stream through incoming batches, validate, index, and write to partitioned Parquet files.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        if corridor_id not in CORRIDORS:
            raise ValueError(f"Corridor {corridor_id} not found in configuration.")
        corridor = CORRIDORS[corridor_id]

        total_input_records = 0
        total_valid_records = 0
        total_dropped_records = 0
        chunks_processed = 0

        t0 = time.time()

        for chunk in table_generator:
            total_input_records += len(chunk)
            clean_chunk, dropped = self.clean_and_index_table(chunk, corridor)
            total_dropped_records += dropped
            total_valid_records += len(clean_chunk)

            if len(clean_chunk) > 0:
                # Write partitioned by corridor_id and date
                ds.write_dataset(
                    clean_chunk,
                    base_dir=str(out_path),
                    format="parquet",
                    partitioning=["corridor_id", "date"],
                    partitioning_flavor="hive",
                    existing_data_behavior="overwrite_or_ignore",
                    basename_template=f"chunk_{chunks_processed}_{{i}}.parquet",
                    file_options=ds.ParquetFileFormat().make_write_options(
                        compression=self.compression,
                        compression_level=self.compression_level,
                    ),
                    max_rows_per_group=250_000,
                    max_rows_per_file=500_000,
                )
            chunks_processed += 1

        elapsed = time.time() - t0
        throughput = total_input_records / elapsed if elapsed > 0 else 0

        # Calculate total storage footprint
        total_bytes = sum(f.stat().st_size for f in out_path.rglob("*.parquet"))

        return {
            "chunks_processed": chunks_processed,
            "total_input_records": total_input_records,
            "total_valid_records": total_valid_records,
            "total_dropped_records": total_dropped_records,
            "drop_rate_pct": round((total_dropped_records / total_input_records * 100), 3) if total_input_records > 0 else 0.0,
            "storage_size_mb": round(total_bytes / (1024 * 1024), 2),
            "elapsed_time_sec": round(elapsed, 2),
            "throughput_records_per_sec": round(throughput, 1),
        }
