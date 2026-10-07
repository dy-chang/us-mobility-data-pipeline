"""
Calibrated High-Throughput Mobility Probe Generator.
Generates production-scale GPS probe telematics trajectories with realistic diurnal patterns,
freight vs passenger splits, bottleneck shockwaves, and sensor noise artifacts.
"""

import datetime
import math
from typing import Generator, Optional
import numpy as np
import pyarrow as pa
from src.config import CORRIDORS, CorridorConfig, CorridorSegment


class ProbeTrajectoryGenerator:
    """
    Vectorized streaming probe generator designed to produce millions of trajectory records
    with realistic transportation physics and federal MAP-21 congestion profiles.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def _generate_chunk(
        self,
        chunk_size: int,
        base_date: datetime.date,
        corridor: CorridorConfig,
    ) -> pa.Table:
        """Generate a single in-memory PyArrow Table chunk using vectorized NumPy arrays."""
        segments = corridor.segments
        num_segs = len(segments)

        # 1. Random segment assignment
        seg_indices = self.rng.integers(0, num_segs, size=chunk_size)
        
        # 2. Vehicle types: 85% passenger cars, 15% freight commercial trucks
        is_truck = self.rng.random(size=chunk_size) < 0.15
        vehicle_types = np.where(is_truck, "commercial_truck", "passenger")

        # 3. Timestamps: distributed across a 7-day study window
        day_offsets = self.rng.integers(0, 7, size=chunk_size)
        # Diurnal distribution: bimodal morning peak (8am) and evening peak (17:30)
        # Mixture of uniforms and peaks
        mode_choice = self.rng.choice(["flat", "am_peak", "pm_peak"], size=chunk_size, p=[0.4, 0.3, 0.3])
        
        hours = np.zeros(chunk_size, dtype=np.float64)
        flat_mask = mode_choice == "flat"
        am_mask = mode_choice == "am_peak"
        pm_mask = mode_choice == "pm_peak"
        
        hours[flat_mask] = self.rng.uniform(0.0, 24.0, size=flat_mask.sum())
        hours[am_mask] = np.clip(self.rng.normal(8.0, 1.2, size=am_mask.sum()), 6.0, 10.5)
        hours[pm_mask] = np.clip(self.rng.normal(17.5, 1.3, size=pm_mask.sum()), 15.5, 20.5)

        seconds_in_day = (hours * 3600).astype(np.int64)
        base_epoch = int(datetime.datetime.combine(base_date, datetime.time.min).replace(tzinfo=datetime.timezone.utc).timestamp())
        timestamps = base_epoch + (day_offsets * 86400) + seconds_in_day

        # 4. Spatial Interpolation along corridor segments
        # Extract segment coordinates
        seg_start_lats = np.array([s.start_lat for s in segments])
        seg_start_lons = np.array([s.start_lon for s in segments])
        seg_end_lats = np.array([s.end_lat for s in segments])
        seg_end_lons = np.array([s.end_lon for s in segments])
        seg_lengths = np.array([s.length_miles for s in segments])
        seg_ids = np.array([s.segment_id for s in segments])

        t_fraction = self.rng.uniform(0.0, 1.0, size=chunk_size)
        lats = seg_start_lats[seg_indices] + t_fraction * (seg_end_lats[seg_indices] - seg_start_lats[seg_indices])
        lons = seg_start_lons[seg_indices] + t_fraction * (seg_end_lons[seg_indices] - seg_start_lons[seg_indices])
        
        # Add micro-scale GPS jitter (std ~ 3-5 meters in degrees)
        lat_jitter = self.rng.normal(0, 0.00004, size=chunk_size)
        lon_jitter = self.rng.normal(0, 0.00004, size=chunk_size)
        lats += lat_jitter
        lons += lon_jitter

        # 5. Physics-Based Speed Calculation
        # Base free-flow speed
        free_flow = corridor.free_flow_speed_mph
        base_speeds = np.full(chunk_size, free_flow, dtype=np.float32)

        # Commercial trucks travel ~5 mph below speed limit
        base_speeds[is_truck] -= 5.0

        # Congestion factor based on time of day (AM Peak ~ 7-9:30, PM Peak ~ 16:30-19:00)
        # Recurrent bottleneck in segment index 2 and 3 (e.g. Golden Glades or Downtown)
        is_bottleneck_seg = np.isin(seg_indices, [2, 3])
        
        is_am = (hours >= 7.0) & (hours <= 9.5)
        is_pm = (hours >= 16.0) & (hours <= 19.0)
        is_weekday = (day_offsets < 5)

        # Speed reduction factor
        speed_factor = np.ones(chunk_size, dtype=np.float32)
        congestion_am = is_weekday & is_am
        congestion_pm = is_weekday & is_pm

        speed_factor[congestion_am] *= self.rng.uniform(0.35, 0.65, size=congestion_am.sum())
        speed_factor[congestion_pm] *= self.rng.uniform(0.30, 0.55, size=congestion_pm.sum())
        
        # Severe shockwave bottleneck reduction
        severe_am = congestion_am & is_bottleneck_seg
        severe_pm = congestion_pm & is_bottleneck_seg
        speed_factor[severe_am] *= self.rng.uniform(0.4, 0.7, size=severe_am.sum())
        speed_factor[severe_pm] *= self.rng.uniform(0.35, 0.65, size=severe_pm.sum())

        speeds = np.clip(base_speeds * speed_factor + self.rng.normal(0, 2.5, size=chunk_size), 3.0, 85.0).astype(np.float32)

        # 6. Bearing / Heading (degrees)
        d_lon = seg_end_lons[seg_indices] - seg_start_lons[seg_indices]
        d_lat = seg_end_lats[seg_indices] - seg_start_lats[seg_indices]
        headings = (np.degrees(np.arctan2(d_lon, d_lat)) + 360) % 360
        headings = (headings + self.rng.normal(0, 3.0, size=chunk_size)) % 360

        # 7. Add controlled GPS sensor artifacts (0.2% invalid coordinates/teleportation for ETL testing)
        anomaly_mask = self.rng.random(size=chunk_size) < 0.002
        speeds[anomaly_mask] = self.rng.choice([0.0, 135.0, 180.0], size=anomaly_mask.sum())
        lats[anomaly_mask] = self.rng.choice([0.0, 99.99, -45.0], size=anomaly_mask.sum())

        # 8. Device IDs
        device_pool_size = max(1000, chunk_size // 20)
        device_ints = self.rng.integers(1, device_pool_size, size=chunk_size)
        device_ids = [f"dev_{i:06d}" for i in device_ints]

        # 9. Build PyArrow Columnar Table
        table = pa.Table.from_arrays(
            [
                pa.array(device_ids, type=pa.string()),
                pa.array(timestamps * 1_000_000, type=pa.timestamp("us")),
                pa.array(lats, type=pa.float64()),
                pa.array(lons, type=pa.float64()),
                pa.array(speeds, type=pa.float32()),
                pa.array(headings.astype(np.float32), type=pa.float32()),
                pa.array([corridor.corridor_id] * chunk_size, type=pa.string()),
                pa.array(seg_ids[seg_indices], type=pa.string()),
                pa.array(vehicle_types, type=pa.string()),
            ],
            names=[
                "device_id",
                "timestamp",
                "latitude",
                "longitude",
                "speed_mph",
                "heading",
                "corridor_id",
                "segment_id",
                "vehicle_type",
            ]
        )
        return table

    def stream_dataset(
        self,
        total_records: int,
        corridor_id: str = "I-95_MIA",
        chunk_size: int = 500_000,
        base_date: Optional[datetime.date] = None,
    ) -> Generator[pa.Table, None, None]:
        """
        Stream chunks of probe data yielding PyArrow Tables.
        Enables out-of-core generation for 10M, 50M, or 100M+ rows with constant bounded RAM.
        """
        if corridor_id not in CORRIDORS:
            raise ValueError(f"Unknown corridor_id: {corridor_id}. Available: {list(CORRIDORS.keys())}")
        
        corridor = CORRIDORS[corridor_id]
        if base_date is None:
            base_date = datetime.date(2026, 10, 1)

        generated = 0
        while generated < total_records:
            current_chunk = min(chunk_size, total_records - generated)
            yield self._generate_chunk(current_chunk, base_date, corridor)
            generated += current_chunk
