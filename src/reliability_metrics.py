"""
Federal Highway Administration (FHWA) MAP-21 / PM3 Rule Performance Metrics
and Kinematic Wave Traffic Physics Engine.

Implements:
1. Level of Travel Time Reliability (LOTTR) - 23 CFR § 490.511
2. Truck Travel Time Reliability (TTTR) - 23 CFR § 490.611
3. Planning Time Index (PTI) & Buffer Time Index (BTI)
4. Lighthill-Whitham-Richards (LWR) Shockwave Velocity (omega = dq / dk)
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
import duckdb
import numpy as np
import pandas as pd
from src.config import (
    CORRIDORS,
    CorridorConfig,
    CorridorSegment,
    LOTTR_PERIODS,
    LOTTR_RELIABLE_THRESHOLD,
    TTTR_PERIODS,
)


@dataclass
class LOTTRResult:
    corridor_id: str
    segment_id: str
    segment_name: str
    period: str
    period_name: str
    travel_time_50th_min: float
    travel_time_80th_min: float
    lottr_ratio: float
    is_reliable: bool


@dataclass
class TTTRResult:
    corridor_id: str
    segment_id: str
    segment_name: str
    period: str
    period_name: str
    travel_time_50th_min: float
    travel_time_95th_min: float
    tttr_ratio: float


@dataclass
class CorridorReliabilitySummary:
    corridor_id: str
    corridor_name: str
    total_segments: int
    reliable_segments: int
    corridor_lottr_pct: float  # Percentage of segments with LOTTR < 1.50
    system_tttr_index: float   # Length-weighted truck reliability index
    mean_pti: float            # Planning Time Index
    mean_bti_pct: float        # Buffer Time Index %


class MobilityReliabilityEngine:
    """
    Computes federal mobility reliability metrics and shockwave dynamics
    using DuckDB high-performance SQL on partitioned Parquet datasets.
    """

    def __init__(self, duckdb_conn: duckdb.DuckDBPyConnection):
        self.conn = duckdb_conn

    def compute_segment_lottr(
        self,
        corridor_id: str,
        view_name: str = "mobility_probe",
    ) -> List[LOTTRResult]:
        """
        Compute Level of Travel Time Reliability (LOTTR) for each segment
        across the 4 federal MAP-21 time periods.
        """
        if corridor_id not in CORRIDORS:
            raise ValueError(f"Unknown corridor_id: {corridor_id}")
        
        corridor = CORRIDORS[corridor_id]
        seg_map = {s.segment_id: s for s in corridor.segments}

        results: List[LOTTRResult] = []

        for period_key, pinfo in LOTTR_PERIODS.items():
            days_sql = ", ".join(str(d) for d in pinfo["days"])
            start_h = pinfo["start_hour"]
            end_h = pinfo["end_hour"]

            query = f"""
            WITH seg_travel_times AS (
                SELECT 
                    segment_id,
                    speed_mph,
                    -- Travel time in minutes: (length_miles / speed_mph) * 60
                    CASE 
                        {" ".join([f"WHEN segment_id = '{s.segment_id}' THEN ({s.length_miles} / NULLIF(speed_mph, 0)) * 60.0" for s in corridor.segments])}
                        ELSE 2.0
                    END AS travel_time_min
                FROM {view_name}
                WHERE corridor_id = '{corridor_id}'
                  AND vehicle_type = 'passenger'
                  AND EXTRACT(DOW FROM timestamp) IN ({days_sql})
                  AND hour >= {start_h} AND hour < {end_h}
            )
            SELECT 
                segment_id,
                ROUND(QUANTILE_CONT(travel_time_min, 0.50), 3) AS tt_50,
                ROUND(QUANTILE_CONT(travel_time_min, 0.80), 3) AS tt_80,
                ROUND(QUANTILE_CONT(travel_time_min, 0.80) / NULLIF(QUANTILE_CONT(travel_time_min, 0.50), 0), 3) AS lottr
            FROM seg_travel_times
            GROUP BY segment_id;
            """
            
            df = self.conn.execute(query).fetchdf()
            for _, row in df.iterrows():
                seg_id = str(row["segment_id"])
                seg_name = seg_map[seg_id].name if seg_id in seg_map else seg_id
                lottr_val = float(row["lottr"]) if pd.notna(row["lottr"]) else 1.0
                tt_50 = float(row["tt_50"]) if pd.notna(row["tt_50"]) else 0.0
                tt_80 = float(row["tt_80"]) if pd.notna(row["tt_80"]) else 0.0

                results.append(
                    LOTTRResult(
                        corridor_id=corridor_id,
                        segment_id=seg_id,
                        segment_name=seg_name,
                        period=period_key,
                        period_name=pinfo["name"],
                        travel_time_50th_min=tt_50,
                        travel_time_80th_min=tt_80,
                        lottr_ratio=lottr_val,
                        is_reliable=(lottr_val < LOTTR_RELIABLE_THRESHOLD),
                    )
                )

        return results

    def compute_segment_tttr(
        self,
        corridor_id: str,
        view_name: str = "mobility_probe",
    ) -> List[TTTRResult]:
        """
        Compute Truck Travel Time Reliability (TTTR) for commercial freight
        across the 5 federal MAP-21 periods (23 CFR § 490.611).
        """
        if corridor_id not in CORRIDORS:
            raise ValueError(f"Unknown corridor_id: {corridor_id}")
        
        corridor = CORRIDORS[corridor_id]
        seg_map = {s.segment_id: s for s in corridor.segments}

        results: List[TTTRResult] = []

        for period_key, pinfo in TTTR_PERIODS.items():
            days_sql = ", ".join(str(d) for d in pinfo["days"])
            start_h = pinfo["start_hour"]
            end_h = pinfo["end_hour"]

            # Overnight crosses midnight
            if start_h > end_h:
                hour_clause = f"(hour >= {start_h} OR hour < {end_h})"
            else:
                hour_clause = f"(hour >= {start_h} AND hour < {end_h})"

            query = f"""
            WITH truck_times AS (
                SELECT 
                    segment_id,
                    CASE 
                        {" ".join([f"WHEN segment_id = '{s.segment_id}' THEN ({s.length_miles} / NULLIF(speed_mph, 0)) * 60.0" for s in corridor.segments])}
                        ELSE 2.0
                    END AS travel_time_min
                FROM {view_name}
                WHERE corridor_id = '{corridor_id}'
                  AND vehicle_type = 'commercial_truck'
                  AND EXTRACT(DOW FROM timestamp) IN ({days_sql})
                  AND {hour_clause}
            )
            SELECT 
                segment_id,
                ROUND(QUANTILE_CONT(travel_time_min, 0.50), 3) AS tt_50,
                ROUND(QUANTILE_CONT(travel_time_min, 0.95), 3) AS tt_95,
                ROUND(QUANTILE_CONT(travel_time_min, 0.95) / NULLIF(QUANTILE_CONT(travel_time_min, 0.50), 0), 3) AS tttr
            FROM truck_times
            GROUP BY segment_id;
            """

            df = self.conn.execute(query).fetchdf()
            for _, row in df.iterrows():
                seg_id = str(row["segment_id"])
                seg_name = seg_map[seg_id].name if seg_id in seg_map else seg_id
                tttr_val = float(row["tttr"]) if pd.notna(row["tttr"]) else 1.0
                tt_50 = float(row["tt_50"]) if pd.notna(row["tt_50"]) else 0.0
                tt_95 = float(row["tt_95"]) if pd.notna(row["tt_95"]) else 0.0

                results.append(
                    TTTRResult(
                        corridor_id=corridor_id,
                        segment_id=seg_id,
                        segment_name=seg_name,
                        period=period_key,
                        period_name=pinfo["name"],
                        travel_time_50th_min=tt_50,
                        travel_time_95th_min=tt_95,
                        tttr_ratio=tttr_val,
                    )
                )

        return results

    def compute_corridor_summary(
        self,
        corridor_id: str,
        view_name: str = "mobility_probe",
    ) -> CorridorReliabilitySummary:
        """
        Generate comprehensive corridor-wide reliability score conforming to FHWA rules.
        """
        corridor = CORRIDORS[corridor_id]
        lottr_list = self.compute_segment_lottr(corridor_id, view_name)
        tttr_list = self.compute_segment_tttr(corridor_id, view_name)

        # 1. LOTTR Compliance: A segment is reliable only if ALL periods have LOTTR < 1.50
        seg_lottr_max: Dict[str, float] = {}
        for item in lottr_list:
            seg_lottr_max[item.segment_id] = max(seg_lottr_max.get(item.segment_id, 0.0), item.lottr_ratio)

        total_segs = len(corridor.segments)
        reliable_count = sum(1 for seg_id in seg_lottr_max if seg_lottr_max[seg_id] < LOTTR_RELIABLE_THRESHOLD)
        lottr_pct = round((reliable_count / total_segs * 100.0), 2) if total_segs > 0 else 0.0

        # 2. System TTTR Index: Length-weighted sum of max TTTR
        seg_tttr_max: Dict[str, float] = {}
        for item in tttr_list:
            seg_tttr_max[item.segment_id] = max(seg_tttr_max.get(item.segment_id, 0.0), item.tttr_ratio)

        weighted_tttr_sum = 0.0
        total_length = sum(s.length_miles for s in corridor.segments)
        for s in corridor.segments:
            max_tttr = seg_tttr_max.get(s.segment_id, 1.0)
            weighted_tttr_sum += s.length_miles * max_tttr

        system_tttr = round(weighted_tttr_sum / total_length, 2) if total_length > 0 else 1.0

        # 3. Corridor PTI and BTI
        # Free flow travel time in minutes
        ff_time_min = sum((s.length_miles / s.free_flow_speed_mph) * 60.0 for s in corridor.segments)

        query = f"""
        SELECT 
            ROUND(AVG(speed_mph), 2) AS mean_speed,
            ROUND(QUANTILE_CONT(speed_mph, 0.05), 2) AS congested_speed_p05,
            ROUND(QUANTILE_CONT(speed_mph, 0.50), 2) AS median_speed
        FROM {view_name}
        WHERE corridor_id = '{corridor_id}';
        """
        speed_stats = self.conn.execute(query).fetchdf().iloc[0]
        mean_speed = float(speed_stats["mean_speed"]) if pd.notna(speed_stats["mean_speed"]) else corridor.free_flow_speed_mph
        p05_speed = float(speed_stats["congested_speed_p05"]) if pd.notna(speed_stats["congested_speed_p05"]) else 25.0

        corridor_tt_mean = (total_length / max(mean_speed, 1.0)) * 60.0
        corridor_tt_p95 = (total_length / max(p05_speed, 1.0)) * 60.0

        pti = round(corridor_tt_p95 / max(ff_time_min, 0.1), 2)
        bti_pct = round(((corridor_tt_p95 - corridor_tt_mean) / max(corridor_tt_mean, 0.1)) * 100.0, 1)

        return CorridorReliabilitySummary(
            corridor_id=corridor_id,
            corridor_name=corridor.name,
            total_segments=total_segs,
            reliable_segments=reliable_count,
            corridor_lottr_pct=lottr_pct,
            system_tttr_index=system_tttr,
            mean_pti=pti,
            mean_bti_pct=bti_pct,
        )

    def compute_lwr_shockwave(
        self,
        upstream_speed_mph: float,
        downstream_speed_mph: float,
        free_flow_speed_mph: float = 65.0,
        jam_density_vpm: float = 220.0,  # vehicles per mile
    ) -> Dict[str, float]:
        """
        Compute Lighthill-Whitham-Richards (LWR) kinematic wave shockwave velocity (omega = dq / dk).
        Uses Greenshields fundamental diagram: k = k_jam * (1 - v / v_ff), q = k * v.
        """
        # Upstream state 1
        v1 = min(upstream_speed_mph, free_flow_speed_mph)
        k1 = jam_density_vpm * (1.0 - (v1 / free_flow_speed_mph))
        q1 = k1 * v1

        # Downstream bottleneck state 2
        v2 = min(downstream_speed_mph, free_flow_speed_mph)
        k2 = jam_density_vpm * (1.0 - (v2 / free_flow_speed_mph))
        q2 = k2 * v2

        # Shockwave speed omega = delta_q / delta_k
        delta_k = k2 - k1
        delta_q = q2 - q1

        if abs(delta_k) < 1e-4:
            omega_mph = 0.0
        else:
            omega_mph = delta_q / delta_k

        return {
            "upstream_speed_mph": round(v1, 1),
            "downstream_speed_mph": round(v2, 1),
            "upstream_density_vpm": round(k1, 1),
            "downstream_density_vpm": round(k2, 1),
            "upstream_flow_vph": round(q1, 0),
            "downstream_flow_vph": round(q2, 0),
            "shockwave_speed_mph": round(omega_mph, 2),
            "propagation_direction": "Upstream (Backward Forming)" if omega_mph < 0 else "Downstream (Forward Recovery)",
        }
