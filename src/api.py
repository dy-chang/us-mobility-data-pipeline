"""
FastAPI Production Microservice for US-Scale Mobility Data Pipeline.
Exposes high-performance REST endpoints for corridor queries, FHWA MAP-21
reliability compliance metrics, and Uber H3 spatial hexbins.
"""

import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import psutil

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CORRIDORS, CorridorConfig, LOTTR_RELIABLE_THRESHOLD
from src.duckdb_engine import DuckDBMobilityEngine
from src.reliability_metrics import MobilityReliabilityEngine

app = FastAPI(
    title="US-Scale Mobility Data Pipeline API",
    description="High-Performance Columnar Mobility API delivering FHWA MAP-21 Reliability & Uber H3 Analytics",
    version="0.1.0",
)

# Global database connection cache
DATA_DIR = Path(os.environ.get("MOBILITY_DATA_DIR", "data/benchmark_data"))
_engine: Optional[DuckDBMobilityEngine] = None
_rel_engine: Optional[MobilityReliabilityEngine] = None


def get_duckdb() -> DuckDBMobilityEngine:
    global _engine, _rel_engine
    if _engine is None:
        _engine = DuckDBMobilityEngine(max_memory="2GB", threads=4)
        if DATA_DIR.exists():
            _engine.register_dataset(DATA_DIR)
        _rel_engine = MobilityReliabilityEngine(_engine.conn)
    return _engine


def get_rel_engine() -> MobilityReliabilityEngine:
    get_duckdb()
    assert _rel_engine is not None
    return _rel_engine


class HealthResponse(BaseModel):
    status: str
    version: str
    memory_usage_mb: float
    cpu_percent: float
    dataset_registered: bool


@app.get("/", tags=["System"])
def root():
    return {
        "service": "US-Scale Mobility Data Pipeline API",
        "version": "0.1.0",
        "docs": "/docs",
        "supported_corridors": list(CORRIDORS.keys()),
    }


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    proc = psutil.Process()
    has_data = DATA_DIR.exists() and any(DATA_DIR.rglob("*.parquet"))
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        memory_usage_mb=round(proc.memory_info().rss / (1024 * 1024), 2),
        cpu_percent=proc.cpu_percent(interval=None),
        dataset_registered=has_data,
    )


@app.get("/corridors", tags=["Corridors"])
def list_corridors():
    return [
        {
            "corridor_id": c.corridor_id,
            "name": c.name,
            "state": c.state,
            "segments_count": len(c.segments),
            "free_flow_speed_mph": c.free_flow_speed_mph,
            "bbox": {
                "min_lat": c.min_lat,
                "max_lat": c.max_lat,
                "min_lon": c.min_lon,
                "max_lon": c.max_lon,
            }
        }
        for c in CORRIDORS.values()
    ]


@app.get("/corridors/{corridor_id}/summary", tags=["Corridors"])
def corridor_summary(corridor_id: str):
    if corridor_id not in CORRIDORS:
        raise HTTPException(status_code=404, detail=f"Corridor {corridor_id} not found")
    
    engine = get_duckdb()
    try:
        summaries = engine.get_corridor_summary()
        matched = [s for s in summaries if s.get("corridor_id") == corridor_id]
        if not matched:
            raise HTTPException(status_code=404, detail=f"No probe records ingested for corridor {corridor_id}")
        return matched[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/corridors/{corridor_id}/reliability", tags=["FHWA MAP-21"])
def corridor_reliability(corridor_id: str):
    if corridor_id not in CORRIDORS:
        raise HTTPException(status_code=404, detail=f"Corridor {corridor_id} not found")
    
    rel_eng = get_rel_engine()
    try:
        summary = rel_eng.compute_corridor_summary(corridor_id)
        return {
            "corridor_id": summary.corridor_id,
            "corridor_name": summary.corridor_name,
            "total_segments": summary.total_segments,
            "reliable_segments": summary.reliable_segments,
            "lottr_compliance_pct": summary.corridor_lottr_pct,
            "system_tttr_index": summary.system_tttr_index,
            "planning_time_index_pti": summary.mean_pti,
            "buffer_time_index_pct": summary.mean_bti_pct,
            "status": "Compliant" if summary.corridor_lottr_pct >= 80.0 else "Action Required",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/corridors/{corridor_id}/lottr", tags=["FHWA MAP-21"])
def get_lottr(corridor_id: str):
    if corridor_id not in CORRIDORS:
        raise HTTPException(status_code=404, detail=f"Corridor {corridor_id} not found")
    rel_eng = get_rel_engine()
    try:
        results = rel_eng.compute_segment_lottr(corridor_id)
        return [
            {
                "segment_id": r.segment_id,
                "segment_name": r.segment_name,
                "period": r.period,
                "period_name": r.period_name,
                "tt_50th_min": r.travel_time_50th_min,
                "tt_80th_min": r.travel_time_80th_min,
                "lottr_ratio": r.lottr_ratio,
                "is_reliable": r.is_reliable,
            }
            for r in results
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/corridors/{corridor_id}/hexbins", tags=["Spatial Indexing"])
def get_hexbins(corridor_id: str, resolution: str = Query("res7", pattern=r"^(res7|res8)$")):
    if corridor_id not in CORRIDORS:
        raise HTTPException(status_code=404, detail=f"Corridor {corridor_id} not found")
    engine = get_duckdb()
    try:
        return engine.compute_h3_spatial_grid(corridor_id, resolution=resolution)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/physics/shockwave", tags=["Traffic Physics"])
def calculate_shockwave(
    upstream_speed_mph: float = Query(65.0, ge=5.0, le=90.0),
    downstream_speed_mph: float = Query(25.0, ge=1.0, le=85.0),
    free_flow_speed_mph: float = Query(65.0, ge=40.0, le=80.0),
):
    rel_eng = get_rel_engine()
    return rel_eng.compute_lwr_shockwave(
        upstream_speed_mph=upstream_speed_mph,
        downstream_speed_mph=downstream_speed_mph,
        free_flow_speed_mph=free_flow_speed_mph,
    )
