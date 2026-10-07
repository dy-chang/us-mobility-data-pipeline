"""
Comprehensive Test Suite for US-Scale Mobility Data Pipeline.
Validates:
1. Probe Trajectory Generation & Diurnal Physics
2. Vectorized ETL Filtering & Uber H3 Hexagonal Indexing
3. Partitioned Parquet Storage & Schema Integrity
4. DuckDB Out-of-Core Queries & Analytical Views
5. Polars Multi-Threaded Space-Time Processing
6. FHWA MAP-21 / PM3 Federal Reliability Calculations (LOTTR, TTTR, PTI, BTI)
7. Lighthill-Whitham-Richards (LWR) Kinematic Shockwave Dynamics
8. FastAPI Microservice Endpoints
"""

from pathlib import Path
import shutil
import pytest
import pyarrow as pa
from starlette.testclient import TestClient

from src.config import CORRIDORS, LOTTR_RELIABLE_THRESHOLD
from src.generator import ProbeTrajectoryGenerator
from src.etl import MobilityETLPipeline
from src.duckdb_engine import DuckDBMobilityEngine
from src.polars_engine import PolarsMobilityEngine
from src.reliability_metrics import MobilityReliabilityEngine
from src.api import app


@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory):
    """Session fixture generating a clean partitioned Parquet test dataset."""
    temp_dir = tmp_path_factory.mktemp("mobility_test_dataset")
    corridor_id = "I-95_MIA"

    gen = ProbeTrajectoryGenerator(seed=123)
    etl = MobilityETLPipeline()

    # Generate 150,000 records in 2 chunks
    stream = gen.stream_dataset(total_records=150_000, corridor_id=corridor_id, chunk_size=75_000)
    stats = etl.process_and_persist(stream, corridor_id=corridor_id, output_dir=temp_dir)
    assert stats["total_valid_records"] > 140_000
    assert stats["storage_size_mb"] > 0

    return temp_dir


def test_generator_physics():
    """Test trajectory generator data contracts and physical boundaries."""
    gen = ProbeTrajectoryGenerator(seed=99)
    table = next(gen.stream_dataset(total_records=10_000, corridor_id="I-95_MIA", chunk_size=10_000))
    
    assert isinstance(table, pa.Table)
    assert len(table) == 10_000
    
    schema_cols = table.schema.names
    for expected_col in ["device_id", "timestamp", "latitude", "longitude", "speed_mph", "corridor_id", "segment_id", "vehicle_type"]:
        assert expected_col in schema_cols

    speeds = table["speed_mph"].to_numpy()
    assert speeds.mean() > 20.0
    assert speeds.max() > 60.0


def test_etl_filtering_and_h3(test_data_dir):
    """Verify that Parquet files contain valid H3 hexagonal columns and Hive partitions."""
    parquet_files = list(Path(test_data_dir).rglob("*.parquet"))
    assert len(parquet_files) > 0

    sample_table = pa.parquet.read_table(parquet_files[0])
    cols = sample_table.schema.names
    assert "h3_res7" in cols
    assert "h3_res8" in cols
    assert "hour" in cols

    import pyarrow.dataset as ds
    dataset = ds.dataset(str(test_data_dir), partitioning="hive")
    ds_cols = dataset.schema.names
    assert "corridor_id" in ds_cols
    assert "date" in ds_cols


def test_duckdb_aggregations(test_data_dir):
    """Verify DuckDB out-of-core scans and analytical summary queries."""
    engine = DuckDBMobilityEngine(max_memory="1GB", threads=2)
    engine.register_dataset(test_data_dir, view_name="test_mobility")

    summary = engine.get_corridor_summary(view_name="test_mobility")
    assert len(summary) >= 1
    corr_stats = summary[0]
    assert corr_stats["corridor_id"] == "I-95_MIA"
    assert corr_stats["record_count"] > 140_000
    assert corr_stats["unique_devices"] > 500
    assert corr_stats["unique_h3_hexagons"] > 10


def test_polars_spacetime_matrix(test_data_dir):
    """Verify Polars multi-threaded space-time speed matrix."""
    polars_engine = PolarsMobilityEngine(test_data_dir)
    matrix = polars_engine.compute_spacetime_speed_matrix("I-95_MIA", time_bucket_minutes=30)
    
    assert matrix.shape[0] > 0
    assert "median_speed_mph" in matrix.columns
    assert "p15_speed_mph" in matrix.columns


def test_fhwa_lottr_and_tttr(test_data_dir):
    """Verify FHWA MAP-21 LOTTR and TTTR reliability calculations."""
    engine = DuckDBMobilityEngine(max_memory="1GB", threads=2)
    engine.register_dataset(test_data_dir, view_name="test_mobility")
    rel_engine = MobilityReliabilityEngine(engine.conn)

    lottr = rel_engine.compute_segment_lottr("I-95_MIA", view_name="test_mobility")
    assert len(lottr) > 0
    for r in lottr:
        assert r.lottr_ratio >= 1.0
        assert r.travel_time_80th_min >= r.travel_time_50th_min

    tttr = rel_engine.compute_segment_tttr("I-95_MIA", view_name="test_mobility")
    assert len(tttr) > 0
    for r in tttr:
        assert r.tttr_ratio >= 1.0
        assert r.travel_time_95th_min >= r.travel_time_50th_min

    corridor_summary = rel_engine.compute_corridor_summary("I-95_MIA", view_name="test_mobility")
    assert corridor_summary.total_segments == len(CORRIDORS["I-95_MIA"].segments)
    assert corridor_summary.mean_pti >= 1.0
    assert corridor_summary.system_tttr_index >= 1.0


def test_lwr_shockwave_physics():
    """Verify Lighthill-Whitham-Richards shockwave velocity equations."""
    engine = DuckDBMobilityEngine(max_memory="1GB", threads=2)
    rel_engine = MobilityReliabilityEngine(engine.conn)

    # Congestion onset: fast upstream (65 mph) approaching queue bottleneck (20 mph)
    wave = rel_engine.compute_lwr_shockwave(
        upstream_speed_mph=65.0,
        downstream_speed_mph=20.0,
        free_flow_speed_mph=65.0,
        jam_density_vpm=220.0
    )
    assert "shockwave_speed_mph" in wave
    assert "upstream_density_vpm" in wave
    assert wave["upstream_density_vpm"] < wave["downstream_density_vpm"]
    assert wave["upstream_flow_vph"] >= 0


def test_fastapi_endpoints(test_data_dir, monkeypatch):
    """Test FastAPI REST endpoints using Starlette TestClient."""
    import src.api
    monkeypatch.setattr(src.api, "DATA_DIR", Path(test_data_dir))
    src.api._engine = None
    src.api._rel_engine = None

    client = TestClient(app)

    res_root = client.get("/")
    assert res_root.status_code == 200
    assert "service" in res_root.json()

    res_health = client.get("/health")
    assert res_health.status_code == 200
    assert res_health.json()["status"] == "healthy"

    res_corridors = client.get("/corridors")
    assert res_corridors.status_code == 200
    corridors = res_corridors.json()
    assert len(corridors) >= 3

    res_summary = client.get("/corridors/I-95_MIA/summary")
    assert res_summary.status_code == 200
    assert res_summary.json()["corridor_id"] == "I-95_MIA"

    res_rel = client.get("/corridors/I-95_MIA/reliability")
    assert res_rel.status_code == 200
    assert "lottr_compliance_pct" in res_rel.json()

    res_shockwave = client.get("/physics/shockwave?upstream_speed_mph=60&downstream_speed_mph=25")
    assert res_shockwave.status_code == 200
    assert "shockwave_speed_mph" in res_shockwave.json()
