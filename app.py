"""
Interactive Mobility Intelligence Dashboard
Streamlit + PyDeck application for large-scale probe trajectory visualization,
FHWA MAP-21 reliability analytics, and LWR traffic shockwave simulation.
"""

import os
from pathlib import Path
import sys
import time

# Ensure repo root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import duckdb
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st

from src.config import CORRIDORS, LOTTR_PERIODS, LOTTR_RELIABLE_THRESHOLD, TTTR_PERIODS
from src.duckdb_engine import DuckDBMobilityEngine
from src.etl import MobilityETLPipeline
from src.generator import ProbeTrajectoryGenerator
from src.polars_engine import PolarsMobilityEngine
from src.reliability_metrics import MobilityReliabilityEngine

st.set_page_config(
    page_title="US-Scale Mobility Data Pipeline",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Application Header
st.title("🚗 US-Scale Mobility Data Pipeline & Network Intelligence")
st.markdown("""
**High-Performance Columnar Architecture for 100M+ Trajectory Records**  
Powered by **DuckDB (Out-of-Core SQL)**, **Apache Parquet (Hive Partitioning)**, **Polars**, **Uber H3**, and **FHWA MAP-21 / PM3 Rule Engine**.
""")

# Sidebar Configuration
st.sidebar.header("⚙️ Corridor & Execution Controls")

corridor_id = st.sidebar.selectbox(
    "Target Highway Corridor:",
    options=list(CORRIDORS.keys()),
    format_func=lambda x: f"{x}: {CORRIDORS[x].name}"
)
corridor = CORRIDORS[corridor_id]

# Data Management
data_scale = st.sidebar.select_slider(
    "Probe Sample Scale:",
    options=[100_000, 500_000, 1_000_000, 2_000_000, 5_000_000],
    value=500_000,
    format_func=lambda x: f"{x:,} Records"
)

DATA_BASE_DIR = Path("data/demo_cache")


@st.cache_resource(show_spinner="Generating calibrated probe trajectories & writing Parquet...")
def load_corridor_data(c_id: str, sample_size: int):
    c_dir = DATA_BASE_DIR / f"{c_id}_{sample_size}"
    if not (c_dir / f"corridor_id={c_id}").exists():
        gen = ProbeTrajectoryGenerator(seed=42)
        etl = MobilityETLPipeline()
        stream = gen.stream_dataset(total_records=sample_size, corridor_id=c_id, chunk_size=250_000)
        etl.process_and_persist(stream, corridor_id=c_id, output_dir=c_dir)
    
    # Initialize DuckDB Engine
    engine = DuckDBMobilityEngine(max_memory="2GB", threads=4)
    engine.register_dataset(c_dir, view_name="mobility_probe")
    rel_engine = MobilityReliabilityEngine(engine.conn)
    return c_dir, engine, rel_engine


try:
    c_dir, duck_engine, rel_engine = load_corridor_data(corridor_id, data_scale)
except Exception as e:
    st.error(f"Error initializing pipeline: {str(e)}")
    st.stop()

# Compute Key Reliability Metrics
with st.spinner("Computing FHWA MAP-21 Reliability & Hexagonal Density..."):
    summary = rel_engine.compute_corridor_summary(corridor_id)
    lottr_data = rel_engine.compute_segment_lottr(corridor_id)
    tttr_data = rel_engine.compute_segment_tttr(corridor_id)
    hex_data = duck_engine.compute_h3_spatial_grid(corridor_id, resolution="res7")

# Top KPI Metric Banners
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
with kpi1:
    st.metric(
        label="Total Trajectory Points",
        value=f"{data_scale:,}",
        delta=f"{len(corridor.segments)} Segments"
    )
with kpi2:
    compliance_color = "normal" if summary.corridor_lottr_pct >= 80.0 else "inverse"
    st.metric(
        label="LOTTR Compliance",
        value=f"{summary.corridor_lottr_pct:.1f}%",
        delta=f"{summary.reliable_segments}/{summary.total_segments} Segments Reliable",
        delta_color=compliance_color
    )
with kpi3:
    st.metric(
        label="Freight TTTR Index",
        value=f"{summary.system_tttr_index:.2f}",
        delta="Target < 1.50",
        delta_color="inverse" if summary.system_tttr_index > 1.50 else "normal"
    )
with kpi4:
    st.metric(
        label="Planning Time Index (PTI)",
        value=f"{summary.mean_pti:.2f}",
        delta=f"{summary.corridor_name.split()[0]} Corridor"
    )
with kpi5:
    st.metric(
        label="Buffer Time Index (BTI)",
        value=f"{summary.mean_bti_pct:.1f}%",
        delta="Buffer Cushion Required"
    )

st.markdown("---")

# Main Dashboard Tabs
tab_map, tab_spacetime, tab_map21, tab_physics, tab_benchmark = st.tabs([
    "🗺️ 3D Spatial Hexagons (PyDeck)",
    "⏱️ Space-Time Bottlenecks",
    "📋 FHWA MAP-21 Compliance",
    "🌊 LWR Kinematic Shockwaves",
    "⚡ Columnar Benchmarks"
])

with tab_map:
    st.subheader(f"Spatial Coverage & Probe Congestion Intensity: {corridor.name}")
    
    col_map_opts, col_map_disp = st.columns([1, 4])
    with col_map_opts:
        st.markdown("**Spatial Filtering**")
        render_mode = st.radio("Visualization Layer:", ["3D Hexagon Density", "Congestion % Points"])
        hex_df = pd.DataFrame(hex_data)
        st.write(f"**Unique H3 Hexagons**: {len(hex_df)}")
        st.write(f"**Avg Hex Speed**: {hex_df['mean_speed'].mean():.1f} mph")
        st.write(f"**Max Point Density**: {hex_df['point_density'].max():,} pts")

    with col_map_disp:
        if not hex_df.empty:
            mid_lat = (corridor.min_lat + corridor.max_lat) / 2
            mid_lon = (corridor.min_lon + corridor.max_lon) / 2

            if render_mode == "3D Hexagon Density":
                layer = pdk.Layer(
                    "ColumnLayer",
                    data=hex_df,
                    get_position=["center_lon", "center_lat"],
                    get_elevation="point_density",
                    elevation_scale=0.08,
                    radius=500,
                    get_fill_color="[255 - mean_speed * 2.5, mean_speed * 3.0, 80, 190]",
                    pickable=True,
                    auto_highlight=True,
                )
            else:
                layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=hex_df,
                    get_position=["center_lon", "center_lat"],
                    get_radius="point_density * 0.15",
                    get_fill_color="[230, 40, 40, 180]",
                    pickable=True,
                )

            view_state = pdk.ViewState(
                latitude=mid_lat,
                longitude=mid_lon,
                zoom=10.5,
                pitch=45,
                bearing=15,
            )

            r = pdk.Deck(
                layers=[layer],
                initial_view_state=view_state,
                tooltip={"html": "<b>H3 Cell:</b> {h3_index}<br/><b>Density:</b> {point_density} pts<br/><b>Mean Speed:</b> {mean_speed} mph<br/><b>Congestion %:</b> {congestion_pct}%"},
                map_style="mapbox://styles/mapbox/light-v10",
            )
            st.pydeck_chart(r)

with tab_spacetime:
    st.subheader("Space-Time Velocity Profile (Lighthill-Whitham-Richards Bottleneck Formation)")
    st.markdown("""
    Space-Time matrix aggregated into 15-minute intervals across corridor segments using Polars.
    Observe how recurrent bottleneck queues form and propagate during morning and evening peak hours.
    """)
    
    polars_engine = PolarsMobilityEngine(c_dir)
    st_matrix = polars_engine.compute_spacetime_speed_matrix(corridor_id, time_bucket_minutes=15)
    
    st_df = st_matrix.to_pandas()
    if not st_df.empty:
        st_df["time_str"] = st_df.apply(lambda r: f"{int(r['hour']):02d}:{int(r['minute']):02d}", axis=1)
        
        fig_st = px.density_heatmap(
            st_df,
            x="time_str",
            y="segment_id",
            z="median_speed_mph",
            histfunc="avg",
            color_continuous_scale="Turbo_r",
            title=f"{corridor.name} - Space-Time Velocity Contour (mph)",
            labels={"time_str": "Time of Day (15-min Intervals)", "segment_id": "Corridor Segment", "median_speed_mph": "Speed (mph)"}
        )
        fig_st.update_layout(height=420)
        st.plotly_chart(fig_st, use_container_width=True)

with tab_map21:
    st.subheader("Federal Highway Administration (FHWA) MAP-21 / PM3 Rule Compliance")
    st.markdown(f"""
    **Federal Standard (23 CFR § 490.511)**: A segment is classified as **Reliable** if 
    LOTTR is strictly under {LOTTR_RELIABLE_THRESHOLD} across all 4 time periods.
    """)

    lottr_df = pd.DataFrame([
        {
            "Segment": r.segment_name,
            "Period": r.period_name,
            "TT 50th (min)": r.travel_time_50th_min,
            "TT 80th (min)": r.travel_time_80th_min,
            "LOTTR Ratio": r.lottr_ratio,
            "Compliance": "PASS (< 1.50)" if r.is_reliable else "FAIL (>= 1.50)",
        }
        for r in lottr_data
    ])

    c_left, c_right = st.columns([1, 1])
    with c_left:
        fig_lottr = px.bar(
            lottr_df,
            x="Segment",
            y="LOTTR Ratio",
            color="Compliance",
            barmode="group",
            color_discrete_map={"PASS (< 1.50)": "#2ca02c", "FAIL (>= 1.50)": "#d62728"},
            title="Segment LOTTR by Federal Time Period",
        )
        fig_lottr.add_hline(y=1.50, line_dash="dash", line_color="black", annotation_text="Federal Threshold (1.50)")
        fig_lottr.update_layout(height=400)
        st.plotly_chart(fig_lottr, use_container_width=True)

    with c_right:
        tttr_df = pd.DataFrame([
            {
                "Segment": r.segment_name,
                "Period": r.period_name,
                "TT 50th (min)": r.travel_time_50th_min,
                "TT 95th (min)": r.travel_time_95th_min,
                "TTTR Ratio": r.tttr_ratio,
            }
            for r in tttr_data
        ])
        fig_tttr = px.line(
            tttr_df,
            x="Period",
            y="TTTR Ratio",
            color="Segment",
            title="Truck Travel Time Reliability (TTTR - 23 CFR § 490.611)",
            markers=True,
        )
        fig_tttr.update_layout(height=400)
        st.plotly_chart(fig_tttr, use_container_width=True)

    st.dataframe(lottr_df, use_container_width=True)

with tab_physics:
    st.subheader("Lighthill-Whitham-Richards (LWR) Kinematic Shockwave Simulator")
    st.markdown("""
    Compute shockwave velocity between upstream free-flow conditions
    and downstream queue bottleneck using Greenshields fundamental diagram.
    """)

    col_sim_in, col_sim_out = st.columns([1, 1])
    with col_sim_in:
        up_spd = st.slider("Upstream Approach Speed (mph)", min_value=30.0, max_value=75.0, value=65.0, step=1.0)
        down_spd = st.slider("Bottleneck Queue Speed (mph)", min_value=5.0, max_value=50.0, value=20.0, step=1.0)
        ff_spd = st.number_input("Corridor Free-Flow Speed (mph)", min_value=50.0, max_value=80.0, value=float(corridor.free_flow_speed_mph))

    wave = rel_engine.compute_lwr_shockwave(
        upstream_speed_mph=up_spd,
        downstream_speed_mph=down_spd,
        free_flow_speed_mph=ff_spd,
    )

    with col_sim_out:
        st.markdown(f"""
        <div style="background-color: #f8f9fa; border-radius: 8px; padding: 16px; border-left: 4px solid #1f77b4;">
            <h4>Calculated Shockwave Characteristics</h4>
            <p><b>Shockwave Speed (&omega;):</b> {wave['shockwave_speed_mph']} mph</p>
            <p><b>Wave Propagation:</b> <span style="color: {'#d62728' if wave['shockwave_speed_mph'] < 0 else '#2ca02c'}; font-weight: bold;">{wave['propagation_direction']}</span></p>
            <p><b>Upstream Density (k1):</b> {wave['upstream_density_vpm']} veh/mi | <b>Flow (q1):</b> {wave['upstream_flow_vph']} veh/hr</p>
            <p><b>Queue Density (k2):</b> {wave['downstream_density_vpm']} veh/mi | <b>Flow (q2):</b> {wave['downstream_flow_vph']} veh/hr</p>
        </div>
        """, unsafe_allow_html=True)

with tab_benchmark:
    st.subheader("Performance Benchmarking: Out-of-Core Columnar Scaling")
    st.markdown("""
    Empirical benchmark comparison between **DuckDB (Streaming Columnar SQL)**, **Polars (Multi-Threaded Vectorized)**,
    and **Vanilla Pandas (In-Memory Object Model)** on 10,000,000 trajectory records.
    """)

    bench_summary_path = Path("benchmarks/results_summary.md")
    if bench_summary_path.exists():
        st.markdown(bench_summary_path.read_text())
    else:
        st.info("Run  to populate benchmark results.")
