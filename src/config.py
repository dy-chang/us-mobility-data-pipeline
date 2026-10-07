"""
Corridor definitions, H3 spatial parameters, and FHWA MAP-21 performance thresholds.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class CorridorSegment:
    segment_id: str
    corridor_id: str
    name: str
    start_lat: float
    start_lon: float
    end_lat: float
    end_lon: float
    length_miles: float
    free_flow_speed_mph: float
    functional_class: str = "Interstate"


@dataclass
class CorridorConfig:
    corridor_id: str
    name: str
    state: str
    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float
    free_flow_speed_mph: float
    segments: List[CorridorSegment] = field(default_factory=list)


# Major US Interstate Corridors for Calibration and Demonstration
CORRIDORS: Dict[str, CorridorConfig] = {
    "I-95_MIA": CorridorConfig(
        corridor_id="I-95_MIA",
        name="Interstate 95 (Miami - Fort Lauderdale Metro)",
        state="FL",
        min_lat=25.70,
        max_lat=26.30,
        min_lon=-80.30,
        max_lon=-80.10,
        free_flow_speed_mph=65.0,
        segments=[
            CorridorSegment("I95_MIA_01", "I-95_MIA", "Downtown Miami to Midtown", 25.778, -80.201, 25.815, -80.205, 2.6, 65.0),
            CorridorSegment("I95_MIA_02", "I-95_MIA", "Midtown to 79th St", 25.815, -80.205, 25.848, -80.208, 2.3, 65.0),
            CorridorSegment("I95_MIA_03", "I-95_MIA", "79th St to Golden Glades", 25.848, -80.208, 25.928, -80.209, 5.5, 65.0),
            CorridorSegment("I95_MIA_04", "I-95_MIA", "Golden Glades to Hallandale", 25.928, -80.209, 25.986, -80.165, 4.8, 65.0),
            CorridorSegment("I95_MIA_05", "I-95_MIA", "Hallandale to Hollywood Blvd", 25.986, -80.165, 26.012, -80.163, 1.8, 65.0),
            CorridorSegment("I95_MIA_06", "I-95_MIA", "Hollywood to Fort Lauderdale Int Airport", 26.012, -80.163, 26.071, -80.160, 4.2, 65.0),
            CorridorSegment("I95_MIA_07", "I-95_MIA", "FLL Airport to Broward Blvd", 26.071, -80.160, 26.122, -80.166, 3.6, 65.0),
            CorridorSegment("I95_MIA_08", "I-95_MIA", "Broward Blvd to Commercial Blvd", 26.122, -80.166, 26.190, -80.168, 4.7, 65.0),
        ]
    ),
    "I-4_ORL": CorridorConfig(
        corridor_id="I-4_ORL",
        name="Interstate 4 (Orlando Metro Area)",
        state="FL",
        min_lat=28.35,
        max_lat=28.70,
        min_lon=-81.55,
        max_lon=-81.30,
        free_flow_speed_mph=60.0,
        segments=[
            CorridorSegment("I4_ORL_01", "I-4_ORL", "Universal / Kirkman Rd to Florida Turnpike", 28.460, -81.455, 28.489, -81.432, 2.7, 60.0),
            CorridorSegment("I4_ORL_02", "I-4_ORL", "Turnpike to Conroy / Millenia", 28.489, -81.432, 28.502, -81.417, 1.3, 60.0),
            CorridorSegment("I4_ORL_03", "I-4_ORL", "Millenia to Downtown Orlando (SR 408)", 28.502, -81.417, 28.539, -81.381, 3.5, 60.0),
            CorridorSegment("I4_ORL_04", "I-4_ORL", "Downtown Orlando to Ivanhoe / Princeton", 28.539, -81.381, 28.571, -81.378, 2.2, 60.0),
            CorridorSegment("I4_ORL_05", "I-4_ORL", "Princeton to Lee Rd / Winter Park", 28.571, -81.378, 28.608, -81.393, 2.9, 60.0),
            CorridorSegment("I4_ORL_06", "I-4_ORL", "Winter Park to Maitland Blvd / SR 414", 28.608, -81.393, 28.632, -81.398, 1.7, 60.0),
        ]
    ),
    "I-75_TPA": CorridorConfig(
        corridor_id="I-75_TPA",
        name="Interstate 75 (Tampa Metro / Hillsborough Bypass)",
        state="FL",
        min_lat=27.80,
        max_lat=28.20,
        min_lon=-82.45,
        max_lon=-82.25,
        free_flow_speed_mph=70.0,
        segments=[
            CorridorSegment("I75_TPA_01", "I-75_TPA", "Gibsonton Dr to US 301 / Riverview", 27.868, -82.355, 27.915, -82.352, 3.3, 70.0),
            CorridorSegment("I75_TPA_02", "I-75_TPA", "US 301 to SR 60 / Brandon", 27.915, -82.352, 27.947, -82.353, 2.2, 70.0),
            CorridorSegment("I75_TPA_03", "I-75_TPA", "SR 60 to I-4 Interchange", 27.947, -82.353, 28.003, -82.354, 3.9, 70.0),
            CorridorSegment("I75_TPA_04", "I-75_TPA", "I-4 Interchange to Fletcher Ave", 28.003, -82.354, 28.072, -82.361, 4.8, 70.0),
            CorridorSegment("I75_TPA_05", "I-75_TPA", "Fletcher Ave to Bruce B Downs Blvd", 28.072, -82.361, 28.125, -82.375, 3.8, 70.0),
        ]
    )
}

# Uber H3 Spatial Parameters
H3_MACRO_RES = 7  # Area ~5.16 km^2, Edge ~1.22 km (Macro Corridor Binning)
H3_MICRO_RES = 8  # Area ~0.74 km^2, Edge ~461 m (Bottleneck & Shockwave Resolution)

# Federal Highway Administration (FHWA) MAP-21 / PM3 Rule (23 CFR Part 490)
# Level of Travel Time Reliability (LOTTR) Thresholds
LOTTR_RELIABLE_THRESHOLD = 1.50

# Federal Time Periods for LOTTR (4 Periods)
LOTTR_PERIODS = {
    "AM_PEAK": {"name": "Weekday Morning Peak", "days": [0, 1, 2, 3, 4], "start_hour": 6, "end_hour": 10},
    "MIDDAY": {"name": "Weekday Midday", "days": [0, 1, 2, 3, 4], "start_hour": 10, "end_hour": 16},
    "PM_PEAK": {"name": "Weekday Evening Peak", "days": [0, 1, 2, 3, 4], "start_hour": 16, "end_hour": 20},
    "WEEKEND": {"name": "Weekend Day", "days": [5, 6], "start_hour": 6, "end_hour": 20},
}

# Truck Travel Time Reliability (TTTR) - 5 Federal Periods
TTTR_PERIODS = {
    "AM_PEAK": {"name": "Weekday Morning Peak", "days": [0, 1, 2, 3, 4], "start_hour": 6, "end_hour": 10},
    "MIDDAY": {"name": "Weekday Midday", "days": [0, 1, 2, 3, 4], "start_hour": 10, "end_hour": 16},
    "PM_PEAK": {"name": "Weekday Afternoon Peak", "days": [0, 1, 2, 3, 4], "start_hour": 16, "end_hour": 20},
    "OVERNIGHT": {"name": "Weekday Overnight", "days": [0, 1, 2, 3, 4], "start_hour": 20, "end_hour": 6},
    "WEEKEND": {"name": "Weekend Period", "days": [5, 6], "start_hour": 6, "end_hour": 20},
}

# Parquet Storage Defaults
DEFAULT_COMPRESSION = "ZSTD"
DEFAULT_COMPRESSION_LEVEL = 7
CHUNK_SIZE_RECORDS = 500_000
