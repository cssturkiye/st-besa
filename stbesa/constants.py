# stbesa/constants.py
"""
ST-BESA Constants and Configuration
SMOD Classification Metadata from JRC Global Human Settlement Layer
"""

# SMOD L1 Classes (Settlement Model Level 1)
SMOD_L1_CLASSES = {
    3: {"name": "URBAN CENTRE", "color": "#FF0000", "rgb": (255, 0, 0), "desc": "Cities (Densely populated area)"},
    2: {"name": "URBAN CLUSTER", "color": "#FFAA00", "rgb": (255, 170, 0), "desc": "Towns & semi-dense area"},
    1: {"name": "RURAL", "color": "#73B273", "rgb": (115, 178, 115), "desc": "Rural areas (Thinly populated area)"},
}

# SMOD L2 Classes (Settlement Model Level 2)
SMOD_L2_CLASSES = {
    30: {"name": "Urban Centre", "l1": 3, "color": "#FF0000"},
    23: {"name": "Dense Urban Cluster", "l1": 2, "color": "#732600"},
    22: {"name": "Semi-dense Urban Cluster", "l1": 2, "color": "#A87000"},
    21: {"name": "Suburban or peri-urban", "l1": 2, "color": "#FFFF00"},
    13: {"name": "Rural Cluster", "l1": 1, "color": "#375623"},
    12: {"name": "Low density rural", "l1": 1, "color": "#ABCD66"},
    11: {"name": "Very low density rural", "l1": 1, "color": "#CDF57A"},
    10: {"name": "Water or no data", "l1": 1, "color": "#7AB6F5"},
}

# L2 Color Map for visualization
L2_COLOR_MAP = {
    10: '#7AB6F5', 11: '#CDF57A', 12: '#ABCD66', 13: '#375623',
    21: '#FFFF00', 22: '#A87000', 23: '#732600', 30: '#FF0000'
}

# Turbo palette for continuous data
TURBO_PALETTE = [
    '#30123b', '#4145ab', '#2e7de0', '#1db6d7', '#27d39b',
    '#7be151', '#c0e620', '#f9e14b', '#fca72c', '#f96814', '#a31e1e'
]

# All analysis years
YEARS_ALL = list(range(1975, 2035, 5))  # 12 years: 1975, 1980, ..., 2030


class SMOD:
    """
    SMOD (Global Human Settlement Layer) metadata and class definitions.
    Backward compatible class structure.
    """
    BAND_NAME = 'smod_code'

    # SMOD Class Values (L2)
    WATER = 10
    NO_BUILDING = 11
    VERY_LOW_DENSITY_RURAL = 12
    LOW_DENSITY_RURAL = 13
    RURAL_CLUSTER = 21
    SUBURBAN = 22
    DENSE_URBAN = 23
    URBAN_CENTRE = 30

    # Human-readable labels (L1 codes)
    LABELS = {
        1: "Rural",
        2: "Urban Cluster", 
        3: "Urban Centre",
        # L2 codes
        10: "Water / No Data",
        11: "Very Low Density Rural",
        12: "Low Density Rural",
        13: "Rural Cluster",
        21: "Suburban / Peri-urban",
        22: "Semi-dense Urban",
        23: "Dense Urban Cluster",
        30: "Urban Centre"
    }

    # Groups for simplified analysis
    GROUPS = {
        "Rural": [11, 12, 13],
        "Urban": [21, 22, 23, 30]
    }

    PALETTE = [
        '#7AB6F5',  # Water
        '#CDF57A',  # Very Low Rural
        '#ABCD66',  # Low Rural
        '#375623',  # Rural Cluster
        '#FFFF00',  # Suburban
        '#A87000',  # Semi-dense
        '#732600',  # Dense Urban
        '#FF0000'   # Urban Centre
    ]


# Export Configuration
EXPORT_CONFIG = {
    "excel": {
        "sheet_name_stats": "Overall_Statistics",
        "sheet_name_l1": "SMOD_L1_Statistics",
        "sheet_name_l2": "SMOD_L2_Statistics",
        "sheet_name_meta": "Metadata",
        "sheet_name_dict": "Data_Dictionary"
    },
    "plot": {
        "dpi": 300,
        "width_mm": 174,
        "aspect_ratio": 0.78
    },
    "map": {
        "width_px": 4110,  # 174mm @ 600 DPI
        "dpi": 600
    }
}

# Column name mapping: programmatic -> user-friendly
COLUMN_NAMES = {
    'il': 'Province',
    'ilce_idx': 'District Index',
    'yil': 'Year',
    'buvol_m3': 'Building Volume (m³)',
    'buvol_sur_m2': 'Building Surface (m²)',
    'pop_person': 'Population (people)',
    'bvpc_m3_per_person': 'BVPC (m³/person)',
    'bspc_m2_per_person': 'BSPC (m²/person)',
    'vol_sur_ratio': 'Volume/Surface Ratio',
    'smod_l1_code': 'SMOD L1 Code',
    'smod_l1_name': 'SMOD L1 Class',
    'smod_l2_code': 'SMOD L2 Code',
    'smod_l2_name': 'SMOD L2 Class',
    'smod_l1_parent': 'Parent L1 Code'
}

# Column descriptions for data dictionary
COLUMN_DESCRIPTIONS = {
    'il': 'Province name (administrative level 1)',
    'ilce_idx': 'District index within the province, or ALL if all districts selected',
    'yil': 'Year of observation (1975-2030 in 5-year intervals)',
    'buvol_m3': 'Total building volume in cubic meters from JRC GHSL GHS_BUILT_V',
    'buvol_sur_m2': 'Total building surface area in square meters from JRC GHSL GHS_BUILT_S',
    'pop_person': 'Total population count from JRC GHSL GHS_POP',
    'bvpc_m3_per_person': 'Building Volume Per Capita: average building volume per person (m³/person)',
    'bspc_m2_per_person': 'Building Surface Per Capita: average building surface per person (m²/person)',
    'vol_sur_ratio': 'Ratio of building volume to building surface, proxy for average building height',
    'smod_l1_code': 'SMOD Level 1 classification code (1=Rural, 2=Urban Cluster, 3=Urban Centre)',
    'smod_l1_name': 'SMOD Level 1 class name',
    'smod_l2_code': 'SMOD Level 2 classification code (10-30 range, detailed urban/rural classification)',
    'smod_l2_name': 'SMOD Level 2 class name (detailed settlement type)',
    'smod_l1_parent': 'Parent SMOD L1 code for this L2 class'
}
