# ST-BESA Multi-Country Boundary Configuration Guide

This document explains how administrative boundary datasets are configured for the ST-BESA (Spatio-Temporal Built Environment & Settlement Analytics) platform. It provides guidelines for adding new countries and understanding the existing configuration structure.

## Table of Contents

1. [Overview](#overview)
2. [Configuration File Structure](#configuration-file-structure)
3. [Data Loading Modes](#data-loading-modes)
4. [Currently Configured Datasets](#currently-configured-datasets)
5. [Adding a New Country](#adding-a-new-country)
6. [Troubleshooting Common Issues](#troubleshooting-common-issues)
7. [Academic Methodology Notes](#academic-methodology-notes)

---

## Overview

ST-BESA supports analysis at two administrative levels:
- **ADM1 (Primary Level)**: Typically provinces, states, regions, or cities.
- **ADM2 (Secondary Level)**: Typically districts, municipalities, boroughs, or neighborhoods.

The platform uses a **registry-based configuration** (`datasets.json`) to define how each country's boundary data should be loaded and interpreted.

### Key Components

| Component | Location | Purpose |
|-----------|----------|---------|
| `datasets.json` | Project root | Registry of all configured datasets |
| `boundaries/` | Project root | Directory containing boundary files |
| `stbesa/converters.py` | Package | Data loading and standardization logic |
| `stbesa/service.py` | Package | Service layer for accessing loaded data |

---

## Configuration File Structure

The `datasets.json` file is a JSON object where each key is a **country/dataset code** (e.g., "TUR", "DEU", "NLD") and the value contains configuration parameters.

### Basic Structure

```json
{
    "COUNTRY_CODE": {
        "name": "Human-readable name",
        "file": "filename.gpkg",
        "format": "ocha" | "kontur",
        // ... format-specific parameters
        "adm1_label": "Label for ADM1 dropdown",
        "adm2_label": "Label for ADM2 dropdown"
    }
}
```

### Common Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `name` | string | Yes | Display name shown in the UI dropdown |
| `file` | string | Yes | Filename (relative to `boundaries/`) |
| `format` | string | Yes | Either `"ocha"` or `"kontur"` |
| `adm1_label` | string | No | Custom label for ADM1 dropdown (default: "Region") |
| `adm2_label` | string | No | Custom label for ADM2 dropdown (default: "Sub-region") |

---

## Data Loading Modes

The `GenericDatasetLoader` supports two primary loading modes based on data structure:

### Mode A: Column-Based (OCHA Format)

Used when the dataset has **explicit columns** for administrative names (e.g., `NAME_1`, `NAME_2`).

**Typical Sources**: UN OCHA COD (Common Operational Datasets), GADM, Natural Earth

**Configuration Parameters:**

```json
{
    "format": "ocha",
    "layer": "ADM2",           // Optional: specific layer in multi-layer GPKG
    "adm1_col": "NAME_1",      // Column containing ADM1 names
    "adm2_col": "NAME_2"       // Column containing ADM2 names
}
```

**Example (Turkey):**
```json
"TUR": {
    "name": "Turkey (Türkiye)",
    "file": "ocha_cod_tur.gpkg",
    "format": "ocha",
    "layer": "ADM2",
    "adm1_col": "NAME_1",
    "adm2_col": "NAME_2",
    "adm1_label": "Province (İl)",
    "adm2_label": "District (İlçe)"
}
```

### Mode B: Level-Based (Kontur/OSM Format)

Used when the dataset uses an **`admin_level`** column to distinguish administrative tiers (common in OpenStreetMap-derived data).

**Typical Sources**: Kontur Boundaries, OpenStreetMap exports, Geofabrik

**Configuration Parameters:**

```json
{
    "format": "kontur",
    "adm1_level": [4, 7],      // Admin level(s) for ADM1 (can be int or array)
    "adm2_level": [8, 9, 10],  // Admin level(s) for ADM2 (can be int or array)
    "name_col": "name",        // Column containing feature names
    "level_col": "admin_level" // Optional: column containing level info (default: "admin_level")
}
```

**Example (Germany):**
```json
"DEU": {
    "name": "Germany (Deutschland)",
    "file": "kontur_boundaries_DE_20230628.gpkg",
    "format": "kontur",
    "adm1_level": [4, 7],
    "adm2_level": [4, 5, 6, 7, 8, 9, 10, 11],
    "name_col": "name",
    "adm1_label": "Region / City / State",
    "adm2_label": "Sub-division (District/Gemeinde)"
}
```

### Understanding Admin Levels

Admin levels follow the OpenStreetMap convention:

| Level | Typical Entity (varies by country) |
|-------|-----------------------------------|
| 2 | Country |
| 3-4 | States, Regions, Provinces |
| 5-6 | Districts, Counties |
| 7-8 | Municipalities, Cities |
| 9-10 | Boroughs, Wards |
| 11+ | Neighborhoods, Localities |

**Important**: Admin levels are **country-specific**. Always inspect the actual data before configuration.

---

## Currently Configured Datasets

### Turkey (TUR)
- **Source**: UN OCHA COD
- **Format**: Column-based (OCHA)
- **ADM1**: 81 Provinces (İl)
- **ADM2**: 973 Districts (İlçe)
- **Notes**: Uses ADM2 layer which contains both province and district names.

### Germany (DEU)
- **Source**: Kontur Boundaries
- **Format**: Level-based
- **ADM1**: States (Bundesländer, Level 4) + Districts (Kreise, Level 7)
- **ADM2**: Municipalities and sub-divisions (Levels 4-11)
- **Notes**: 
  - Berlin, Hamburg, Bremen are "City-States" at Level 4
  - Major cities (München, Köln) are "Kreisfreie Städte" at Level 7
  - Self-referencing levels allow single-unit entities to appear

### Netherlands (NLD)
- **Source**: Kontur Boundaries
- **Format**: Level-based
- **ADM1**: Municipalities (Gemeenten, Level 10)
- **ADM2**: Settlements (Woonplaatsen, Level 12, 14)
- **Notes**:
  - Self-referencing (Level 10 in ADM2) ensures all municipalities appear
  - Detailed neighborhood (Wijk/Buurt) data may require separate CBS datasets

---

## Adding a New Country

Follow this step-by-step process to add a new country to the platform:

### Step 1: Obtain Boundary Data

**Recommended Sources:**
1. **UN OCHA HDX**: https://data.humdata.org/ (COD datasets)
2. **Kontur Boundaries**: https://www.kontur.io/portfolio/population-data/
3. **GADM**: https://gadm.org/
4. **Geofabrik**: https://download.geofabrik.de/

**Preferred Format**: GeoPackage (`.gpkg`) for multi-layer support.

### Step 2: Inspect the Data Structure

Run the following Python commands to understand the data:

```python
import geopandas as gpd
import fiona

# List layers (for multi-layer files)
print("Layers:", fiona.listlayers("your_file.gpkg"))

# Read and inspect
gdf = gpd.read_file("your_file.gpkg")  # or specify layer="LAYER_NAME"
print("Columns:", list(gdf.columns))
print("Sample:\n", gdf.head())

# For level-based data
if 'admin_level' in gdf.columns:
    levels = sorted(gdf['admin_level'].dropna().unique().astype(int).tolist())
    print("Admin Levels:", levels)
    for lvl in levels:
        count = len(gdf[gdf['admin_level'] == lvl])
        sample = gdf[gdf['admin_level'] == lvl]['name'].head(3).tolist()
        print(f"  Level {lvl}: {count} features, e.g., {sample}")
```

### Step 3: Identify ADM1 and ADM2 Levels

**For Column-Based (OCHA):**
- Look for columns like `NAME_1`, `NAME_2`, `ADM1_EN`, `ADM2_EN`
- Check if layers exist: `ADM0`, `ADM1`, `ADM2`

**For Level-Based (Kontur/OSM):**
- Identify which levels correspond to your desired "Region" (ADM1)
- Identify which levels correspond to "Sub-region" (ADM2)
- **Common patterns:**
  - Federal countries: Level 4 (States), Level 6-7 (Districts)
  - Unitary countries: Level 4-6 (Regions), Level 8-10 (Municipalities)

### Step 4: Test Spatial Relationships

Verify that ADM2 features are spatially "within" ADM1 features:

```python
# For a city (e.g., "Berlin")
city = gdf[(gdf['admin_level'] == 4) & (gdf['name'] == 'Berlin')]
sub_units = gdf[gdf['admin_level'] == 10]
joined = gpd.sjoin(sub_units, city[['geometry']], predicate='within')
print(f"Berlin has {len(joined)} sub-units at Level 10")
```

### Step 5: Handle Edge Cases

**Problem: Major cities missing from list**
- **Cause**: City-states or "Kreisfreie Städte" may be at a different level.
- **Solution**: Include multiple levels in `adm1_level` array: `[4, 7]`

**Problem: Cities without sub-divisions**
- **Cause**: Spatial join excludes parent entities with no children.
- **Solution**: Add the ADM1 level to `adm2_level` array for self-referencing.

**Example for self-referencing:**
```json
"adm1_level": [10],
"adm2_level": [10, 12, 14]  // 10 included for cities without sub-units
```

### Step 6: Add Configuration

Add the entry to `datasets.json`:

```json
"NEW": {
    "name": "New Country",
    "file": "boundaries_new.gpkg",
    "format": "kontur",
    "adm1_level": [LEVEL_A, LEVEL_B],
    "adm2_level": [LEVEL_A, LEVEL_B, LEVEL_C, ...],
    "name_col": "name",
    "adm1_label": "Region",
    "adm2_label": "Sub-region"
}
```

### Step 7: Place the File

Copy the boundary file to `boundaries/` directory.

### Step 8: Test

1. Restart the Gradio app
2. Select the new country
3. Verify ADM1 list populates correctly
4. Select a region and verify ADM2 list populates
5. Run a test analysis

---

## Troubleshooting Common Issues

### Issue: "No rows found for ADM2 level X"

**Cause**: The specified level doesn't exist in the dataset.

**Solution**: Inspect available levels:
```python
print(sorted(gdf['admin_level'].dropna().unique().astype(int).tolist()))
```

### Issue: City/Region not appearing in dropdown

**Cause**: The entity exists at a different admin level, or has no children in ADM2 levels.

**Solution**:
1. Search for the entity across all levels:
   ```python
   print(gdf[gdf['name'].str.contains('CityName', na=False)][['name', 'admin_level']])
   ```
2. Add the found level to `adm1_level` array
3. Add self-referencing level to `adm2_level`

### Issue: Duplicate entries in dropdown

**Cause**: Same name exists at multiple levels.

**Solution**: This is usually acceptable. The system merges geometries with identical names when selected.

### Issue: "Unknown" appears as region name

**Cause**: Spatial join failed (ADM2 feature not within any ADM1 feature).

**Solution**:
1. Check coordinate systems (both should be WGS84)
2. Use `intersects` predicate if boundaries have gaps
3. Manually verify spatial relationships in QGIS/ArcGIS

---

## Academic Methodology Notes

### NUTS/LAU Framework Alignment

The ST-BESA configuration attempts to align with the European NUTS/LAU hierarchy:

| ST-BESA Level | NUTS/LAU Equivalent | Turkey | Germany | Netherlands |
|---------------|---------------------|--------|---------|-------------|
| ADM1 | NUTS-3 / LAU-1 | İl (Province) | Kreis | Gemeente |
| ADM2 | LAU-2 | İlçe (District) | Gemeinde | Wijk/Woonplaats |

### Methodological Considerations

1. **Comparability**: When comparing across countries, ensure similar administrative granularity (e.g., Population size ranges).

2. **Temporal Consistency**: Administrative boundaries change over time. Document the dataset date (e.g., "Kontur 2023-06-28").

3. **Citation**: Include data source citations in publications:
   - OCHA COD: "OCHA Common Operational Datasets, UN Office for the Coordination of Humanitarian Affairs"
   - Kontur: "Kontur Population & Boundaries Dataset, Kontur Inc."

4. **Limitations**: Document that neighborhood-level analysis may require specialized datasets beyond standard administrative boundaries.

---

## File Checklist for New Dataset

- [ ] Boundary file in `boundaries/`
- [ ] Entry added to `datasets.json`
- [ ] Admin levels documented
- [ ] Name column identified
- [ ] Self-referencing levels added (if needed)
- [ ] Labels localized (ADM1/ADM2 names)
- [ ] Test analysis completed

---

*Last Updated: 2026-01-01*
*Author: ST-BESA Development Team*
