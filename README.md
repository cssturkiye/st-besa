# ST-BESA: Spatio-Temporal Built Environment & Settlement Analysis Platform
![ST-BESA Banner](img/st-besa-banner.png)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

Purpose
-------
This project is designed for non‑programmer social scientists. It is part of CSS Türkiye and aligns with the Fast CSS Tool ecosystem, making advanced geospatial analysis accessible without coding. See the Fast CSS Tool project for related work: https://github.com/cssturkiye/fastcsstool

This repository implements a reproducible workflow for computing built-environment indicators (building volume, building surface, population) and derived metrics (BVPC, BSPC, volume/surface ratio) for administrative units in Türkiye (province/district) using Google Earth Engine (GEE) and the JRC GHSL datasets. The project provides an interactive Jupyter notebook UI (`app.ipynb`) driven by `stbesa_service.py` which: 

- Loads and validates administrative boundaries (districts / provinces)
- Sends polygons to Earth Engine to extract raster-derived statistics
- Renders interactive web maps and publication-quality time-series plots
- Exports multi-sheet Excel reports with data dictionaries and SMOD breakdowns

This video explains the main flow.

![ST-BESA Demo](img/ST-BESA.gif)


Contents / Repository layout
---------------------------
- `stbesa_service.py` — Core service: data loaders, Earth Engine wrappers, UI builder, plotting and export functions.
- `app.ipynb` — Notebook that instantiates the UI and runs the interactive analysis.
- `BOUNDRY_DATA.md` — Boundary dataset provenance and notes (HDX OCHA COD-AB 2025).
- Other large binary resources (shapefile ZIPs) are expected to be downloaded on demand by the loader and are not committed to the repo.

Primary data sources
--------------------
1. Administrative boundaries (district/province):
   - `tur_adm_2025_ab_shp.zip` (HDX / OCHA COD-AB 2025) — used because it is curated and provides consistent district polygons needed for accurate aggregations. See `BOUNDRY_DATA.md` for details.
   - Loader supports both local ZIP and remote URL references and caches extracted shapefiles.

2. JRC Global Human Settlement Layer (GHSL) on Google Earth Engine:
   - Building volume: `JRC/GHSL/P2023A/GHS_BUILT_V/<year>`
   - Building surface (built-up area): `JRC/GHSL/P2023A/GHS_BUILT_S/<year>`
   - Population count: `JRC/GHSL/P2023A/GHS_POP/<year>`
   - SMOD settlement model: `JRC/GHSL/P2023A/GHS_SMOD/<year>` (L1 and L2 classes)

3. Other: local HGM GPKG (optional) and auxiliary lookup tables.

Install example (Conda / pip):
```bash
conda create -n bvpc python=3.11
conda activate bvpc
pip install -r requirements.txt
pip install earthengine-api geemap geopandas pandas matplotlib ipywidgets folium ipyleaflet openpyxl xlsxwriter
```

Authentication & Google Cloud project
------------------------------------
Earth Engine requires a Google account and a Google Cloud project context. For this repository:

- What you need: a Google Cloud Project ID with the "Earth Engine API" enabled. No API key is required for Earth Engine OAuth in this notebook (API keys are for other Google services; optional if you integrate those).

Quick steps (once):
1. Create a Google Cloud project in the Cloud Console.
2. Enable "Earth Engine API" for that project.
3. Make sure your Google account has Earth Engine access (sign up if needed) and note the Project ID.

How the code authenticates
- In notebooks (Colab/Jupyter), initialization will attempt these flows in order:
  1) Use existing credentials: `ee.Initialize(project='<PROJECT_ID>')`
  2) Interactive link/token flow: `ee.Authenticate()` then `ee.Initialize(...)`
  3) Hosted-notebook popup flow: `ee.Authenticate(auth_mode='notebook')` then `ee.Initialize(...)`

If all fail, the code raises a clear error explaining how to enable Earth Engine API and retry.

Local developers (CLI alternative)
- You can also authenticate once from a terminal:
```bash
earthengine authenticate
```
Then the notebook/app can call `ee.Initialize(project='<PROJECT_ID>')` without prompts.

High-level workflow
-------------------
1. Load administrative boundaries and build GeoDataFrames for provinces and districts.
2. The UI allows selecting province(s), district(s), a target year (1975–2030 in 5 year steps), and visualization parameters.
3. When the user runs the analysis, the selected polygons are converted to Earth Engine `ee.Geometry` (preserving WGS84), and GHSL images for the selected year are requested.
4. Raster layers are optionally reprojected/resampled to align 1 km SMOD (or GHSL sources) to the 100 m grid used by built/pop rasters.
5. Per-band reductions (sum, percentile) and grouped reductions (per-class totals for SMOD) are performed; results are cached in-memory for UI responsiveness.
6. Map layers and time-series plots are rendered; an Excel export computes multi-year SMOD breakdowns if requested and writes several sheets: overall stats, L1 stats, L2 stats, and a data dictionary.

Key modules and functions (code walk-through)
-------------------------------------------
Note: this is a conceptual overview of the code present in `stbesa_service.py` and how the pieces fit together.

- OCHACODLoader
  - Purpose: download/unpack administrative boundary ZIPs (local or remote), validate contents, and prepare `GeoDataFrame` objects for use by the rest of the pipeline.
  - Behavior: when provided a URL it downloads to a `tempfile` work dir; when provided a local zip path it extracts into `_ocha_tmp` (or a similar cache folder). If the folder already exists with shapefiles, it will reuse them to avoid repeated downloads.
  - Topology handling: performs basic checks and applies simple fixes (buffer(0) or dissolve) to correct minor topology issues that would otherwise break polygonization or area calculations.

- STBESAAnalysis (Earth Engine wrapper)
  - Purpose: encapsulates the Earth Engine interactions and compute routines.
  - Main responsibilities:
    - `get_images_for_year(year)`: loads `ee.Image` objects for volume, surface, population, and SMOD for the requested year.
    - `compute_basic_stats(geom, year)`: performs `reduceRegion` with `ee.Reducer.sum()` for `volume`, `surface`, and `population` bands to compute totals within a geometry.
    - `compute_smod_statistics(geom, year, level='L1'|'L2')`: computes per-class aggregated statistics for SMOD classes. Implementation notes:
      - SMOD original resolution is 1 km; to align with 100 m GHSL rasters the code reprojects/resamples as needed, using `reproject()` with an explicit CRS and scale. For categorical data, nearest-neighbor is required — the code avoids invalid `resample('near')` calls and instead relies on `reproject()`'s default nearest behavior.
      - Grouped reductions in Earth Engine: `Reducer.group()` requires an image with exactly two bands `[value, group]` where `groupField=1`. To compute totals for multiple metrics (volume, surface, population) the code runs separate `reduceRegion` calls per metric, each combining the metric band with the SMOD class band to produce per-class sums. This avoids the `Need 2 bands for Reducer.group` error and the `Group input must come after weighted inputs` ordering issue.

- UI builder (`build_picker_ui`)
  - Purpose: construct the interactive Jupyter widgets (using `ipywidgets`) and outputs (`out_map`, `out_tbl`, `out_plot`) and bind handlers.
  - Important components and behavior:
    - Selection controls: province dropdown, district multi-select, year slider, scale controls (auto/manual), performance settings (Fast toggle which switches map backend between Folium and ipyleaflet). The district dropdown displays indices in the label for clarity (e.g. `Kadıköy (1)`).
    - Action buttons: `Run Analysis` (runs on the selected year and region), `Export XLSX` (generates full multi-year SMOD breakdowns and writes an Excel file), `Apply scale` (applies manual color ranges), and a `Save image` button inside the map (folium backend) when the Fast option is off.
    - Busy state / UX: all long-running actions are wrapped with a `set_busy(True/False)` helper that disables interactive controls and displays a transient overlay with a spinner message. Status messages update progressively during multi-step operations and are scheduled to auto-dismiss after a short interval using `asyncio` to avoid background thread `ContextVar` issues.
    - Map rendering: uses `geemap` to produce either a Folium or ipyleaflet map depending on the Fast toggle. Map layer order is controlled to render (top-to-bottom): Boundary, Building volume, Building surface, Population, SMOD L1, SMOD L2.
    - Legends: continuous legends for volume/surface/population are injected as compact HTML colorbars into the folium root and positioned using absolute CSS (`right` and `bottom`). Categorical legends for SMOD L1 and L2 are injected as separate HTML blocks positioned at the bottom-left. The UI code exposes tuning values for these positions and has been iterated to avoid overlap with other map controls.

- Plotting helpers
  - Purpose: create publication-quality time-series plots (3x2 grid) for the six metrics: building volume (m³), building surface (m²), population (people), BVPC (m³/person), BSPC (m²/person), and Volume/Surface ratio.
  - Implementation notes:
    - Uses Matplotlib with global rcParams tuned for publication quality: `figure.dpi=300`, `savefig.dpi=300`, width set to 174 mm converted to inches (~6.85), and height scaled by 2/3 for compact subplots.
    - Legends for L1 and L2 class breakdowns are embedded inside the figure area, positioned below subplots to avoid overlap, and rendered as compact 3x2 color-box grids.
    - Care is taken to avoid duplicate rendering by centralizing plot creation and ensuring `plt.close(fig)` is called after embedding the figure into the notebook output.

Caching and performance
-----------------------
- In-memory cache: computed per-year results are stored in a `cache` dictionary keyed by year and dataset name to avoid redundant EE calls when the user toggles visualization settings or re-renders the map for the same year.
- SMOD performance: because per-class SMOD reductions across many years can be slow, the UI computes SMOD stats only for the current year on `Run` (fast UX). Full multi-year SMOD computations are deferred until `Export XLSX` where the code computes all years required and writes them into the Excel file.

Excel export
------------
The export routine produces a multi-sheet Excel workbook with the following sheets:
- `overall` — overall totals per year (vol, sur, pop, BVPC, BSPC, ratio) using programmatic column names (machine-friendly).
- `smod_l1` — per-year and per-class totals for SMOD L1 (ordered by `smod_l1_code`, then `year`) to make selection and plotting easy.
- `smod_l2` — per-year and per-class totals for SMOD L2 (ordered by `smod_l2_code`, then `year`).
- `data_dictionary` — mapping between programmatic column names and UX-friendly labels and a textual description column.

Exported files follow the naming convention: `BVPC_Export_<province>_<district>_<YYYYMMDD_HHMMSS>.xlsx` with logic to handle the district placeholder:
- If all districts are selected: `ALL` used for the district part.
- If multiple but not all and <= 5: district names are concatenated with commas (sanitized for filenames).
- If multiple and > 5: only indices are included to keep filenames short.



License and attribution
-----------------------
- This project is released under the MIT License (see LICENSE file for details).
- Data sources include HDX/OCHA (administrative boundaries) and JRC GHSL (built/pop/SMOD) via Google Earth Engine. 
- When using this software, please respect the data providers' licenses:
  - HDX/OCHA data: Creative Commons Attribution for Intergovernmental Organisations (CC BY-IGO) 3.0
  - JRC GHSL data: Creative Commons Attribution 4.0 International (CC BY 4.0)
- For derivative products, ensure proper attribution to both this software and the underlying data sources.