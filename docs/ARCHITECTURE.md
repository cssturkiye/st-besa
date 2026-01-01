# Architecture

This document describes the software architecture of ST-BESA.

## Design Principles

1. **Modularity**: Separation of concerns into discrete modules.
2. **Portability**: Runs locally, in Jupyter notebooks, and on Google Colab.
3. **Statelessness**: Session state is managed per-user via Gradio's `gr.State`.

## Package Structure

```
stbesa/
├── __init__.py        # Package initialization
├── analysis.py        # Earth Engine computation logic
├── constants.py       # Configuration constants and lookup tables
├── converters.py      # Data transformation utilities
├── exporter.py        # Export handlers (Excel, PNG, ZIP)
├── gradio_app.py      # User interface (Gradio)
└── service.py         # Service integration layer
```

## Module Responsibilities

### `analysis.py`
**Purpose**: Encapsulates all Google Earth Engine interactions.

| Class | Responsibility |
|-------|----------------|
| `STBESAAnalysis` | Handles EE authentication, image retrieval, and statistical reductions. |

Key methods:
- `initialize_ee()`: Implements 3-step robust authentication.
- `compute_basic_stats()`: Computes sum statistics (volume, surface, population) for a geometry.
- `compute_smod_statistics()`: Computes per-class aggregates for SMOD L1/L2.
- `dynamic_stretch()`: Calculates percentile-based visualization bounds.

### `exporter.py`
**Purpose**: Handles all export operations.

| Class | Responsibility |
|-------|----------------|
| `STBESAExporter` | Exports Excel workbooks and plot images. |
| `LayerExporter` | Exports high-resolution map layers (600 DPI). |

Export pipeline for layers:
1. Compute geometry bounds.
2. Generate EE visualization images.
3. Download images in parallel (ThreadPoolExecutor).
4. Create legends and Photoshop script.
5. Package into ZIP archive.
6. Clean up temporary directory.

### `gradio_app.py`
**Purpose**: Builds and manages the web-based user interface.

Components:
- Configuration panel (dataset, province, district, year).
- Visualization controls (auto-scale, manual ranges).
- Output tabs (Map, Charts, Data).
- Export buttons (Excel, Plots, Layers).

State management:
- Uses `gr.State` to store analysis results per session.
- Enables instant year switching without re-computation.
- Isolates user sessions from each other.

### `service.py`
**Purpose**: Integrates boundary data loading with analysis operations.

| Class | Responsibility |
|-------|----------------|
| `STBESAService` | Loads datasets from `datasets.json`, auto-downloads missing files, provides province/district lookups. |

Key features:
- **Auto-download**: If a boundary file is missing locally, downloads from the URL specified in `datasets.json`.
- **SHA256 Verification**: Validates downloaded files against stored checksums to ensure data integrity.
- **Progress Reporting**: Reports download and loading progress to the UI via callback.

### `constants.py`
**Purpose**: Defines project-wide constants.

Contents:
- Column name mappings (programmatic to user-friendly).
- Column descriptions for data dictionary.
- SMOD class definitions (L1 and L2).
- Color palettes for visualization.

### `converters.py`
**Purpose**: Data transformation utilities.

Functions:
- GeoDataFrame to Earth Engine geometry conversion.
- Coordinate reference system transformations.
- Data validation and cleaning.

## Data Flow

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│   service   │───▶│   analysis   │───▶│   exporter  │
│  (Boundary) │    │  (EE Stats)  │    │  (Output)   │
└─────────────┘    └──────────────┘    └─────────────┘
        │                  │                   │
        └──────────────────┼───────────────────┘
                           │
                    ┌──────▼──────┐
                    │ gradio_app  │
                    │    (UI)     │
                    └─────────────┘
```

## Session State Structure

The `gr.State` object stores:

| Key | Type | Description |
|-----|------|-------------|
| `overall` | DataFrame | Yearly aggregate statistics |
| `l1` | DataFrame | SMOD L1 breakdown |
| `l2` | DataFrame | SMOD L2 breakdown |
| `geom` | ee.Geometry | Selected region geometry |
| `maps` | Dict | Pre-rendered map HTML by year |
| `vis_params` | Dict | Visualization parameters |
| `meta` | Dict | Analysis metadata |

## Technology Stack

| Component | Technology |
|-----------|------------|
| UI Framework | Gradio 4.x |
| Geospatial Backend | Google Earth Engine |
| Map Rendering | geemap, Folium |
| Data Processing | pandas, geopandas, NumPy |
| Visualization | Matplotlib |
| Export | openpyxl, Pillow, zipfile |

## Concurrency Model

- **Analysis**: Uses `ThreadPoolExecutor` for parallel year processing.
- **Layer Export**: Uses `ThreadPoolExecutor` for parallel image downloads.
- **UI**: Gradio handles concurrent user sessions via async handlers.

## Error Handling

| Layer | Strategy |
|-------|----------|
| Earth Engine | Exponential backoff for rate limits (429 errors). |
| Export | Try/except with warning logs; non-critical failures do not halt execution. |
| UI | `gr.Error` for user-facing errors; modal overlay for progress feedback. |

## Boundary Data Management

Large boundary files (`.gpkg`) are not stored in the Git repository to keep clone sizes small. Instead:

1. **GitHub Releases**: Boundary files are hosted as release assets.
2. **Registry Configuration**: `datasets.json` stores download URLs and SHA256 checksums.
3. **On-Demand Download**: When a user selects a dataset, the service layer checks if the file exists locally. If not, it downloads automatically.
4. **Integrity Check**: Downloaded files are validated against SHA256 hashes before use.

Configuration example (`datasets.json`):
```json
{
    "TUR": {
        "file": "ocha_cod_tur.gpkg",
        "boundary_url": "https://github.com/.../releases/download/v1.0-data/ocha_cod_tur.gpkg",
        "boundary_sha256": "sha256:572a91f01e3120eeb08d78f4f14ddcedc8ae713988c8ebee8735ad26ac6fade5"
    }
}
```

This approach ensures:
- Fast repository cloning.
- Reproducible data provenance.
- Transparent user experience (progress bar during download).
