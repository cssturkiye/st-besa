# User Guide

This document describes how to use the ST-BESA platform for spatio-temporal settlement analysis.

## Overview

ST-BESA computes built-environment indicators (building volume, building surface, population) and derived metrics (BVPC, BSPC, volume-to-surface ratio) for administrative units using Google Earth Engine and JRC GHSL datasets.

## Before You Start

For the step-by-step first-run workflow, see the [README Quick Start](../README.md#quick-start). For detailed local setup and authentication, see the [Installation Guide](INSTALLATION.md). This guide assumes ST-BESA is already running and focuses on using the interface.

## Workflow

```text
Select Region -> Configure Parameters -> Run Analysis -> View Results -> Export Data
```

## 1. Region Selection

| Control | Description |
|---------|-------------|
| Dataset | Select the administrative boundary dataset (for example, Turkey OCHA 2025). |
| Province | Select one or more provinces or Level 1 administrative units. |
| District | Select one or more districts or Level 2 administrative units. |

## 2. Parameter Configuration

### Year Selection

- Use the **Year Slider** to select the target year (1975-2030, 5-year intervals).
- After the initial analysis, the slider enables instant switching between years without re-running computations.

### Visualization Scaling

| Option | Behavior |
|--------|----------|
| Auto Scale (default) | Computes min/max values using a 5th-99th percentile stretch. |
| Manual Controls | Allows user-defined min/max values for Volume, Surface, and Population layers. |

When Auto Scale is disabled, manual input fields appear:

- **Volume**: Min/Max values in cubic meters (`m^3`).
- **Surface**: Min/Max values in square meters (`m^2`).
- **Population**: Min/Max values in people count.

The **Normalize** button for each layer recalculates suitable values based on the current data.

## 3. Running the Analysis

1. Click **Run Analysis**.
2. A modal overlay displays progress:
   - Initializing
   - Authenticating
   - Processing Data (Year X of N)
   - Rendering Maps
3. When processing is complete, the modal closes and the results are displayed.

## 4. Viewing Results

### Map Tab

Displays an interactive map with the following layers:

- Boundary outline
- Building Volume
- Building Surface
- Population
- SMOD L1 (Degree of Urbanization)
- SMOD L2 (Settlement Classes)

### Chart Tab

Displays time-series plots:

- **L1 Panel**: Metrics aggregated by SMOD Level 1 classes.
- **L2 Panel**: Metrics aggregated by SMOD Level 2 classes.

### Data Tab

Displays tabular statistics for the selected year.

## 5. Exporting Data

| Button | Output | Format |
|--------|--------|--------|
| Excel | Multi-sheet workbook with Overall, L1, L2 statistics and a data dictionary. | `.xlsx` |
| Plots | Time-series charts (L1 and L2, 190 mm width at 1000 DPI). | `.zip` containing `.png` files |
| Layers | High-resolution map layers (190 mm width at 600 DPI). | `.zip` containing `.png` files |

Exported files are saved to the `exports/` directory. This runtime directory is created when exports are generated and stores the artifacts produced by the current run. It is separate from the curated archived example outputs stored under `case-studies/` in the repository. A notification displays the full path upon completion.

## 6. Instant Year Switching

After running the initial analysis:

1. Adjust the **Year Slider**.
2. The map and charts update instantly using cached data.
3. No re-authentication or re-computation is required.

## Notes

- The platform caches computed statistics in memory. Restarting the application clears the cache.
- For large regions or slow network connections, the initial analysis may take several minutes.
