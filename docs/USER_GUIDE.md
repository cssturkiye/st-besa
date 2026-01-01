# User Guide

This document describes how to use the ST-BESA platform for spatio-temporal settlement analysis.

## Overview

ST-BESA computes built-environment indicators (building volume, building surface, population) and derived metrics (BVPC, BSPC, volume/surface ratio) for administrative units using Google Earth Engine and JRC GHSL datasets.

## Workflow

```
Select Region → Configure Parameters → Run Analysis → View Results → Export Data
```

## 1. Region Selection

| Control | Description |
|---------|-------------|
| Dataset | Select the administrative boundary dataset (e.g., Turkey OCHA 2025). |
| Province | Select one or more provinces (Level 1 administrative unit). |
| District | Select one or more districts (Level 2 administrative unit). |

## 2. Parameter Configuration

### Year Selection
- Use the **Year Slider** to select the target year (1975–2030, 5-year intervals).
- After initial analysis, the slider enables instant switching between years without re-running computations.

### Visualization Scaling

| Option | Behavior |
|--------|----------|
| Auto Scale (default) | Computes optimal min/max values using 5th–99th percentile stretch. |
| Manual Controls | Allows user-defined min/max values for Volume, Surface, and Population layers. |

When Auto Scale is disabled, manual input fields appear:
- **Volume**: Min/Max values in cubic meters (m³).
- **Surface**: Min/Max values in square meters (m²).
- **Population**: Min/Max values in people count.

The **Normalize** button for each layer recalculates optimal values based on the current data.

## 3. Running the Analysis

1. Click **Run Analysis**.
2. A modal overlay displays progress:
   - Initializing
   - Authenticating
   - Processing Data (Year X of N)
   - Rendering Maps
3. Upon completion, the modal closes and results are displayed.

## 4. Viewing Results

### Map Tab
Displays an interactive map with the following layers (top to bottom):
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
| Excel | Multi-sheet workbook with Overall, L1, L2 statistics and data dictionary. | `.xlsx` |
| Plots | Time-series charts (L1 and L2). | `.zip` containing `.png` files |
| Layers | High-resolution map layers (600 DPI, 174 mm width). | `.zip` containing `.png` files |

Exported files are saved to the `exports/` directory. A notification displays the full path upon completion.

## 6. Instant Year Switching

After running the initial analysis:
1. Adjust the **Year Slider**.
2. The map and charts update instantly using cached data.
3. No re-authentication or re-computation is required.

## Notes

- The platform caches computed statistics in memory. Restarting the application clears the cache.
- For large regions or slow network connections, initial analysis may take several minutes.
