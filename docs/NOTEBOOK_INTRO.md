# ST-BESA Notebook - Overview & How to Use
> Spatio-Temporal Built Environment & Settlement Analysis Platform

![ST-BESA Banner](img/st-besa-banner.png)

## What this application does

- Analyzes the built environment across years (1975–2030) for selected Turkish provinces/districts.
- Computes and visualizes: Building Volume, Building Surface, Population, BVPC (m³/person), BSPC (m²/person), Volume/Surface Ratio.
- Adds SMOD Settlement Model (L1/L2) class breakdowns on the map and in time-series plots.
- Exports a multi-sheet Excel report (overall + SMOD L1/L2 + data dictionary).

## How to run

1) Installation (first cell, named "Installation")
- This cell silently installs dependencies and loads the application code. No user action or code edits are required.

2) Application (second cell, named "Application")
- Click the play button on this cell to open the interactive interface.
- Choose Province and District(s), pick a Year, optionally adjust visualization settings.
- Click "Run Analysis" to render the map and plots.
- Click "Export XLSX" to generate the Excel file.

## First‑time Google access

- You will use your Google account to authorize access to Google Earth Engine.
- If required, a browser prompt will appear automatically when you press the "Application" cell.

Google Cloud project (Project ID)
- You need a Google Cloud Project ID associated with your Earth Engine account for quota/billing context.
- Quick steps (once):
  1. Go to Google Cloud Console and create a new project.
  2. Enable the "Earth Engine API" for that project.
  3. Note the Project ID; the app uses it when initializing Earth Engine.
- API key: not required for Earth Engine in this notebook. Only the Project ID is needed.

Tips
- Export may take longer than a single-year analysis because it computes full multi‑year SMOD stats.
- If something fails, re-run the "Application" cell once; any required Google sign-in will re-prompt.