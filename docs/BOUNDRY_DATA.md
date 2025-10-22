L1: Türkiye - Subnational Administrative Boundaries

This repository uses the Türkiye administrative boundaries provided by Humanitarian Data Exchange (HDX / OCHA). The source package we rely on is the 2025 COD-AB release distributed as a shapefile ZIP archive.

Download (primary distribution): `https://data.humdata.org/dataset/d74086a0-f398-4474-9e12-1b9a70907bd0/resource/d8636d50-2afb-44d5-b989-2ef812da6725/download/tur_adm_2025_ab_shp.zip`

Why this dataset is used
- Many public boundary datasets (various national and international sources) contain mismatches, topology errors, or outdated labels for Türkiye's administrative subdivisions. For BVPC analyses we require consistent, topologically-correct polygons at the district (`İlçe`) level because our Google Earth Engine reductions, area calculations, and per-district aggregations are sensitive to boundary precision.
- The HDX/OCHA COD-AB 2025 package is curated for humanitarian and operational use and has been processed to align with official mapping products. It is the dataset used by the project because it provides better consistency and metadata than several alternative sources.

Archive contents and expected files
- The ZIP archive contains shapefile sets for administrative levels 0, 1 and 2. Common file names appearing inside the archive (may vary by mirror but typical):
  - `TUR_adm0.*` (country boundary)
  - `TUR_adm1.*` (province boundaries; 81 features)
  - `TUR_adm2.*` (district boundaries; ~973 features)
- Each shapefile set typically contains `.shp`, `.shx`, `.dbf`, `.prj` and ancillary files.

Coordinate reference systems and usage
- The shapefiles are provided in geographic coordinates (WGS84 / EPSG:4326). This is the format expected when converting geometries to Earth Engine `ee.Geometry` objects and for web mapping.
- For planar area calculations (e.g., square meters) the loader/project code projects geometries to an appropriate projected CRS (EPSG:3857 or a local equal-area CRS) before computing areas.

Key attributes and mapping
- The COD-AB shapefiles contain standard OCHA/COD attributes such as `GID_*`, `NAME_*`, `ENGTYPE_*`, and administrative codes. In this repository we map:
  - `NAME_1` → province name (İl)
  - `NAME_2` → district name (İlçe)
- Additional attributes may include unique administrative identifiers which are preserved where available.

Provenance and versioning
- Provider: General Command of Mapping (HGM), prepared and distributed through OCHA/HDX.
- Dataset: "Türkiye - Subnational Administrative Boundaries (COD-AB) - 2025" (the archive above)
- Last-checked in this repository: January 17, 2025 (see repository metadata and notes).

Usage notes for this project
- The repository's loader (`stbesa_service.py` / `OCHACODLoader`) supports both local ZIP archives and the remote HDX URL. The loader:
  - Downloads and extracts the ZIP when a remote URL is provided.
  - Caches the extracted shapefiles in a local work directory to avoid repeated downloads.
  - Validates basic topology and attempts simple fixes (buffering, dissolve) when small topology issues are detected.
- If you plan to update the boundary data, download the new ZIP and point the loader to the local file or URL. The loader will reuse the local copy when present.

Attribution and license
- Source/Attribution: Humanitarian Data Exchange (HDX) / OCHA — dataset page and metadata contain the canonical attribution and license information. When redistributing any derived product, keep the original dataset attribution.

Contact
- For questions about dataset provenance or suitability, consult the HDX dataset page or the original contributor listed on HDX.