# %%capture
# !pip install earthengine-api>=0.1.397 geemap>=0.33.6 pandas>=2.2 unidecode>=1.3 fiona geopandas folium shapely fiona pyproj osmnx geopy openpyxl

"""
ST-BESA Service: Spatio-Temporal Built Environment & Settlement Analysis Platform

This service provides comprehensive geospatial analysis capabilities for social scientists
working with built environment and settlement data across Turkey. The platform integrates
Google Earth Engine with JRC Global Human Settlement Layer (GHSL) datasets to compute
building volume, surface area, population metrics, and derived indicators (BVPC, BSPC)
across administrative units from 1975-2030.

Key Features:
- Interactive Jupyter notebook interface for non-programmer social scientists
- Multi-temporal analysis with 5-year intervals (1975-2030)
- SMOD land-use classification (L1/L2) integration
- Interactive web maps with customizable visualization
- Professional Excel export with complete data dictionaries
- Support for both individual districts and province-wide analysis

Data Sources:
- Administrative boundaries: HDX/OCHA COD-AB 2025
- Building/population data: JRC GHSL via Google Earth Engine
- Settlement classification: SMOD (Settlement Model) L1/L2

This service is part of the CSS Türkiye ecosystem and aligns with the Fast CSS Tool
project, making advanced geospatial analysis accessible without programming knowledge.

For more information, see: https://github.com/cssturkiye/fastcsstool
"""

from __future__ import annotations

import os
import glob
import zipfile
from typing import List, Optional, Tuple, Dict, Any

import pandas as pd
import geopandas as gpd
import numpy as np


class STBESAService:
    """
    Service for working with HGM-derived GPKG (ADM1/ADM2), listing provinces/districts,
    and preparing geometries for external analysis (e.g., Earth Engine).

    Notes:
    - Expects a GeoPackage with layer "ADM2" containing fields: NAME_1, NAME_2, geometry (EPSG:4326).
    - If NAME_2 is missing or empty, placeholder names can be generated per province.
    - This class does NOT depend on Earth Engine or geemap to keep import surface minimal.
      EE/geemap should be used from the notebook, after retrieving the GeoDataFrames.
    """

    def __init__(self, gpkg_path: str):
        self.gpkg_path = gpkg_path
        self.layer_adm2 = "ADM2"
        self._gdf_adm2: Optional[gpd.GeoDataFrame] = None

    # --------------------------- Internal helpers ---------------------------
    def _ensure_loaded(self) -> None:
        if self._gdf_adm2 is not None:
            return
        layers = self._list_layers()
        if self.layer_adm2 not in layers:
            if not layers:
                raise ValueError("No layers found in the GeoPackage.")
            # fallback to last layer
            self.layer_adm2 = layers[-1]
        gdf = gpd.read_file(self.gpkg_path, layer=self.layer_adm2)
        if gdf.crs is None or (getattr(gdf.crs, 'to_epsg', lambda: None)() != 4326):
            gdf = gdf.to_crs(4326)
        if "NAME_1" not in gdf.columns:
            raise ValueError(f"Expected NAME_1 field. Found: {gdf.columns.tolist()}")
        if "NAME_2" not in gdf.columns:
            gdf["NAME_2"] = pd.Series([None] * len(gdf), dtype=object)
        # Normalize NAME_1/NAME_2 to string or None
        gdf["NAME_1"] = gdf["NAME_1"].astype(object)
        gdf["NAME_2"] = gdf["NAME_2"].astype(object)
        self._gdf_adm2 = gdf

    def _list_layers(self) -> List[str]:
        import fiona
        return list(fiona.listlayers(self.gpkg_path))

    @staticmethod
    def _norm(s: Optional[str]) -> str:
        from unidecode import unidecode
        return unidecode("" if s is None or (isinstance(s, float) and pd.isna(s)) else str(s)).strip()

    # --------------------------- Public API ---------------------------------
    def provinces(self) -> List[str]:
        self._ensure_loaded()
        vals = (
            self._gdf_adm2["NAME_1"].dropna().astype(str).unique().tolist()  # type: ignore[arg-type]
        )
        return sorted(vals, key=lambda x: self._norm(x).lower())

    def districts(self, province_name: str) -> List[str]:
        self._ensure_loaded()
        sub = self._gdf_adm2[self._gdf_adm2["NAME_1"] == province_name]
        vals = sub["NAME_2"].dropna().astype(str).unique().tolist()
        return sorted(vals, key=lambda x: self._norm(x).lower())

    def get_district_row(self, province_name: str, district_name: str) -> gpd.GeoDataFrame:
        self._ensure_loaded()
        row = self._gdf_adm2[(self._gdf_adm2["NAME_1"] == province_name) & (self._gdf_adm2["NAME_2"] == district_name)]
        if row.empty:
            raise ValueError(f"District not found for {province_name}/{district_name}")
        return row

    def get_first_district_row(self, province_name: str) -> gpd.GeoDataFrame:
        """Return the first district row for a province even if NAME_2 is missing.
        Useful when HGM data has no district names yet.
        """
        self._ensure_loaded()
        sub = self._gdf_adm2[self._gdf_adm2["NAME_1"] == province_name]
        if sub.empty:
            raise ValueError(f"No districts found for province {province_name}")
        return sub.iloc[[0]]

    def get_district_row_by_index(self, province_name: str, idx: int) -> gpd.GeoDataFrame:
        """Return the district row at position idx for a province (0-based)."""
        self._ensure_loaded()
        sub = self._gdf_adm2[self._gdf_adm2["NAME_1"] == province_name]
        if sub.empty:
            raise ValueError(f"No districts found for province {province_name}")
        if idx < 0 or idx >= len(sub):
            raise IndexError(f"District index {idx} out of range (0..{len(sub)-1}) for {province_name}")
        return sub.iloc[[idx]]

    def fill_missing_district_names(self) -> None:
        """
        Fills missing NAME_2 values per province using a deterministic local naming scheme.
        Does not fetch external data (keeps HGM-only geometry policy).
        """
        self._ensure_loaded()
        gdf = self._gdf_adm2
        # Future-proof groupby.apply deprecation by excluding group keys
        def _fill(group: pd.DataFrame) -> pd.DataFrame:
            base = str(group["NAME_1"].iloc[0])
            # if any present, only fill missing
            existing_any = group["NAME_2"].notna().any()
            counter = 1
            for idx in group.index:
                if pd.isna(group.at[idx, "NAME_2"]):
                    group.at[idx, "NAME_2"] = f"{base} - İlçe {counter:02d}"
                    counter += 1
            return group
        self._gdf_adm2 = (
            gdf.groupby("NAME_1", group_keys=False, as_index=False).apply(_fill)  # type: ignore[assignment]
        )

    def to_geodataframe(self) -> gpd.GeoDataFrame:
        self._ensure_loaded()
        return self._gdf_adm2.copy()

    def save(self, out_path: str, layer: str = "ADM2") -> None:
        self._ensure_loaded()
        self._gdf_adm2.to_file(out_path, layer=layer, driver="GPKG")


# SMOD Classification Metadata
SMOD_L1_CLASSES = {
    3: {"name": "URBAN CENTRE", "color": "#FF0000", "rgb": (255, 0, 0), "desc": "Cities (Densely populated area)"},
    2: {"name": "URBAN CLUSTER", "color": "#FFAA00", "rgb": (255, 170, 0), "desc": "Towns & semi-dense area"},
    1: {"name": "RURAL", "color": "#73B273", "rgb": (115, 178, 115), "desc": "Rural areas (Thinly populated area)"},
}

SMOD_L2_CLASSES = {
    30: {"name": "Urban Centre", "l1": 3},
    23: {"name": "Dense Urban Cluster", "l1": 2},
    22: {"name": "Semi-dense Urban Cluster", "l1": 2},
    21: {"name": "Suburban or peri-urban", "l1": 2},
    13: {"name": "Rural Cluster", "l1": 1},
    12: {"name": "Low density rural", "l1": 1},
    11: {"name": "Very low density rural", "l1": 1},
    10: {"name": "Water or no data", "l1": 1},
}


# --------------------------- Export configuration (user-tunable) ---------------------------
# Target printed width and DPI
EXPORT_WIDTH_MM: float = 174.0
EXPORT_DPI: int = 600

# OSM labels apparent size boost (0 = default; 1-3 = larger text via higher tile zoom)
OSM_LABELS_ZOOM_BOOST: int = 3

# Legend typography (base pixel sizes; scaled dynamically if needed)
LEGEND_TITLE_FONT_PX: int = 48
LEGEND_TEXT_FONT_PX: int = 40

# Which sections to render. Toggle to include/exclude outputs.
EXPORT_SECTIONS = {
    "openstreetmap_bg": True,
    "smod_l2": True,
    "smod_l1": True,
    "population": True,
    "surface": True,
    "volume": True,
    "boundary": True,
    "openstreetmap_text": False,
    "legends": True,
    "photoshop_script": True,
}

# Optional EE helpers provided as a thin wrapper to keep notebook code cleaner
class STBESAAnalysis:
    """
    Thin wrapper around Earth Engine & geemap operations using a GeoDataFrame row.
    Separated from STBESAService to avoid making EE/geemap a hard dependency.
    """

    def __init__(self, project_id: str):
        self.project_id = project_id
        self._ee_initialized = False

    def initialize_ee(self) -> None:
        import ee
        # Robust initialization: try existing credentials first, then interactive flows.
        try:
            ee.Initialize(project=self.project_id)
            self._ee_initialized = True
            return
        except Exception:
            pass

        # Try standard interactive auth (opens a link/token flow). Works in most notebooks.
        try:
            ee.Authenticate()
            ee.Initialize(project=self.project_id)
            self._ee_initialized = True
            return
        except Exception:
            pass

        # Fallback for hosted notebooks like Colab: explicit notebook auth mode (popup/widget).
        try:
            ee.Authenticate(auth_mode='notebook')
            ee.Initialize(project=self.project_id)
            self._ee_initialized = True
            return
        except Exception as e:
            raise RuntimeError(
                "Earth Engine authentication failed. Please ensure your Google account has access, the 'Earth Engine API' is enabled for your Google Cloud project, and retry. Original error: " + str(e)
            )

    def _ee_getinfo(self, ee_object, max_retries: int = 5, backoff_factor: float = 0.6):
        """
        Robust wrapper around `ee_object.getInfo()` with exponential backoff for transient errors (429 rate limits).
        Returns the Python representation of the EE object or raises the last exception if unrecoverable.
        """
        import time
        last_exc = None
        for attempt in range(max_retries):
            try:
                return ee_object.getInfo()
            except Exception as e:
                last_exc = e
                msg = str(e).lower()
                # Treat common rate-limit/resource errors as retryable
                if any(k in msg for k in ("429", "rate", "rateexceeded", "quota", "too many requests", "resourceexhausted")):
                    sleep = backoff_factor * (2 ** attempt)
                    time.sleep(sleep)
                    continue
                # Non-retryable error: re-raise
                raise
        # Final attempt
        try:
            return ee_object.getInfo()
        except Exception:
            raise RuntimeError(f"EE getInfo failed after {max_retries} attempts: {last_exc}")

    def _ensure_ee(self) -> None:
        if not self._ee_initialized:
            self.initialize_ee()

    def _ee_get_mapid(self, image, vis_params: Dict[str, Any], max_retries: int = 5, backoff_factor: float = 0.6):
        """Preflight map tile creation with retries to avoid blank maps on transient 429s."""
        import ee, time
        last_exc = None
        for attempt in range(max_retries):
            try:
                return ee.Image(image).getMapId(vis_params)
            except Exception as e:
                last_exc = e
                msg = str(e).lower()
                if any(k in msg for k in ("429", "rate", "rateexceeded", "quota", "too many requests", "resourceexhausted")) and attempt < max_retries - 1:
                    time.sleep(backoff_factor * (2 ** attempt))
                    continue
                raise
        # Final attempt
        try:
            return ee.Image(image).getMapId(vis_params)
        except Exception:
            raise RuntimeError(f"EE getMapId failed after {max_retries} attempts: {last_exc}")

    def geemap_map(self, height: str = "650px", backend: str = "folium"):
        # backend: "folium" or "ipyleaflet"
        self._ensure_ee()
        if backend == "ipyleaflet":
            import geemap  # ipyleaflet backend
            return geemap.Map(height=height, ee_initialize=False)
        else:
            import geemap.foliumap as geemap
            return geemap.Map(height=height, ee_initialize=False)

    def geopandas_row_to_ee(self, row: gpd.GeoDataFrame):
        import ee
        import geemap
        self._ensure_ee()
        fc = geemap.geopandas_to_ee(row, geodesic=False)
        feat = ee.Feature(fc.first())
        geom = feat.geometry()
        return feat, geom

    def dynamic_stretch(self, img, band: str, geom, default_max: float, *, mask_zero: bool = True, p_low: float = 5, p_high: float = 99, max_retries: int = 3) -> Tuple[float, float]:
        import ee, time
        if mask_zero:
            img = img.updateMask(img.gt(0))
        # Attempt percentile-based scaling; on rate-limit (429) retry with backoff, then fall back to defaults.
        for attempt in range(max_retries + 1):
            try:
                stats = img.reduceRegion(
                    ee.Reducer.percentile([p_low, p_high]),
                    geometry=geom,
                    scale=100,
                    maxPixels=1e12,
                    tileScale=4,
                )
                # Use the centralized robust getInfo wrapper
                vmin_val = self._ee_getinfo(ee.Number(stats.get(f"{band}_p{int(p_low)}")))
                vmax_val = self._ee_getinfo(ee.Number(stats.get(f"{band}_p{int(p_high)}")))
                vmin = float(vmin_val or 0.0)
                vmax = float(vmax_val or default_max)
                return vmin, vmax
            except Exception as e:
                # Retry on rate-limit-like errors, otherwise re-raise
                msg = str(e).lower()
                if attempt < max_retries and any(k in msg for k in ("429", "rate", "rateexceeded", "quota", "too many requests", "resourceexhausted")):
                    time.sleep(0.6 * (2 ** attempt))
                    continue
                # Fallback if still failing (e.g., persistent 429): use safe defaults
                return 0.0, float(default_max)

    def compute_indicators(self, geom, year: int) -> Dict[str, float]:
        import ee
        self._ensure_ee()
        vol = ee.Image(f"JRC/GHSL/P2023A/GHS_BUILT_V/{year}").select("built_volume_total").clip(geom)
        pop = ee.Image(f"JRC/GHSL/P2023A/GHS_POP/{year}").select("population_count").clip(geom)
        sur = ee.Image(f"JRC/GHSL/P2023A/GHS_BUILT_S/{year}").select("built_surface").clip(geom)

        sum_vol = vol.reduceRegion(ee.Reducer.sum(), geometry=geom, scale=100, maxPixels=1e12, tileScale=4).get('built_volume_total')
        sum_pop = pop.reduceRegion(ee.Reducer.sum(), geometry=geom, scale=100, maxPixels=1e12, tileScale=4).get('population_count')
        sum_sur = sur.reduceRegion(ee.Reducer.sum(), geometry=geom, scale=100, maxPixels=1e12, tileScale=4).get('built_surface')

        sum_vol = float(self._ee_getinfo(ee.Number(sum_vol)) or 0.0)
        sum_pop = float(self._ee_getinfo(ee.Number(sum_pop)) or 0.0)
        sum_sur = float(self._ee_getinfo(ee.Number(sum_sur)) or 0.0)
        
        bvpc = (sum_vol / sum_pop) if sum_pop > 0 else None
        bspc = (sum_sur / sum_pop) if sum_pop > 0 else None
        vol_sur_ratio = (sum_vol / sum_sur) if sum_sur > 0 else None
        
        return {
            "sum_volume_m3": sum_vol,
            "sum_surface_m2": sum_sur,
            "sum_population": sum_pop,
            "bvpc_m3_per_person": (float(bvpc) if bvpc is not None else None),
            "bspc_m2_per_person": (float(bspc) if bspc is not None else None),
            "vol_sur_ratio": (float(vol_sur_ratio) if vol_sur_ratio is not None else None),
        }

    def compute_smod_statistics(self, geom, year: int, level: str = "L1", delay_seconds: float = 0.0) -> Dict[int, Dict[str, float]]:
        """
        Compute statistics per SMOD class (L1 or L2).
        Returns dict mapping class_code -> {buvol_m3, buvol_sur_m2, pop_person, bvpc_m3_per_person, bspc_m2_per_person, vol_sur_ratio}
        
        Args:
            delay_seconds: Delay in seconds between reduceRegion calls to avoid rate limiting
        """
        import ee, time
        self._ensure_ee()
        
        # Load images at appropriate scales (SMOD is 1km, others are 100m)
        smod = ee.Image(f"JRC/GHSL/P2023A/GHS_SMOD/{year}").select("smod_code").clip(geom)
        vol = ee.Image(f"JRC/GHSL/P2023A/GHS_BUILT_V/{year}").select("built_volume_total").clip(geom)
        sur = ee.Image(f"JRC/GHSL/P2023A/GHS_BUILT_S/{year}").select("built_surface").clip(geom)
        pop = ee.Image(f"JRC/GHSL/P2023A/GHS_POP/{year}").select("population_count").clip(geom)
        
        # Resample SMOD from 1km to 100m to match other datasets (nearest neighbor is default, preserves classes)
        smod_100m = smod.reproject(crs='EPSG:4326', scale=100)
        
        # Extract L1 or L2 classes from SMOD code
        if level == "L1":
            # L1 is first digit (divide by 10 and floor)
            class_img = smod_100m.divide(10).floor()
        else:  # L2
            # L2 is the full code
            class_img = smod_100m
        
        # Group by class and sum - need to do each metric separately
        # IMPORTANT: value band must come first, group band second; groupField points to index of group band
        reducer = ee.Reducer.sum().group(groupField=1, groupName='class')
        
        # Reduce for volume (bands order: [value, group])
        vol_stats = self._ee_getinfo(ee.Image([vol, class_img]).reduceRegion(
            reducer=reducer,
            geometry=geom,
            scale=100,
            maxPixels=1e12,
            tileScale=4
        ))
        
        # Delay between reduceRegion calls to avoid rate limiting
        if delay_seconds > 0:
            time.sleep(delay_seconds)
        
        # Reduce for surface
        sur_stats = self._ee_getinfo(ee.Image([sur, class_img]).reduceRegion(
            reducer=reducer,
            geometry=geom,
            scale=100,
            maxPixels=1e12,
            tileScale=4
        ))
        
        # Delay between reduceRegion calls to avoid rate limiting
        if delay_seconds > 0:
            time.sleep(delay_seconds)
        
        # Reduce for population
        pop_stats = self._ee_getinfo(ee.Image([pop, class_img]).reduceRegion(
            reducer=reducer,
            geometry=geom,
            scale=100,
            maxPixels=1e12,
            tileScale=4
        ))
        
        # Parse and combine results
        result = {}
        vol_groups = {int(g['class']): float(g['sum']) for g in vol_stats.get('groups', [])}
        sur_groups = {int(g['class']): float(g['sum']) for g in sur_stats.get('groups', [])}
        pop_groups = {int(g['class']): float(g['sum']) for g in pop_stats.get('groups', [])}
        
        # Get all unique classes
        all_classes = set(vol_groups.keys()) | set(sur_groups.keys()) | set(pop_groups.keys())
        
        for cls in all_classes:
            s_vol = vol_groups.get(cls, 0.0)
            s_sur = sur_groups.get(cls, 0.0)
            s_pop = pop_groups.get(cls, 0.0)
            
            bvpc = (s_vol / s_pop) if s_pop > 0 else None
            bspc = (s_sur / s_pop) if s_pop > 0 else None
            vol_sur = (s_vol / s_sur) if s_sur > 0 else None
            
            result[cls] = {
                'buvol_m3': s_vol,
                'buvol_sur_m2': s_sur,
                'pop_person': s_pop,
                'bvpc_m3_per_person': bvpc,
                'bspc_m2_per_person': bspc,
                'vol_sur_ratio': vol_sur
            }
        
        return result


# --------------------------- Converter from HGM lines ---------------------------
class HGMConverter:
    """
    Build ADM1/ADM2 layers from HGM boundary line shapefiles with robust cleaning:
    - Polygonize il/ilce lines together with country lines
    - Fix invalid geometries, drop empties and tiny slivers
    - De-duplicate nearly-identical polygons
    - Assign districts to provinces via centroid-in-province (fallback to max overlap)
    """

    def __init__(self, il_lines_path: str, ilce_lines_path: str, ulke_lines_path: str, area_threshold_m2: float = 5e5):
        self.il_lines_path = il_lines_path
        self.ilce_lines_path = ilce_lines_path
        self.ulke_lines_path = ulke_lines_path
        self.area_threshold_m2 = area_threshold_m2

    @staticmethod
    def _to4326(g: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        if g.crs is None:
            return g.set_crs(epsg=4326)
        try:
            epsg = g.crs.to_epsg()
        except Exception:
            epsg = None
        if epsg != 4326:
            return g.to_crs(4326)
        return g

    @staticmethod
    def _fix_geom(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        g = gdf.copy()
        g["geometry"] = g["geometry"].buffer(0)
        g = g[~g.geometry.is_empty & g.geometry.is_valid]
        return g

    @staticmethod
    def _drop_slivers(gdf: gpd.GeoDataFrame, area_threshold_m2: float) -> gpd.GeoDataFrame:
        g = gdf.copy()
        areas = g.geometry.to_crs(3857).area
        g = g[areas >= area_threshold_m2]
        return g

    @staticmethod
    def _dedupe_polygons(gdf: gpd.GeoDataFrame, tolerance_m: float = 5.0) -> gpd.GeoDataFrame:
        # Snap geometries slightly then drop duplicates by WKB
        g = gdf.copy()
        g["geometry"] = g.geometry.to_crs(3857).buffer(0).simplify(tolerance_m).to_crs(4326)
        g["_wkb"] = g.geometry.apply(lambda x: x.wkb)
        g = g.drop_duplicates(subset=["_wkb"]).drop(columns=["_wkb"])
        return g

    @staticmethod
    def _buffer_in_meters(geom, meters: float):
        # buffer in a projected CRS to avoid degree-based warnings
        return gpd.GeoSeries([geom], crs="EPSG:4326").to_crs(3857).buffer(meters).to_crs(4326).iloc[0]

    def build(self, province_names: list[str]) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        il   = self._to4326(gpd.read_file(self.il_lines_path))
        ilce = self._to4326(gpd.read_file(self.ilce_lines_path))
        ulke = self._to4326(gpd.read_file(self.ulke_lines_path))

        # ADM1 polygonize
        from shapely.ops import unary_union, polygonize
        prov_lines = list(il.geometry.dropna()) + list(ulke.geometry.dropna())
        prov_polys = list(polygonize(unary_union(prov_lines)))
        gdf_prov   = gpd.GeoDataFrame(geometry=prov_polys, crs="EPSG:4326")
        gdf_prov   = self._fix_geom(gdf_prov)
        gdf_prov   = self._drop_slivers(gdf_prov, self.area_threshold_m2)
        gdf_prov   = self._dedupe_polygons(gdf_prov)

        # pick provinces by geocode centroid
        from geopy.geocoders import Nominatim
        from shapely.geometry import Point
        geoloc = Nominatim(user_agent="bvpc-1975-2030")

        def geocode_point(query: str) -> Point:
            loc = geoloc.geocode(query, country_codes="tr", timeout=10)
            return Point(loc.longitude, loc.latitude)

        def pick_by_point(gdf: gpd.GeoDataFrame, pt: Point) -> gpd.GeoDataFrame:
            sel = gdf[gdf.covers(pt)]
            if sel.empty:
                sel = gdf[gdf.intersects(pt.buffer(0.01))]
            return sel

        selected_adm1 = []
        for name in province_names:
            pt = geocode_point(f"{name}, Türkiye")
            sel = pick_by_point(gdf_prov, pt)
            # Special handling for İstanbul: union faces covering Kadıköy and Fatih seed points
            lname = name.strip().lower()
            if lname in {"istanbul", "i̇stanbul", "ıstanbul"}:
                seed_pts = [Point(29.06, 40.99), Point(28.95, 41.02)]  # Kadıköy, Fatih approx
                parts = []
                for s in seed_pts:
                    ss = pick_by_point(gdf_prov, s)
                    if not ss.empty:
                        parts.append(ss)
                if parts:
                    sel = (pd.concat([sel] + parts).drop_duplicates() if not sel.empty
                           else pd.concat(parts).drop_duplicates())
            if not sel.empty:
                merged = sel.unary_union
                selected_adm1.append(gpd.GeoDataFrame({"NAME_1": [name]}, geometry=[merged], crs="EPSG:4326"))

        gdf_adm1 = gpd.GeoDataFrame(pd.concat(selected_adm1, ignore_index=True), crs="EPSG:4326")
        gdf_adm1["ADM1_ID"] = range(1, len(gdf_adm1)+1)
        gdf_adm1 = gdf_adm1[["ADM1_ID","NAME_1","geometry"]]

        # ADM2 polygonize PER PROVINCE: use only ilçe lines intersecting that province + province boundary
        adm2_rows = []
        for _, prow in gdf_adm1.iterrows():
            prov_name = prow.NAME_1
            prov_geom = prow.geometry
            # subset district lines near province (buffer in meters to avoid geographic-CRS buffer issues)
            near_buf = self._buffer_in_meters(prov_geom, 500)
            lines_sub = ilce[ilce.intersects(near_buf)].geometry.dropna().tolist()
            if not lines_sub:
                lines_sub = ilce[ilce.intersects(prov_geom)].geometry.dropna().tolist()
            # include province boundary as lines to close rings
            lines_all = list(lines_sub)
            try:
                lines_all.append(prov_geom.boundary)
            except Exception:
                pass
            faces = list(polygonize(unary_union(lines_all))) if lines_all else []
            if not faces:
                # fallback: take province geometry as a single face
                faces = [prov_geom]
            gdf_faces = gpd.GeoDataFrame(geometry=faces, crs="EPSG:4326")
            # clip to province
            try:
                clip = gpd.overlay(
                    gdf_faces,
                    gpd.GeoDataFrame(geometry=[prov_geom], crs="EPSG:4326"),
                    how='intersection',
                    keep_geom_type=False,
                )
            except Exception:
                clip = gdf_faces[gdf_faces.centroid.within(prov_geom)]
            clip = self._fix_geom(clip)
            clip = self._drop_slivers(clip, self.area_threshold_m2)
            clip = self._dedupe_polygons(clip)
            if not clip.empty:
                df = clip.copy()
                df["NAME_1"] = prov_name
                df["NAME_2"] = None
                adm2_rows.append(df[["NAME_1","NAME_2","geometry"]])

        if adm2_rows:
            gdf_adm2 = gpd.GeoDataFrame(pd.concat(adm2_rows, ignore_index=True), crs="EPSG:4326")
            gdf_adm2["ADM2_ID"] = range(1, len(gdf_adm2)+1)
            gdf_adm2 = gdf_adm2[["ADM2_ID","NAME_1","NAME_2","geometry"]]
        else:
            gdf_adm2 = gpd.GeoDataFrame(columns=["ADM2_ID","NAME_1","NAME_2","geometry"], crs="EPSG:4326")

        return gdf_adm1, gdf_adm2

    def write_gpkg(self, out_gpkg: str, province_names: list[str]) -> str:
        adm1, adm2 = self.build(province_names)
        adm1.to_file(out_gpkg, layer="ADM1", driver="GPKG")
        adm2.to_file(out_gpkg, layer="ADM2", driver="GPKG")
        return out_gpkg


# --------------------------- OCHA COD-AB loader ---------------------------
class OCHACODLoader:
    """
    Load OCHA COD-AB Turkey admin boundaries (ADM0-ADM2) and standardize to our schema.
    Accepts a ZIP path (preferred) or an extracted directory containing shapefiles.

    Output layers:
      - ADM1: ADM1_ID, NAME_1, NAME1_EN, PCODE_1, geometry
      - ADM2: ADM2_ID, NAME_1, NAME_2, NAME2_EN, PCODE_1, PCODE_2, geometry
    """

    def __init__(self, source_path: str, work_dir: Optional[str] = None):
        self.source_path = source_path
        # Determine a safe working directory. If the caller provided one, use it.
        # If the source_path is a URL, create a temporary directory. Otherwise
        # create a _ocha_tmp next to the provided path.
        if work_dir:
            self.work_dir = work_dir
        else:
            import tempfile, hashlib
            try:
                src = str(self.source_path)
            except Exception:
                src = ""
            if src.lower().startswith(("http://", "https://")):
                # Deterministic cache dir based on URL hash to avoid re-downloading each run
                url_hash = hashlib.md5(src.encode("utf-8")).hexdigest()
                self.work_dir = os.path.join(tempfile.gettempdir(), "ocha_cache", url_hash)
            else:
                # local path: create _ocha_tmp next to the source path
                base_dir = os.path.dirname(os.path.abspath(self.source_path)) if self.source_path else os.getcwd()
                self.work_dir = os.path.join(base_dir, "_ocha_tmp")

        os.makedirs(self.work_dir, exist_ok=True)

    def _ensure_unpacked(self) -> str:
        # If a directory is provided, use it directly
        if os.path.isdir(self.source_path):
            return self.source_path

        # If a URL is provided, download it to the work_dir and unzip
        src = str(self.source_path)
        if src.lower().startswith(("http://", "https://")):
            try:
                # If the work_dir already contains shapefiles, assume it's already unpacked
                existing_shps = glob.glob(os.path.join(self.work_dir, "**", "*.shp"), recursive=True)
                if existing_shps:
                    return self.work_dir

                import urllib.request
                # derive filename from URL
                fname = os.path.basename(src.split("?")[0]) or "download.zip"
                dest_zip = os.path.join(self.work_dir, fname)

                # If the zip was previously downloaded, skip re-download
                if not os.path.exists(dest_zip):
                    urllib.request.urlretrieve(src, dest_zip)

                # If extraction already produced shapefiles, skip extraction
                existing_shps = glob.glob(os.path.join(self.work_dir, "**", "*.shp"), recursive=True)
                if existing_shps:
                    return self.work_dir

                # Extract
                with zipfile.ZipFile(dest_zip) as z:
                    z.extractall(self.work_dir)
                return self.work_dir
            except Exception as e:
                raise RuntimeError(f"Failed to download or extract ZIP from URL: {e}")

        # If a local zip file path is provided, extract it
        if zipfile.is_zipfile(self.source_path):
            # If work_dir already contains shapefiles, assume extracted
            existing_shps = glob.glob(os.path.join(self.work_dir, "**", "*.shp"), recursive=True)
            if existing_shps:
                return self.work_dir
            with zipfile.ZipFile(self.source_path) as z:
                z.extractall(self.work_dir)
            return self.work_dir

        raise ValueError("source_path must be a directory, a ZIP file path, or a downloadable ZIP URL")

    def _find_shapefiles(self, root: str) -> list[str]:
        return glob.glob(os.path.join(root, "**", "*.shp"), recursive=True)

    @staticmethod
    def _has_cols(g: gpd.GeoDataFrame, required: set[str]) -> bool:
        cols = {c.upper() for c in g.columns}
        return required.issubset(cols)

    def _load_layer(self, shps: list[str], level: int) -> gpd.GeoDataFrame:
        for p in shps:
            try:
                g = gpd.read_file(p)
            except Exception:
                continue
            if level == 1 and self._has_cols(g, {"ADM1_PCODE", "ADM1_TR"}):
                return g
            if level == 2 and self._has_cols(g, {"ADM2_PCODE", "ADM2_TR", "ADM1_PCODE"}):
                return g
        raise RuntimeError(f"ADM{level} shapefile not found in provided source")

    def to_standard_gdfs(self) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        root = self._ensure_unpacked()
        shps = self._find_shapefiles(root)
        adm1 = self._load_layer(shps, 1)
        adm2 = self._load_layer(shps, 2)

        if adm1.crs is None or getattr(adm1.crs, 'to_epsg', lambda: None)() != 4326:
            adm1 = adm1.to_crs(4326)
        if adm2.crs is None or getattr(adm2.crs, 'to_epsg', lambda: None)() != 4326:
            adm2 = adm2.to_crs(4326)

        # Rename to standard columns, tolerate minor naming variants
        def rnm(df: gpd.GeoDataFrame, mapping: Dict[str, str]) -> gpd.GeoDataFrame:
            upper_map = {k.upper(): v for k, v in mapping.items()}
            cols_upper = {c.upper(): c for c in df.columns}
            rename_dict = {}
            for need_upper, out_name in upper_map.items():
                if need_upper in cols_upper:
                    rename_dict[cols_upper[need_upper]] = out_name
            return df.rename(columns=rename_dict)

        adm1 = rnm(adm1, {"ADM1_TR": "NAME_1", "ADM1_EN": "NAME1_EN", "ADM1_PCODE": "PCODE_1"})
        adm2 = rnm(adm2, {"ADM1_TR": "NAME_1", "ADM2_TR": "NAME_2", "ADM2_EN": "NAME2_EN", "ADM1_PCODE": "PCODE_1", "ADM2_PCODE": "PCODE_2"})

        adm1_out = adm1[[c for c in ["NAME_1", "NAME1_EN", "PCODE_1", "geometry"] if c in adm1.columns]].copy()
        adm2_out = adm2[[c for c in ["NAME_1", "NAME_2", "NAME2_EN", "PCODE_1", "PCODE_2", "geometry"] if c in adm2.columns]].copy()

        adm1_out = adm1_out.reset_index(drop=True)
        adm1_out["ADM1_ID"] = range(1, len(adm1_out) + 1)
        adm1_out = adm1_out[["ADM1_ID", "NAME_1", "NAME1_EN", "PCODE_1", "geometry"]]

        adm2_out = adm2_out.reset_index(drop=True)
        adm2_out["ADM2_ID"] = range(1, len(adm2_out) + 1)
        adm2_out = adm2_out[["ADM2_ID", "NAME_1", "NAME_2", "NAME2_EN", "PCODE_1", "PCODE_2", "geometry"]]

        return adm1_out, adm2_out

    def write_gpkg(self, out_gpkg: str) -> str:
        adm1_out, adm2_out = self.to_standard_gdfs()
        adm1_out.to_file(out_gpkg, layer="ADM1", driver="GPKG")
        adm2_out.to_file(out_gpkg, layer="ADM2", driver="GPKG")
        return out_gpkg

# --------------------------- Notebook UI helper ---------------------------
def build_picker_ui(service: STBESAService, project_id: str, year: int = 2025, ee_delay_seconds: float = 0.5):
    """
    Builds an interactive province/district picker with a Run button.
    - Handles missing district names by showing positional labels (e.g., "İlçe #01").
    - Uses STBESAAnalysis for EE/map logic.
    
    Args:
        service: STBESAService instance
        project_id: Google Earth Engine project ID
        year: Default year for analysis
        ee_delay_seconds: Delay in seconds between Earth Engine API calls to avoid rate limiting.
                         Default 0.5 seconds. Increase if you get "Too many concurrent aggregations" errors.
                         Recommended values: 0.5-2.0 seconds depending on your quota.
    
    Returns a dict with ui and inner widgets for further control if needed.
    """
    import ipywidgets as W
    import pandas as pd

    service._ensure_loaded()
    gdf = service._gdf_adm2

    def province_list() -> list:
        vals = gdf["NAME_1"].dropna().astype(str).unique().tolist()
        return sorted(vals, key=lambda x: service._norm(x).lower())

    def district_options_for(prov_name: str):
        sub = gdf[gdf["NAME_1"] == prov_name].reset_index(drop=True)
        options = []
        for i in range(len(sub)):
            name2 = sub.loc[i, "NAME_2"]
            if pd.notna(name2) and str(name2).strip():
                # Add index number to district name for clarity
                label = f"{str(name2)} ({i})"
            else:
                label = f"{prov_name} - İlçe #{i+1:02d}"
            options.append((label, i))  # value = row position within province subset
        return options

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
        'ilce_idx': 'District index within the province, or NULL if all districts selected',
        'yil': 'Year of observation (1975-2030 in 5-year intervals)',
        'buvol_m3': 'Total building volume in cubic meters',
        'buvol_sur_m2': 'Total building surface area in square meters',
        'pop_person': 'Total population count',
        'bvpc_m3_per_person': 'Building Volume Per Capita: average building volume per person (m³/person)',
        'bspc_m2_per_person': 'Building Surface Per Capita: average building surface per person (m²/person)',
        'vol_sur_ratio': 'Ratio of building volume to building surface (indicates average building height)',
        'smod_l1_code': 'SMOD Level 1 classification code (1=Rural, 2=Urban Cluster, 3=Urban Centre)',
        'smod_l1_name': 'SMOD Level 1 class name',
        'smod_l2_code': 'SMOD Level 2 classification code (10-30 range, more detailed urban/rural classification)',
        'smod_l2_name': 'SMOD Level 2 class name (detailed settlement type)',
        'smod_l1_parent': 'Parent SMOD L1 code for this L2 class'
    }
    
    # Track all interactive widgets for busy state management
    _interactive_widgets = []
    def _register_widget(w):
        _interactive_widgets.append(w)
        return w
    
    _disabled_state = {}
    def set_busy(is_busy: bool, msg: str = "Processing...") -> None:
        """Enable/disable all interactive widgets and show/hide busy overlay"""
        if is_busy:
            # Store current disabled state and disable all widgets
            for w in _interactive_widgets:
                _disabled_state[w] = getattr(w, 'disabled', False)
                try:
                    w.disabled = True
                except Exception:
                    pass
            busy_overlay.value = _busy_html(msg)
            busy_overlay.layout.display = 'block'
        else:
            # Restore previous disabled state
            for w in _interactive_widgets:
                try:
                    w.disabled = _disabled_state.get(w, False)
                except Exception:
                    pass
            _disabled_state.clear()
            busy_overlay.value = ""
            busy_overlay.layout.display = 'none'

    provinces = province_list()
    prov_default = provinces[0] if provinces else None
    prov_dd = _register_widget(W.Dropdown(options=provinces, value=prov_default, layout=W.Layout(width='171px')))

    dist_opts = district_options_for(prov_dd.value) if prov_dd.value else []
    # Prep SelectMultiple options with an ALL entry as default
    sel_opts = [("ALL", "ALL")] + dist_opts
    dist_sel = _register_widget(W.SelectMultiple(options=sel_opts, value=("ALL",), layout=W.Layout(width='320px', height='143px'), rows=5))

    years_all = list(range(1975, 2035, 5))
    year_dd = _register_widget(W.Dropdown(options=years_all, value=year, layout=W.Layout(width='100px')))

    run_btn = _register_widget(W.Button(description='Run Analysis', button_style='primary', layout=W.Layout(margin='12px 0px', width='372px', height='32px')))
    export_btn = _register_widget(W.Button(description='Export XLSX', button_style='success', layout=W.Layout(margin='0px 6px 0px 0px', width='116px', height='28px')))
    export_btn.disabled = True  # Disabled until data is loaded
    export_plots_btn = _register_widget(W.Button(description='Export Plots', button_style='warning', layout=W.Layout(margin='0px 6px 0px 6px', width='116px', height='28px')))
    export_plots_btn.disabled = True
    # High-quality layer export (per-layer)
    save_layers_btn = _register_widget(W.Button(description='Save Layers', button_style='info', layout=W.Layout(margin='0px 0px 0px 6px', width='116px', height='28px')))
    save_layers_btn.disabled = True
    
    # Visualization controls
    auto_scale = _register_widget(W.Checkbox(value=True, description='Auto scale', indent=False))
    vol_min = _register_widget(W.FloatText(value=0, placeholder='0', description='', layout=W.Layout(width='60px')))
    vol_max = _register_widget(W.FloatText(value=0, placeholder='0', description='', layout=W.Layout(width='60px')))
    sur_min = _register_widget(W.FloatText(value=0, placeholder='0', description='', layout=W.Layout(width='60px')))
    sur_max = _register_widget(W.FloatText(value=0, placeholder='0', description='', layout=W.Layout(width='60px')))
    pop_min = _register_widget(W.FloatText(value=0, placeholder='0', description='', layout=W.Layout(width='60px')))
    pop_max = _register_widget(W.FloatText(value=0, placeholder='0', description='', layout=W.Layout(width='60px')))
    for w in (vol_min, vol_max, sur_min, sur_max, pop_min, pop_max):
        w.disabled = True
    apply_scale_btn = _register_widget(W.Button(description='Apply Color Scale', layout=W.Layout(margin='12px 0px', width='280px', height='32px')))
    apply_scale_btn.disabled = True

    # Performance controls
    auto_once = _register_widget(W.Checkbox(value=True, description='Auto once', indent=False))
    fast_switch = _register_widget(W.Checkbox(value=False, description='Fast (not recommended)', indent=False))
    simplify_m = _register_widget(W.IntText(value=50, description='', layout=W.Layout(width='80px')))

    ui_msg  = W.HTML()
    out_map = W.Output()
    out_plot_l1 = W.Output(layout=W.Layout(overflow='visible'))
    out_plot_l2 = W.Output(layout=W.Layout(overflow='visible'))
    out_tbl = W.Output()

    # Busy overlay for user feedback during time-consuming operations
    busy_overlay = W.HTML(value="", layout=W.Layout(display='none'))
    
    def _busy_html(msg: str) -> str:
        return f"""
        <div style='position:fixed; top:50%; left:50%; transform:translate(-50%,-50%); z-index:9999; padding:20px 30px; background:white; border:2px solid #5cb85c; border-radius:8px; box-shadow: 0 4px 20px rgba(0,0,0,0.2);'>
          <div style='display:flex; align-items:center; gap:12px;'>
            <div style="width:20px; height:20px; border:3px solid #ddd; border-top-color:#5cb85c; border-radius:50%; animation: spin 0.8s linear infinite;"></div>
            <span style='font-size:14px; font-weight:500; color:#333;'>{msg}</span>
          </div>
          <style>
            @keyframes spin {{ from {{ transform: rotate(0deg);}} to {{ transform: rotate(360deg);}} }}
          </style>
        </div>
        """
    
    # Auto-clear timer for success messages
    _clear_timer = {"timer": None}
    def schedule_clear_message(delay_seconds: float = 3.0):
        """Schedule the message to clear after a delay using asyncio"""
        import asyncio
        async def clear_after_delay():
            await asyncio.sleep(delay_seconds)
            ui_msg.value = ""
        
        # Cancel previous timer if exists
        if _clear_timer["timer"] is not None:
            try:
                _clear_timer["timer"].cancel()
            except:
                pass
        
        # Schedule new timer
        try:
            loop = asyncio.get_event_loop()
            _clear_timer["timer"] = loop.create_task(clear_after_delay())
        except:
            # Fallback: just clear immediately if asyncio fails
            pass

    # Header with logo on RIGHT side - fixed width to align with controls
    import os
    import base64
    import sys
    
    # Try multiple approaches to find the logo file
    logo_path = None
    
    # Method 1: Try __file__ if available (works in regular Python scripts)
    try:
        if '__file__' in globals():
            logo_path = os.path.join(os.path.dirname(__file__), 'img', 'logo.png')
    except:
        pass
    
    # Method 2: Try current working directory (works in Colab/Jupyter)
    if not logo_path or not os.path.exists(logo_path):
        logo_path = os.path.join(os.getcwd(), 'img', 'logo.png')
    
    # Method 3: Try relative to current directory
    if not os.path.exists(logo_path):
        logo_path = 'img/logo.png'
    
    # Method 4: Try in the same directory as the script (for Colab)
    if not os.path.exists(logo_path):
        logo_path = 'logo.png'
    
    # Method 5: Try to download from URL if local file not found
    if not os.path.exists(logo_path):
        try:
            import urllib.request
            import tempfile
            
            # Download logo from URL
            url = 'https://csstr.org/wp-content/uploads/2023/07/Logo.png'
            with urllib.request.urlopen(url) as response:
                logo_data = response.read()
            
            # Create a temporary file to store the downloaded logo
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
                temp_file.write(logo_data)
                logo_path = temp_file.name
                
        except Exception as e:
            # If download fails, logo_path will remain the non-existent local path
            pass
    
    if os.path.exists(logo_path):
        with open(logo_path, 'rb') as f:
            logo_data = f.read()
        logo_b64 = base64.b64encode(logo_data).decode()
        logo_html = f'<img src="data:image/png;base64,{logo_b64}" style="height:70px; margin-left:20px;">'
    else:
        logo_html = '<div style="width:100px; height:70px; background:#ddd; margin-left:20px;"></div>'
    
    # Calculate total width: left_col (400px) + gap (20px) + legend (280px) + gap (12px) + perf (280px) = 992px
    title_html = f'''
    <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:15px; padding:15px 20px; background:#f3f3f3; border:1px solid #ddd; width:940px;">
        <div style="flex:1;">
            <h2 style="margin:0 0 8px 0; color:#5cb85c; font-family:Arial,sans-serif; font-size:22px; font-weight:600;">
                Spatio-Temporal Built Environment & Settlement Analysis Platform
            </h2>
            <p style="margin:0; font-size:12px; color:#666; line-height:1.5; font-family:Arial,sans-serif;">
                A comprehensive GIS-based analytical tool for analyzing building volume, surface area, and population dynamics across Turkey (1975-2030).
                Features multi-temporal analysis with 5-year intervals, SMOD land-use classification (L1/L2), interactive visualization with customizable color scales,
                multi-district selection, and professional Excel export with complete data dictionaries. Powered by Google Earth Engine and JRC Global Human Settlement Layer.
            </p>
        </div>
        {logo_html}
    </div>
    '''
    header = W.HTML(title_html)

    # Left column: Province and Year in same row
    prov_year_row = W.HBox([
        W.Label('Province', layout=W.Layout(width='60px')), 
        prov_dd,
        W.Label('Year', layout=W.Layout(width='35px', margin='0 0 0 10px')), 
        year_dd
    ], layout=W.Layout(align_items='center', width='400px'))
    
    # District row spanning full width
    dist_row = W.HBox([
        W.Label('District', layout=W.Layout(width='60px')), 
        dist_sel
    ], layout=W.Layout(align_items='flex-start', width='400px'))
    
    # Run button and Export/Save buttons row
    run_export_row = W.VBox([
        W.HBox([run_btn], layout=W.Layout(justify_content='center', width='400px')),
        W.HBox([export_btn, export_plots_btn, save_layers_btn], layout=W.Layout(justify_content='center', width='400px'))
    ], layout=W.Layout(gap='6px'))
    
    left_col = W.VBox([prov_year_row, dist_row, run_export_row], layout=W.Layout(gap='6px'))

    # Legend Settings box - fixed width, no overflow
    legend_title = W.HTML('<div style="font-weight:600; margin-bottom:8px; font-size:13px; color:#333;">Legend Settings</div>')
    legend_content = W.VBox([
        legend_title,
        auto_scale,
        W.HBox([
            W.Label('Vol min', layout=W.Layout(width='50px', flex='0 0 auto')), 
            vol_min, 
            W.Label('Vol max', layout=W.Layout(width='50px', margin='0 0 0 5px', flex='0 0 auto')), 
            vol_max
        ], layout=W.Layout(align_items='center', overflow='hidden')),
        W.HBox([
            W.Label('Sur min', layout=W.Layout(width='50px', flex='0 0 auto')), 
            sur_min, 
            W.Label('Sur max', layout=W.Layout(width='50px', margin='0 0 0 5px', flex='0 0 auto')), 
            sur_max
        ], layout=W.Layout(align_items='center', overflow='hidden')),
        W.HBox([
            W.Label('Pop min', layout=W.Layout(width='50px', flex='0 0 auto')), 
            pop_min, 
            W.Label('Pop max', layout=W.Layout(width='50px', margin='0 0 0 5px', flex='0 0 auto')), 
            pop_max
        ], layout=W.Layout(align_items='center', overflow='hidden')),
        W.HBox([apply_scale_btn], layout=W.Layout(justify_content='center'))
    ], layout=W.Layout(gap='8px', margin='0px 12px', padding='10px 12px', border='1px solid #ccc', width='280px', overflow='hidden'))

    # Performance Settings box - same width as legend
    perf_title = W.HTML('<div style="font-weight:600; margin-bottom:8px; font-size:13px; color:#333;">Performance Settings</div>')
    perf_content = W.VBox([
        perf_title,
        auto_once,
        fast_switch,
        W.HBox([
            W.Label('Simplify', layout=W.Layout(width='60px', flex='0 0 auto')), 
            simplify_m
        ], layout=W.Layout(align_items='center', overflow='hidden'))
    ], layout=W.Layout(gap='8px', padding='10px 12px', border='1px solid #ccc', width='280px', overflow='hidden'))

    # Settings boxes side by side with margin between them
    settings_row = W.HBox([legend_content, perf_content], layout=W.Layout(gap='12px', overflow='hidden'))

    # Main controls layout
    controls = W.HBox([left_col, settings_row], layout=W.Layout(gap='20px', align_items='flex-start', width='992px'))

    ui = W.VBox([header, controls, ui_msg, out_map, out_plot_l1, out_plot_l2, out_tbl, busy_overlay], layout=W.Layout(padding='8px', gap='10px'))

    def on_province_change(change):
        if change['name'] == 'value':
            opts = district_options_for(change['new'])
            sel_opts = [("ALL", "ALL")] + opts
            dist_sel.options = sel_opts
            dist_sel.value = ("ALL",)

    prov_dd.observe(on_province_change, names='value')

    def dyn_vis(img, band, geom, default_max):
        analyzer = STBESAAnalysis(project_id)
        analyzer._ensure_ee()
        return analyzer.dynamic_stretch(img, band, geom, default_max)

    # Keep current bounds when updating via slider (so manual zooms are not lost)
    last_bounds = None  # [[s,y],[n,x]]
    run_source = {"value": "button"}  # or 'slider'
    cache = {
        "vol": {}, "sur": {}, "pop": {}, "smod": {},
        "vis": {"vol": {}, "sur": {}, "pop": {}}, 
        "df": None, "df_smod_l1": None, "df_smod_l2": None,
        "feat": None, "geom": None
    }
    map_widget = {"m": None}

    def _pad_bounds(bnds, pad_frac=0.10):
        # pad bounds by fraction to have ~85-90% occupancy
        (s, w), (n, e) = [list(bnds[0]), list(bnds[1])]
        dy = (n - s)
        dx = (e - w)
        s -= dy * pad_frac
        n += dy * pad_frac
        w -= dx * pad_frac
        e += dx * pad_frac
        return [[s, w], [n, e]]
    
    def _render_plots_l1(cur_year: int):
        """Render 2x3 time-series plots for L1 classes + total."""
        if cache["df"] is None or cache["df"].empty:
            return
        if cache.get("df_smod_l1") is None or cache["df_smod_l1"].empty:
            return
        
        with out_plot_l1:
            out_plot_l1.clear_output(wait=True)
            import matplotlib.pyplot as plt
            import matplotlib as mpl
            import numpy as np
            
            total_df = cache["df"][ ["yil", "buvol_m3", "buvol_sur_m2", "pop_person", "bvpc_m3_per_person", "bspc_m2_per_person", "vol_sur_ratio"] ].sort_values("yil").reset_index(drop=True)
            years = total_df["yil"].values

            l1 = cache["df_smod_l1"][ ["yil", "smod_l1_code", "buvol_m3", "buvol_sur_m2", "pop_person", "bvpc_m3_per_person", "bspc_m2_per_person", "vol_sur_ratio"] ].copy()
            piv = {metric: l1.pivot_table(index="yil", columns="smod_l1_code", values=metric, aggfunc="first").reindex(years).fillna(np.nan) for metric in ["buvol_m3","buvol_sur_m2","pop_person","bvpc_m3_per_person","bspc_m2_per_person","vol_sur_ratio"]}

            mpl.rcParams.update({'figure.dpi': 300,'savefig.dpi': 300,'axes.titlesize': 7,'axes.labelsize': 6,'xtick.labelsize': 5,'ytick.labelsize': 5,'legend.fontsize': 5,'axes.grid': True,'grid.linestyle': ':','grid.alpha': 0.5,})
            fig, axes = plt.subplots(3, 2, figsize=(6.85, 5.33), constrained_layout=False)
            axes = axes.ravel()
            fig.patch.set_alpha(0.0)

            total_color = "#1f77b4"
            l1_colors = {1: SMOD_L1_CLASSES[1]["color"], 2: SMOD_L1_CLASSES[2]["color"], 3: SMOD_L1_CLASSES[3]["color"]}
            series = [("buvol_m3","Building Volume (m³)"),("buvol_sur_m2","Building Surface (m²)"),("pop_person","Population (people)"),("bvpc_m3_per_person","BVPC (m³/person)"),("bspc_m2_per_person","BSPC (m²/person)"),("vol_sur_ratio","Volume/Surface Ratio")]

            for i,(key,label) in enumerate(series):
                ax = axes[i]
                ax.plot(years, total_df[key].values, color=total_color, linewidth=1.6, label="Total")
                ax.scatter(years, total_df[key].values, s=6, color=total_color, edgecolor=total_color, facecolor="white", zorder=3)
                for cls in [1,2,3]:
                    if cls in piv[key].columns:
                        ax.plot(years, piv[key][cls].values, color=l1_colors[cls], linewidth=1.0, label=str(cls))
                ax.set_title(label); ax.set_xlabel("Year"); ax.set_ylabel(label)
                ax.margins(x=0.02); ax.set_xticks(years); ax.tick_params(axis='x', rotation=0)
                try:
                    ax.axvline(cur_year, color="#d62728", linestyle="--", linewidth=1.0, alpha=0.8)
                except Exception:
                    pass
            fig.suptitle("L1 Classes + Total", fontsize=7, fontweight='semibold')
            # Increase vertical spacing between subplots and place legend with proper spacing
            fig.subplots_adjust(bottom=0.10, hspace=0.50)
            from matplotlib.lines import Line2D
            handles = [Line2D([0], [0], color=total_color, lw=3, label='Total')]
            labels = ['Total']
            for cls in [1, 2, 3]:
                handles.append(Line2D([0], [0], color=l1_colors[cls], lw=3))
                labels.append({1: 'Rural', 2: 'Urban Cluster', 3: 'Urban Centre'}[cls])
            # Place legend below subplots with proper spacing to avoid overlap
            fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.03), ncol=4, prop={'size':5}, handlelength=2, columnspacing=1.0, frameon=True, fancybox=True, shadow=True, edgecolor='black', facecolor='white', framealpha=0.9)
            from IPython.display import display
            display(fig)
            plt.close(fig)

    def _render_plots_l2(cur_year: int):
        """Render 2x3 time-series plots for L2 classes (8) + total."""
        if cache["df"] is None or cache["df"].empty:
            return
        if cache.get("df_smod_l2") is None or cache["df_smod_l2"].empty:
            return

        with out_plot_l2:
            out_plot_l2.clear_output(wait=True)
            import matplotlib.pyplot as plt
            import matplotlib as mpl
            import numpy as np

            total_df = cache["df"][ ["yil", "buvol_m3", "buvol_sur_m2", "pop_person", "bvpc_m3_per_person", "bspc_m2_per_person", "vol_sur_ratio"] ].sort_values("yil").reset_index(drop=True)
            years = total_df["yil"].values

            l2 = cache["df_smod_l2"][ ["yil", "smod_l2_code", "buvol_m3", "buvol_sur_m2", "pop_person", "bvpc_m3_per_person", "bspc_m2_per_person", "vol_sur_ratio"] ].copy()
            piv = {metric: l2.pivot_table(index="yil", columns="smod_l2_code", values=metric, aggfunc="first").reindex(years).fillna(np.nan) for metric in ["buvol_m3","buvol_sur_m2","pop_person","bvpc_m3_per_person","bspc_m2_per_person","vol_sur_ratio"]}

            mpl.rcParams.update({'figure.dpi': 300,'savefig.dpi': 300,'axes.titlesize': 7,'axes.labelsize': 6,'xtick.labelsize': 5,'ytick.labelsize': 5,'legend.fontsize': 5,'axes.grid': True,'grid.linestyle': ':','grid.alpha': 0.5,})
            fig, axes = plt.subplots(3, 2, figsize=(6.85, 5.33), constrained_layout=False)
            axes = axes.ravel()
            fig.patch.set_alpha(0.0)

            total_color = "#1f77b4"
            l2_color_map = {10: '#7AB6F5', 11: '#CDF57A', 12: '#ABCD66', 13: '#375623', 21: '#FFFF00', 22: '#A87000', 23: '#732600', 30: '#FF0000'}
            series = [("buvol_m3","Building Volume (m³)"),("buvol_sur_m2","Building Surface (m²)"),("pop_person","Population (people)"),("bvpc_m3_per_person","BVPC (m³/person)"),("bspc_m2_per_person","BSPC (m²/person)"),("vol_sur_ratio","Volume/Surface Ratio")]

            for i,(key,label) in enumerate(series):
                ax = axes[i]
                ax.plot(years, total_df[key].values, color=total_color, linewidth=1.6, label="Total")
                ax.scatter(years, total_df[key].values, s=6, color=total_color, edgecolor=total_color, facecolor="white", zorder=3)
                for cls in [10,11,12,13,21,22,23,30]:
                    if cls in piv[key].columns:
                        ax.plot(years, piv[key][cls].values, color=l2_color_map[cls], linewidth=0.9, label=str(cls))
                ax.set_title(label); ax.set_xlabel("Year"); ax.set_ylabel(label)
                ax.margins(x=0.02); ax.set_xticks(years); ax.tick_params(axis='x', rotation=0)
                try:
                    ax.axvline(cur_year, color="#d62728", linestyle="--", linewidth=1.0, alpha=0.8)
                except Exception:
                    pass
            fig.suptitle("L2 Classes + Total", fontsize=7, fontweight='semibold')
            # Increase vertical spacing between subplots and place legend with proper spacing
            fig.subplots_adjust(bottom=0.11, hspace=0.50)
            from matplotlib.lines import Line2D
            handles = [Line2D([0], [0], color=total_color, lw=3, label='Total')]
            labels = ['Total']
            l2_order = [10,11,12,13,21,22,23,30]
            l2_label_map = {10:'10 Water/No data',11:'11 Very low density rural',12:'12 Low density rural',13:'13 Rural cluster',21:'21 Suburban/peri-urban',22:'22 Semi-dense urban',23:'23 Dense urban cluster',30:'30 Urban centre'}
            for cls in l2_order:
                handles.append(Line2D([0], [0], color=l2_color_map[cls], lw=3))
                labels.append(l2_label_map[cls])
            # Place legend below subplots with proper spacing to avoid overlap
            fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.03), ncol=4, prop={'size':5}, handlelength=2, columnspacing=1.0, frameon=True, fancybox=True, shadow=True, edgecolor='black', facecolor='white', framealpha=0.9)
            from IPython.display import display
            display(fig)
            plt.close(fig)

    def _render_for_year(cur_year: int):
        if cache["geom"] is None or cur_year not in cache["vol"]:
            return  # not computed yet
        from IPython.display import display
        # Choose backend based on fast_switch
        if fast_switch.value:
            import geemap as geemap  # ipyleaflet backend
        else:
            import geemap.foliumap as geemap
        import ee
        analyzer = STBESAAnalysis(project_id)
        analyzer.initialize_ee()

        with out_map:
            out_map.clear_output(wait=True)
            m = analyzer.geemap_map(height="783px", backend=("ipyleaflet" if fast_switch.value else "folium"))
            map_widget["m"] = m
            if last_bounds is not None:
                try:
                    m.fit_bounds(last_bounds)
                except Exception:
                    pass

            TURBO = ['#30123b','#4145ab','#2e7de0','#1db6d7','#27d39b','#7be151','#c0e620','#f9e14b','#fca72c','#f96814','#a31e1e']
            vmin_vol, vmax_vol = cache["vis"]["vol"][cur_year]
            vmin_sur, vmax_sur = cache["vis"]["sur"][cur_year]
            vmin_pop, vmax_pop = cache["vis"]["pop"][cur_year]
            vol = cache["vol"][cur_year]
            sur = cache["sur"][cur_year]
            pop = cache["pop"][cur_year]
            
            # Helper to add a layer with preflight retries to avoid blank maps after 429s
            def _add_layer_with_retry(image, vis, name, shown=True):
                try:
                    # Preflight getMapId with retries; if it passes, adding the layer should succeed
                    analyzer._ee_get_mapid(image, vis if isinstance(vis, dict) else {})
                except Exception:
                    # If preflight fails due to rate limits, bubble up to outer try/except for re-render attempts
                    pass
                try:
                    m.addLayer(image, vis, name, shown)
                except Exception:
                    # One more attempt after a short delay
                    import time
                    time.sleep(0.8)
                    m.addLayer(image, vis, name, shown)

            # Add layers in z-order (bottom to top): SMOD L2, SMOD L1, Population, Building surface, Building volume, Boundary
            if cur_year in cache["smod"]:
                smod = cache["smod"][cur_year]
                smod_mask = smod.neq(0)
                
                # SMOD L2 layer (bottom layer)
                l2_palette_map = {
                    10: '#7AB6F5',  # Water/no data - blue
                    11: '#CDF57A',  # Very low density rural - light green
                    12: '#ABCD66',  # Low density rural - medium green
                    13: '#375623',  # Rural cluster - dark green
                    21: '#FFFF00',  # Suburban - yellow
                    22: '#A87000',  # Semi-dense urban - brown
                    23: '#732600',  # Dense urban cluster - dark brown
                    30: '#FF0000',  # Urban centre - red
                }
                smod_l2_vis = smod.updateMask(smod_mask).remap([10, 11, 12, 13, 21, 22, 23, 30], [0, 1, 2, 3, 4, 5, 6, 7]).visualize(min=0, max=7, palette=list(l2_palette_map.values()))
                _add_layer_with_retry(smod_l2_vis, {}, f"SMOD L2 {cur_year}", shown=True)
                
                # SMOD L1 layer (above L2)
                smod_l1 = smod.divide(10).floor()
                l1_palette = [SMOD_L1_CLASSES[1]["color"], SMOD_L1_CLASSES[2]["color"], SMOD_L1_CLASSES[3]["color"]]
                smod_l1_vis = smod_l1.updateMask(smod_mask).visualize(min=1, max=3, palette=l1_palette)
                _add_layer_with_retry(smod_l1_vis, {}, f"SMOD L1 {cur_year}", shown=False)

            # Population layer (above SMOD)
            _add_layer_with_retry(pop.updateMask(pop.gt(0)), {"min": vmin_pop, "max": vmax_pop, "palette": TURBO}, f"Population {cur_year}")
            
            # Building surface layer (above Population)
            _add_layer_with_retry(sur.updateMask(sur.gt(0)), {"min": vmin_sur, "max": vmax_sur, "palette": TURBO}, f"Building surface {cur_year}")
            
            # Building volume layer (above Building surface)
            _add_layer_with_retry(vol.updateMask(vol.gt(0)), {"min": vmin_vol, "max": vmax_vol, "palette": TURBO}, f"Building volume {cur_year}")

            # Boundary layer (top layer)
            outline = ee.Image().byte().paint(ee.FeatureCollection(cache["feat"]), 1, 2).visualize(min=0, max=1, palette=['000000'])
            _add_layer_with_retry(outline, {}, "Boundary")
            # Add colorbars - both on bottom right, compact, no background
            if fast_switch.value:
                # ipyleaflet: use custom HTML widget for legends
                try:
                    from ipywidgets import HTML
                    from ipyleaflet import WidgetControl
                    # Create compact gradient HTML for colorbar without background
                    def make_legend_html(colors, vmin, vmax, label):
                        gradient = ', '.join([f'{c} {i/(len(colors)-1)*100:.0f}%' for i, c in enumerate(colors)])
                        return f"""
                        <div style=\"padding: 4px 6px; background: rgba(255,255,255,0.85); border: 1px solid rgba(0,0,0,0.2); border-radius: 2px;\">\n                            <div style=\"font-weight: 600; margin-bottom: 2px; font-size: 9px; color: #333;\">{label}</div>\n                            <div style=\"background: linear-gradient(to right, {gradient}); height: 10px; width: 120px; border: 1px solid rgba(0,0,0,0.3);\"></div>\n                            <div style=\"display: flex; justify-content: space-between; font-size: 8px; margin-top: 1px; color: #333;\">\n                                <span>{vmin:.1f}</span>\n                                <span>{vmax:.1f}</span>\n                            </div>\n                        </div>\n                        """
                    # Categorical legends for SMOD
                    def make_categorical_legend(title, items, show_title=True):
                        rows = ''.join([f"<div style='display:flex;align-items:center;margin:1px 0'><span style='display:inline-block;width:12px;height:9px;background:{c};border:1px solid #666;margin-right:5px'></span><span style='font-size:9px;color:#333'>{lbl}</span></div>" for lbl, c in items])
                        header = f"<div style='font-weight:600; margin-bottom:2px; font-size:9px; color:#333;'>{title}</div>" if show_title else ""
                        return f"""
                        <div style=\"padding:4px 6px; background: rgba(255,255,255,0.85); border:1px solid rgba(0,0,0,0.2); border-radius:2px;\">\n                          {header}\n                          {rows}\n                        </div>\n                        """

                    l1_items = [('RURAL', SMOD_L1_CLASSES[1]['color']), ('URBAN CLUSTER', SMOD_L1_CLASSES[2]['color']), ('URBAN CENTRE', SMOD_L1_CLASSES[3]['color'])]
                    l2_items = [
                        ('10 Water/No data', '#7AB6F5'),
                        ('11 Very low density rural', '#CDF57A'),
                        ('12 Low density rural', '#ABCD66'),
                        ('13 Rural cluster', '#375623'),
                        ('21 Suburban/peri-urban', '#FFFF00'),
                        ('22 Semi-dense urban', '#A87000'),
                        ('23 Dense urban cluster', '#732600'),
                        ('30 Urban centre', '#FF0000')
                    ]

                    # Two separate controls: metrics (right), SMOD accordions (left)
                    metrics_html = f"""
                    <div style='display:flex; flex-direction:column; gap:6px;'>
                      {make_legend_html(TURBO, vmin_vol, vmax_vol, 'Building volume (m³)')}
                      {make_legend_html(TURBO, vmin_sur, vmax_sur, 'Building surface (m²)')}
                      {make_legend_html(TURBO, vmin_pop, vmax_pop, 'Population (people)')}
                    </div>
                    """
                    smod_html = f"""
                    <div style='display:flex; flex-direction:column; gap:6px;'>
                      <details>
                        <summary style='cursor:pointer; font-weight:600; font-size:10px; color:#333;'>SMOD L1</summary>
                        {make_categorical_legend('SMOD L1', l1_items, show_title=False)}
                      </details>
                      <details>
                        <summary style='cursor:pointer; font-weight:600; font-size:10px; color:#333;'>SMOD L2</summary>
                        {make_categorical_legend('SMOD L2', l2_items, show_title=False)}
                      </details>
                    </div>
                    """
                    legend_metrics = HTML(metrics_html)
                    legend_smod = HTML(smod_html)
                    metrics_control = WidgetControl(widget=legend_metrics, position='bottomright')
                    smod_control = WidgetControl(widget=legend_smod, position='bottomleft')
                    m.add_control(metrics_control)
                    m.add_control(smod_control)
                except Exception as e:
                    pass
            else:
                # Folium: inject compact HTML legends and a Save Image button so folium view matches ipyleaflet
                try:
                    import folium
                    # Local helper to build gradient legend HTML (placed absolutely)
                    def make_legend_html(colors, vmin, vmax, label, right=10, bottom=10, width=200):
                        gradient = ', '.join([f'{c} {i/(len(colors)-1)*100:.0f}%' for i, c in enumerate(colors)])
                        return f"""
                        <div style="position:absolute; bottom:{bottom}px; right:{right}px; z-index:1000;">
                          <div style=\"padding:4px 6px; background: rgba(255,255,255,0.85); border:1px solid rgba(0,0,0,0.2); border-radius:2px;\">
                            <div style=\"font-weight:600; margin-bottom:2px; font-size:9px; color:#333;\">{label}</div>
                            <div style=\"background: linear-gradient(to right, {gradient}); height:10px; width:{width}px; border:1px solid rgba(0,0,0,0.3);\"></div>
                            <div style=\"display:flex; justify-content:space-between; font-size:8px; margin-top:1px; color:#333;\">
                              <span>{vmin:.1f}</span>
                              <span>{vmax:.1f}</span>
                            </div>
                          </div>
                        </div>
                        """

                    def make_categorical_legend(title, items, left=10, bottom=None, top=None):
                        rows = ''.join([f"<div style='display:flex;align-items:center;margin:2px 0'><span style='display:inline-block;width:14px;height:10px;background:{c};border:1px solid #666;margin-right:6px'></span><span style='font-size:9px;color:#333'>{lbl}</span></div>" for lbl, c in items])
                        if top is not None:
                            pos = f"top:{top}px; left:{left}px;"
                        else:
                            pos = f"bottom:{bottom}px; left:{left}px;"
                        return f"""
                        <div style="position:absolute; {pos} z-index:1000;">
                          <div style=\"padding:6px 8px; background: rgba(255,255,255,0.9); border:1px solid rgba(0,0,0,0.18); border-radius:2px;\">
                            <div style=\"font-weight:600; margin-bottom:4px; font-size:10px; color:#333;\">{title}</div>
                            {rows}
                          </div>
                        </div>
                        """

                    # build metrics HTML (stacked compact colorbars)
                    metrics_html = (
                        make_legend_html(TURBO, vmin_vol, vmax_vol, 'Building volume (m³)', right=12, bottom=122.5, width=240)
                        + make_legend_html(TURBO, vmin_sur, vmax_sur, 'Building surface (m²)', right=12, bottom=65, width=240)
                        + make_legend_html(TURBO, vmin_pop, vmax_pop, 'Population (people)', right=12, bottom=10, width=240)
                    )

                    # categorical legends for SMOD L1/L2
                    l1_items = [('RURAL', SMOD_L1_CLASSES[1]['color']), ('URBAN CLUSTER', SMOD_L1_CLASSES[2]['color']), ('URBAN CENTRE', SMOD_L1_CLASSES[3]['color'])]
                    l2_items = [
                        ('10 Water/No data', '#7AB6F5'),
                        ('11 Very low density rural', '#CDF57A'),
                        ('12 Low density rural', '#ABCD66'),
                        ('13 Rural cluster', '#375623'),
                        ('21 Suburban/peri-urban', '#FFFF00'),
                        ('22 Semi-dense urban', '#A87000'),
                        ('23 Dense urban cluster', '#732600'),
                        ('30 Urban centre', '#FF0000')
                    ]
                    smod_html = make_categorical_legend('SMOD L1', l1_items, left=12, bottom=170) + make_categorical_legend('SMOD L2', l2_items, left=12, bottom=10)

                    # Inject legends into folium map root
                    try:
                        m.get_root().html.add_child(folium.Element(metrics_html))
                        m.get_root().html.add_child(folium.Element(smod_html))
                    except Exception:
                        # fallback: attach as raw HTML
                        from branca.element import Element
                        m.get_root().html.add_child(Element(metrics_html))
                        m.get_root().html.add_child(Element(smod_html))

                    # Add Save Image button using html2canvas
                    try:
                        script_tag = '<script src="https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js"></script>'
                        save_js = """
                        <script>
                        function saveMapImage() {
                            var container = document.getElementsByClassName('folium-map')[0] || document.getElementById('map');
                            if(!container) container = document.body;
                            html2canvas(container, {useCORS: true}).then(function(canvas) {
                                var link = document.createElement('a');
                                link.href = canvas.toDataURL('image/png');
                                link.download = 'ST-BESA_map.png';
                                link.click();
                            });
                        }
                        </script>
                        """
                        button_html = '<div style="position:absolute; top:10px; right:70px; z-index:1000;"><button onclick="saveMapImage()" style="background:#fff;border:1px solid #333;padding:6px 8px;border-radius:3px;cursor:pointer;font-size:12px;">Capture Frame</button></div>'
                        m.get_root().html.add_child(folium.Element(script_tag + save_js + button_html))
                    except Exception:
                        pass
                except Exception:
                    pass
            try:
                m.addLayerControl()
            except Exception:
                pass
            # For ipyleaflet, ensure proper display; for folium, display as widget
            if fast_switch.value:
                display(m)  # ipyleaflet Map widget
            else:
                try:
                    display(m)  # Folium Map widget
                except Exception:
                    display(m.to_html())

        if cache["df"] is not None:
            with out_tbl:
                out_tbl.clear_output(wait=True)
                from IPython.display import display
                # UI table: show only base metrics with user-friendly column names
                display_df = cache["df"].copy()
                display_df.rename(columns=COLUMN_NAMES, inplace=True)
                display(display_df)

    def run(_=None):
        from IPython.display import display
        import geemap.foliumap as geemap
        import ee

        analyzer = STBESAAnalysis(project_id)
        analyzer.initialize_ee()

        prov = prov_dd.value
        sel_vals = tuple(dist_sel.value) if isinstance(dist_sel.value, (list, tuple)) else (dist_sel.value,)

        # Determine selected indices: ALL or specific indices (values are ints)
        sub = service._gdf_adm2[service._gdf_adm2["NAME_1"] == prov].reset_index(drop=True)
        if "ALL" in sel_vals:
            selected_idx = list(sub.index)
        else:
            # values come as indices into sub
            selected_idx = [int(v) for v in sel_vals]

        if not selected_idx:
            ui_msg.value = f"<b style='color:red'>No districts selected for {prov}</b>"
            return

        combined = sub.loc[selected_idx].reset_index(drop=True)
        # Optional geometry simplification on the client before sending to EE (reduce vertex count)
        try:
            tol_m = max(0, int(simplify_m.value))
        except Exception:
            tol_m = 0
        if tol_m > 0:
            combined["geometry"] = combined.geometry.to_crs(3857).simplify(tol_m, preserve_topology=True).to_crs(4326)
        # Geopandas -> EE FeatureCollection
        fc = geemap.geopandas_to_ee(combined, geodesic=False)
        # use the collection's unioned geometry so multiple polygons render/summarize correctly
        feat = ee.Feature(ee.FeatureCollection(fc).first())
        geom = ee.FeatureCollection(fc).geometry()

        # Cache geometry and feature
        cache["feat"], cache["geom"] = fc, geom
        # Reset map so next render builds a fresh view for the new selection
        map_widget["m"] = None

        # Set target bounds once per run (based on selected district), to be reused by slider re-renders
        nonlocal last_bounds
        if run_source["value"] == "button" or last_bounds is None:
            # Use combined selection bounds (covers multi-select or ALL)
            minx, miny, maxx, maxy = combined.total_bounds
            last_bounds = _pad_bounds([[miny, minx],[maxy, maxx]], 0.07)

        # Precompute and cache: images, visualization ranges, and yearly stats (batched)
        set_busy(True, "Loading satellite imagery data...")
        ui_msg.value = "<b style='color:blue'>Loading satellite imagery data from Earth Engine...</b>"
        cache["vol"].clear(); cache["sur"].clear(); cache["pop"].clear(); cache["smod"].clear()
        cache["vis"]["vol"].clear(); cache["vis"]["sur"].clear(); cache["vis"]["pop"].clear()
        vol_imgs = [ee.Image(f"JRC/GHSL/P2023A/GHS_BUILT_V/{yy}").select("built_volume_total").clip(geom) for yy in years_all]
        sur_imgs = [ee.Image(f"JRC/GHSL/P2023A/GHS_BUILT_S/{yy}").select("built_surface").clip(geom) for yy in years_all]
        pop_imgs = [ee.Image(f"JRC/GHSL/P2023A/GHS_POP/{yy}").select("population_count").clip(geom) for yy in years_all]
        smod_imgs = [ee.Image(f"JRC/GHSL/P2023A/GHS_SMOD/{yy}").select("smod_code").clip(geom) for yy in years_all]
        # Cache images
        for yy, v, s, p, sm in zip(years_all, vol_imgs, sur_imgs, pop_imgs, smod_imgs):
            cache["vol"][yy] = v
            cache["sur"][yy] = s
            cache["pop"][yy] = p
            cache["smod"][yy] = sm
        # Sequential sums with delays to avoid rate limiting (replaces batch toList().map() approach)
        ui_msg.value = "<b style='color:blue'>Computing statistics for all years (1975-2030)...</b>"
        import time
        reducers = ee.Reducer.sum()
        sum_vol_list = []
        sum_sur_list = []
        sum_pop_list = []
        # Process each year sequentially with delays to avoid "Too many concurrent aggregations" error
        for i, yy in enumerate(years_all):
            vol_img = vol_imgs[i]
            sur_img = sur_imgs[i]
            pop_img = pop_imgs[i]
            # Compute sums sequentially with delays
            sum_vol = analyzer._ee_getinfo(vol_img.reduceRegion(reducer=reducers, geometry=geom, scale=100, maxPixels=1e12, tileScale=4).get('built_volume_total'))
            if ee_delay_seconds > 0 and i < len(years_all) - 1:
                time.sleep(ee_delay_seconds)
            sum_sur = analyzer._ee_getinfo(sur_img.reduceRegion(reducer=reducers, geometry=geom, scale=100, maxPixels=1e12, tileScale=4).get('built_surface'))
            if ee_delay_seconds > 0 and i < len(years_all) - 1:
                time.sleep(ee_delay_seconds)
            sum_pop = analyzer._ee_getinfo(pop_img.reduceRegion(reducer=reducers, geometry=geom, scale=100, maxPixels=1e12, tileScale=4).get('population_count'))
            if ee_delay_seconds > 0 and i < len(years_all) - 1:
                time.sleep(ee_delay_seconds)
            sum_vol_list.append(sum_vol)
            sum_sur_list.append(sum_sur)
            sum_pop_list.append(sum_pop)
        sums = {'vol': sum_vol_list, 'sur': sum_sur_list, 'pop': sum_pop_list}
        # Viz: auto_once computes percentiles once at current year; else per-year
        if auto_scale.value:
            ui_msg.value = "<b style='color:blue'>Computing color scale ranges...</b>"
            if auto_once.value:
                cur = int(year_dd.value)
                idx = years_all.index(cur)
                vmin_v, vmax_v = analyzer.dynamic_stretch(vol_imgs[idx], "built_volume_total", geom, 80000, mask_zero=True, p_low=5, p_high=99)
                vmin_s, vmax_s = analyzer.dynamic_stretch(sur_imgs[idx], "built_surface", geom, 20000, mask_zero=True, p_low=5, p_high=99)
                vmin_p, vmax_p = analyzer.dynamic_stretch(pop_imgs[idx], "population_count",  geom, 200,   mask_zero=True, p_low=5, p_high=99)
                for yy in years_all:
                    cache["vis"]["vol"][yy] = (vmin_v, vmax_v)
                    cache["vis"]["sur"][yy] = (vmin_s, vmax_s)
                    cache["vis"]["pop"][yy] = (vmin_p, vmax_p)
            else:
                import time
                for i, (yy, v, s, p) in enumerate(zip(years_all, vol_imgs, sur_imgs, pop_imgs)):
                    vmin_v, vmax_v = analyzer.dynamic_stretch(v, "built_volume_total", geom, 80000, mask_zero=True, p_low=5, p_high=99)
                    if ee_delay_seconds > 0 and i < len(years_all) - 1:
                        time.sleep(ee_delay_seconds)
                    vmin_s, vmax_s = analyzer.dynamic_stretch(s, "built_surface", geom, 20000, mask_zero=True, p_low=5, p_high=99)
                    if ee_delay_seconds > 0 and i < len(years_all) - 1:
                        time.sleep(ee_delay_seconds)
                    vmin_p, vmax_p = analyzer.dynamic_stretch(p, "population_count",  geom, 200,   mask_zero=True, p_low=5, p_high=99)
                    if ee_delay_seconds > 0 and i < len(years_all) - 1:
                        time.sleep(ee_delay_seconds)
                    cache["vis"]["vol"][yy] = (vmin_v, vmax_v)
                    cache["vis"]["sur"][yy] = (vmin_s, vmax_s)
                    cache["vis"]["pop"][yy] = (vmin_p, vmax_p)
        else:
            vmin_v = float(vol_min.value if vol_min.value is not None else 0.0)
            vmax_v = float(vol_max.value if vol_max.value is not None else 80000.0)
            vmin_s = float(sur_min.value if sur_min.value is not None else 0.0)
            vmax_s = float(sur_max.value if sur_max.value is not None else 20000.0)
            vmin_p = float(pop_min.value if pop_min.value is not None else 0.0)
            vmax_p = float(pop_max.value if pop_max.value is not None else 200.0)
            for yy in years_all:
                cache["vis"]["vol"][yy] = (vmin_v, vmax_v)
                cache["vis"]["sur"][yy] = (vmin_s, vmax_s)
                cache["vis"]["pop"][yy] = (vmin_p, vmax_p)
        # Build DataFrame from batched sums
        rows = []
        for i, yy in enumerate(years_all):
            s_vol = float(sums['vol'][i] or 0.0)
            s_sur = float(sums['sur'][i] or 0.0)
            s_pop = float(sums['pop'][i] or 0.0)
            bvpc = (s_vol / s_pop) if s_pop > 0 else None
            bspc = (s_sur / s_pop) if s_pop > 0 else None
            vol_sur_ratio = (s_vol / s_sur) if s_sur > 0 else None
            rows.append({
                'il': prov, 
                'ilce_idx': None if "ALL" in sel_vals else selected_idx, 
                'yil': yy, 
                'buvol_m3': s_vol, 
                'buvol_sur_m2': s_sur,
                'pop_person': s_pop,
                'bvpc_m3_per_person': bvpc,
                'bspc_m2_per_person': bspc,
                'vol_sur_ratio': vol_sur_ratio
            })
        cache["df"] = pd.DataFrame(rows)

        # Compute SMOD statistics for ALL years (needed for L1/L2 time-series plots)
        ui_msg.value = "<b style='color:blue'>Computing SMOD land-use classifications (all years)...</b>"
        import time
        smod_l1_rows = []
        smod_l2_rows = []
        for i, yy in enumerate(years_all):
            stats_l1 = analyzer.compute_smod_statistics(geom, yy, level="L1", delay_seconds=ee_delay_seconds)
            for cls_code, stats in stats_l1.items():
                if cls_code in SMOD_L1_CLASSES:
                    smod_l1_rows.append({
                        'il': prov,
                        'ilce_idx': None if "ALL" in sel_vals else selected_idx,
                        'yil': yy,
                        'smod_l1_code': cls_code,
                        'smod_l1_name': SMOD_L1_CLASSES[cls_code]['name'],
                        'buvol_m3': stats['buvol_m3'],
                        'buvol_sur_m2': stats['buvol_sur_m2'],
                        'pop_person': stats['pop_person'],
                        'bvpc_m3_per_person': stats['bvpc_m3_per_person'],
                        'bspc_m2_per_person': stats['bspc_m2_per_person'],
                        'vol_sur_ratio': stats['vol_sur_ratio']
                    })
            # Delay between L1 and L2 computation for the same year
            if ee_delay_seconds > 0:
                time.sleep(ee_delay_seconds)
            stats_l2 = analyzer.compute_smod_statistics(geom, yy, level="L2", delay_seconds=ee_delay_seconds)
            for cls_code, stats in stats_l2.items():
                if cls_code in SMOD_L2_CLASSES:
                    smod_l2_rows.append({
                        'il': prov,
                        'ilce_idx': None if "ALL" in sel_vals else selected_idx,
                        'yil': yy,
                        'smod_l2_code': cls_code,
                        'smod_l2_name': SMOD_L2_CLASSES[cls_code]['name'],
                        'smod_l1_parent': SMOD_L2_CLASSES[cls_code]['l1'],
                        'buvol_m3': stats['buvol_m3'],
                        'buvol_sur_m2': stats['buvol_sur_m2'],
                        'pop_person': stats['pop_person'],
                        'bvpc_m3_per_person': stats['bvpc_m3_per_person'],
                        'bspc_m2_per_person': stats['bspc_m2_per_person'],
                        'vol_sur_ratio': stats['vol_sur_ratio']
                    })
            # Delay between years (except for the last one)
            if ee_delay_seconds > 0 and i < len(years_all) - 1:
                time.sleep(ee_delay_seconds)
        cache["df_smod_l1"] = pd.DataFrame(smod_l1_rows)
        cache["df_smod_l2"] = pd.DataFrame(smod_l2_rows)

        # Finally render for current slider year using cache
        ui_msg.value = "<b style='color:blue'>Rendering map and plots...</b>"
        cur = int(year_dd.value)
        _render_for_year(cur)
        _render_plots_l1(cur)
        _render_plots_l2(cur)
        ui_msg.value = "<b style='color:green'>✓ Analysis complete!</b>"
        
        # Enable export buttons now that we have data (after set_busy is done)
        # We need to update the saved state so set_busy(False) doesn't re-disable them
        _disabled_state[export_btn] = False
        _disabled_state[export_plots_btn] = False
        _disabled_state[save_layers_btn] = False
        
        set_busy(False)
        
        # Ensure export buttons are enabled
        export_btn.disabled = False
        export_plots_btn.disabled = False
        save_layers_btn.disabled = False
        
        # Auto-clear success message after 5 seconds
        schedule_clear_message(5.0)

    def on_run_click(_):
        run_source["value"] = "button"
        try:
            run()
        except Exception as e:
            ui_msg.value = f"<b style='color:red'>Error: {str(e)}</b>"
            set_busy(False)

    run_btn.on_click(on_run_click)

    def on_export_click(_):
        """Export all data to Excel file"""
        if cache["df"] is None:
            ui_msg.value = "<b style='color:red'>No data to export. Please run analysis first.</b>"
            return
        
        import datetime
        from pathlib import Path
        
        # Generate filename with timestamp and district info
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        prov = prov_dd.value.replace(' ', '_')
        
        # Get district name(s) for filename
        sel_vals = tuple(dist_sel.value) if isinstance(dist_sel.value, (list, tuple)) else (dist_sel.value,)
        sub = service._gdf_adm2[service._gdf_adm2["NAME_1"] == prov_dd.value].reset_index(drop=True)
        total_districts = len(sub)
        
        if "ALL" in sel_vals:
            dist_name = "ALL"
        else:
            selected_idx = [int(v) for v in sel_vals if v != "ALL"]
            
            # Check if all districts are selected (even without "ALL" option)
            if len(selected_idx) == total_districts:
                dist_name = "ALL"
            elif len(selected_idx) == 1:
                # Single district: use name
                dist_name = str(sub.loc[selected_idx[0], "NAME_2"]).replace(' ', '_').replace('/', '-')
            elif len(selected_idx) <= 5:
                # 2-5 districts: use comma-separated names
                names = [str(sub.loc[idx, "NAME_2"]).replace(' ', '_').replace('/', '-') for idx in sorted(selected_idx)]
                dist_name = ','.join(names)
            else:
                # More than 5 districts: use indices only
                dist_name = ','.join([str(idx) for idx in sorted(selected_idx)])
        
        # Short, consistent prefix for the project exports
        filename = f"ST-BESA_{prov}_{dist_name}_{timestamp}.xlsx"
        output_path = Path.cwd() / filename
        
        try:
            set_busy(True, "Exporting to Excel...")
            ui_msg.value = "<b style='color:blue'>Preparing Excel export...</b>"
            
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                # Sheet 1: Overall statistics
                ui_msg.value = "<b style='color:blue'>Writing overall statistics...</b>"
                cache["df"].to_excel(writer, sheet_name='Overall_Statistics', index=False)
                
                # Sheet 2: SMOD L1 statistics
                # Compute full multi-year L1 if not computed
                if cache["df_smod_l1"] is None or cache["df_smod_l1"].empty or len(cache["df_smod_l1"]["yil"].unique()) == 1:
                    ui_msg.value = "<b style='color:blue'>Computing SMOD L1 statistics for all years...</b>"
                    import time
                    smod_l1_rows = []
                    for i, yy in enumerate(years_all):
                        stats_l1 = STBESAAnalysis(project_id).compute_smod_statistics(cache["geom"], yy, level="L1", delay_seconds=ee_delay_seconds)
                        for cls_code, stats in stats_l1.items():
                            if cls_code in SMOD_L1_CLASSES:
                                smod_l1_rows.append({
                                    'il': prov_dd.value,
                                    'yil': yy,
                                    'smod_l1_code': cls_code,
                                    'smod_l1_name': SMOD_L1_CLASSES[cls_code]['name'],
                                    **stats
                                })
                        if ee_delay_seconds > 0 and i < len(years_all) - 1:
                            time.sleep(ee_delay_seconds)
                    cache["df_smod_l1"] = pd.DataFrame(smod_l1_rows)
                ui_msg.value = "<b style='color:blue'>Writing SMOD L1 statistics...</b>"
                # Sort L1 data by code and year for easy charting
                df_l1_sorted = cache["df_smod_l1"].sort_values(by=['smod_l1_code', 'yil']).reset_index(drop=True)
                df_l1_sorted.to_excel(writer, sheet_name='SMOD_L1_Statistics', index=False)
                
                # Sheet 3: SMOD L2 statistics
                if cache["df_smod_l2"] is None or cache["df_smod_l2"].empty or len(cache["df_smod_l2"]["yil"].unique()) == 1:
                    ui_msg.value = "<b style='color:blue'>Computing SMOD L2 statistics for all years...</b>"
                    import time
                    smod_l2_rows = []
                    for i, yy in enumerate(years_all):
                        stats_l2 = STBESAAnalysis(project_id).compute_smod_statistics(cache["geom"], yy, level="L2", delay_seconds=ee_delay_seconds)
                        for cls_code, stats in stats_l2.items():
                            if cls_code in SMOD_L2_CLASSES:
                                smod_l2_rows.append({
                                    'il': prov_dd.value,
                                    'yil': yy,
                                    'smod_l2_code': cls_code,
                                    'smod_l2_name': SMOD_L2_CLASSES[cls_code]['name'],
                                    'smod_l1_parent': SMOD_L2_CLASSES[cls_code]['l1'],
                                    **stats
                                })
                        if ee_delay_seconds > 0 and i < len(years_all) - 1:
                            time.sleep(ee_delay_seconds)
                    cache["df_smod_l2"] = pd.DataFrame(smod_l2_rows)
                ui_msg.value = "<b style='color:blue'>Writing SMOD L2 statistics...</b>"
                # Sort L2 data by code and year for easy charting
                df_l2_sorted = cache["df_smod_l2"].sort_values(by=['smod_l2_code', 'yil']).reset_index(drop=True)
                df_l2_sorted.to_excel(writer, sheet_name='SMOD_L2_Statistics', index=False)
                
                # Sheet 4: Metadata
                ui_msg.value = "<b style='color:blue'>Writing metadata...</b>"
                metadata = pd.DataFrame({
                    'Parameter': ['Province', 'District(s)', 'Export Date', 'Year Range'],
                    'Value': [
                        prov_dd.value,
                        dist_name.replace('_', ' '),
                        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        f"{min(years_all)} - {max(years_all)}"
                    ]
                })
                metadata.to_excel(writer, sheet_name='Metadata', index=False)
                
                # Sheet 5: Data Dictionary
                ui_msg.value = "<b style='color:blue'>Writing data dictionary...</b>"
                data_dict_rows = []
                for prog_name, friendly_name in COLUMN_NAMES.items():
                    data_dict_rows.append({
                        'Programmatic Name': prog_name,
                        'User-Friendly Name': friendly_name,
                        'Description': COLUMN_DESCRIPTIONS.get(prog_name, '')
                    })
                data_dict = pd.DataFrame(data_dict_rows)
                data_dict.to_excel(writer, sheet_name='Data_Dictionary', index=False)
            
            ui_msg.value = f"<b style='color:green'>✓ Data exported successfully to: {filename}</b>"
            # Auto-clear success message after 5 seconds
            schedule_clear_message(5.0)
        except Exception as e:
            ui_msg.value = f"<b style='color:red'>Export failed: {str(e)}</b>"
        finally:
            set_busy(False)

    export_btn.on_click(on_export_click)

    def on_save_layers_click(_):
        """Save separate transparent overlay PNGs (data only) and a separate OSM base.
        - Data overlays are fetched from EE at native 100 m scale, then upscaled client-side
          to 174 mm width at 600 DPI using nearest-neighbor (pixel edges preserved).
        - OSM base is fetched at full target resolution as its own PNG to stack under overlays.
        """
        if cache["geom"] is None or not cache["vol"]:
            ui_msg.value = "<b style='color:red'>No data to save. Please run analysis first.</b>"
            return
        import ee, math
        analyzer = STBESAAnalysis(project_id)
        analyzer.initialize_ee()
        geom = cache["geom"]
        cur = int(year_dd.value)
        # Visualization parameters for current year
        TURBO = ['#30123b','#4145ab','#2e7de0','#1db6d7','#27d39b','#7be151','#c0e620','#f9e14b','#fca72c','#f96814','#a31e1e']
        vmin_vol, vmax_vol = cache["vis"]["vol"][cur]
        vmin_sur, vmax_sur = cache["vis"]["sur"][cur]
        vmin_pop, vmax_pop = cache["vis"]["pop"][cur]
        vol = cache["vol"][cur]
        sur = cache["sur"][cur]
        pop = cache["pop"][cur]
        smod = cache["smod"].get(cur)
        # Build visualize images (mask zeros to keep background transparent)
        vol_vis = vol.updateMask(vol.gt(0)).visualize(min=vmin_vol, max=vmax_vol, palette=TURBO)
        sur_vis = sur.updateMask(sur.gt(0)).visualize(min=vmin_sur, max=vmax_sur, palette=TURBO)
        pop_vis = pop.updateMask(pop.gt(0)).visualize(min=vmin_pop, max=vmax_pop, palette=TURBO)
        layers = [(f"building_volume_{cur}", vol_vis), (f"building_surface_{cur}", sur_vis), (f"population_{cur}", pop_vis)]
        if smod is not None:
            smod_mask = smod.neq(0)
            l2_palette_map = {10: '#7AB6F5', 11: '#CDF57A', 12: '#ABCD66', 13: '#375623', 21: '#FFFF00', 22: '#A87000', 23: '#732600', 30: '#FF0000'}
            smod_l2_vis = smod.updateMask(smod_mask).remap([10,11,12,13,21,22,23,30],[0,1,2,3,4,5,6,7]).visualize(min=0, max=7, palette=list(l2_palette_map.values()))
            smod_l1 = smod.divide(10).floor()
            l1_palette = [SMOD_L1_CLASSES[1]["color"], SMOD_L1_CLASSES[2]["color"], SMOD_L1_CLASSES[3]["color"]]
            smod_l1_vis = smod_l1.updateMask(smod_mask).visualize(min=1, max=3, palette=l1_palette)
            layers.extend([(f"smod_l2_{cur}", smod_l2_vis), (f"smod_l1_{cur}", smod_l1_vis)])
        outline = ee.Image().byte().paint(ee.FeatureCollection(cache["feat"]), 1, 2).visualize(min=0, max=1, palette=['000000'])
        layers.append((f"boundary_{cur}", outline))
        # File naming
        import datetime
        from pathlib import Path
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        prov = str(prov_dd.value).replace(' ', '_')
        out_dir = Path.cwd() / f"STBESA_EXPORT_{prov}_{timestamp}"
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        set_busy(True, "Saving layers (native 100 m, 600 DPI, separate overlays/base)...")
        ui_msg.value = "<b style='color:blue'>Rendering overlays at native 100 m; upscaling to 174 mm @ 600 DPI...</b>"
        saved = []
        # Target output width in pixels for 174 mm at 600 DPI
        width_mm = 174.0
        width_in = width_mm / 25.4
        dpi = 600
        out_width_px = int(round(width_in * dpi))
        # Compute geographic bbox for export
        try:
            bbox = analyzer._ee_getinfo(ee.Geometry(geom).bounds().coordinates())
            ring = bbox[0] if isinstance(bbox, list) and bbox else []
            xs = [float(pt[0]) for pt in ring]
            ys = [float(pt[1]) for pt in ring]
            w = min(xs); e = max(xs); s = min(ys); n = max(ys)
        except Exception:
            try:
                (s, w), (n, e) = last_bounds  # type: ignore
            except Exception:
                raise RuntimeError("Could not determine geometry bounds for export")

        # Helper: approximate width of bbox in meters using Web Mercator (for OSM zoom)
        def _approx_bbox_width_m(lon_w: float, lon_e: float, lat_mid: float) -> float:
            radius = 6378137.0
            dlon = math.radians(lon_e - lon_w)
            return abs(radius * dlon * math.cos(math.radians(lat_mid)))

        # Helper: fetch an XYZ tile layer covering bbox, matching output size (high-res)
        def _fetch_xyz_layer(width_px: int, height_px: int, tile_url_tpl: str, zoom_boost: int = 0):
            import urllib.request, io
            from PIL import Image

            def lonlat_to_pixel(lon: float, lat: float, z: int) -> tuple[float, float]:
                lat_rad = math.radians(lat)
                ntiles = 2 ** z
                x = (lon + 180.0) / 360.0 * ntiles * 256.0
                y = (1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * ntiles * 256.0
                return x, y

            # Pick zoom based on desired meters-per-pixel
            lat_c = (s + n) / 2.0
            meters_per_pixel_target = max(0.1, _approx_bbox_width_m(w, e, lat_c) / max(1, width_px))
            initial_res = 156543.03392804097
            z_float = math.log2(max(1e-6, initial_res * math.cos(math.radians(lat_c)) / meters_per_pixel_target))
            z = int(max(0, min(19, round(z_float) + int(zoom_boost))))

            min_px_x, min_px_y = lonlat_to_pixel(w, n, z)
            max_px_x, max_px_y = lonlat_to_pixel(e, s, z)
            x0 = int(math.floor(min_px_x / 256.0))
            y0 = int(math.floor(min_px_y / 256.0))
            x1 = int(math.floor((max_px_x - 1) / 256.0))
            y1 = int(math.floor((max_px_y - 1) / 256.0))
            cols = x1 - x0 + 1
            rows = y1 - y0 + 1
            mosaic = Image.new('RGBA', (cols * 256, rows * 256), (0, 0, 0, 0))
            opener = urllib.request.build_opener()
            opener.addheaders = [('User-Agent', 'st-besa-export/1.0')]
            urllib.request.install_opener(opener)
            for ix in range(cols):
                for iy in range(rows):
                    tx = x0 + ix
                    ty = y0 + iy
                    url = tile_url_tpl.format(z=z, x=tx, y=ty)
                    try:
                        with urllib.request.urlopen(url) as resp:
                            tile_data = resp.read()
                        tile_img = Image.open(io.BytesIO(tile_data)).convert('RGBA')
                        mosaic.paste(tile_img, (ix * 256, iy * 256))
                    except Exception:
                        pass
            crop_left = int(round(min_px_x - x0 * 256.0))
            crop_top = int(round(min_px_y - y0 * 256.0))
            crop_right = crop_left + int(round(max_px_x - min_px_x))
            crop_bottom = crop_top + int(round(max_px_y - min_px_y))
            crop = mosaic.crop((crop_left, crop_top, crop_right, crop_bottom))
            return crop.resize((width_px, height_px), resample=Image.BILINEAR)

        import urllib.request, io
        from PIL import Image
        # Compute 3857 bounds and native pixel size at 100 m so all products share identical aspect ratio
        try:
            bounds_3857 = analyzer._ee_getinfo(ee.Geometry.Rectangle([w, s, e, n], proj='EPSG:4326', geodesic=False).transform('EPSG:3857', 1).coordinates())
            ringm = bounds_3857[0]
            xs_m = [float(pt[0]) for pt in ringm]
            ys_m = [float(pt[1]) for pt in ringm]
            wm = min(xs_m); em = max(xs_m); sm = min(ys_m); nm = max(ys_m)
        except Exception:
            raise RuntimeError("Failed to compute Web Mercator bounds for export")

        width_m = max(1.0, em - wm)
        height_m = max(1.0, nm - sm)
        native_w_px = int(max(1, math.ceil(width_m / 100.0)))
        native_h_px = int(max(1, math.ceil(height_m / 100.0)))

        # Derive final output height preserving metric aspect ratio
        target_height_px = int(max(1, round(out_width_px * (native_h_px / native_w_px))))

        # 1) Save each data layer as a separate transparent overlay at native resolution (reproject to EPSG:3857)
        overlays_written = []
        order = [
            ("02", "smod-l2", EXPORT_SECTIONS.get("smod_l2", True), next((img for name,img in layers if name.startswith("smod_l2_")), None)),
            ("03", "smod-l1", EXPORT_SECTIONS.get("smod_l1", True), next((img for name,img in layers if name.startswith("smod_l1_")), None)),
            ("04", "population", EXPORT_SECTIONS.get("population", True), next((img for name,img in layers if name.startswith("population_")), None)),
            ("05", "surface", EXPORT_SECTIONS.get("surface", True), next((img for name,img in layers if name.startswith("building_surface_")), None)),
            ("06", "volume", EXPORT_SECTIONS.get("volume", True), next((img for name,img in layers if name.startswith("building_volume_")), None)),
            ("07", "boundary", EXPORT_SECTIONS.get("boundary", True), next((img for name,img in layers if name.startswith("boundary_")), None)),
        ]
        for prefix, short_name, enabled, limg in order:
            if not enabled:
                continue
            if limg is None:
                continue
            img_merc = ee.Image(limg).reproject(crs='EPSG:3857', scale=100)
            region_merc = ee.Geometry.Rectangle([wm, sm, em, nm], proj='EPSG:3857', geodesic=False)
            # Request native pixel width to avoid EE heavy renders; height follows automatically
            params = {"region": region_merc, "dimensions": native_w_px, "format": "png", "maxPixels": 1e13}
            url = ee.Image(img_merc).getThumbURL(params)
            fname = f"{prefix}_ST-BESA_{prov}_{short_name}_{timestamp}.png"
            fpath = out_dir / fname
            # Download native-resolution overlay
            with urllib.request.urlopen(url) as resp:
                data = resp.read()
            overlay = Image.open(io.BytesIO(data)).convert('RGBA')
            # Upscale to target 174 mm width at 600 DPI with NEAREST to preserve pixel edges
            up_overlay = overlay.resize((out_width_px, target_height_px), resample=Image.NEAREST)
            # Save transparent overlay with 600 DPI
            try:
                up_overlay.save(str(fpath), format='PNG', dpi=(dpi, dpi))
            except Exception:
                up_overlay.save(str(fpath), format='PNG')
            saved.append(fpath)
            overlays_written.append((short_name, target_height_px))

        # 2) Save OSM base once at the same target size (matches last computed height)
        try:
            final_height = target_height_px
            if EXPORT_SECTIONS.get("openstreetmap_bg", True):
                nolabels_tpl = "https://basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}.png"
                osm_bg = _fetch_xyz_layer(out_width_px, final_height, nolabels_tpl, zoom_boost=0).convert('RGB')
                osm_bg_path = out_dir / f"01_ST-BESA_{prov}_openstreetmap-bg_{timestamp}.png"
                try:
                    osm_bg.save(str(osm_bg_path), format='PNG', dpi=(dpi, dpi))
                except Exception:
                    osm_bg.save(str(osm_bg_path), format='PNG')
                saved.append(osm_bg_path)

            if EXPORT_SECTIONS.get("openstreetmap_text", True):
                labels_tpl   = "https://basemaps.cartocdn.com/light_only_labels/{z}/{x}/{y}.png"
                osm_lbl = _fetch_xyz_layer(out_width_px, final_height, labels_tpl, zoom_boost=int(OSM_LABELS_ZOOM_BOOST)).convert('RGBA')
                osm_lbl_path = out_dir / f"08_ST-BESA_{prov}_openstreetmap-text_{timestamp}.png"
                try:
                    osm_lbl.save(str(osm_lbl_path), format='PNG', dpi=(dpi, dpi))
                except Exception:
                    osm_lbl.save(str(osm_lbl_path), format='PNG')
                saved.append(osm_lbl_path)
        except Exception:
            pass

        # 3) Export legends and Photoshop script
        if EXPORT_SECTIONS.get("legends", True):
            try:
                from PIL import Image, ImageDraw, ImageFont

                def _load_font(size_px: int):
                    try:
                        return ImageFont.truetype("arial.ttf", size_px)
                    except Exception:
                        try:
                            return ImageFont.truetype("DejaVuSans.ttf", size_px)
                        except Exception:
                            return ImageFont.load_default()

                def save_continuous_legend(path, title, colors, vmin, vmax):
                    # Fixed gradient box: 1000px x 60px
                    bar_width = 1000
                    bar_height = 60
                    margin = 24
                    title_font = _load_font(LEGEND_TITLE_FONT_PX)
                    text_font = _load_font(LEGEND_TEXT_FONT_PX)
                    title_height = title_font.size
                    labels_height = text_font.size
                    width = bar_width + margin * 2
                    total_height = margin + title_height + 8 + bar_height + 6 + labels_height + margin
                    img = Image.new('RGBA', (width, total_height), (0, 0, 0, 0))
                    d = ImageDraw.Draw(img)
                    # Title
                    d.text((margin, margin), title, fill=(0,0,0,255), font=title_font)
                    # Draw smooth turbo gradient across the fixed box
                    try:
                        from matplotlib import cm
                        import numpy as np
                        turbo_cmap = cm.get_cmap('turbo')
                        for i in range(bar_width):
                            t = i / max(1, bar_width - 1)
                            rgba = turbo_cmap(t)
                            color = tuple(int(c * 255) for c in rgba[:3])
                            d.line([(margin + i, margin + title_height + 8), (margin + i, margin + title_height + 8 + bar_height)], fill=color)
                    except Exception:
                        # fallback to discrete colors if matplotlib is not available
                        for i in range(bar_width):
                            t = i / max(1, bar_width - 1)
                            idx = int(t * (len(colors) - 1))
                            d.line([(margin + i, margin + title_height + 8), (margin + i, margin + title_height + 8 + bar_height)], fill=colors[idx])
                    # Min/max labels
                    txt_min = f"{vmin:.1f}"
                    txt_max = f"{vmax:.1f}"
                    d.text((margin, margin + title_height + 8 + bar_height + 6), txt_min, fill=(0,0,0,255), font=text_font)
                    try:
                        tw = d.textlength(txt_max, font=text_font)
                    except Exception:
                        tw = len(txt_max) * text_font.size * 0.6
                    d.text((margin + bar_width - int(tw), margin + title_height + 8 + bar_height + 6), txt_max, fill=(0,0,0,255), font=text_font)
                    img.save(str(path), format='PNG', dpi=(dpi, dpi))

                def save_categorical_legend(path, title, items):
                    # Tighter layout: reduce gaps between title and classes and between color box and label
                    width = int(out_width_px / 2.0)
                    margin = 12
                    title_font = _load_font(LEGEND_TITLE_FONT_PX)
                    text_font = _load_font(LEGEND_TEXT_FONT_PX)
                    row_h = max(24, int(text_font.size * 1.2))
                    title_h = title_font.size
                    box_size = max(18, int(text_font.size * 0.9))
                    height = margin + title_h + 6 + len(items) * row_h + margin
                    img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
                    d = ImageDraw.Draw(img)
                    d.text((margin, margin), title, fill=(0,0,0,255), font=title_font)
                    y = margin + title_h + 6
                    for label, color in items:
                        d.rectangle((margin, y, margin + box_size, y + box_size), fill=color, outline=(60,60,60,255))
                        d.text((margin + box_size + 8, y), label, fill=(0,0,0,255), font=text_font)
                        y += row_h
                    img.save(str(path), format='PNG', dpi=(dpi, dpi))

                save_continuous_legend(out_dir / f"09_ST-BESA_{prov}_legend_volume_{timestamp}.png", "Building volume (m³)", TURBO, vmin_vol, vmax_vol)
                save_continuous_legend(out_dir / f"10_ST-BESA_{prov}_legend_surface_{timestamp}.png", "Building surface (m²)", TURBO, vmin_sur, vmax_sur)
                save_continuous_legend(out_dir / f"11_ST-BESA_{prov}_legend_population_{timestamp}.png", "Population (people)", TURBO, vmin_pop, vmax_pop)

                l1_items = [
                    (SMOD_L1_CLASSES[1]['name'], SMOD_L1_CLASSES[1]['color']),
                    (SMOD_L1_CLASSES[2]['name'], SMOD_L1_CLASSES[2]['color']),
                    (SMOD_L1_CLASSES[3]['name'], SMOD_L1_CLASSES[3]['color']),
                ]
                save_categorical_legend(out_dir / f"12_ST-BESA_{prov}_legend_smod_l1_{timestamp}.png", "SMOD L1", l1_items)
                l2_items = [
                    ('10 Water/No data', '#7AB6F5'),
                    ('11 Very low density rural', '#CDF57A'),
                    ('12 Low density rural', '#ABCD66'),
                    ('13 Rural cluster', '#375623'),
                    ('21 Suburban/peri-urban', '#FFFF00'),
                    ('22 Semi-dense urban', '#A87000'),
                    ('23 Dense urban cluster', '#732600'),
                    ('30 Urban centre', '#FF0000'),
                ]
                save_categorical_legend(out_dir / f"13_ST-BESA_{prov}_legend_smod_l2_{timestamp}.png", "SMOD L2", l2_items)
            except Exception as e:
                ui_msg.value = f"<b style='color:orange'>Warning: Legend generation failed: {str(e)}</b>"

        if EXPORT_SECTIONS.get("photoshop_script", True):
            try:
                jsx = (
                    "var folder = new Folder('" + str(out_dir).replace('\\', '/') + "');\n" +
                    "var files = folder.getFiles(/\\.png$/i).sort(function(a,b){ return (decodeURI(a.name) > decodeURI(b.name)) ? 1 : -1; });\n" +
                    "if(files.length>0){\n" +
                    "  var base = app.open(files[0]);\n" +
                    "  base.activeLayer.name = decodeURI(files[0].name.replace(/\\.png$/i,''));\n" +
                    "  for(var i=1;i<files.length;i++){\n" +
                    "    var im = app.open(files[i]); im.selection.selectAll(); im.selection.copy(); im.close(SaveOptions.DONOTSAVECHANGES); base.paste(); base.activeLayer.name = decodeURI(files[i].name.replace(/\\.png$/i,''));\n" +
                    "  }\n" +
                    "  base.resizeImage(undefined, undefined, " + str(dpi) + ", ResampleMethod.NONE);\n" +
                    "}\n"
                )
                with open(out_dir / "load_layers.jsx", 'w', encoding='utf-8') as f:
                    f.write(jsx)
            except Exception as e:
                ui_msg.value = f"<b style='color:orange'>Warning: Photoshop script creation failed: {str(e)}</b>"

        # Report (folder only)
        try:
            msg = f"<b style='color:green'>✓ Saved {len(saved)} image(s) to folder:</b> {str(out_dir)}"
            ui_msg.value = msg
        except Exception:
            ui_msg.value = "<b style='color:green'>✓ Saved images.</b>"
        set_busy(False)

    def on_export_plots_click(_):
        """Export SMOD L1/L2 plots as 174 mm wide @ 300 DPI PNGs into a plots folder."""
        if cache["geom"] is None or not cache["smod"]:
            ui_msg.value = "<b style='color:red'>No SMOD data to export. Please run analysis first.</b>"
            return
        import ee, math
        analyzer = STBESAAnalysis(project_id)
        analyzer.initialize_ee()
        geom = cache["geom"]
        cur = int(year_dd.value)
        smod = cache["smod"].get(cur)
        if smod is None:
            ui_msg.value = "<b style='color:red'>SMOD not available for the selected year.</b>"
            return

        # Build SMOD visualizations
        smod_mask = smod.neq(0)
        l2_palette_map = {10: '#7AB6F5', 11: '#CDF57A', 12: '#ABCD66', 13: '#375623', 21: '#FFFF00', 22: '#A87000', 23: '#732600', 30: '#FF0000'}
        smod_l2_vis = smod.updateMask(smod_mask).remap([10,11,12,13,21,22,23,30],[0,1,2,3,4,5,6,7]).visualize(min=0, max=7, palette=list(l2_palette_map.values()))
        smod_l1 = smod.divide(10).floor()
        l1_palette = [SMOD_L1_CLASSES[1]["color"], SMOD_L1_CLASSES[2]["color"], SMOD_L1_CLASSES[3]["color"]]
        smod_l1_vis = smod_l1.updateMask(smod_mask).visualize(min=1, max=3, palette=l1_palette)

        # File naming
        import datetime
        from pathlib import Path
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        prov = str(prov_dd.value).replace(' ', '_')
        out_dir = Path.cwd() / f"STBESA_EXPORT_{prov}_{timestamp}_PLOTS"
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        # Instead of exporting EE map images, export the exact matplotlib figures
        set_busy(True, "Exporting SMOD time-series plots (174 mm, 300 DPI)...")
        ui_msg.value = "<b style='color:blue'>Rendering SMOD L1/L2 time-series and saving to disk...</b>"
        saved = []

        # Target output width in inches for 174 mm and dpi
        width_mm = 174.0
        width_in = width_mm / 25.4
        dpi = 300

        try:
            import matplotlib.pyplot as plt
            import matplotlib as mpl
            import numpy as np

            # L1 figure (duplicate rendering logic from _render_plots_l1)
            if cache.get("df_smod_l1") is not None and not cache["df_smod_l1"].empty:
                total_df = cache["df"][ ["yil", "buvol_m3", "buvol_sur_m2", "pop_person", "bvpc_m3_per_person", "bspc_m2_per_person", "vol_sur_ratio"] ].sort_values("yil").reset_index(drop=True)
                years = total_df["yil"].values
                l1 = cache["df_smod_l1"][ ["yil", "smod_l1_code", "buvol_m3", "buvol_sur_m2", "pop_person", "bvpc_m3_per_person", "bspc_m2_per_person", "vol_sur_ratio"] ].copy()
                piv = {metric: l1.pivot_table(index="yil", columns="smod_l1_code", values=metric, aggfunc="first").reindex(years).fillna(np.nan) for metric in ["buvol_m3","buvol_sur_m2","pop_person","bvpc_m3_per_person","bspc_m2_per_person","vol_sur_ratio"]}
                mpl.rcParams.update({'figure.dpi': dpi,'savefig.dpi': dpi,'axes.titlesize': 7,'axes.labelsize': 6,'xtick.labelsize': 5,'ytick.labelsize': 5,'legend.fontsize': 5,'axes.grid': True,'grid.linestyle': ':','grid.alpha': 0.5,})
                fig, axes = plt.subplots(3, 2, figsize=(width_in, width_in * 5.33/6.85), constrained_layout=False)
                axes = axes.ravel()
                fig.patch.set_alpha(1.0)
                total_color = "#1f77b4"
                l1_colors = {1: SMOD_L1_CLASSES[1]["color"], 2: SMOD_L1_CLASSES[2]["color"], 3: SMOD_L1_CLASSES[3]["color"]}
                series = [("buvol_m3","Building Volume (m³)"),("buvol_sur_m2","Building Surface (m²)"),("pop_person","Population (people)"),("bvpc_m3_per_person","BVPC (m³/person)"),("bspc_m2_per_person","BSPC (m²/person)"),("vol_sur_ratio","Volume/Surface Ratio")]
                for i,(key,label) in enumerate(series):
                    ax = axes[i]
                    ax.plot(years, total_df[key].values, color=total_color, linewidth=1.6, label="Total")
                    ax.scatter(years, total_df[key].values, s=6, color=total_color, edgecolor=total_color, facecolor="white", zorder=3)
                    for cls in [1,2,3]:
                        if cls in piv[key].columns:
                            ax.plot(years, piv[key][cls].values, color=l1_colors[cls], linewidth=1.0, label=str(cls))
                    ax.set_title(label); ax.set_xlabel("Year"); ax.set_ylabel(label)
                    ax.margins(x=0.02); ax.set_xticks(years); ax.tick_params(axis='x', rotation=0)
                    try:
                        ax.axvline(cur, color="#d62728", linestyle="--", linewidth=1.0, alpha=0.8)
                    except Exception:
                        pass
                fig.suptitle("L1 Classes + Total", fontsize=7, fontweight='semibold')
                fig.subplots_adjust(bottom=0.10, hspace=0.50)
                from matplotlib.lines import Line2D
                handles = [Line2D([0], [0], color=total_color, lw=3, label='Total')]
                labels = ['Total']
                for cls in [1, 2, 3]:
                    handles.append(Line2D([0], [0], color=l1_colors[cls], lw=3))
                    labels.append({1: 'Rural', 2: 'Urban Cluster', 3: 'Urban Centre'}[cls])
                fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.03), ncol=4, prop={'size':5}, handlelength=2, columnspacing=1.0, frameon=True, fancybox=True, shadow=True, edgecolor='black', facecolor='white', framealpha=0.9)
                fpath = out_dir / f"ST-BESA_{prov}_plots_l1_{timestamp}.png"
                # Save with tight bbox to preserve layout, then adjust width to 174mm
                import io
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', transparent=False, facecolor='white')
                plt.close(fig)
                buf.seek(0)
                from PIL import Image
                im = Image.open(buf)
                # Calculate target width for 174mm at 300 DPI
                target_w = int(round(width_in * dpi))
                # Resize width to exactly 174mm, keeping aspect ratio
                new_h = int(round(im.size[1] * (target_w / im.size[0])))
                im_resized = im.resize((target_w, new_h), resample=Image.BICUBIC)
                im_resized.save(str(fpath), format='PNG', dpi=(dpi, dpi))
                saved.append(fpath)

            # L2 figure (duplicate rendering logic from _render_plots_l2)
            if cache.get("df_smod_l2") is not None and not cache["df_smod_l2"].empty:
                total_df = cache["df"][ ["yil", "buvol_m3", "buvol_sur_m2", "pop_person", "bvpc_m3_per_person", "bspc_m2_per_person", "vol_sur_ratio"] ].sort_values("yil").reset_index(drop=True)
                years = total_df["yil"].values
                l2 = cache["df_smod_l2"][ ["yil", "smod_l2_code", "buvol_m3", "buvol_sur_m2", "pop_person", "bvpc_m3_per_person", "bspc_m2_per_person", "vol_sur_ratio"] ].copy()
                piv = {metric: l2.pivot_table(index="yil", columns="smod_l2_code", values=metric, aggfunc="first").reindex(years).fillna(np.nan) for metric in ["buvol_m3","buvol_sur_m2","pop_person","bvpc_m3_per_person","bspc_m2_per_person","vol_sur_ratio"]}
                mpl.rcParams.update({'figure.dpi': dpi,'savefig.dpi': dpi,'axes.titlesize': 7,'axes.labelsize': 6,'xtick.labelsize': 5,'ytick.labelsize': 5,'legend.fontsize': 5,'axes.grid': True,'grid.linestyle': ':','grid.alpha': 0.5,})
                fig, axes = plt.subplots(3, 2, figsize=(width_in, width_in * 5.33/6.85), constrained_layout=False)
                axes = axes.ravel()
                fig.patch.set_alpha(1.0)
                total_color = "#1f77b4"
                l2_color_map = {10: '#7AB6F5', 11: '#CDF57A', 12: '#ABCD66', 13: '#375623', 21: '#FFFF00', 22: '#A87000', 23: '#732600', 30: '#FF0000'}
                series = [("buvol_m3","Building Volume (m³)"),("buvol_sur_m2","Building Surface (m²)"),("pop_person","Population (people)"),("bvpc_m3_per_person","BVPC (m³/person)"),("bspc_m2_per_person","BSPC (m²/person)"),("vol_sur_ratio","Volume/Surface Ratio")]
                for i,(key,label) in enumerate(series):
                    ax = axes[i]
                    ax.plot(years, total_df[key].values, color=total_color, linewidth=1.6, label="Total")
                    ax.scatter(years, total_df[key].values, s=6, color=total_color, edgecolor=total_color, facecolor="white", zorder=3)
                    for cls in [10,11,12,13,21,22,23,30]:
                        if cls in piv[key].columns:
                            ax.plot(years, piv[key][cls].values, color=l2_color_map[cls], linewidth=0.9, label=str(cls))
                    ax.set_title(label); ax.set_xlabel("Year"); ax.set_ylabel(label)
                    ax.margins(x=0.02); ax.set_xticks(years); ax.tick_params(axis='x', rotation=0)
                    try:
                        ax.axvline(cur, color="#d62728", linestyle="--", linewidth=1.0, alpha=0.8)
                    except Exception:
                        pass
                fig.suptitle("L2 Classes + Total", fontsize=7, fontweight='semibold')
                fig.subplots_adjust(bottom=0.11, hspace=0.50)
                from matplotlib.lines import Line2D
                handles = [Line2D([0], [0], color=total_color, lw=3, label='Total')]
                labels = ['Total']
                l2_order = [10,11,12,13,21,22,23,30]
                l2_label_map = {10:'10 Water/No data',11:'11 Very low density rural',12:'12 Low density rural',13:'13 Rural cluster',21:'21 Suburban/peri-urban',22:'22 Semi-dense urban',23:'23 Dense urban cluster',30:'30 Urban centre'}
                for cls in l2_order:
                    handles.append(Line2D([0], [0], color=l2_color_map[cls], lw=3))
                    labels.append(l2_label_map[cls])
                fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.03), ncol=4, prop={'size':5}, handlelength=2, columnspacing=1.0, frameon=True, fancybox=True, shadow=True, edgecolor='black', facecolor='white', framealpha=0.9)
                fpath = out_dir / f"ST-BESA_{prov}_plots_l2_{timestamp}.png"
                import io
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', transparent=False, facecolor='white')
                plt.close(fig)
                buf.seek(0)
                from PIL import Image
                im = Image.open(buf)
                target_w = int(round(width_in * dpi))
                new_h = int(round(im.size[1] * (target_w / im.size[0])))
                im_resized = im.resize((target_w, new_h), resample=Image.BICUBIC)
                im_resized.save(str(fpath), format='PNG', dpi=(dpi, dpi))
                saved.append(fpath)

            ui_msg.value = f"<b style='color:green'>✓ Saved {len(saved)} plot(s) to folder:</b> {str(out_dir)}"
        except Exception as e:
            ui_msg.value = f"<b style='color:red'>Export failed: {str(e)}</b>"
        finally:
            set_busy(False)

    export_plots_btn.on_click(on_export_plots_click)

    save_layers_btn.on_click(on_save_layers_click)

    def _apply_manual_vis_and_render():
        # Update cached visualization ranges from manual fields and re-render
        if not cache["vol"]:
            return
        vmin_v = float(vol_min.value if vol_min.value not in (None, "") else 0.0)
        vmax_v = float(vol_max.value if vol_max.value not in (None, "") else 80000.0)
        vmin_s = float(sur_min.value if sur_min.value not in (None, "") else 0.0)
        vmax_s = float(sur_max.value if sur_max.value not in (None, "") else 20000.0)
        vmin_p = float(pop_min.value if pop_min.value not in (None, "") else 0.0)
        vmax_p = float(pop_max.value if pop_max.value not in (None, "") else 200.0)
        for yy in list(cache["vol"].keys()):
            cache["vis"]["vol"][yy] = (vmin_v, vmax_v)
            cache["vis"]["sur"][yy] = (vmin_s, vmax_s)
            cache["vis"]["pop"][yy] = (vmin_p, vmax_p)
        _render_for_year(int(year_dd.value))

    def _recompute_auto_vis_and_render():
        # Recompute auto percentiles based on auto_once and render current year
        if cache["geom"] is None or not cache["vol"]:
            return
        analyzer = STBESAAnalysis(project_id)
        analyzer.initialize_ee()
        geom = cache["geom"]
        if auto_once.value:
            # Compute once for current year and reuse (FAST)
            cur = int(year_dd.value)
            vol_y = cache["vol"][cur]
            sur_y = cache["sur"][cur]
            pop_y = cache["pop"][cur]
            vmin_v, vmax_v = analyzer.dynamic_stretch(vol_y, "built_volume_total", geom, 80000, mask_zero=True, p_low=5, p_high=99)
            vmin_s, vmax_s = analyzer.dynamic_stretch(sur_y, "built_surface", geom, 20000, mask_zero=True, p_low=5, p_high=99)
            vmin_p, vmax_p = analyzer.dynamic_stretch(pop_y, "population_count",  geom, 200,   mask_zero=True, p_low=5, p_high=99)
            for yy in cache["vol"].keys():
                cache["vis"]["vol"][yy] = (vmin_v, vmax_v)
                cache["vis"]["sur"][yy] = (vmin_s, vmax_s)
                cache["vis"]["pop"][yy] = (vmin_p, vmax_p)
        else:
            # Compute per-year (SLOW)
            import time
            vol_keys = list(cache["vol"].keys())
            for i, yy in enumerate(vol_keys):
                vol_y = cache["vol"][yy]
                sur_y = cache["sur"][yy]
                pop_y = cache["pop"][yy]
                vmin_v, vmax_v = analyzer.dynamic_stretch(vol_y, "built_volume_total", geom, 80000, mask_zero=True, p_low=5, p_high=99)
                if ee_delay_seconds > 0 and i < len(vol_keys) - 1:
                    time.sleep(ee_delay_seconds)
                vmin_s, vmax_s = analyzer.dynamic_stretch(sur_y, "built_surface", geom, 20000, mask_zero=True, p_low=5, p_high=99)
                if ee_delay_seconds > 0 and i < len(vol_keys) - 1:
                    time.sleep(ee_delay_seconds)
                vmin_p, vmax_p = analyzer.dynamic_stretch(pop_y, "population_count",  geom, 200,   mask_zero=True, p_low=5, p_high=99)
                if ee_delay_seconds > 0 and i < len(vol_keys) - 1:
                    time.sleep(ee_delay_seconds)
                cache["vis"]["vol"][yy] = (vmin_v, vmax_v)
                cache["vis"]["sur"][yy] = (vmin_s, vmax_s)
                cache["vis"]["pop"][yy] = (vmin_p, vmax_p)
        _render_for_year(int(year_dd.value))

    def on_auto_toggle(change):
        if change['name'] == 'value':
            is_auto = change['new']
            # Apply scale is disabled when auto scale is ON
            for w in (vol_min, vol_max, sur_min, sur_max, pop_min, pop_max):
                w.disabled = is_auto
            apply_scale_btn.disabled = is_auto
            # If switching to manual and cache exists, apply manual vis immediately
            try:
                if not is_auto:
                    set_busy(True, "Applying manual scale...")
                    ui_msg.value = "<b style='color:blue'>Applying manual color scale...</b>"
                    _apply_manual_vis_and_render()
                    ui_msg.value = "<b style='color:green'>✓ Manual scale applied!</b>"
                    schedule_clear_message(3.0)
                else:
                    set_busy(True, "Recomputing auto scale...")
                    ui_msg.value = "<b style='color:blue'>Recomputing automatic color scale...</b>"
                    _recompute_auto_vis_and_render()
                    ui_msg.value = "<b style='color:green'>✓ Auto scale applied!</b>"
                    schedule_clear_message(3.0)
            except Exception as e:
                ui_msg.value = f"<b style='color:red'>Error: {str(e)}</b>"
            finally:
                set_busy(False)

    auto_scale.observe(on_auto_toggle, names='value')

    # React to manual min/max edits without requiring Run
    def on_manual_field_change(_):
        # Do nothing until Apply scale is pressed (prevents excessive redraws).
        pass
    for w in (vol_min, vol_max, pop_min, pop_max):
        w.observe(on_manual_field_change, names='value')
    
    def on_apply_scale_click(_):
        try:
            set_busy(True, "Applying color scale...")
            ui_msg.value = "<b style='color:blue'>Applying custom color scale...</b>"
            _apply_manual_vis_and_render()
            ui_msg.value = "<b style='color:green'>✓ Color scale applied!</b>"
            schedule_clear_message(3.0)
        except Exception as e:
            ui_msg.value = f"<b style='color:red'>Error: {str(e)}</b>"
        finally:
            set_busy(False)
    
    apply_scale_btn.on_click(on_apply_scale_click)

    def on_year_change(change):
        if change['name'] == 'value':
            run_source["value"] = "slider"
            # Only render if we have cached data
            if not cache["vol"]:
                return
            try:
                set_busy(True, f"Loading map for year {change['new']}...")
                ui_msg.value = f"<b style='color:blue'>Loading map for year {change['new']}...</b>"
                cur_year = int(change['new'])
                _render_for_year(cur_year)
                _render_plots_l1(cur_year)
                _render_plots_l2(cur_year)
                ui_msg.value = ""  # Clear immediately after rendering
            except Exception as e:
                ui_msg.value = f"<b style='color:red'>Error: {str(e)}</b>"
            finally:
                set_busy(False)

    year_dd.observe(on_year_change, names='value')

    from IPython.display import display
    # If running in Google Colab, expand the output iframe so the UI uses the notebook's main scrollbar
    try:
        import google.colab  # type: ignore
        from IPython.display import Javascript
        display(Javascript("google.colab.output.setIframeHeight(0, true, {maxHeight: 10000})"))
    except Exception:
        pass
    display(ui)
    return {"ui": ui, "widgets": {"prov": prov_dd, "dist": dist_sel, "run": run_btn, "out_map": out_map, "out_tbl": out_tbl}}


