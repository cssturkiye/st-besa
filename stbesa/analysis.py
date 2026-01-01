# stbesa/analysis.py

import time
from typing import Dict, Any, Tuple
import pandas as pd
import geopandas as gpd
import numpy as np

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
        import ee
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

    def geemap_map(self, height: str = "700px", backend: str = "folium"):
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
