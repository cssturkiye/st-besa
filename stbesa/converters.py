# stbesa/converters.py

import os
import glob
import zipfile
import shutil
import tempfile
import urllib.request
import geopandas as gpd
import pandas as pd
from unidecode import unidecode
from typing import Optional, List, Dict
from pathlib import Path

class GenericDatasetLoader:
    """
    Helper to load any administrative boundary file (GPKG, SHP, GeoJSON) 
    and normalize it to the expected internal schema (NAME_1, NAME_2, geometry).
    Supports two modes:
    1. Column-based (OCHA): Separate columns for ADM1 and ADM2 (e.g., 'NAME_1', 'NAME_2').
    2. Level-based (Kontur): Single layer with 'admin_level' column to distinguish hierarchies.
    """
    @staticmethod
    def load(path: str, 
             adm1_col: Optional[str] = None, adm2_col: Optional[str] = None,
             adm1_level: Optional[int] = None, adm2_level: Optional[int] = None,
             level_col: str = "admin_level", name_col: str = "name",
             layer: Optional[str] = None) -> gpd.GeoDataFrame:
        """
        Loads and standardizes a dataset.
        
        Args:
            path: Path to file (gpkg, shp, geojson, zip).
            adm1_col: Column name for ADM1 (Province) - used for Column-based.
            adm2_col: Column name for ADM2 (District) - used for Column-based.
            adm1_level: 'admin_level' value for ADM1 (e.g. 4) - used for Level-based.
            adm2_level: 'admin_level' value for ADM2 (e.g. 6) - used for Level-based.
            level_col: Column name containing level info (default 'admin_level').
            name_col: Column name containing region names (used when level-based).
            layer: Specific layer name to read from multi-layer GPKG files.
        """
        
        # 1. READ FILE
        if path.endswith('.zip'):
            # Basic zip handling (same as before)
            temp_dir = tempfile.mkdtemp()
            with zipfile.ZipFile(path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # Find shapefile
            shps = glob.glob(os.path.join(temp_dir, "**/*.shp"), recursive=True)
            if not shps:
                shutil.rmtree(temp_dir)
                raise ValueError("No .shp found in zip")
            file_to_read = shps[0]
        else:
            file_to_read = path

        try:
            # Read with optional layer specification
            if layer:
                gdf = gpd.read_file(file_to_read, layer=layer)
            else:
                gdf = gpd.read_file(file_to_read)
        except Exception as e:
            if 'temp_dir' in locals(): shutil.rmtree(temp_dir)
            raise ValueError(f"Failed to read file: {e}")

        # Ensure WGS84
        if gdf.crs != "EPSG:4326":
            gdf = gdf.to_crs("EPSG:4326")

        # 2. STANDARDIZE COLUMNS
        new_gdf = gpd.GeoDataFrame(geometry=gdf.geometry, crs=gdf.crs)
        col_map = {c.upper(): c for c in gdf.columns}

        # MODE A: LEVEL-BASED (Kontur)
        if adm1_level is not None:
            if level_col.upper() not in col_map:
                raise ValueError(f"Level column '{level_col}' not found for level-based loading.")
            real_level_col = col_map[level_col.upper()]
            
            if name_col.upper() not in col_map:
                raise ValueError(f"Name column '{name_col}' not found.")
            real_name_col = col_map[name_col.upper()]
            
            # Fallback Logic: If name_col is e.g. 'NAME_EN' and it has nulls, fill with 'NAME' if exists
            if name_col.lower() != "name" and "NAME" in col_map:
                fallback_col = col_map["NAME"]
                # Create a temporary coalesced column
                temp_name_col = f"__coalesced_{name_col}__"
                gdf[temp_name_col] = gdf[real_name_col].fillna(gdf[fallback_col])
                real_name_col = temp_name_col

            # Helper to filter by level (int or list)
            def filter_level(df, col, level):
                if isinstance(level, list):
                    return df[df[col].isin(level)].copy()
                return df[df[col] == level].copy()

            # Let's try to do a spatial join if we have both levels.
            if adm2_level is not None:
                adm2_rows = filter_level(gdf, real_level_col, adm2_level)
                adm1_rows = filter_level(gdf, real_level_col, adm1_level)
                
                if adm2_rows.empty:
                     # Fallback if adm2 not found (maybe level is wrong)
                     raise ValueError(f"No rows found for ADM2 level {adm2_level}")

                if adm1_rows.empty:
                    new_gdf = gpd.GeoDataFrame(geometry=adm2_rows.geometry, crs=adm2_rows.crs)
                    new_gdf["NAME_2"] = adm2_rows[real_name_col].astype(str).str.strip()
                    new_gdf["NAME_1"] = "Unknown"
                else:
                    # Spatial join to assign ADM1 name to ADM2 units
                    joined = gpd.sjoin(adm2_rows, adm1_rows[[real_name_col, 'geometry']], how='left', predicate='within')
                    
                    new_gdf = gpd.GeoDataFrame(geometry=joined.geometry, crs=joined.crs)
                    n_col = real_name_col
                    left_col = n_col + '_left' if n_col + '_left' in joined.columns else n_col
                    right_col = n_col + '_right' if n_col + '_right' in joined.columns else 'index_right'
                    
                    new_gdf["NAME_2"] = joined[left_col].astype(str).str.strip()
                    if n_col + '_right' in joined.columns:
                        new_gdf["NAME_1"] = joined[n_col + '_right'].fillna("Unknown").astype(str).str.strip()
                    else:
                        new_gdf["NAME_1"] = "Unknown"
                
            else:
                # Only ADM1 level (treat as primary)
                rows = gdf[gdf[real_level_col] == adm1_level].copy()
                new_gdf = gpd.GeoDataFrame(geometry=rows.geometry, crs=rows.crs)
                new_gdf["NAME_1"] = rows[real_name_col].astype(str).str.strip()
                new_gdf["NAME_2"] = "All"

        # MODE B: COLUMN-BASED (OCHA)
        else:
            if adm1_col is None:
                raise ValueError("Must provide either adm1_level or adm1_col")
                
            if adm1_col.upper() not in col_map:
                raise ValueError(f"Region column '{adm1_col}' not found. Available: {list(gdf.columns)}")
            real_adm1 = col_map[adm1_col.upper()]
            
            new_gdf["NAME_1"] = gdf[real_adm1].astype(str).str.strip()
            
            if adm2_col and adm2_col.upper() in col_map:
                real_adm2 = col_map[adm2_col.upper()]
                new_gdf["NAME_2"] = gdf[real_adm2].astype(str).str.strip()
            else:
                new_gdf["NAME_2"] = [f"Feature {i+1}" for i in range(len(gdf))]
        
        # Cleanup
        if 'temp_dir' in locals(): shutil.rmtree(temp_dir)
        return new_gdf

class OCHACODLoader:
    """
    Specific loader for OCHA COD (Common Operational Datasets) structure.
    Usually simplified into GenericDatasetLoader, but kept for legacy or specific OCHA logic.
    """
    @staticmethod
    def load_turkey(gpkg_path: str) -> gpd.GeoDataFrame:
        # Pre-configured for standard Turkey OCHA COD file
        return GenericDatasetLoader.load(gpkg_path, adm1_col="NAME_1", adm2_col="NAME_2")

class HGMConverter:
    """
    Legacy converter for General Directorate of Mapping (HGM) Excel/XLS data.
    """
    @staticmethod
    def convert_xls_to_gpkg_and_geojson(xls_path: str, output_gpkg: str, output_geojson: str = None):
        pass
