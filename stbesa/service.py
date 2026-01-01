# stbesa/service.py

import json
import os
import pandas as pd
import geopandas as gpd
from typing import List, Optional, Dict, Tuple
from unidecode import unidecode
from stbesa.converters import GenericDatasetLoader
from stbesa.constants import SMOD

class STBESAService:
    """
    Core service for managing settlement analysis datasets and operations.
    Handles loading of data from local registry or custom files.
    """
    
    def __init__(self, registry_path: str = "datasets.json"):
        # Load registry if exists in current dir or fallback
        if os.path.exists(registry_path):
            with open(registry_path, 'r', encoding='utf-8') as f:
                self.registry = json.load(f)
        else:
            self.registry = {}
            
        self.active_dataset_code: Optional[str] = None
        self.active_gdf: Optional[gpd.GeoDataFrame] = None
        self.active_meta: Dict = {}

    def get_available_datasets(self) -> List[Tuple[str, str]]:
        """Returns list of (code, label) for available datasets."""
        return [(conf.get('name', code), code) for code, conf in self.registry.items()]

    def load_dataset(self, code: str, custom_path: str = None, progress_callback=None) -> None:
        """
        Loads a dataset into active memory.
        If code is a registry key, loads from registry config.
        If code is 'CUSTOM' (or similar), loads from custom_path using default heuristic.
        
        Args:
            code: Dataset code (e.g. "TUR", "DEU")
            custom_path: Path to custom file if code is "CUSTOM"
            progress_callback: Optional callable(float, str) to report progress (0.0-1.0, message)
        """
        if progress_callback: progress_callback(0.1, "Initializing dataset loading...")
        
        if code in self.registry:
            conf = self.registry[code]
            path = conf.get('file')
            # Handle relative paths assumes they are in 'boundaries' or relative to CWD
            # If not found, check boundaries
            
            # Auto-download logic
            target_path = path
            if not os.path.exists(target_path):
                boundary_path = os.path.join("boundaries", path)
                if os.path.exists(boundary_path):
                    target_path = boundary_path
                else:
                    # File not found locally. Check if we can download it.
                    url = conf.get('boundary_url')
                    if url:
                        print(f"Dataset file not found locally. Attempting download from: {url}")
                        try:
                            import urllib.request
                            import hashlib
                            
                            if progress_callback: progress_callback(0.2, "Downloading boundary data.")
                            
                            # Ensure boundaries directory
                            os.makedirs("boundaries", exist_ok=True)
                            download_dest = os.path.join("boundaries", os.path.basename(path))
                            
                            def download_reporthook(block_num, block_size, total_size):
                                if progress_callback and total_size > 0:
                                    percent = min(0.2 + (block_num * block_size / total_size) * 0.4, 0.6) # 0.2 to 0.6 range for download
                                    # Limit updates to avoid spamming UI
                                    if block_num % 100 == 0: 
                                        progress_callback(percent, f"Downloading: {block_num * block_size // 1024} KB / {total_size // 1024} KB")

                            print(f"Downloading to {download_dest}...")
                            urllib.request.urlretrieve(url, download_dest, reporthook=download_reporthook)
                            print("Download complete.")
                            
                            if progress_callback: progress_callback(0.6, "Verifying file integrity.")
                            
                            # Optional SHA256 verification
                            expected_sha = conf.get('boundary_sha256')
                            if expected_sha and expected_sha.startswith("sha256:"):
                                expected_sha = expected_sha.split(":")[1]
                                print("Verifying SHA256 header...")
                                sha256_hash = hashlib.sha256()
                                with open(download_dest, "rb") as f:
                                    for byte_block in iter(lambda: f.read(4096), b""):
                                        sha256_hash.update(byte_block)
                                file_hash = sha256_hash.hexdigest()
                                if file_hash.lower() == expected_sha.lower():
                                    print("✅ SHA256 verification successful.")
                                else:
                                    print(f"❌ SHA256 verification FAILED. Expected {expected_sha}, got {file_hash}")
                                    # We warn but don't delete/block for now to be permissive, or raise?
                                    # Raising is safer for security.
                                    raise ValueError("Downloaded file checksum mismatch.")
                            
                            target_path = download_dest
                            
                        except Exception as e:
                           raise RuntimeError(f"Failed to download dataset: {e}")
                    else:
                        pass # Let it fail below if still not found
            
            path = target_path # Update path pointer
            
            if not os.path.exists(path) and os.path.exists(os.path.join("boundaries", path)):
                path = os.path.join("boundaries", path)
            
            if progress_callback: progress_callback(0.7, "Loading geometry")
                

            kwargs = {}
            if "adm1_col" in conf: kwargs["adm1_col"] = conf["adm1_col"]
            if "adm2_col" in conf: kwargs["adm2_col"] = conf["adm2_col"]
            if "adm1_level" in conf: kwargs["adm1_level"] = conf["adm1_level"]
            if "adm2_level" in conf: kwargs["adm2_level"] = conf["adm2_level"]
            if "name_col" in conf: kwargs["name_col"] = conf["name_col"]
            if "level_col" in conf: kwargs["level_col"] = conf["level_col"]
            if "layer" in conf: kwargs["layer"] = conf["layer"]
            
            self.active_gdf = GenericDatasetLoader.load(path, **kwargs)
            self.active_dataset_code = code
            self.active_meta = conf
            
        else:
            # Custom file load
            if code == "CUSTOM":
                if not custom_path:
                    raise ValueError("Must provide custom_path for custom dataset")
            else:
                if not custom_path:
                     raise ValueError(f"Dataset code '{code}' not found in registry and no custom path provided.")

            self.active_gdf = GenericDatasetLoader.load(custom_path, adm1_col="NAME_1", adm2_col="NAME_2")
            self.active_dataset_code = "CUSTOM"
            self.active_meta = {"name": "Custom Dataset", "adm1_label": "Region", "adm2_label": "Sub-Region"}

    @property
    def adm1_label(self) -> str:
        return self.active_meta.get('adm1_label', 'Province')

    @property
    def adm2_label(self) -> str:
        return self.active_meta.get('adm2_label', 'District')

    def _ensure_loaded(self) -> None:
        if self.active_gdf is None:
            raise RuntimeError("No dataset loaded. Call load_dataset() first.")

    @staticmethod
    def _norm(s: Optional[str]) -> str:
        return unidecode("" if s is None or (isinstance(s, float) and pd.isna(s)) else str(s)).strip()

    def provinces(self) -> List[str]:
        self._ensure_loaded()
        vals = self.active_gdf["NAME_1"].dropna().astype(str).unique().tolist()
        return sorted(vals, key=lambda x: self._norm(x).lower())

    def districts(self, province_name: str) -> List[str]:
        self._ensure_loaded()
        sub = self.active_gdf[self.active_gdf["NAME_1"] == province_name]
        vals = sub["NAME_2"].dropna().astype(str).unique().tolist()
        return sorted(vals, key=lambda x: self._norm(x).lower())

    def get_district_row(self, province_name: str, district_name: str) -> gpd.GeoDataFrame:
        self._ensure_loaded()
        row = self.active_gdf[(self.active_gdf["NAME_1"] == province_name) & (self.active_gdf["NAME_2"] == district_name)]
        if row.empty:
            raise ValueError(f"District not found for {province_name}/{district_name}")
        return row

    def get_rows_by_province(self, province_name: str) -> gpd.GeoDataFrame:
        self._ensure_loaded()
        rows = self.active_gdf[self.active_gdf["NAME_1"] == province_name]
        if rows.empty:
            raise ValueError(f"No districts found for {province_name}")
        return rows
