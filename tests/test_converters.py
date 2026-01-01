"""
Unit tests for stbesa.converters module.

Tests the GenericDatasetLoader for all configured datasets.
Dynamically runs tests for each country in datasets.json.
"""

import pytest
import geopandas as gpd
from pathlib import Path
from stbesa.converters import GenericDatasetLoader


class TestGenericDatasetLoaderAllDatasets:
    """Tests that run for each configured dataset."""
    
    def test_load_returns_geodataframe(self, dataset_code, datasets_config, boundaries_dir):
        """Loading any configured dataset should return a GeoDataFrame."""
        config = datasets_config[dataset_code]
        file_path = boundaries_dir / config["file"]
        
        if not file_path.exists():
            pytest.skip(f"Boundary file not downloaded: {file_path}")
        
        # Prepare kwargs based on format
        kwargs = {"layer": config.get("layer")}
        
        if config.get("format") == "ocha":
            kwargs["adm1_col"] = config.get("adm1_col")
            kwargs["adm2_col"] = config.get("adm2_col")
        else:  # kontur format
            kwargs["adm1_level"] = config.get("adm1_level")
            kwargs["adm2_level"] = config.get("adm2_level")
            kwargs["name_col"] = config.get("name_col", "name")
        
        result = GenericDatasetLoader.load(str(file_path), **kwargs)
        
        assert isinstance(result, gpd.GeoDataFrame)
    
    def test_result_has_required_columns(self, dataset_code, datasets_config, boundaries_dir):
        """Result should have NAME_1, NAME_2, and geometry columns."""
        config = datasets_config[dataset_code]
        file_path = boundaries_dir / config["file"]
        
        if not file_path.exists():
            pytest.skip(f"Boundary file not downloaded: {file_path}")
        
        kwargs = {"layer": config.get("layer")}
        
        if config.get("format") == "ocha":
            kwargs["adm1_col"] = config.get("adm1_col")
            kwargs["adm2_col"] = config.get("adm2_col")
        else:
            kwargs["adm1_level"] = config.get("adm1_level")
            kwargs["adm2_level"] = config.get("adm2_level")
            kwargs["name_col"] = config.get("name_col", "name")
        
        result = GenericDatasetLoader.load(str(file_path), **kwargs)
        
        assert "NAME_1" in result.columns, f"{dataset_code}: NAME_1 column missing"
        assert "NAME_2" in result.columns, f"{dataset_code}: NAME_2 column missing"
        assert "geometry" in result.columns, f"{dataset_code}: geometry column missing"
    
    def test_result_has_correct_crs(self, dataset_code, datasets_config, boundaries_dir):
        """Result should be in WGS84 (EPSG:4326)."""
        config = datasets_config[dataset_code]
        file_path = boundaries_dir / config["file"]
        
        if not file_path.exists():
            pytest.skip(f"Boundary file not downloaded: {file_path}")
        
        kwargs = {"layer": config.get("layer")}
        
        if config.get("format") == "ocha":
            kwargs["adm1_col"] = config.get("adm1_col")
            kwargs["adm2_col"] = config.get("adm2_col")
        else:
            kwargs["adm1_level"] = config.get("adm1_level")
            kwargs["adm2_level"] = config.get("adm2_level")
            kwargs["name_col"] = config.get("name_col", "name")
        
        result = GenericDatasetLoader.load(str(file_path), **kwargs)
        
        assert result.crs is not None, f"{dataset_code}: CRS is None"
        assert result.crs.to_epsg() == 4326, f"{dataset_code}: CRS is not WGS84"
    
    def test_result_has_valid_geometries(self, dataset_code, datasets_config, boundaries_dir):
        """All geometries should be valid."""
        config = datasets_config[dataset_code]
        file_path = boundaries_dir / config["file"]
        
        if not file_path.exists():
            pytest.skip(f"Boundary file not downloaded: {file_path}")
        
        kwargs = {"layer": config.get("layer")}
        
        if config.get("format") == "ocha":
            kwargs["adm1_col"] = config.get("adm1_col")
            kwargs["adm2_col"] = config.get("adm2_col")
        else:
            kwargs["adm1_level"] = config.get("adm1_level")
            kwargs["adm2_level"] = config.get("adm2_level")
            kwargs["name_col"] = config.get("name_col", "name")
        
        result = GenericDatasetLoader.load(str(file_path), **kwargs)
        
        invalid_count = (~result.geometry.is_valid).sum()
        assert invalid_count == 0, f"{dataset_code}: {invalid_count} invalid geometries"
    
    def test_result_has_provinces(self, dataset_code, datasets_config, boundaries_dir):
        """Result should have at least one province (NAME_1)."""
        config = datasets_config[dataset_code]
        file_path = boundaries_dir / config["file"]
        
        if not file_path.exists():
            pytest.skip(f"Boundary file not downloaded: {file_path}")
        
        kwargs = {"layer": config.get("layer")}
        
        if config.get("format") == "ocha":
            kwargs["adm1_col"] = config.get("adm1_col")
            kwargs["adm2_col"] = config.get("adm2_col")
        else:
            kwargs["adm1_level"] = config.get("adm1_level")
            kwargs["adm2_level"] = config.get("adm2_level")
            kwargs["name_col"] = config.get("name_col", "name")
        
        result = GenericDatasetLoader.load(str(file_path), **kwargs)
        
        provinces = result["NAME_1"].dropna().unique()
        assert len(provinces) > 0, f"{dataset_code}: No provinces found"


class TestGenericDatasetLoaderErrors:
    """Tests for error handling (non-parametrized)."""
    
    def test_load_nonexistent_file_raises_error(self):
        """Loading a non-existent file should raise an error."""
        with pytest.raises(ValueError, match="Failed to read file"):
            GenericDatasetLoader.load(
                "nonexistent_file.gpkg",
                adm1_col="NAME_1",
                adm2_col="NAME_2"
            )
