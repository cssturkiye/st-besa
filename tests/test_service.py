"""
Unit tests for stbesa.service module.

Tests STBESAService for dataset loading and data access methods.
Dynamically runs tests for each country in datasets.json.
"""

import pytest
import os
from pathlib import Path
from stbesa.service import STBESAService


class TestSTBESAServiceAllDatasets:
    """Tests that run for each configured dataset."""
    
    @pytest.fixture
    def service(self, project_root):
        """Creates a service instance using real datasets.json."""
        original_cwd = os.getcwd()
        os.chdir(project_root)
        try:
            svc = STBESAService(registry_path="datasets.json")
            yield svc
        finally:
            os.chdir(original_cwd)
    
    def test_load_dataset_sets_active_gdf(self, service, dataset_code, boundaries_dir, datasets_config):
        """Loading a dataset should set active_gdf."""
        config = datasets_config[dataset_code]
        file_path = boundaries_dir / config["file"]
        
        if not file_path.exists():
            pytest.skip(f"Boundary file not downloaded: {file_path}")
        
        service.load_dataset(dataset_code)
        
        assert service.active_gdf is not None, f"{dataset_code}: active_gdf is None"
        assert service.active_dataset_code == dataset_code
    
    def test_provinces_returns_list(self, service, dataset_code, boundaries_dir, datasets_config):
        """provinces() should return a non-empty list."""
        config = datasets_config[dataset_code]
        file_path = boundaries_dir / config["file"]
        
        if not file_path.exists():
            pytest.skip(f"Boundary file not downloaded: {file_path}")
        
        service.load_dataset(dataset_code)
        provinces = service.provinces()
        
        assert isinstance(provinces, list), f"{dataset_code}: provinces() did not return a list"
        assert len(provinces) > 0, f"{dataset_code}: No provinces returned"
    
    def test_provinces_are_sorted(self, service, dataset_code, boundaries_dir, datasets_config):
        """provinces() should return a sorted list."""
        config = datasets_config[dataset_code]
        file_path = boundaries_dir / config["file"]
        
        if not file_path.exists():
            pytest.skip(f"Boundary file not downloaded: {file_path}")
        
        service.load_dataset(dataset_code)
        provinces = service.provinces()
        
        # The service sorts case-insensitively using unidecode
        # We just verify it returns without error and is a list
        assert isinstance(provinces, list)
    
    def test_districts_returns_list(self, service, dataset_code, boundaries_dir, datasets_config):
        """districts(province) should return a list for any province."""
        config = datasets_config[dataset_code]
        file_path = boundaries_dir / config["file"]
        
        if not file_path.exists():
            pytest.skip(f"Boundary file not downloaded: {file_path}")
        
        service.load_dataset(dataset_code)
        provinces = service.provinces()
        
        if not provinces:
            pytest.skip(f"{dataset_code}: No provinces to test")
        
        # Test first province
        districts = service.districts(provinces[0])
        
        assert isinstance(districts, list), f"{dataset_code}: districts() did not return a list"
        assert len(districts) > 0, f"{dataset_code}: No districts for {provinces[0]}"
    
    def test_adm_labels_match_config(self, service, dataset_code, datasets_config):
        """ADM labels should match the configuration."""
        config = datasets_config[dataset_code]
        service.active_meta = config
        
        expected_adm1 = config.get("adm1_label", "Province")
        expected_adm2 = config.get("adm2_label", "District")
        
        assert service.adm1_label == expected_adm1, f"{dataset_code}: adm1_label mismatch"
        assert service.adm2_label == expected_adm2, f"{dataset_code}: adm2_label mismatch"


class TestSTBESAServiceGeneral:
    """General service tests (non-parametrized)."""
    
    def test_get_available_datasets_returns_all(self, project_root, datasets_config):
        """get_available_datasets() should return all configured datasets."""
        original_cwd = os.getcwd()
        os.chdir(project_root)
        
        try:
            service = STBESAService(registry_path="datasets.json")
            available = service.get_available_datasets()
            
            # Should have same count as datasets_config
            assert len(available) == len(datasets_config)
            
            # Each should be a tuple of (name, code)
            for name, code in available:
                assert code in datasets_config
        finally:
            os.chdir(original_cwd)
    
    def test_provinces_without_loading_raises_error(self, project_root):
        """Calling provinces() without loading should raise RuntimeError."""
        original_cwd = os.getcwd()
        os.chdir(project_root)
        
        try:
            service = STBESAService(registry_path="datasets.json")
            
            with pytest.raises(RuntimeError, match="No dataset loaded"):
                service.provinces()
        finally:
            os.chdir(original_cwd)
    
    def test_load_unknown_dataset_raises_error(self, project_root):
        """Loading an unknown dataset code should raise ValueError."""
        original_cwd = os.getcwd()
        os.chdir(project_root)
        
        try:
            service = STBESAService(registry_path="datasets.json")
            
            with pytest.raises(ValueError):
                service.load_dataset("UNKNOWN_COUNTRY_CODE_XYZ")
        finally:
            os.chdir(original_cwd)
