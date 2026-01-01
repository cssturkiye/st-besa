"""
Pytest configuration and shared fixtures for ST-BESA tests.

Uses the real datasets.json configuration and boundaries folder.
Tests dynamically cover all configured countries.
"""

import pytest
import json
import os
import sys
from pathlib import Path


# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Add project root to sys.path for stbesa imports
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="session")
def project_root():
    """Returns the project root directory."""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def datasets_config():
    """Loads the real datasets.json configuration."""
    config_path = PROJECT_ROOT / "datasets.json"
    if not config_path.exists():
        pytest.skip("datasets.json not found in project root")
    
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def dataset_codes(datasets_config):
    """Returns list of all configured dataset codes."""
    return list(datasets_config.keys())


@pytest.fixture(scope="session")
def boundaries_dir():
    """Returns the boundaries directory path."""
    return PROJECT_ROOT / "boundaries"


def pytest_generate_tests(metafunc):
    """
    Dynamically parametrize tests to run for each configured country.
    Automatically downloads missing boundary files before running tests.
    """
    if "dataset_code" in metafunc.fixturenames:
        config_path = PROJECT_ROOT / "datasets.json"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            
            # Auto-download missing files
            import urllib.request
            import hashlib

            def download_file(url, dest, sha256=None):
                print(f"Downloading from {url}...")
                urllib.request.urlretrieve(url, dest)
                
                if sha256 and sha256.startswith("sha256:"):
                    expected = sha256.split(":")[1]
                    hasher = hashlib.sha256()
                    with open(dest, "rb") as f:
                        for chunk in iter(lambda: f.read(4096), b""):
                            hasher.update(chunk)
                    if hasher.hexdigest().lower() != expected.lower():
                        print(f"Warning: SHA256 mismatch for {dest}")

            for code, conf in config.items():
                file_path = PROJECT_ROOT / "boundaries" / conf["file"]
                if not file_path.exists():
                    print(f"\n[Test Setup] Downloading missing boundary for {code}...")
                    try:
                        (PROJECT_ROOT / "boundaries").mkdir(exist_ok=True)
                        download_file(
                            conf["boundary_url"], 
                            str(file_path),
                            sha256=conf.get("boundary_sha256")
                        )
                        print(f"[Test Setup] Successfully downloaded {code}.")
                    except Exception as e:
                        print(f"[Test Setup] Failed to download {code}: {e}")
            
            metafunc.parametrize("dataset_code", list(config.keys()))
        else:
            metafunc.parametrize("dataset_code", [])
