"""
Focused unit tests for the analytical engine.

These tests avoid live Earth Engine calls by replacing the imported `ee`
module with lightweight fakes.
"""

from types import SimpleNamespace
from pathlib import Path
import builtins
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stbesa.analysis import STBESAAnalysis


class _FakeReduceResult:
    def __init__(self, mapping):
        self._mapping = mapping

    def get(self, key):
        return self._mapping[key]


class _FakeImage:
    def __init__(self, asset, values):
        self.asset = asset
        self.values = values
        self.band = None

    def select(self, band):
        self.band = band
        return self

    def clip(self, _geom):
        return self

    def reduceRegion(self, _reducer, **_kwargs):
        return _FakeReduceResult({self.band: self.values[(self.asset, self.band)]})


def test_compute_indicators_derives_ratios_and_zero_guards(monkeypatch):
    values = {
        ("JRC/GHSL/P2023A/GHS_BUILT_V/2020", "built_volume_total"): 200.0,
        ("JRC/GHSL/P2023A/GHS_POP/2020", "population_count"): 10.0,
        ("JRC/GHSL/P2023A/GHS_BUILT_S/2020", "built_surface"): 50.0,
        ("JRC/GHSL/P2023A/GHS_BUILT_V/2025", "built_volume_total"): 120.0,
        ("JRC/GHSL/P2023A/GHS_POP/2025", "population_count"): 0.0,
        ("JRC/GHSL/P2023A/GHS_BUILT_S/2025", "built_surface"): 30.0,
    }

    fake_ee = SimpleNamespace(
        Image=lambda asset: _FakeImage(asset, values),
        Number=lambda value: value,
        Reducer=SimpleNamespace(sum=lambda: "sum"),
    )
    monkeypatch.setitem(sys.modules, "ee", fake_ee)

    analysis = STBESAAnalysis("test-project")
    analysis._ee_initialized = True
    monkeypatch.setattr(analysis, "_ee_getinfo", lambda value: value)

    result_2020 = analysis.compute_indicators("fake-geom", 2020)
    assert result_2020["sum_volume_m3"] == 200.0
    assert result_2020["sum_surface_m2"] == 50.0
    assert result_2020["sum_population"] == 10.0
    assert result_2020["bvpc_m3_per_person"] == 20.0
    assert result_2020["bspc_m2_per_person"] == 5.0
    assert result_2020["vol_sur_ratio"] == 4.0

    result_2025 = analysis.compute_indicators("fake-geom", 2025)
    assert result_2025["sum_population"] == 0.0
    assert result_2025["bvpc_m3_per_person"] is None
    assert result_2025["bspc_m2_per_person"] is None
    assert result_2025["vol_sur_ratio"] == 4.0


def test_default_map_backend_does_not_import_geemap_foliumap(monkeypatch):
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "geemap.foliumap" or name.startswith("geemap.foliumap."):
            raise AssertionError("geemap.foliumap should not be imported")
        return real_import(name, *args, **kwargs)

    analysis = STBESAAnalysis("test-project")
    analysis._ee_initialized = True
    monkeypatch.setattr(builtins, "__import__", guarded_import)

    rendered_map = analysis.geemap_map(height="700px")

    assert hasattr(rendered_map, "addLayer")
    assert hasattr(rendered_map, "centerObject")
    assert hasattr(rendered_map, "addLayerControl")
    assert "leaflet" in rendered_map._repr_html_().lower()


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__]))
