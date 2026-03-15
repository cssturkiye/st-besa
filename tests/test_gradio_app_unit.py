"""
Focused unit tests for the Gradio orchestration layer.

The test imports the app module dynamically so the working directory can be
set to the project root before module-level registry loading happens.
"""

from importlib import import_module
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class _FakeRows:
    def dissolve(self):
        return "fake-geom-gdf"


class _FakeAnalysisService:
    def initialize_ee(self):
        return None

    def geopandas_row_to_ee(self, _geom_gdf):
        return "fake-feature", "fake-ee-geom"

    def compute_indicators(self, _geom, year):
        return {
            "sum_volume_m3": 100.0 + year,
            "sum_surface_m2": 50.0,
            "sum_population": 10.0,
            "bvpc_m3_per_person": 10.0,
            "bspc_m2_per_person": 5.0,
            "vol_sur_ratio": 2.0,
        }

    def compute_smod_statistics(self, _geom, _year, level="L1", delay_seconds=0.0):
        if level == "L1":
            return {
                1: {
                    "buvol_m3": 20.0,
                    "buvol_sur_m2": 10.0,
                    "pop_person": 5.0,
                    "bvpc_m3_per_person": 4.0,
                    "bspc_m2_per_person": 2.0,
                    "vol_sur_ratio": 2.0,
                }
            }
        return {
            30: {
                "buvol_m3": 80.0,
                "buvol_sur_m2": 40.0,
                "pop_person": 5.0,
                "bvpc_m3_per_person": 16.0,
                "bspc_m2_per_person": 8.0,
                "vol_sur_ratio": 2.0,
            }
        }


def test_run_analysis_builds_state_for_a_typical_all_district_run(monkeypatch):
    project_root = PROJECT_ROOT
    monkeypatch.chdir(project_root)
    sys.modules.pop("stbesa.gradio_app", None)
    gradio_app = import_module("stbesa.gradio_app")

    fake_analysis = _FakeAnalysisService()
    monkeypatch.setattr(gradio_app, "YEARS_ALL", [2020])
    monkeypatch.setattr(
        gradio_app,
        "init_analysis_service",
        lambda project_id: setattr(gradio_app, "analysis_service", fake_analysis),
    )
    monkeypatch.setattr(
        gradio_app,
        "service",
        type(
            "FakeService",
            (),
            {"get_rows_by_province": staticmethod(lambda province: _FakeRows())},
        )(),
    )
    monkeypatch.setattr(gradio_app, "_generate_map_html", lambda *args: "<map-html>")
    monkeypatch.setattr(gradio_app, "_render_plots_l1", lambda *args: "FIG_L1")
    monkeypatch.setattr(gradio_app, "_render_plots_l2", lambda *args: "FIG_L2")
    monkeypatch.setattr(gradio_app.time, "sleep", lambda _seconds: None)

    outputs = list(
        gradio_app.run_analysis(
            project_id="demo-project",
            dataset_code="TUR",
            province="Izmir",
            districts=["ALL"],
            year=2020,
            max_workers=1,
            auto_scale=False,
            vol_min=1,
            vol_max=2,
            sur_min=3,
            sur_max=4,
            pop_min=5,
            pop_max=6,
        )
    )

    final = outputs[-1]
    results_state = final[12]

    assert final[1] == "<map-html>"
    assert "Analysis Complete: Izmir (All)" in final[11]
    assert results_state["meta"]["Dataset"] == "TUR"
    assert results_state["meta"]["Province"] == "Izmir"
    assert results_state["meta"]["Districts"] == "ALL"
    assert results_state["vis_params"] == {"vol": (1, 2), "sur": (3, 4), "pop": (5, 6)}
    assert results_state["overall"]["yil"].tolist() == [2020]
    assert results_state["l1"]["smod_l1_code"].tolist() == [1]
    assert results_state["l2"]["smod_l2_code"].tolist() == [30]


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__]))
