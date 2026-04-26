"""
Focused unit tests for the export layer.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from stbesa.exporter import (
    EXPORT_DPI,
    EXPORT_WIDTH_MM,
    PLOT_EXPORT_DPI,
    PLOT_EXPORT_WIDTH_MM,
    OSM_TEXT_ZOOM_OFFSET,
    LayerExporter,
    STBESAExporter,
)


def test_export_excel_report_creates_expected_workbook_structure(tmp_path: Path):
    output_path = tmp_path / "report.xlsx"

    overall = pd.DataFrame(
        [
            {"yil": 2025, "buvol_m3": 3.0, "buvol_sur_m2": 2.0, "pop_person": 1.0},
            {"yil": 2020, "buvol_m3": 2.0, "buvol_sur_m2": 1.0, "pop_person": 1.0},
        ]
    )
    smod_l1 = pd.DataFrame(
        [
            {"yil": 2025, "smod_l1_code": 2, "buvol_m3": 7.0},
            {"yil": 2020, "smod_l1_code": 1, "buvol_m3": 5.0},
        ]
    )
    smod_l2 = pd.DataFrame(
        [
            {"yil": 2025, "smod_l2_code": 30, "buvol_m3": 9.0},
            {"yil": 2020, "smod_l2_code": 21, "buvol_m3": 6.0},
        ]
    )
    metadata = {"Dataset": "TUR", "Province": "Izmir"}

    returned_path = STBESAExporter.export_excel_report(
        str(output_path), overall, smod_l1, smod_l2, metadata
    )

    assert returned_path == str(output_path)
    assert output_path.exists()

    workbook = pd.ExcelFile(output_path)
    assert workbook.sheet_names == [
        "Overall_Statistics",
        "SMOD_L1_Statistics",
        "SMOD_L2_Statistics",
        "Metadata",
        "Data_Dictionary",
    ]

    overall_sheet = pd.read_excel(output_path, sheet_name="Overall_Statistics")
    metadata_sheet = pd.read_excel(output_path, sheet_name="Metadata")
    dictionary_sheet = pd.read_excel(output_path, sheet_name="Data_Dictionary")

    assert overall_sheet["yil"].tolist() == [2020, 2025]
    assert metadata_sheet["Parameter"].tolist() == ["Dataset", "Province"]
    assert "Programmatic Name" in dictionary_sheet.columns
    assert not dictionary_sheet.empty


def test_export_resolution_settings_are_publication_defaults():
    assert EXPORT_WIDTH_MM == 190.0
    assert EXPORT_DPI == 600
    assert PLOT_EXPORT_WIDTH_MM == 190.0
    assert PLOT_EXPORT_DPI == 1000
    assert OSM_TEXT_ZOOM_OFFSET == 0


def test_export_plots_as_png_uses_configured_width_and_dpi(tmp_path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    output_path = tmp_path / "plot.png"
    fig, ax = plt.subplots(figsize=(2, 1))
    ax.plot([0, 1], [0, 1])

    try:
        returned_path = STBESAExporter.export_plots_as_png(fig, str(output_path))
    finally:
        plt.close(fig)

    assert returned_path == str(output_path)

    with Image.open(output_path) as image:
        expected_width_px = round((PLOT_EXPORT_WIDTH_MM / 25.4) * PLOT_EXPORT_DPI)
        assert image.width == expected_width_px
        assert round(image.info["dpi"][0]) == PLOT_EXPORT_DPI
        assert round(image.info["dpi"][1]) == PLOT_EXPORT_DPI


def test_export_plots_as_png_forces_white_background(tmp_path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    output_path = tmp_path / "plot-white-bg.png"
    fig, ax = plt.subplots(figsize=(2, 1))
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    ax.plot([0, 1], [0, 1])

    try:
        STBESAExporter.export_plots_as_png(fig, str(output_path))
    finally:
        plt.close(fig)

    with Image.open(output_path) as image:
        assert image.mode == "RGB"
        assert image.getpixel((0, 0)) == (255, 255, 255)


def test_osm_text_task_is_opt_in():
    exporter = LayerExporter.__new__(LayerExporter)
    exporter.include_osm_text = False

    task_names = [name for name, _task in exporter._build_osm_tasks(Path("."), "20260101")]
    assert task_names == ["OSM Background"]

    exporter.include_osm_text = True
    task_names = [name for name, _task in exporter._build_osm_tasks(Path("."), "20260101")]
    assert task_names == ["OSM Background", "OSM Text"]


def test_photoshop_script_uses_relative_script_folder(tmp_path: Path):
    exporter = LayerExporter.__new__(LayerExporter)
    exporter.dpi = EXPORT_DPI
    out_dir = tmp_path / "STBESA_LAYERS_Test_20260101_120000"
    out_dir.mkdir()

    jsx_path = exporter._generate_photoshop_script(out_dir)
    content = jsx_path.read_text(encoding="utf-8")

    assert "var scriptFile = new File($.fileName);" in content
    assert "var folder = scriptFile.parent;" in content
    assert str(tmp_path) not in content
    assert "STBESA_EXPORT_" not in content


def test_vector_boundary_layer_draws_internal_district_lines(tmp_path: Path):
    import geopandas as gpd
    from PIL import Image
    from shapely.geometry import box

    boundary_gdf = gpd.GeoDataFrame(
        geometry=[
            box(0, 0, 100, 100),
            box(100, 0, 200, 100),
        ],
        crs="EPSG:3857",
    )
    exporter = LayerExporter.__new__(LayerExporter)
    exporter.boundary_gdf = boundary_gdf
    exporter.bounds_3857 = {"w": 0, "e": 200, "s": 0, "n": 100}
    exporter.out_width_px = 200
    exporter.target_height_px = 100
    exporter.province = "Test"
    exporter.dpi = EXPORT_DPI

    path = exporter._save_vector_boundary_layer(tmp_path, "20260101_120000")

    with Image.open(path) as image:
        assert image.size == (200, 100)
        # The shared edge between the two source polygons should be visible.
        assert any(image.getpixel((100, y))[3] > 0 for y in range(10, 90))


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__]))
