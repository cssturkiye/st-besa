"""
Focused unit tests for the export layer.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from stbesa.exporter import STBESAExporter


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


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__]))
