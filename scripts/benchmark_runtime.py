from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from unidecode import unidecode

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stbesa.analysis import STBESAAnalysis
from stbesa.service import STBESAService


YEARS_ALL = list(range(1975, 2035, 5))
REQUEST_DELAY_SECONDS = 0.2


@dataclass(frozen=True)
class RegionSpec:
    scale_label: str
    region_label: str
    dataset_code: str
    province_name: str
    unit_type: str
    notes: str


DEFAULT_REGIONS = [
    RegionSpec(
        scale_label="Small",
        region_label="Berlin",
        dataset_code="DEU",
        province_name="Berlin",
        unit_type="State / City",
        notes="Compact city-state boundary",
    ),
    RegionSpec(
        scale_label="Medium",
        region_label="Izmir Province",
        dataset_code="TUR",
        province_name="Izmir",
        unit_type="Province",
        notes="Mixed metropolitan core with peripheral and rural districts",
    ),
    RegionSpec(
        scale_label="Large",
        region_label="Konya Province",
        dataset_code="TUR",
        province_name="Konya",
        unit_type="Province",
        notes="Large-area provincial benchmark",
    ),
]


def normalize_name(value: str) -> str:
    return unidecode(value).strip().lower()


def resolve_province_name(service: STBESAService, requested_name: str) -> str:
    requested_norm = normalize_name(requested_name)
    provinces = service.provinces()
    exact_matches = [name for name in provinces if normalize_name(name) == requested_norm]
    if exact_matches:
        return exact_matches[0]

    contains_matches = [name for name in provinces if requested_norm in normalize_name(name)]
    if contains_matches:
        return contains_matches[0]

    preview = ", ".join(provinces[:10])
    raise ValueError(f"Province '{requested_name}' not found. Sample available names: {preview}")


def compute_area_km2(geom_gdf) -> float:
    return float(geom_gdf.to_crs(6933).geometry.area.sum() / 1_000_000.0)


def quota_like_error(message: str) -> bool:
    msg = message.lower()
    needles = ("429", "rate", "rateexceeded", "quota", "too many requests", "resourceexhausted")
    return any(token in msg for token in needles)


def build_region_geometry(service: STBESAService, region: RegionSpec) -> tuple[Any, float, str]:
    service.load_dataset(region.dataset_code)
    resolved_name = resolve_province_name(service, region.province_name)
    rows = service.get_rows_by_province(resolved_name)
    geom_gdf = rows.dissolve()
    area_km2 = compute_area_km2(geom_gdf)
    return geom_gdf, area_km2, resolved_name


def run_single_analysis(analysis: STBESAAnalysis, geom_gdf, workers: int) -> float:
    start = time.perf_counter()
    _, ee_geom = analysis.geopandas_row_to_ee(geom_gdf)

    def process_year(year: int) -> None:
        analysis.compute_indicators(ee_geom, year)
        time.sleep(REQUEST_DELAY_SECONDS)
        analysis.compute_smod_statistics(ee_geom, year, level="L1", delay_seconds=REQUEST_DELAY_SECONDS)
        time.sleep(REQUEST_DELAY_SECONDS)
        analysis.compute_smod_statistics(ee_geom, year, level="L2", delay_seconds=REQUEST_DELAY_SECONDS)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process_year, year) for year in YEARS_ALL]
        for future in concurrent.futures.as_completed(futures):
            future.result()

    return time.perf_counter() - start


def summarize_timings(values: list[float]) -> tuple[float, float, float]:
    return statistics.median(values), min(values), max(values)


def benchmark_region(
    service: STBESAService,
    analysis: STBESAAnalysis,
    region: RegionSpec,
    workers: int,
    repeats: int,
) -> dict[str, Any]:
    geom_gdf, area_km2, resolved_name = build_region_geometry(service, region)
    timings: list[float] = []
    failures: list[str] = []

    for run_index in range(1, repeats + 1):
        try:
            elapsed = run_single_analysis(analysis, geom_gdf, workers)
            timings.append(elapsed)
            print(
                f"[OK] {region.region_label} run {run_index}/{repeats}: {elapsed:.1f}s",
                flush=True,
            )
        except Exception as exc:  # pragma: no cover - runtime script
            message = str(exc)
            failures.append(message)
            print(
                f"[FAIL] {region.region_label} run {run_index}/{repeats}: {message}",
                flush=True,
            )
            break

    result: dict[str, Any] = {
        "scale_label": region.scale_label,
        "region_label": region.region_label,
        "dataset_code": region.dataset_code,
        "province_name": resolved_name,
        "unit_type": region.unit_type,
        "area_km2": round(area_km2, 1),
        "years_computed": len(YEARS_ALL),
        "workers": workers,
        "notes": region.notes,
        "timings_seconds": [round(value, 2) for value in timings],
        "status": "ok" if len(timings) == repeats else "failed",
    }

    if timings:
        median_sec, min_sec, max_sec = summarize_timings(timings)
        result["median_seconds"] = round(median_sec, 2)
        result["runtime_range_seconds"] = [round(min_sec, 2), round(max_sec, 2)]

    if failures:
        result["failures"] = failures
        result["quota_like_failure"] = any(quota_like_error(msg) for msg in failures)

    return result


def write_json(results: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")


def print_summary(results: dict[str, Any]) -> None:
    print("\nBenchmark summary", flush=True)
    for item in results["regions"]:
        status = item["status"]
        if status == "ok":
            lo, hi = item["runtime_range_seconds"]
            print(
                f"- {item['scale_label']}: {item['region_label']} | "
                f"workers={item['workers']} | median={item['median_seconds']}s | "
                f"range={lo}-{hi}s | area={item['area_km2']} km^2",
                flush=True,
            )
        else:
            print(
                f"- {item['scale_label']}: {item['region_label']} | workers={item['workers']} | FAILED",
                flush=True,
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark ST-BESA analysis runtimes for small/medium/large regions.")
    parser.add_argument("--project-id", required=True, help="Google Cloud project id with Earth Engine API enabled.")
    parser.add_argument("--workers", type=int, default=5, help="Parallel worker count. Default: 5.")
    parser.add_argument("--repeats", type=int, default=3, help="Number of benchmark repeats per region. Default: 3.")
    parser.add_argument(
        "--output-json",
        default=str(PROJECT_ROOT / "exports" / "benchmark_runtime_results.json"),
        help="Where to write benchmark results as JSON.",
    )
    args = parser.parse_args()

    service = STBESAService(registry_path=str(PROJECT_ROOT / "datasets.json"))
    analysis = STBESAAnalysis(args.project_id)
    analysis.initialize_ee()

    results: dict[str, Any] = {
        "project_id": args.project_id,
        "workers": args.workers,
        "repeats": args.repeats,
        "years": YEARS_ALL,
        "request_delay_seconds": REQUEST_DELAY_SECONDS,
        "regions": [],
    }

    for region in DEFAULT_REGIONS:
        results["regions"].append(
            benchmark_region(
                service=service,
                analysis=analysis,
                region=region,
                workers=args.workers,
                repeats=args.repeats,
            )
        )

    write_json(results, Path(args.output_json))
    print_summary(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
