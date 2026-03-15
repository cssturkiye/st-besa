"""
Extract a single administrative unit from a local GHS-DUC R2023A package
across all available epochs.

Usage:
    python tests/extract_ghs_duc_timeseries.py <package_dir> <gid> <level>

Example:
    python tests/extract_ghs_duc_timeseries.py GHS_DUC_MT_GLOBE_R2023A_V2_0 DEU.3_1 1
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("package_dir", help="Path to GHS_DUC_MT_GLOBE_R2023A_V2_0")
    parser.add_argument("gid", help="Administrative unit ID, e.g. DEU.3_1")
    parser.add_argument("level", type=int, help="GADM level, e.g. 1")
    args = parser.parse_args()

    package_dir = Path(args.package_dir)
    gid_col = f"GID_{args.level}"
    files = sorted(package_dir.glob(f"GHS_DUC_GLOBE_R2023A_V2_0_GADM41_*_level{args.level}.csv"))
    if not files:
        print(f"No level {args.level} files found in {package_dir}")
        return 1

    print(
        "Year,"
        "Tot_Pop,"
        "UCentre_Pop,"
        "UCluster_Pop,"
        "Rural_Pop,"
        "UCentre_share,"
        "UCluster_share,"
        "Rural_share,"
        "DEGURBA_L1,"
        "DEGURBA_L2"
    )

    found_any = False
    for csv_path in files:
        year = int(csv_path.stem.split("_GADM41_")[1].split("_level")[0])
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if row.get(gid_col) == args.gid:
                    found_any = True
                    print(
                        ",".join(
                            [
                                str(year),
                                row["Tot_Pop"],
                                row["UCentre_Pop"],
                                row["UCluster_Pop"],
                                row["Rural_Pop"],
                                row["UCentre_share"],
                                row["UCluster_share"],
                                row["Rural_share"],
                                row["DEGURBA_L1"],
                                row["DEGURBA_L2"],
                            ]
                        )
                    )
                    break

    if not found_any:
        print(f"No rows found for {args.gid} at level {args.level}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
