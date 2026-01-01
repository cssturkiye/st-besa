"""
Script to inspect the administrative levels of a Kontur/OSM GeoPackage.

This script loads the specified GPKG file, identifies all unique 'admin_level' values,
and prints a summary for each level, including the total count of features and 
a sample of names. This helps in configuring the 'datasets.json' file correctly.

Usage:
    python tests/inspect_gpkg.py <path_to_gpkg>
"""

import geopandas as gpd
import pandas as pd
import sys
import os

def inspect_gpkg(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    print(f"--- Inspecting: {os.path.basename(file_path)} ---")
    try:
        gdf = gpd.read_file(file_path)
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    print("Columns found:", list(gdf.columns))

    if 'admin_level' not in gdf.columns:
        print("Error: 'admin_level' column not found in dataset.")
        print("Available columns:", list(gdf.columns))
        return

    # Convert to numeric, handling NaNs if necessary
    levels = sorted(gdf['admin_level'].dropna().astype(int).unique())
    
    print(f"Found {len(levels)} unique admin levels: {levels}\n")

    # Helper to get name
    def get_name(row):
        nm = ""
        if 'name_en' in row and pd.notna(row['name_en']): nm = row['name_en']
        elif 'name' in row and pd.notna(row['name']): nm = row['name']
        return nm

    for level in levels:
        subset = gdf[gdf['admin_level'].astype(int) == level]
        count = len(subset)
        
        print(f"Level {level} (Total: {count}):")
        # Sort for readability
        names = sorted([get_name(row) for _, row in subset.iterrows()])
        
        # Show first 10 names for all levels
        if count <= 10:
            print(f"  Names: {', '.join(names)}")
        else:
            print(f"  First 10 Names: {', '.join(names[:10])}")
            print(f"  ... and {count - 10} more")

        print("-" * 30)

    # SPATIAL CONTAINMENT CHECK
    # Automatically check containment between adjacent admin levels
    
    print("\n--- Spatial Containment Analysis ---")
    print("Checking which levels are spatially contained within others...\n")
    
    # Check containment for common level pairs (potential ADM1 -> ADM2 relationships)
    level_pairs_to_check = []
    for i, lower_level in enumerate(levels):
        for higher_level in levels[i+1:]:
            # Check levels that are 1-4 steps apart (common administrative hierarchies)
            if 1 <= (higher_level - lower_level) <= 4:
                level_pairs_to_check.append((lower_level, higher_level))
    
    for parent_level, child_level in level_pairs_to_check[:5]:  # Limit to first 5 pairs
        parents = gdf[gdf['admin_level'] == parent_level]
        children = gdf[gdf['admin_level'] == child_level]
        
        if parents.empty or children.empty:
            continue
            
        print(f"Checking: Level {child_level} ({len(children)} units) within Level {parent_level} ({len(parents)} units)?")
        
        # Sample check (first 3 children)
        sample_children = children.head(3)
        
        for _, child in sample_children.iterrows():
            child_name = get_name(child)
            # Find parent containing this child
            containing = parents[parents.geometry.contains(child.geometry.centroid)]
            
            if not containing.empty:
                parent_name = get_name(containing.iloc[0])
                print(f"  ✓ '{child_name}' is inside '{parent_name}'")
            else:
                print(f"  ✗ '{child_name}' is NOT inside any Level {parent_level} unit")
        
        print()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tests/inspect_gpkg.py <path_to_gpkg>")
        print("\nExample:")
        print("  python tests/inspect_gpkg.py boundaries/kontur_boundaries_DE_20230628.gpkg")
    else:
        inspect_gpkg(sys.argv[1])
