# stbesa/gradio_app.py
"""
ST-BESA Gradio Application - Feature-Complete Implementation
Includes: Multi-year analysis, map data layers, time-series plots, SMOD L1/L2
"""

import gradio as gr
import pandas as pd
import geopandas as gpd
import numpy as np
import io
import time
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from stbesa.service import STBESAService
from stbesa.analysis import STBESAAnalysis
from stbesa.exporter import STBESAExporter

# SMOD Classification Metadata (from backup)
SMOD_L1_CLASSES = {
    3: {"name": "URBAN CENTRE", "color": "#FF0000", "rgb": (255, 0, 0), "desc": "Cities (Densely populated area)"},
    2: {"name": "URBAN CLUSTER", "color": "#FFAA00", "rgb": (255, 170, 0), "desc": "Towns & semi-dense area"},
    1: {"name": "RURAL", "color": "#73B273", "rgb": (115, 178, 115), "desc": "Rural areas (Thinly populated area)"},
}

SMOD_L2_CLASSES = {
    30: {"name": "Urban Centre", "l1": 3},
    23: {"name": "Dense Urban Cluster", "l1": 2},
    22: {"name": "Semi-dense Urban Cluster", "l1": 2},
    21: {"name": "Suburban or peri-urban", "l1": 2},
    13: {"name": "Rural Cluster", "l1": 1},
    12: {"name": "Low density rural", "l1": 1},
    11: {"name": "Very low density rural", "l1": 1},
    10: {"name": "Water or no data", "l1": 1},
}

L2_COLOR_MAP = {
    10: '#7AB6F5', 11: '#CDF57A', 12: '#ABCD66', 13: '#375623',
    21: '#FFFF00', 22: '#A87000', 23: '#732600', 30: '#FF0000'
}

TURBO_PALETTE = ['#30123b','#4145ab','#2e7de0','#1db6d7','#27d39b','#7be151','#c0e620','#f9e14b','#fca72c','#f96814','#a31e1e']

YEARS_ALL = list(range(1975, 2035, 5))  # 12 years: 1975, 1980, ..., 2030

# Global service instances
service = STBESAService()
analysis_service: STBESAAnalysis = None


def init_analysis_service(project_id: str):
    """Initialize or reinitialize the analysis service with the given project ID."""
    global analysis_service
    if project_id and project_id.strip():
        analysis_service = STBESAAnalysis(project_id.strip())
    else:
        analysis_service = None

def get_dataset_choices():
    """Returns list of (label, value) tuples for Gradio dropdown."""
    choices = []
    for code, conf in service.registry.items():
        label = conf.get('name', code)
        choices.append((label, code))
    return choices

def on_dataset_change(code, progress=gr.Progress()):
    """Load dataset and return updated province choices and labels."""
    try:
        if not code:
            return gr.update(choices=[], value=None), gr.update(label="Region"), gr.update(label="Sub-Region")
        
        # Wrapper to adapt progress callback to Gradio's progress object
        def progress_wrapper(p, msg):
            progress(p, desc=msg)

        service.load_dataset(code, progress_callback=progress_wrapper)
        provs = service.provinces()
        label1 = service.adm1_label
        label2 = service.adm2_label
        
        return (
            gr.update(choices=provs, value=None, label=label1),
            gr.update(label=label1),
            gr.update(label=label2)
        )
    except Exception as e:
        raise gr.Error(f"Error loading dataset: {str(e)}")

def on_province_change(prov):
    """Return updated district choices for the selected province."""
    if not prov:
        return gr.update(choices=[], value=[])
    try:
        dists = service.districts(prov)
        dists.insert(0, "ALL")
        return gr.update(choices=dists, value=[])
    except Exception as e:
        raise gr.Error(f"Error loading districts: {str(e)}")

def _render_plots_l1(df_overall: pd.DataFrame, df_l1: pd.DataFrame, current_year: int):
    """Render 3x2 time-series plots for L1 classes + total (Exact replica of original logic)."""
    if df_overall is None or df_overall.empty:
        return None
    if df_l1 is None or df_l1.empty:
        return None
    
    total_df = df_overall[["yil", "buvol_m3", "buvol_sur_m2", "pop_person", "bvpc_m3_per_person", "bspc_m2_per_person", "vol_sur_ratio"]].sort_values("yil").reset_index(drop=True)
    years = total_df["yil"].values
    
    l1 = df_l1[["yil", "smod_l1_code", "buvol_m3", "buvol_sur_m2", "pop_person", "bvpc_m3_per_person", "bspc_m2_per_person", "vol_sur_ratio"]].copy()
    piv = {metric: l1.pivot_table(index="yil", columns="smod_l1_code", values=metric, aggfunc="first").reindex(years).fillna(np.nan) 
           for metric in ["buvol_m3", "buvol_sur_m2", "pop_person", "bvpc_m3_per_person", "bspc_m2_per_person", "vol_sur_ratio"]}
    
    mpl.rcParams.update({
        'figure.dpi': 300, 'savefig.dpi': 300, 'axes.titlesize': 7, 'axes.labelsize': 6,
        'xtick.labelsize': 5, 'ytick.labelsize': 5, 'legend.fontsize': 5,
        'axes.grid': True, 'grid.linestyle': ':', 'grid.alpha': 0.5
    })
    
    fig, axes = plt.subplots(3, 2, figsize=(6.85, 5.33), constrained_layout=False)
    axes = axes.ravel()
    fig.patch.set_alpha(0.0)
    
    total_color = "#1f77b4"
    l1_colors = {1: SMOD_L1_CLASSES[1]["color"], 2: SMOD_L1_CLASSES[2]["color"], 3: SMOD_L1_CLASSES[3]["color"]}
    
    series = [
        ("buvol_m3", "Building Volume (m³)"),
        ("buvol_sur_m2", "Building Surface (m²)"),
        ("pop_person", "Population (people)"),
        ("bvpc_m3_per_person", "BVPC (m³/person)"),
        ("bspc_m2_per_person", "BSPC (m²/person)"),
        ("vol_sur_ratio", "Volume/Surface Ratio")
    ]
    
    for i, (key, label) in enumerate(series):
        ax = axes[i]
        ax.plot(years, total_df[key].values, color=total_color, linewidth=1.6, label="Total")
        ax.scatter(years, total_df[key].values, s=6, color=total_color, edgecolor=total_color, facecolor="white", zorder=3)
        for cls in [1, 2, 3]:
            if cls in piv[key].columns:
                ax.plot(years, piv[key][cls].values, color=l1_colors[cls], linewidth=1.0, label=str(cls))
        ax.set_title(label)
        ax.set_xlabel("Year")
        ax.set_ylabel(label)
        ax.margins(x=0.02)
        ax.set_xticks(years)
        ax.tick_params(axis='x', rotation=0)
        try:
            ax.axvline(current_year, color="#d62728", linestyle="--", linewidth=1.0, alpha=0.8)
        except Exception:
            pass
    
    fig.suptitle("L1 Classes + Total", fontsize=7, fontweight='semibold')
    
    # ADJUSTMENT FOR SCREEN: Increase bottom margin and move legend INSIDE canvas (y > 0)
    # Original export used y=-0.03 which is clipped on screen.
    fig.subplots_adjust(bottom=0.18, hspace=0.50)
    
    handles = [Line2D([0], [0], color=total_color, lw=3, label='Total')]
    labels = ['Total']
    for cls in [1, 2, 3]:
        handles.append(Line2D([0], [0], color=l1_colors[cls], lw=3))
        labels.append({1: 'Rural', 2: 'Urban Cluster', 3: 'Urban Centre'}[cls])
    
    # Legend at y=0.02 ensures it is within the figure bounds
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=4, 
               prop={'size': 5}, handlelength=2, columnspacing=1.0, frameon=True, 
               fancybox=True, shadow=True, edgecolor='black', facecolor='white', framealpha=0.9)
    
    return fig


def _render_plots_l2(df_overall: pd.DataFrame, df_l2: pd.DataFrame, current_year: int):
    """Render 3x2 time-series plots for L2 classes (8) + total (Exact replica of original logic)."""
    if df_overall is None or df_overall.empty:
        return None
    if df_l2 is None or df_l2.empty:
        return None
    
    total_df = df_overall[["yil", "buvol_m3", "buvol_sur_m2", "pop_person", "bvpc_m3_per_person", "bspc_m2_per_person", "vol_sur_ratio"]].sort_values("yil").reset_index(drop=True)
    years = total_df["yil"].values
    
    l2 = df_l2[["yil", "smod_l2_code", "buvol_m3", "buvol_sur_m2", "pop_person", "bvpc_m3_per_person", "bspc_m2_per_person", "vol_sur_ratio"]].copy()
    piv = {metric: l2.pivot_table(index="yil", columns="smod_l2_code", values=metric, aggfunc="first").reindex(years).fillna(np.nan) 
           for metric in ["buvol_m3", "buvol_sur_m2", "pop_person", "bvpc_m3_per_person", "bspc_m2_per_person", "vol_sur_ratio"]}
    
    mpl.rcParams.update({
        'figure.dpi': 300, 'savefig.dpi': 300, 'axes.titlesize': 7, 'axes.labelsize': 6,
        'xtick.labelsize': 5, 'ytick.labelsize': 5, 'legend.fontsize': 5,
        'axes.grid': True, 'grid.linestyle': ':', 'grid.alpha': 0.5
    })
    
    fig, axes = plt.subplots(3, 2, figsize=(6.85, 5.33), constrained_layout=False)
    axes = axes.ravel()
    fig.patch.set_alpha(0.0)
    
    total_color = "#1f77b4"
    l2_color_map = {10: '#7AB6F5', 11: '#CDF57A', 12: '#ABCD66', 13: '#375623', 
                    21: '#FFFF00', 22: '#A87000', 23: '#732600', 30: '#FF0000'}
    
    series = [
        ("buvol_m3", "Building Volume (m³)"),
        ("buvol_sur_m2", "Building Surface (m²)"),
        ("pop_person", "Population (people)"),
        ("bvpc_m3_per_person", "BVPC (m³/person)"),
        ("bspc_m2_per_person", "BSPC (m²/person)"),
        ("vol_sur_ratio", "Volume/Surface Ratio")
    ]
    
    for i, (key, label) in enumerate(series):
        ax = axes[i]
        ax.plot(years, total_df[key].values, color=total_color, linewidth=1.6, label="Total")
        ax.scatter(years, total_df[key].values, s=6, color=total_color, edgecolor=total_color, facecolor="white", zorder=3)
        for cls in [10, 11, 12, 13, 21, 22, 23, 30]:
            if cls in piv[key].columns:
                ax.plot(years, piv[key][cls].values, color=l2_color_map[cls], linewidth=0.9, label=str(cls))
        ax.set_title(label)
        ax.set_xlabel("Year")
        ax.set_ylabel(label)
        ax.margins(x=0.02)
        ax.set_xticks(years)
        ax.tick_params(axis='x', rotation=0)
        try:
            ax.axvline(current_year, color="#d62728", linestyle="--", linewidth=1.0, alpha=0.8)
        except Exception:
            pass
    
    fig.suptitle("L2 Classes + Total", fontsize=7, fontweight='semibold')
    
    # ADJUSTMENT FOR SCREEN: Increase bottom margin and move legend INSIDE canvas
    fig.subplots_adjust(bottom=0.18, hspace=0.50)
    
    handles = [Line2D([0], [0], color=total_color, lw=3, label='Total')]
    labels = ['Total']
    l2_order = [10, 11, 12, 13, 21, 22, 23, 30]
    l2_label_map = {
        10: '10 Water/No data', 11: '11 Very low density rural', 12: '12 Low density rural',
        13: '13 Rural cluster', 21: '21 Suburban/peri-urban', 22: '22 Semi-dense urban',
        23: '23 Dense urban cluster', 30: '30 Urban centre'
    }
    for cls in l2_order:
        handles.append(Line2D([0], [0], color=l2_color_map[cls], lw=3))
        labels.append(l2_label_map[cls])
    
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=4, 
               prop={'size': 5}, handlelength=2, columnspacing=1.0, frameon=True, 
               fancybox=True, shadow=True, edgecolor='black', facecolor='white', framealpha=0.9)
    
    return fig


def _generate_map_html(analysis_svc, ee_geom, year: int, vis_params: dict) -> str:
    """Generate interactive map with data layers and legends."""
    import ee
    
    m = analysis_svc.geemap_map(height="700px")
    
    try:
        # Load EE images
        vol = ee.Image(f"JRC/GHSL/P2023A/GHS_BUILT_V/{year}").select("built_volume_total").clip(ee_geom)
        sur = ee.Image(f"JRC/GHSL/P2023A/GHS_BUILT_S/{year}").select("built_surface").clip(ee_geom)
        pop = ee.Image(f"JRC/GHSL/P2023A/GHS_POP/{year}").select("population_count").clip(ee_geom)
        smod = ee.Image(f"JRC/GHSL/P2023A/GHS_SMOD/{year}").select("smod_code").clip(ee_geom)
        
        # Get vis ranges (use defaults if not computed)
        vmin_vol, vmax_vol = vis_params.get("vol", (0, 80000))
        vmin_sur, vmax_sur = vis_params.get("sur", (0, 20000))
        vmin_pop, vmax_pop = vis_params.get("pop", (0, 500))
        
        # SMOD L2 layer
        smod_mask = smod.neq(0)
        smod_l2_vis = smod.updateMask(smod_mask).remap(
            [10, 11, 12, 13, 21, 22, 23, 30], [0, 1, 2, 3, 4, 5, 6, 7]
        ).visualize(min=0, max=7, palette=list(L2_COLOR_MAP.values()))
        m.addLayer(smod_l2_vis, {}, f"SMOD L2 {year}", shown=True)
        
        # SMOD L1 layer
        smod_l1 = smod.divide(10).floor()
        l1_palette = [SMOD_L1_CLASSES[1]["color"], SMOD_L1_CLASSES[2]["color"], SMOD_L1_CLASSES[3]["color"]]
        smod_l1_vis = smod_l1.updateMask(smod_mask).visualize(min=1, max=3, palette=l1_palette)
        m.addLayer(smod_l1_vis, {}, f"SMOD L1 {year}", shown=False)
        
        # Population
        m.addLayer(pop.updateMask(pop.gt(0)), 
                  {"min": vmin_pop, "max": vmax_pop, "palette": TURBO_PALETTE}, 
                  f"Population {year}", shown=False)
        
        # Building Surface
        m.addLayer(sur.updateMask(sur.gt(0)), 
                  {"min": vmin_sur, "max": vmax_sur, "palette": TURBO_PALETTE}, 
                  f"Building Surface {year}", shown=False)
        
        # Building Volume
        m.addLayer(vol.updateMask(vol.gt(0)), 
                  {"min": vmin_vol, "max": vmax_vol, "palette": TURBO_PALETTE}, 
                  f"Building Volume {year}", shown=False)
        
        # Boundary
        outline = ee.Image().byte().paint(ee.FeatureCollection([ee.Feature(ee_geom)]), 1, 2)
        m.addLayer(outline.visualize(min=0, max=1, palette=['000000']), {}, "Boundary")
        
        m.centerObject(ee_geom, 9)
        m.addLayerControl()
        
    except Exception as e:
        print(f"[WARN] Map layer error: {e}")
        m.centerObject(ee_geom, 9)
    
    # Generate HTML with embedded legends
    base_html = m._repr_html_()
    
    # Build legend HTML - use absolute positioning within the map wrapper
    def make_gradient_legend(label, colors, vmin, vmax, right, bottom):
        gradient = ', '.join([f'{c} {i/(len(colors)-1)*100:.0f}%' for i, c in enumerate(colors)])
        return f'''
        <div style="position:absolute; bottom:{bottom}px; right:{right}px; z-index:1000; 
                    padding:6px 10px; background:rgba(255,255,255,0.92); border:1px solid #999; 
                    border-radius:4px; font-family:Arial,sans-serif; box-shadow:0 2px 6px rgba(0,0,0,0.2);">
          <div style="font-weight:600; font-size:11px; color:#333; margin-bottom:4px;">{label}</div>
          <div style="background:linear-gradient(to right, {gradient}); height:12px; width:150px; 
                      border:1px solid #666;"></div>
          <div style="display:flex; justify-content:space-between; font-size:9px; color:#333; margin-top:2px;">
            <span>{vmin:.0f}</span><span>{vmax:.0f}</span>
          </div>
        </div>'''
    
    def make_categorical_legend(title, items, left, bottom):
        rows = ''.join([f'''<div style="display:flex;align-items:center;margin:2px 0;">
            <span style="display:inline-block;width:14px;height:10px;background:{c};border:1px solid #666;margin-right:6px;"></span>
            <span style="font-size:10px;color:#333;">{lbl}</span></div>''' for lbl, c in items])
        return f'''
        <div style="position:absolute; bottom:{bottom}px; left:{left}px; z-index:1000;
                    padding:6px 10px; background:rgba(255,255,255,0.92); border:1px solid #999;
                    border-radius:4px; font-family:Arial,sans-serif; box-shadow:0 2px 6px rgba(0,0,0,0.2);">
          <div style="font-weight:600; font-size:11px; color:#333; margin-bottom:4px;">{title}</div>
          {rows}
        </div>'''
    
    # Gradient legends (right side, stacked) - Increased spacing
    legends_html = make_gradient_legend("Building Volume (m³)", TURBO_PALETTE, vmin_vol, vmax_vol, 15, 160)
    legends_html += make_gradient_legend("Building Surface (m²)", TURBO_PALETTE, vmin_sur, vmax_sur, 15, 95)
    legends_html += make_gradient_legend("Population", TURBO_PALETTE, vmin_pop, vmax_pop, 15, 30)
    
    # Categorical legends (left side) - Increased spacing
    l1_items = [('Rural', SMOD_L1_CLASSES[1]['color']), 
                ('Urban Cluster', SMOD_L1_CLASSES[2]['color']), 
                ('Urban Centre', SMOD_L1_CLASSES[3]['color'])]
    legends_html += make_categorical_legend("SMOD L1", l1_items, 15, 200)
    
    l2_items = [('10 Water/No data', '#7AB6F5'), ('11 Very low rural', '#CDF57A'),
                ('12 Low rural', '#ABCD66'), ('13 Rural cluster', '#375623'),
                ('21 Suburban', '#FFFF00'), ('22 Semi-dense', '#A87000'),
                ('23 Dense urban', '#732600'), ('30 Urban centre', '#FF0000')]
    legends_html += make_categorical_legend("SMOD L2", l2_items, 15, 30)
    
    # Wrap the map in a positioned container and add legends inside
    # Ensure map iframe fills height
    # Ensure map iframe fills height (handle various formatting and sizes)
    base_html = base_html.replace('height:700px', 'height:100%').replace('height: 700px', 'height:100%') \
                         .replace('height:600px', 'height:100%').replace('height: 600px', 'height:100%') \
                         .replace('height:650px', 'height:100%').replace('height: 650px', 'height:100%')
    
    wrapped_html = f'''
    <div style="position:relative; width:100%; height:700px;">
        <div style="position:absolute; top:0; left:0; right:0; bottom:0; height:100% !important;">
            {base_html}
        </div>
        {legends_html}
    </div>'''
    
    return wrapped_html


def normalize_layer(state, year, layer):
    """Calculate dynamic stretch for a single layer."""
    if not state or not state.get("geom"):
        raise gr.Error("No analysis results. Run analysis first.")
    
    ee_geom = state["geom"]
    import ee
    
    try:
        if layer == 'vol':
            img = ee.Image(f"JRC/GHSL/P2023A/GHS_BUILT_V/{year}").select("built_volume_total").clip(ee_geom)
            v_min, v_max = analysis_service.dynamic_stretch(img, "built_volume_total", ee_geom, 80000)
            return v_min, v_max
        elif layer == 'sur':
            img = ee.Image(f"JRC/GHSL/P2023A/GHS_BUILT_S/{year}").select("built_surface").clip(ee_geom)
            s_min, s_max = analysis_service.dynamic_stretch(img, "built_surface", ee_geom, 20000)
            return s_min, s_max
        elif layer == 'pop':
            img = ee.Image(f"JRC/GHSL/P2023A/GHS_POP/{year}").select("population_count").clip(ee_geom)
            p_min, p_max = analysis_service.dynamic_stretch(img, "population_count", ee_geom, 500)
            return p_min, p_max
    except Exception as e:
        raise gr.Error(f"Normalization failed: {str(e)}")
    return 0, 100 


def refresh_visualization(state, year, vol_min, vol_max, sur_min, sur_max, pop_min, pop_max):
    """Update map with manual visualization parameters without re-running analysis."""
    if not state or not state.get("geom"):
        raise gr.Error("No analysis results. Please run analysis first.")
    
    # Update vis_params in state
    new_vis_params = {
        "vol": (vol_min, vol_max),
        "sur": (sur_min, sur_max), 
        "pop": (pop_min, pop_max)
    }
    state["vis_params"] = new_vis_params
    
    # Render Map
    try:
        map_html = _generate_map_html(analysis_service, state["geom"], year, new_vis_params)
    except Exception as e:
        map_html = f"<div style='padding:20px;color:red;'>Map update failed: {str(e)}</div>"
        
    return map_html, state


def run_analysis(project_id, dataset_code, province, districts, year: int, max_workers: int = 5, auto_scale: bool = True, vol_min=0, vol_max=80000, sur_min=0, sur_max=20000, pop_min=0, pop_max=500):
    """Main analysis workflow with custom blocking modal."""
    
    # Helper to generate modal HTML
    def get_modal_html(title, text):
        return f"""
        <div id="modal-overlay">
            <div id="modal-content">
                <div class="spinner"></div>
                <div class="modal-title">{title}</div>
                <div class="modal-text">{text}</div>
            </div>
        </div>
        """
    
    # Initial Verification & Yield
    try:
        if not dataset_code or not province:
            raise gr.Error("Please select a dataset and a region.")
        
        if not project_id or not project_id.strip():
            raise gr.Error("Please enter your Google Cloud Project ID.")
        
        # Start specific init steps
        yield (get_modal_html("Initializing...", "Connecting to Earth Engine..."), *[gr.update()] * 18)
        
        init_analysis_service(project_id)
        
        try:
            analysis_service.initialize_ee()
        except Exception as e:
            raise gr.Error(f"Earth Engine Init Failed: {str(e)}")
            
        yield (get_modal_html("Preparing Geometry...", f"Processing {province}..."), *[gr.update()] * 18)
    
        # Get Geometry
        if not districts or "ALL" in districts:
            row = service.get_rows_by_province(province)
            geom_gdf = row.dissolve()
            analysis_title = f"{province} (All)"
        else:
            s = service.active_gdf
            rows = s[(s["NAME_1"] == province) & (s["NAME_2"].isin(districts))]
            geom_gdf = rows.dissolve()
            analysis_title = f"{province} - {', '.join(districts)}"
            
        ee_feat, ee_geom = analysis_service.geopandas_row_to_ee(geom_gdf)
        
    except Exception as e:
        # On error, hide modal and raise
        yield ("", *[gr.update()] * 18)
        raise gr.Error(f"Setup Failed: {str(e)}")

    # Multi-Year Analysis Loop
    yield (get_modal_html("Computing Statistics...", f"Starting analysis with {max_workers} workers..."), *[gr.update()] * 18)
    
    overall_rows = []
    smod_l1_rows = []
    smod_l2_rows = []
    
    import concurrent.futures
    import threading
    
    # Use user-configured max workers
    request_delay = 0.2  # Delay between batches
    lock = threading.Lock()
    
    def process_year(yr):
        """Process a single year - returns (overall_row, l1_rows, l2_rows) or None on error."""
        try:
            # Compute overall indicators
            indicators = analysis_service.compute_indicators(ee_geom, yr)
            
            overall_row = {
                'il': province,
                'ilce_idx': ','.join(districts) if districts else "ALL",
                'yil': yr,
                'buvol_m3': indicators.get("sum_volume_m3", 0),
                'buvol_sur_m2': indicators.get("sum_surface_m2", 0),
                'pop_person': indicators.get("sum_population", 0),
                'bvpc_m3_per_person': indicators.get("bvpc_m3_per_person"),
                'bspc_m2_per_person': indicators.get("bspc_m2_per_person"),
                'vol_sur_ratio': indicators.get("vol_sur_ratio")
            }
            
            time.sleep(request_delay)  # Small delay between metric types
            
            # Compute SMOD L1
            l1_rows = []
            stats_l1 = analysis_service.compute_smod_statistics(ee_geom, yr, level="L1", delay_seconds=request_delay)
            for cls_code, stats in stats_l1.items():
                if cls_code in SMOD_L1_CLASSES:
                    l1_rows.append({
                        'il': province, 'yil': yr,
                        'smod_l1_code': cls_code,
                        'smod_l1_name': SMOD_L1_CLASSES[cls_code]['name'],
                        **stats
                    })
            
            time.sleep(request_delay)
            
            # Compute SMOD L2
            l2_rows = []
            stats_l2 = analysis_service.compute_smod_statistics(ee_geom, yr, level="L2", delay_seconds=request_delay)
            for cls_code, stats in stats_l2.items():
                if cls_code in SMOD_L2_CLASSES:
                    l2_rows.append({
                        'il': province, 'yil': yr,
                        'smod_l2_code': cls_code,
                        'smod_l2_name': SMOD_L2_CLASSES[cls_code]['name'],
                        'smod_l1_parent': SMOD_L2_CLASSES[cls_code]['l1'],
                        **stats
                    })
            
            return (overall_row, l1_rows, l2_rows)
            
        except Exception as e:
            print(f"[WARN] Year {yr} failed: {e}")
            return None
    
    # Process years in parallel with limited concurrency
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_year = {executor.submit(process_year, yr): yr for yr in YEARS_ALL}
        
        pending_count = len(YEARS_ALL)
        
        for future in concurrent.futures.as_completed(future_to_year):
            yr = future_to_year[future]
            completed += 1
            
            # Yield Progress Update
            percentage = int((completed / pending_count) * 100)
            yield (get_modal_html(f"Processing Data ({percentage}%)", f"Completed Year {yr}..."), *[gr.update()] * 18)
            
            result = future.result()
            if result:
                overall_row, l1_rows, l2_rows = result
                with lock:
                    overall_rows.append(overall_row)
                    smod_l1_rows.extend(l1_rows)
                    smod_l2_rows.extend(l2_rows)
    
    # Create DataFrames
    df_overall = pd.DataFrame(overall_rows)
    df_l1 = pd.DataFrame(smod_l1_rows)
    df_l2 = pd.DataFrame(smod_l2_rows)
    
    df_l2 = pd.DataFrame(smod_l2_rows)
    
    yield (get_modal_html("Generating Visuals...", "Creating Maps & Charts..."), *[gr.update()] * 18)
    
    # Compute visualization ranges
    vis_params = {"vol": (0, 80000), "sur": (0, 20000), "pop": (0, 500)}
    
    if auto_scale:
        yield (get_modal_html("Auto-Scaling...", "Calculating optimal color bounds..."), *[gr.update()] * 18)
        try:
            import ee
            # Dynamic stretch using 5th-99th percentile (matching backup logic)
            vol_img = ee.Image(f"JRC/GHSL/P2023A/GHS_BUILT_V/{year}").select("built_volume_total").clip(ee_geom)
            sur_img = ee.Image(f"JRC/GHSL/P2023A/GHS_BUILT_S/{year}").select("built_surface").clip(ee_geom)
            pop_img = ee.Image(f"JRC/GHSL/P2023A/GHS_POP/{year}").select("population_count").clip(ee_geom)
            
            v_min, v_max = analysis_service.dynamic_stretch(vol_img, "built_volume_total", ee_geom, 80000)
            s_min, s_max = analysis_service.dynamic_stretch(sur_img, "built_surface", ee_geom, 20000)
            p_min, p_max = analysis_service.dynamic_stretch(pop_img, "population_count", ee_geom, 500)
            
            vis_params = {
                "vol": (v_min, v_max),
                "sur": (s_min, s_max),
                "pop": (p_min, p_max)
            }
        except Exception as e:
            print(f"[WARN] Auto-scale failed, falling back to heuristic: {e}")
            # Fallback to simple heuristic if specialized stretch fails
            if year in df_overall["yil"].values:
                yr_data = df_overall[df_overall["yil"] == year].iloc[0]
                vol_max = max(1, yr_data.get("buvol_m3", 80000) * 0.1) 
                pop_max = max(1, yr_data.get("pop_person", 500) * 0.01)
                vis_params = {"vol": (0, min(vol_max, 100000)), "sur": (0, 30000), "pop": (0, min(pop_max, 1000))}
            if year in df_overall["yil"].values:
                yr_data = df_overall[df_overall["yil"] == year].iloc[0]
                vol_max = max(1, yr_data.get("buvol_m3", 80000) * 0.1) 
                pop_max = max(1, yr_data.get("pop_person", 500) * 0.01)
                vis_params = {"vol": (0, min(vol_max, 100000)), "sur": (0, 30000), "pop": (0, min(pop_max, 1000))}
    else:
        # User disabled auto-scale: use manual inputs
        vis_params = {
            "vol": (vol_min, vol_max),
            "sur": (sur_min, sur_max),
            "pop": (pop_min, pop_max)
        }
    
    # Generate Map
    yield (get_modal_html("Finalizing...", "Rendering map layers..."), *[gr.update()] * 18)
    try:
        map_html = _generate_map_html(analysis_service, ee_geom, year, vis_params)
    except Exception as e:
        map_html = f"<div style='padding:20px;color:red;'>Map generation failed: {str(e)}</div>"
    
    # Generate Time-Series Plots (L1 and L2 separately)
    yield (get_modal_html("Finalizing...", "Rendering time series charts..."), *[gr.update()] * 18)
    fig_l1 = _render_plots_l1(df_overall, df_l1, year)
    fig_l2 = _render_plots_l2(df_overall, df_l2, year)
    
    # Format Data Table - Show ALL YEARS with all columns (matching original)
    df_display = df_overall[["yil", "buvol_m3", "buvol_sur_m2", "pop_person", "bvpc_m3_per_person", "bspc_m2_per_person", "vol_sur_ratio"]].copy()
    df_display.columns = ["Year", "Building Volume (m³)", "Building Surface (m²)", "Population (people)", "BVPC (m³/person)", "BSPC (m²/person)", "Volume/Surface Ratio"]
    df_display = df_display.sort_values("Year").reset_index(drop=True)
    
    # Prepare State for Export
    meta = {
        "Dataset": dataset_code,
        "Province": province,
        "Districts": ','.join(districts) if districts else "ALL",
        "Year Range": f"{min(YEARS_ALL)}-{max(YEARS_ALL)}",
        "Selected Year": year,
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    results_state = {
        "overall": df_overall,
        "l1": df_l1,
        "l2": df_l2,
        "geom": ee_geom,
        "vis_params": vis_params,
        "meta": meta
    }

    v_min_up, v_max_up = vis_params["vol"]
    s_min_up, s_max_up = vis_params["sur"]
    p_min_up, p_max_up = vis_params["pop"]
    
    # Final Yield: Hide Modal ("") + Return all Real Data
    yield (
        "", # Hide Modal
        map_html, 
        gr.update(visible=True, value=df_display), gr.update(visible=False, value=""),
        gr.update(visible=True, value=fig_l1), gr.update(visible=False), gr.update(visible=True),
        gr.update(visible=True, value=fig_l2), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False, value=""), # Clear content explicitly
        f"✓ Analysis Complete: {analysis_title} ({len(YEARS_ALL)} years)", 
        results_state,
        gr.update(value=v_min_up), gr.update(value=v_max_up),
        gr.update(value=s_min_up), gr.update(value=s_max_up),
        gr.update(value=p_min_up), gr.update(value=p_max_up)
    )


def update_display(state, year):
    """Instant update of Map and Plots when Year Slider changes (no re-computation)."""
    if not state or not state.get("geom"):
        return gr.update(), gr.update(), gr.update(), state
    
    # Retrieve data from state
    ee_geom = state["geom"]
    vis_params = state["vis_params"]
    df_overall = state["overall"]
    df_l1 = state["l1"]
    df_l2 = state["l2"]
    
    # Render Map (Fast EE call just for tiles)
    try:
        map_html = _generate_map_html(analysis_service, ee_geom, year, vis_params)
    except Exception as e:
        map_html = f"<div style='padding:20px;color:red;'>Map update failed: {str(e)}</div>"
        
    # Render Plots (Instant local render)
    fig_l1 = _render_plots_l1(df_overall, df_l1, year)
    fig_l2 = _render_plots_l2(df_overall, df_l2, year)
    
    # Update state selected year for exports
    state["meta"]["Selected Year"] = year
    
    return map_html, fig_l1, fig_l2, state


def export_data(state):
    """Export all multi-year data to Excel."""
    if not state:
        raise gr.Error("No analysis results to export. Run analysis first.")
    
    meta = state["meta"]
    province = meta.get('Province', 'Unknown').replace(' ', '_')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"ST-BESA_{province}_{timestamp}.xlsx"
    
    exports_dir = os.path.join(os.getcwd(), "exports")
    os.makedirs(exports_dir, exist_ok=True)
    path = os.path.join(exports_dir, filename)
    
    STBESAExporter.export_excel_report(
        path,
        state["overall"],
        state["l1"],
        state["l2"],
        state["meta"]
    )
    gr.Info(f"✅ Excel Export Saved: {path}", duration=10)
    return path


def export_plots(state):
    """Export time-series plots as high-resolution PNGs (both L1 and L2)."""
    if not state:
        raise gr.Error("No analysis results to export. Run analysis first.")
    
    df_overall = state.get("overall")
    df_l1 = state.get("l1")
    df_l2 = state.get("l2")
    
    if df_overall is None or df_overall.empty:
        raise gr.Error("No data available for plotting.")
    
    meta = state["meta"]
    province = meta.get('Province', 'Unknown').replace(' ', '_')
    # Use selected year from state if updated via slider, else fallback
    current_year = meta.get('Selected Year', 2025)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Generate both plots
    fig_l1 = _render_plots_l1(df_overall, df_l1, current_year)
    fig_l2 = _render_plots_l2(df_overall, df_l2, current_year)
    
    exports_dir = os.path.join(os.getcwd(), "exports")
    os.makedirs(exports_dir, exist_ok=True)
    
    # Save L1 plot
    filename_l1 = f"ST-BESA_{province}_L1_plots_{timestamp}.png"
    path_l1 = os.path.join(exports_dir, filename_l1)
    if fig_l1:
        fig_l1.savefig(path_l1, format='png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig_l1)
    
    # Save L2 plot
    filename_l2 = f"ST-BESA_{province}_L2_plots_{timestamp}.png"
    path_l2 = os.path.join(exports_dir, filename_l2)
    if fig_l2:
        fig_l2.savefig(path_l2, format='png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig_l2)
    
    # Create ZIP with both plots
    import zipfile
    zip_filename = f"ST-BESA_{province}_plots_{timestamp}.zip"
    zip_path = os.path.join(exports_dir, zip_filename)
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        if os.path.exists(path_l1):
            zf.write(path_l1, os.path.basename(path_l1))
            os.remove(path_l1)  # Clean up individual file
        if os.path.exists(path_l2):
            zf.write(path_l2, os.path.basename(path_l2))
            os.remove(path_l2)  # Clean up individual file
    
    gr.Info(f"✅ Plots Export Saved: {zip_path}", duration=10)
    return zip_path


def save_layers(state, year, progress=gr.Progress()):
    """Export high-resolution publication-quality layers (174mm @ 600 DPI)."""
    if not state:
        raise gr.Error("No analysis results to export. Run analysis first.")
    
    if state.get("geom") is None:
        raise gr.Error("No geometry data in state. Please run analysis first.")
    
    from stbesa.exporter import LayerExporter
    
    meta = state["meta"]
    province = meta.get('Province', 'Unknown')
    
    # Get vis params from state if available, otherwise use defaults
    vis_params = state.get("vis_params", {
        'vol': (0, 80000),
        'sur': (0, 20000),
        'pop': (0, 500)
    })
    
    # Create exporter and run export (sequential, more reliable)
    exporter = LayerExporter(
        analysis_service=analysis_service,
        ee_geom=state["geom"],
        year=int(year),
        province=province,
        vis_params=vis_params
    )
    
    progress(0, desc="Exporting layers in parallel (2-5 min)...")
    zip_path = exporter.export_all_layers_parallel(progress=progress)
    
    progress(1.0, desc="Export Complete!")
              
    gr.Info(f"✅ Layers Export Saved: {zip_path}", duration=10)
    return str(zip_path)


# ===================== BUILD GRADIO UI =====================

css = """
/* Hide Default Gradio Progress Checks - REMOVED to allow Exports to show progress */
/*.meta-text, .progress-level, .loader, .wrap.default, .eta-bar, .progress-bar {
    display: none !important;
}*/

/* Custom Blocking Modal Styles */
#modal-overlay {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-color: rgba(0, 0, 0, 0.6);
    z-index: 9999;
    display: flex;
    align-items: center;
    justify-content: center;
    backdrop-filter: blur(4px);
}

#modal-content {
    background: white;
    padding: 40px;
    border-radius: 16px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    text-align: center;
    width: 400px;
    max-width: 90%;
}

.spinner {
    border: 6px solid #f3f3f3;
    border-top: 6px solid #FF7F50; /* Coral color matching theme */
    border-radius: 50%;
    width: 50px;
    height: 50px;
    animation: spin 1s linear infinite;
    margin: 0 auto 20px auto;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

.modal-title {
    font-size: 24px;
    font-weight: bold;
    color: #333;
    margin-bottom: 10px;
}

.modal-text {
    font-size: 16px;
    color: #666;
    margin-bottom: 5px;
}

/* Ensure HTML containers fill space properly */
#map-container, #map-container > div, #data-container, #data-container > div {
    width: 100%;
    min-height: 100px;
}
"""

with gr.Blocks(title="ST-BESA Platform", css=css) as app:
    
    # Custom Blocking Modal Container
    modal_out = gr.HTML(visible=True) # Always visible, content controls visibility via CSS display
    
    analysis_results = gr.State()
    
    gr.Markdown("""
    # 🌍 ST-BESA: Spatio-Temporal Built Environment & Settlement Analytics
    
    Analyze building volume, surface area, and population dynamics across administrative regions (1975-2030).
    Powered by **Google Earth Engine** and **JRC Global Human Settlement Layer**.
    """)
    
    with gr.Row():
        with gr.Column(scale=1, min_width=300):
            gr.Markdown("### ⚙️ Configuration")
            
            # Check for pre-configured Project ID (e.g., from Colab notebook)
            _default_project_id = os.environ.get("STBESA_PROJECT_ID", "")
            
            project_id_input = gr.Textbox(
                label="Google Cloud Project ID", 
                placeholder="e.g., my-gcp-project-123",
                value=_default_project_id,
                info="Required: Your GCP project with Earth Engine API enabled." if not _default_project_id else "Pre-configured from environment."
            )
            
            gr.Markdown("### 📍 Dataset & Region")
            
            dataset_dd = gr.Dropdown(
                choices=get_dataset_choices(), 
                label="Country/Dataset", 
                interactive=True,
                value=None,
                info="Select a country/dataset."
            )
            

            
            prov_dd = gr.Dropdown(choices=[], label="Region", interactive=True)
            dist_dd = gr.Dropdown(choices=[], label="Sub-Region", multiselect=True, interactive=True)
            
            year_slider = gr.Slider(
                minimum=1975, maximum=2030, step=5, value=2025, 
                label="Display Year",
                info="All years are computed; this selects which to display on map."
            )
            
            max_workers_slider = gr.Slider(
                minimum=1, maximum=10, step=1, value=5,
                label="Parallel Workers",
                info="Number of concurrent EE requests (1-10). Higher = faster but may hit rate limits."
            )
            
            auto_scale_chk = gr.Checkbox(
                value=True, 
                label="Auto Scale Visualization",
                info="Dynamically adjust map colors. Uncheck to set manual ranges."
            )
            
            # Manual Scale Controls (Visible when Auto Scale is OFF)
            with gr.Group(visible=False) as manual_viz_group:
                with gr.Row():
                    gr.Markdown("#### 🎨 Manual Ranges")
                
                # Volume Row
                with gr.Row(equal_height=True):
                    with gr.Column(scale=2, min_width=80):
                        vol_min_in = gr.Number(label="Vol Min", value=0, precision=1)
                    with gr.Column(scale=2, min_width=80):
                        vol_max_in = gr.Number(label="Vol Max", value=80000, precision=1)
                    with gr.Column(scale=1, min_width=80):
                        vol_norm_btn = gr.Button("Normalize", size="sm")

                # Surface Row
                with gr.Row(equal_height=True):
                    with gr.Column(scale=2, min_width=80):
                        sur_min_in = gr.Number(label="Sur Min", value=0, precision=1)
                    with gr.Column(scale=2, min_width=80):
                        sur_max_in = gr.Number(label="Sur Max", value=20000, precision=1)
                    with gr.Column(scale=1, min_width=80):
                        sur_norm_btn = gr.Button("Normalize", size="sm")

                # Population Row
                with gr.Row(equal_height=True):
                    with gr.Column(scale=2, min_width=80):
                        pop_min_in = gr.Number(label="Pop Min", value=0, precision=1)
                    with gr.Column(scale=2, min_width=80):
                        pop_max_in = gr.Number(label="Pop Max", value=500, precision=1)
                    with gr.Column(scale=1, min_width=80):
                        pop_norm_btn = gr.Button("Normalize", size="sm")
                
                refresh_btn = gr.Button("🔄 Update Viz Only", size="sm") 
            
            run_btn = gr.Button("🚀 Run Full Analysis (All Years)", variant="primary", size="lg")
            
            gr.Markdown("### 📤 Export")
            with gr.Row():
                export_btn = gr.Button("📥 Excel", size="sm")
                export_plots_btn = gr.Button("📊 Plots", size="sm")
                save_layers_btn = gr.Button("🖼️ Layers", size="sm")
            
            export_out = gr.File(label="Download", height=60)
            
        with gr.Column(scale=3):
            with gr.Tabs():
                with gr.Tab("🗺️ Map"):
                    # Placeholder map HTML to prevent progress bar overlap
                    placeholder_map = """
                    <div style="width:100%; height:700px; background: linear-gradient(135deg, #e8f4f8 0%, #d4e8ed 50%, #c0dce4 100%); 
                                border-radius: 8px; display: flex; align-items: center; justify-content: center; 
                                border: 2px dashed #94b8c5;">
                        <div style="text-align: center; color: #5a7d8a;">
                            <div style="font-size: 48px; margin-bottom: 16px;">🗺️</div>
                            <div style="font-size: 18px; font-weight: 500;">Select a region and run analysis</div>
                            <div style="font-size: 14px; opacity: 0.7; margin-top: 8px;">Map will appear here</div>
                        </div>
                    </div>
                    """
                    map_out = gr.HTML(value=placeholder_map, elem_id="map-container")
                    
                with gr.Tab("📊 Data"):
                    placeholder_data = """
                    <div style="width:100%; height:500px; background: linear-gradient(135deg, #f0f8e8 0%, #e0eed0 50%, #d0e4b8 100%); 
                                border-radius: 8px; display: flex; align-items: center; justify-content: center; 
                                border: 2px dashed #a8c595;">
                        <div style="text-align: center; color: #5a7a4a;">
                            <div style="font-size: 48px; margin-bottom: 16px;">📊</div>
                            <div style="font-size: 18px; font-weight: 500;">Data Analysis Pending</div>
                            <div style="font-size: 14px; opacity: 0.7; margin-top: 8px;">Run analysis to see statistics tables</div>
                        </div>
                    </div>
                    """
                    stats_out = gr.DataFrame(label="All Years Statistics")
                    stats_placeholder_out = gr.HTML(value=placeholder_data, visible=True, elem_id="data-container")
                    stats_out.visible = False
                    
            # L1 Tab
            with gr.Tab("📈 L1 Time Series"):
                with gr.Group() as l1_placeholder_group:
                    gr.HTML("""
                    <div style="width:100%; height:600px; background: linear-gradient(135deg, #f0f4f8 0%, #e0e8f0 50%, #d0dce8 100%); 
                                border-radius: 8px; display: flex; align-items: center; justify-content: center; 
                                border: 2px dashed #95aec5;">
                        <div style="text-align: center; color: #4a6a8a;">
                            <div style="font-size: 48px; margin-bottom: 16px;">📈</div>
                            <div style="font-size: 18px; font-weight: 500;">L1 Plots Waiting</div>
                            <div style="font-size: 14px; opacity: 0.7; margin-top: 8px;">Run analysis to generate time series</div>
                        </div>
                    </div>
                    """)
                
                with gr.Group(visible=False) as l1_result_group:
                    plot_l1_out = gr.Plot(label="L1 Classes + Total (6 metrics)")
                
            # L2 Tab
            with gr.Tab("📈 L2 Time Series"):
                with gr.Group() as l2_placeholder_group:
                    l2_placeholder_html = gr.HTML("""
                    <div style="width:100%; height:600px; background: linear-gradient(135deg, #f8f0f4 0%, #f0e0e8 50%, #e8d0dc 100%); 
                                border-radius: 8px; display: flex; align-items: center; justify-content: center; 
                                border: 2px dashed #c595ae;">
                        <div style="text-align: center; color: #8a4a6a;">
                            <div style="font-size: 48px; margin-bottom: 16px;">📈</div>
                            <div style="font-size: 18px; font-weight: 500;">L2 Plots Waiting</div>
                            <div style="font-size: 14px; opacity: 0.7; margin-top: 8px;">Run analysis to generate time series</div>
                        </div>
                    </div>
                    """)
                
                with gr.Group(visible=False) as l2_result_group:
                    plot_l2_out = gr.Plot(label="L2 Classes + Total (6 metrics)")
                    
            status_out = gr.Textbox(label="Status", interactive=False, lines=1)
    
    # Event Handlers
    
    # Toggle logic for manual controls
    def toggle_manual(auto):
        # Show Manual Inputs ONLY if Auto Scale is FALSE
        return gr.update(visible=not auto)

    auto_scale_chk.change(fn=toggle_manual, inputs=[auto_scale_chk], outputs=[manual_viz_group]) 
    
    dataset_dd.change(on_dataset_change, inputs=[dataset_dd], outputs=[prov_dd, prov_dd, dist_dd])

    prov_dd.change(on_province_change, inputs=[prov_dd], outputs=[dist_dd])
    
    run_btn.click(
        run_analysis, 
        inputs=[project_id_input, dataset_dd, prov_dd, dist_dd, year_slider, max_workers_slider, auto_scale_chk, 
                vol_min_in, vol_max_in, sur_min_in, sur_max_in, pop_min_in, pop_max_in], 
        outputs=[
            modal_out, # FIRST OUTPUT: The Blocking Modal
            map_out, 
            stats_out, stats_placeholder_out,
            plot_l1_out, l1_placeholder_group, l1_result_group,
            plot_l2_out, l2_placeholder_group, l2_result_group, l2_placeholder_html,
            status_out, analysis_results,
            vol_min_in, vol_max_in, sur_min_in, sur_max_in, pop_min_in, pop_max_in
        ]
    )
    
    refresh_btn.click(
        refresh_visualization,
        inputs=[analysis_results, year_slider, vol_min_in, vol_max_in, sur_min_in, sur_max_in, pop_min_in, pop_max_in],
        outputs=[map_out, analysis_results]
    )
    
    # Normalize Buttons
    vol_norm_btn.click(
        lambda s, y: normalize_layer(s, y, 'vol'),
        inputs=[analysis_results, year_slider],
        outputs=[vol_min_in, vol_max_in]
    )
    
    sur_norm_btn.click(
        lambda s, y: normalize_layer(s, y, 'sur'),
        inputs=[analysis_results, year_slider],
        outputs=[sur_min_in, sur_max_in]
    )
    
    pop_norm_btn.click(
        lambda s, y: normalize_layer(s, y, 'pop'),
        inputs=[analysis_results, year_slider],
        outputs=[pop_min_in, pop_max_in]
    )
    
    # Instant Update Wiring
    year_slider.change(
        update_display,
        inputs=[analysis_results, year_slider],
        outputs=[map_out, plot_l1_out, plot_l2_out, analysis_results]
    )
    
    export_btn.click(
        export_data,
        inputs=[analysis_results],
        outputs=[export_out]
    )
    
    export_plots_btn.click(
        export_plots,
        inputs=[analysis_results],
        outputs=[export_out]
    )
    
    save_layers_btn.click(
        save_layers,
        inputs=[analysis_results, year_slider],
        outputs=[export_out]
    )

if __name__ == "__main__":
    app.launch(theme=gr.themes.Soft())

