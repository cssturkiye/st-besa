# stbesa/exporter.py
"""
ST-BESA Export Module
Handles export of analysis results to Excel, high-res PNGs, and other formats.
"""

import pandas as pd
import math
import os
import io
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from stbesa.constants import COLUMN_NAMES, COLUMN_DESCRIPTIONS, SMOD_L1_CLASSES

# Export configuration
TURBO_PALETTE = ['#30123b','#4145ab','#2e7de0','#1db6d7','#27d39b','#7be151','#c0e620','#f9e14b','#fca72c','#f96814','#a31e1e']
L2_PALETTE_MAP = {10: '#7AB6F5', 11: '#CDF57A', 12: '#ABCD66', 13: '#375623', 21: '#FFFF00', 22: '#A87000', 23: '#732600', 30: '#FF0000'}

# Export settings
EXPORT_WIDTH_MM = 174.0
EXPORT_DPI = 300
LEGEND_TITLE_FONT_PX = 48
LEGEND_TEXT_FONT_PX = 36
OSM_LABELS_ZOOM_BOOST = 1


class STBESAExporter:
    """
    Handles export of analysis results to various formats (Excel, etc.).
    """

    @staticmethod
    def export_excel_report(output_path: str, 
                          overall_stats: pd.DataFrame, 
                          smod_l1: pd.DataFrame, 
                          smod_l2: pd.DataFrame, 
                          metadata_dict: Dict[str, Any]) -> str:
        """
        Generates a multi-sheet Excel report matching the backup format.
        
        Sheets:
        1. Overall_Statistics - Yearly aggregate statistics
        2. SMOD_L1_Statistics - SMOD Level 1 breakdown by year
        3. SMOD_L2_Statistics - SMOD Level 2 breakdown by year
        4. Metadata - Analysis metadata
        5. Data_Dictionary - Column definitions
        """
        
        # Prepare Metadata DataFrame
        meta_df = pd.DataFrame({
            'Parameter': list(metadata_dict.keys()),
            'Value': [str(v) for v in metadata_dict.values()]
        })

        # Prepare Data Dictionary DataFrame
        dict_rows = []
        for col, friendly in COLUMN_NAMES.items():
            dict_rows.append({
                'Programmatic Name': col,
                'User-Friendly Name': friendly,
                'Description': COLUMN_DESCRIPTIONS.get(col, '')
            })
        dict_df = pd.DataFrame(dict_rows)

        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Sheet 1: Overall Statistics
            if overall_stats is not None and not overall_stats.empty:
                # Sort by year for easy charting
                sorted_overall = overall_stats.sort_values('yil').reset_index(drop=True)
                sorted_overall.to_excel(writer, sheet_name='Overall_Statistics', index=False)
            
            # Sheet 2: SMOD L1 Statistics
            if smod_l1 is not None and not smod_l1.empty:
                sorted_l1 = smod_l1.sort_values(['smod_l1_code', 'yil']).reset_index(drop=True)
                sorted_l1.to_excel(writer, sheet_name='SMOD_L1_Statistics', index=False)
                
            # Sheet 3: SMOD L2 Statistics
            if smod_l2 is not None and not smod_l2.empty:
                sorted_l2 = smod_l2.sort_values(['smod_l2_code', 'yil']).reset_index(drop=True)
                sorted_l2.to_excel(writer, sheet_name='SMOD_L2_Statistics', index=False)
                
            # Sheet 4: Metadata
            meta_df.to_excel(writer, sheet_name='Metadata', index=False)
            
            # Sheet 5: Data Dictionary
            dict_df.to_excel(writer, sheet_name='Data_Dictionary', index=False)
            
        return output_path
    
    @staticmethod
    def export_plots_as_png(fig, output_path: str, dpi: int = 300, width_mm: float = 174) -> str:
        """
        Export matplotlib figure as PNG at specified DPI and width.
        """
        from PIL import Image
        
        width_in = width_mm / 25.4
        target_w = int(round(width_in * dpi))
        
        # Save to buffer
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', facecolor='white')
        buf.seek(0)
        
        # Resize to exact width
        im = Image.open(buf)
        new_h = int(round(im.size[1] * (target_w / im.size[0])))
        im_resized = im.resize((target_w, new_h), resample=Image.BICUBIC)
        im_resized.save(output_path, format='PNG', dpi=(dpi, dpi))
        
        return output_path


class LayerExporter:
    """
    Exports high-resolution map layers as transparent PNGs for publication.
    Matches backup functionality exactly: 174mm @ 600 DPI.
    """
    
    def __init__(self, analysis_service, ee_geom, year: int, province: str, vis_params: Dict):
        """
        Initialize layer exporter.
        
        Args:
            analysis_service: STBESAAnalysis instance with initialized EE
            ee_geom: Earth Engine geometry for the region
            year: Year to export
            province: Province name for file naming
            vis_params: Visualization parameters dict with 'vol', 'sur', 'pop' keys
        """
        self.analysis = analysis_service
        self.geom = ee_geom
        self.year = year
        self.province = province.replace(' ', '_')
        self.vis_params = vis_params
        
        # Output settings
        self.dpi = EXPORT_DPI
        self.width_mm = EXPORT_WIDTH_MM
        self.width_in = self.width_mm / 25.4
        self.out_width_px = int(round(self.width_in * self.dpi))
        
        # Will be computed from geometry
        self.bbox = None
        self.bounds_3857 = None
        self.target_height_px = None
        
    def _compute_bounds(self):
        """Compute geographic and web mercator bounds for export."""
        import ee
        
        # Get WGS84 bounds
        bbox = self.analysis._ee_getinfo(ee.Geometry(self.geom).bounds().coordinates())
        ring = bbox[0] if isinstance(bbox, list) and bbox else []
        xs = [float(pt[0]) for pt in ring]
        ys = [float(pt[1]) for pt in ring]
        self.bbox = {'w': min(xs), 'e': max(xs), 's': min(ys), 'n': max(ys)}
        
        # Get Web Mercator bounds for upscaling
        bounds_3857 = self.analysis._ee_getinfo(
            ee.Geometry.Rectangle(
                [self.bbox['w'], self.bbox['s'], self.bbox['e'], self.bbox['n']], 
                proj='EPSG:4326', geodesic=False
            ).transform('EPSG:3857', 1).coordinates()
        )
        ringm = bounds_3857[0]
        xs_m = [float(pt[0]) for pt in ringm]
        ys_m = [float(pt[1]) for pt in ringm]
        self.bounds_3857 = {'w': min(xs_m), 'e': max(xs_m), 's': min(ys_m), 'n': max(ys_m)}
        
        # Compute native pixel dimensions at 100m scale
        width_m = max(1.0, self.bounds_3857['e'] - self.bounds_3857['w'])
        height_m = max(1.0, self.bounds_3857['n'] - self.bounds_3857['s'])
        native_w_px = int(max(1, math.ceil(width_m / 100.0)))
        native_h_px = int(max(1, math.ceil(height_m / 100.0)))
        
        # Compute target height preserving aspect ratio
        self.target_height_px = int(max(1, round(self.out_width_px * (native_h_px / native_w_px))))
        self.native_w_px = native_w_px
        self.native_h_px = native_h_px
        
    def _fetch_xyz_layer(self, width_px: int, height_px: int, tile_url_tpl: str, zoom_boost: int = 0):
        """Fetch XYZ tiles covering bbox at target resolution."""
        from PIL import Image
        
        w, e, s, n = self.bbox['w'], self.bbox['e'], self.bbox['s'], self.bbox['n']
        
        def lonlat_to_pixel(lon: float, lat: float, z: int):
            lat_rad = math.radians(lat)
            ntiles = 2 ** z
            x = (lon + 180.0) / 360.0 * ntiles * 256.0
            y = (1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * ntiles * 256.0
            return x, y
        
        def approx_bbox_width_m(lon_w: float, lon_e: float, lat_mid: float) -> float:
            radius = 6378137.0
            dlon = math.radians(lon_e - lon_w)
            return abs(radius * dlon * math.cos(math.radians(lat_mid)))
        
        # Pick zoom based on meters-per-pixel
        lat_c = (s + n) / 2.0
        meters_per_pixel_target = max(0.1, approx_bbox_width_m(w, e, lat_c) / max(1, width_px))
        initial_res = 156543.03392804097
        z_float = math.log2(max(1e-6, initial_res * math.cos(math.radians(lat_c)) / meters_per_pixel_target))
        z = int(max(0, min(19, round(z_float) + int(zoom_boost))))
        
        min_px_x, min_px_y = lonlat_to_pixel(w, n, z)
        max_px_x, max_px_y = lonlat_to_pixel(e, s, z)
        x0 = int(math.floor(min_px_x / 256.0))
        y0 = int(math.floor(min_px_y / 256.0))
        x1 = int(math.floor((max_px_x - 1) / 256.0))
        y1 = int(math.floor((max_px_y - 1) / 256.0))
        cols = x1 - x0 + 1
        rows = y1 - y0 + 1
        
        mosaic = Image.new('RGBA', (cols * 256, rows * 256), (0, 0, 0, 0))
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-Agent', 'st-besa-export/1.0')]
        urllib.request.install_opener(opener)
        
        for ix in range(cols):
            for iy in range(rows):
                tx = x0 + ix
                ty = y0 + iy
                url = tile_url_tpl.format(z=z, x=tx, y=ty)
                try:
                    with urllib.request.urlopen(url) as resp:
                        tile_data = resp.read()
                    tile_img = Image.open(io.BytesIO(tile_data)).convert('RGBA')
                    mosaic.paste(tile_img, (ix * 256, iy * 256))
                except Exception:
                    pass
                    
        crop_left = int(round(min_px_x - x0 * 256.0))
        crop_top = int(round(min_px_y - y0 * 256.0))
        crop_right = crop_left + int(round(max_px_x - min_px_x))
        crop_bottom = crop_top + int(round(max_px_y - min_px_y))
        crop = mosaic.crop((crop_left, crop_top, crop_right, crop_bottom))
        
        return crop.resize((width_px, height_px), resample=Image.BILINEAR)
    
    def _save_ee_layer(self, img, short_name: str, prefix: str, out_dir: Path, timestamp: str) -> Optional[Path]:
        """Save an EE image layer as high-res PNG."""
        import ee
        from PIL import Image
        
        wm, sm, em, nm = self.bounds_3857['w'], self.bounds_3857['s'], self.bounds_3857['e'], self.bounds_3857['n']
        
        img_merc = ee.Image(img).reproject(crs='EPSG:3857', scale=100)
        region_merc = ee.Geometry.Rectangle([wm, sm, em, nm], proj='EPSG:3857', geodesic=False)
        
        params = {"region": region_merc, "dimensions": self.native_w_px, "format": "png", "maxPixels": 1e13}
        
        print(f"[LayerExporter] Getting thumb URL for {short_name}...")
        url = img_merc.getThumbURL(params)
        
        fname = f"{prefix}_ST-BESA_{self.province}_{short_name}_{timestamp}.png"
        fpath = out_dir / fname
        
        # Download native-resolution overlay with timeout
        print(f"[LayerExporter] Downloading {short_name} ({self.native_w_px}px native)...")
        try:
            with urllib.request.urlopen(url, timeout=120) as resp:
                data = resp.read()
        except Exception as e:
            print(f"[LayerExporter] FAILED to download {short_name}: {e}")
            return None
            
        overlay = Image.open(io.BytesIO(data)).convert('RGBA')
        
        # Upscale to target 174mm width at 600 DPI with NEAREST to preserve pixel edges
        up_overlay = overlay.resize((self.out_width_px, self.target_height_px), resample=Image.NEAREST)
        
        # Save transparent overlay with 600 DPI
        try:
            up_overlay.save(str(fpath), format='PNG', dpi=(self.dpi, self.dpi))
        except Exception:
            up_overlay.save(str(fpath), format='PNG')
        
        print(f"[LayerExporter] Saved {short_name} -> {fpath.name}")
        return fpath
    
    def _save_continuous_legend(self, path: Path, title: str, vmin: float, vmax: float):
        """Save a continuous colorbar legend as PNG."""
        from PIL import Image, ImageDraw, ImageFont
        
        def load_font(size_px: int):
            try:
                return ImageFont.truetype("arial.ttf", size_px)
            except Exception:
                try:
                    return ImageFont.truetype("DejaVuSans.ttf", size_px)
                except Exception:
                    return ImageFont.load_default()
        
        bar_width = 1000
        bar_height = 60
        margin = 24
        title_font = load_font(LEGEND_TITLE_FONT_PX)
        text_font = load_font(LEGEND_TEXT_FONT_PX)
        title_height = title_font.size
        labels_height = text_font.size
        width = bar_width + margin * 2
        total_height = margin + title_height + 8 + bar_height + 6 + labels_height + margin
        
        img = Image.new('RGBA', (width, total_height), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        
        # Title
        d.text((margin, margin), title, fill=(0,0,0,255), font=title_font)
        
        # Draw turbo gradient
        try:
            from matplotlib import cm
            turbo_cmap = cm.get_cmap('turbo')
            for i in range(bar_width):
                t = i / max(1, bar_width - 1)
                rgba = turbo_cmap(t)
                color = tuple(int(c * 255) for c in rgba[:3])
                d.line([(margin + i, margin + title_height + 8), (margin + i, margin + title_height + 8 + bar_height)], fill=color)
        except Exception:
            # Fallback to discrete colors
            for i in range(bar_width):
                t = i / max(1, bar_width - 1)
                idx = int(t * (len(TURBO_PALETTE) - 1))
                d.line([(margin + i, margin + title_height + 8), (margin + i, margin + title_height + 8 + bar_height)], fill=TURBO_PALETTE[idx])
        
        # Min/max labels
        txt_min = f"{vmin:.1f}"
        txt_max = f"{vmax:.1f}"
        d.text((margin, margin + title_height + 8 + bar_height + 6), txt_min, fill=(0,0,0,255), font=text_font)
        try:
            tw = d.textlength(txt_max, font=text_font)
        except Exception:
            tw = len(txt_max) * text_font.size * 0.6
        d.text((margin + bar_width - int(tw), margin + title_height + 8 + bar_height + 6), txt_max, fill=(0,0,0,255), font=text_font)
        
        img.save(str(path), format='PNG', dpi=(self.dpi, self.dpi))
        
    def _save_categorical_legend(self, path: Path, title: str, items: List[Tuple[str, str]]):
        """Save a categorical legend as PNG."""
        from PIL import Image, ImageDraw, ImageFont
        
        def load_font(size_px: int):
            try:
                return ImageFont.truetype("arial.ttf", size_px)
            except Exception:
                try:
                    return ImageFont.truetype("DejaVuSans.ttf", size_px)
                except Exception:
                    return ImageFont.load_default()
        
        width = int(self.out_width_px / 2.0)
        margin = 12
        title_font = load_font(LEGEND_TITLE_FONT_PX)
        text_font = load_font(LEGEND_TEXT_FONT_PX)
        row_h = max(24, int(text_font.size * 1.2))
        title_h = title_font.size
        box_size = max(18, int(text_font.size * 0.9))
        height = margin + title_h + 6 + len(items) * row_h + margin
        
        img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.text((margin, margin), title, fill=(0,0,0,255), font=title_font)
        
        y = margin + title_h + 6
        for label, color in items:
            d.rectangle((margin, y, margin + box_size, y + box_size), fill=color, outline=(60,60,60,255))
            d.text((margin + box_size + 8, y), label, fill=(0,0,0,255), font=text_font)
            y += row_h
            
        img.save(str(path), format='PNG', dpi=(self.dpi, self.dpi))
    
    def _generate_photoshop_script(self, out_dir: Path) -> Path:
        """Generate Photoshop JSX script to load and stack layers."""
        jsx = (
            "var folder = new Folder('" + str(out_dir).replace('\\', '/') + "');\n" +
            "var files = folder.getFiles(/\\.png$/i).sort(function(a,b){ return (decodeURI(a.name) > decodeURI(b.name)) ? 1 : -1; });\n" +
            "if(files.length>0){\n" +
            "  var base = app.open(files[0]);\n" +
            "  base.activeLayer.name = decodeURI(files[0].name.replace(/\\.png$/i,''));\n" +
            "  for(var i=1;i<files.length;i++){\n" +
            "    var im = app.open(files[i]); im.selection.selectAll(); im.selection.copy(); im.close(SaveOptions.DONOTSAVECHANGES); base.paste(); base.activeLayer.name = decodeURI(files[i].name.replace(/\\.png$/i,''));\n" +
            "  }\n" +
            f"  base.resizeImage(undefined, undefined, {self.dpi}, ResampleMethod.NONE);\n" +
            "}\n"
        )
        jsx_path = out_dir / "load_layers.jsx"
        with open(jsx_path, 'w', encoding='utf-8') as f:
            f.write(jsx)
        return jsx_path
    
    def export_all_layers(self, cache: Dict) -> Path:
        """
        Export all layers matching backup functionality exactly.
        
        Returns path to output directory containing all files.
        """
        import ee
        from PIL import Image
        
        # Compute bounds
        self._compute_bounds()
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path.cwd() / f"STBESA_EXPORT_{self.province}_{timestamp}"
        out_dir.mkdir(parents=True, exist_ok=True)
        
        saved = []
        
        # Get visualization ranges
        vmin_vol, vmax_vol = self.vis_params.get('vol', (0, 80000))
        vmin_sur, vmax_sur = self.vis_params.get('sur', (0, 20000))
        vmin_pop, vmax_pop = self.vis_params.get('pop', (0, 500))
        
        # Build visualized layers
        vol = ee.Image(f"JRC/GHSL/P2023A/GHS_BUILT_V/{self.year}").select("built_volume_total").clip(self.geom)
        sur = ee.Image(f"JRC/GHSL/P2023A/GHS_BUILT_S/{self.year}").select("built_surface").clip(self.geom)
        pop = ee.Image(f"JRC/GHSL/P2023A/GHS_POP/{self.year}").select("population_count").clip(self.geom)
        smod = ee.Image(f"JRC/GHSL/P2023A/GHS_SMOD/{self.year}").select("smod_code").clip(self.geom)
        
        vol_vis = vol.updateMask(vol.gt(0)).visualize(min=vmin_vol, max=vmax_vol, palette=TURBO_PALETTE)
        sur_vis = sur.updateMask(sur.gt(0)).visualize(min=vmin_sur, max=vmax_sur, palette=TURBO_PALETTE)
        pop_vis = pop.updateMask(pop.gt(0)).visualize(min=vmin_pop, max=vmax_pop, palette=TURBO_PALETTE)
        
        smod_mask = smod.neq(0)
        smod_l2_vis = smod.updateMask(smod_mask).remap([10,11,12,13,21,22,23,30],[0,1,2,3,4,5,6,7]).visualize(min=0, max=7, palette=list(L2_PALETTE_MAP.values()))
        smod_l1 = smod.divide(10).floor()
        l1_palette = [SMOD_L1_CLASSES[1]["color"], SMOD_L1_CLASSES[2]["color"], SMOD_L1_CLASSES[3]["color"]]
        smod_l1_vis = smod_l1.updateMask(smod_mask).visualize(min=1, max=3, palette=l1_palette)
        
        # Boundary
        outline = ee.Image().byte().paint(ee.FeatureCollection([ee.Feature(self.geom)]), 1, 2).visualize(min=0, max=1, palette=['000000'])
        
        # Save data layers
        layers_to_save = [
            ("02", "smod-l2", smod_l2_vis),
            ("03", "smod-l1", smod_l1_vis),
            ("04", "population", pop_vis),
            ("05", "surface", sur_vis),
            ("06", "volume", vol_vis),
            ("07", "boundary", outline),
        ]
        
        for prefix, short_name, img in layers_to_save:
            try:
                fpath = self._save_ee_layer(img, short_name, prefix, out_dir, timestamp)
                if fpath:
                    saved.append(fpath)
            except Exception as e:
                print(f"[WARN] Failed to save layer {short_name}: {e}")
        
        # Save OSM background
        try:
            nolabels_tpl = "https://basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}.png"
            osm_bg = self._fetch_xyz_layer(self.out_width_px, self.target_height_px, nolabels_tpl, zoom_boost=0).convert('RGB')
            osm_bg_path = out_dir / f"01_ST-BESA_{self.province}_openstreetmap-bg_{timestamp}.png"
            osm_bg.save(str(osm_bg_path), format='PNG', dpi=(self.dpi, self.dpi))
            saved.append(osm_bg_path)
        except Exception as e:
            print(f"[WARN] Failed to save OSM background: {e}")
        
        # Save OSM labels
        try:
            labels_tpl = "https://basemaps.cartocdn.com/light_only_labels/{z}/{x}/{y}.png"
            osm_lbl = self._fetch_xyz_layer(self.out_width_px, self.target_height_px, labels_tpl, zoom_boost=OSM_LABELS_ZOOM_BOOST).convert('RGBA')
            osm_lbl_path = out_dir / f"08_ST-BESA_{self.province}_openstreetmap-text_{timestamp}.png"
            osm_lbl.save(str(osm_lbl_path), format='PNG', dpi=(self.dpi, self.dpi))
            saved.append(osm_lbl_path)
        except Exception as e:
            print(f"[WARN] Failed to save OSM labels: {e}")
        
        # Save legends
        try:
            self._save_continuous_legend(out_dir / f"09_ST-BESA_{self.province}_legend_volume_{timestamp}.png", "Building volume (m³)", vmin_vol, vmax_vol)
            self._save_continuous_legend(out_dir / f"10_ST-BESA_{self.province}_legend_surface_{timestamp}.png", "Building surface (m²)", vmin_sur, vmax_sur)
            self._save_continuous_legend(out_dir / f"11_ST-BESA_{self.province}_legend_population_{timestamp}.png", "Population (people)", vmin_pop, vmax_pop)
            
            l1_items = [
                (SMOD_L1_CLASSES[1]['name'], SMOD_L1_CLASSES[1]['color']),
                (SMOD_L1_CLASSES[2]['name'], SMOD_L1_CLASSES[2]['color']),
                (SMOD_L1_CLASSES[3]['name'], SMOD_L1_CLASSES[3]['color']),
            ]
            self._save_categorical_legend(out_dir / f"12_ST-BESA_{self.province}_legend_smod_l1_{timestamp}.png", "SMOD L1", l1_items)
            
            l2_items = [
                ('10 Water/No data', '#7AB6F5'),
                ('11 Very low density rural', '#CDF57A'),
                ('12 Low density rural', '#ABCD66'),
                ('13 Rural cluster', '#375623'),
                ('21 Suburban/peri-urban', '#FFFF00'),
                ('22 Semi-dense urban', '#A87000'),
                ('23 Dense urban cluster', '#732600'),
                ('30 Urban centre', '#FF0000'),
            ]
            self._save_categorical_legend(out_dir / f"13_ST-BESA_{self.province}_legend_smod_l2_{timestamp}.png", "SMOD L2", l2_items)
        except Exception as e:
            print(f"[WARN] Legend generation failed: {e}")
        
        # Generate Photoshop script
        try:
            self._generate_photoshop_script(out_dir)
        except Exception as e:
            print(f"[WARN] Photoshop script creation failed: {e}")
            
        # Create ZIP archive
        import zipfile
        import shutil
        
        zip_filename = f"STBESA_LAYERS_{self.province}_{timestamp}.zip"
        zip_path = Path.cwd() / "exports" / zip_filename
        zip_path.parent.mkdir(exist_ok=True)
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for file_path in out_dir.iterdir():
                if file_path.is_file():
                    zf.write(file_path, file_path.name)
        
        # Cleanup directory
        try:
            shutil.rmtree(out_dir)
        except Exception:
            pass
            
        return zip_path

    def export_all_layers_parallel(self, progress=None) -> Path:
        """
        Export all layers in parallel using ThreadPoolExecutor.
        """
        import ee
        import concurrent.futures
        import zipfile
        import shutil
        from PIL import Image
        
        # Compute bounds
        print("[LayerExporter] Computing geometry bounds...")
        self._compute_bounds()
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dirname = f"STBESA_EXPORT_{self.province}_{timestamp}"
        out_dir = Path.cwd() / "exports" / out_dirname
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Get visualization ranges
        vmin_vol, vmax_vol = self.vis_params.get('vol', (0, 80000))
        vmin_sur, vmax_sur = self.vis_params.get('sur', (0, 20000))
        vmin_pop, vmax_pop = self.vis_params.get('pop', (0, 500))
        
        # Prepare EE Images
        print("[LayerExporter] Preparing Earth Engine layers...")
        vol = ee.Image(f"JRC/GHSL/P2023A/GHS_BUILT_V/{self.year}").select("built_volume_total").clip(self.geom)
        sur = ee.Image(f"JRC/GHSL/P2023A/GHS_BUILT_S/{self.year}").select("built_surface").clip(self.geom)
        pop = ee.Image(f"JRC/GHSL/P2023A/GHS_POP/{self.year}").select("population_count").clip(self.geom)
        smod = ee.Image(f"JRC/GHSL/P2023A/GHS_SMOD/{self.year}").select("smod_code").clip(self.geom)
        
        vol_vis = vol.updateMask(vol.gt(0)).visualize(min=vmin_vol, max=vmax_vol, palette=TURBO_PALETTE)
        sur_vis = sur.updateMask(sur.gt(0)).visualize(min=vmin_sur, max=vmax_sur, palette=TURBO_PALETTE)
        pop_vis = pop.updateMask(pop.gt(0)).visualize(min=vmin_pop, max=vmax_pop, palette=TURBO_PALETTE)
        
        smod_mask = smod.neq(0)
        smod_l2_vis = smod.updateMask(smod_mask).remap([10,11,12,13,21,22,23,30],[0,1,2,3,4,5,6,7]).visualize(min=0, max=7, palette=list(L2_PALETTE_MAP.values()))
        smod_l1 = smod.divide(10).floor()
        l1_palette = [SMOD_L1_CLASSES[1]["color"], SMOD_L1_CLASSES[2]["color"], SMOD_L1_CLASSES[3]["color"]]
        smod_l1_vis = smod_l1.updateMask(smod_mask).visualize(min=1, max=3, palette=l1_palette)
        
        # Boundary
        outline = ee.Image().byte().paint(ee.FeatureCollection([ee.Feature(self.geom)]), 1, 2).visualize(min=0, max=1, palette=['000000'])
        
        # Tasks definitions
        def task_save_ee(args):
            prefix, short_name, img = args
            return self._save_ee_layer(img, short_name, prefix, out_dir, timestamp)
            
        def task_save_osm_bg():
            nolabels_tpl = "https://basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}.png"
            osm_bg = self._fetch_xyz_layer(self.out_width_px, self.target_height_px, nolabels_tpl, zoom_boost=0).convert('RGB')
            osm_bg_path = out_dir / f"01_ST-BESA_{self.province}_openstreetmap-bg_{timestamp}.png"
            osm_bg.save(str(osm_bg_path), format='PNG', dpi=(self.dpi, self.dpi))
            return osm_bg_path
            
        def task_save_osm_lbl():
            labels_tpl = "https://basemaps.cartocdn.com/light_only_labels/{z}/{x}/{y}.png"
            osm_lbl = self._fetch_xyz_layer(self.out_width_px, self.target_height_px, labels_tpl, zoom_boost=OSM_LABELS_ZOOM_BOOST).convert('RGBA')
            osm_lbl_path = out_dir / f"08_ST-BESA_{self.province}_openstreetmap-text_{timestamp}.png"
            osm_lbl.save(str(osm_lbl_path), format='PNG', dpi=(self.dpi, self.dpi))
            return osm_lbl_path
            
        def task_save_legends():
            self._save_continuous_legend(out_dir / f"09_ST-BESA_{self.province}_legend_volume_{timestamp}.png", "Building volume (m³)", vmin_vol, vmax_vol)
            self._save_continuous_legend(out_dir / f"10_ST-BESA_{self.province}_legend_surface_{timestamp}.png", "Building surface (m²)", vmin_sur, vmax_sur)
            self._save_continuous_legend(out_dir / f"11_ST-BESA_{self.province}_legend_population_{timestamp}.png", "Population (people)", vmin_pop, vmax_pop)
            
            l1_items = [
                (SMOD_L1_CLASSES[1]['name'], SMOD_L1_CLASSES[1]['color']),
                (SMOD_L1_CLASSES[2]['name'], SMOD_L1_CLASSES[2]['color']),
                (SMOD_L1_CLASSES[3]['name'], SMOD_L1_CLASSES[3]['color']),
            ]
            self._save_categorical_legend(out_dir / f"12_ST-BESA_{self.province}_legend_smod_l1_{timestamp}.png", "SMOD L1", l1_items)
            
            l2_items = [
                ('10 Water/No data', '#7AB6F5'),
                ('11 Very low density rural', '#CDF57A'),
                ('12 Low density rural', '#ABCD66'),
                ('13 Rural cluster', '#375623'),
                ('21 Suburban/peri-urban', '#FFFF00'),
                ('22 Semi-dense urban', '#A87000'),
                ('23 Dense urban cluster', '#732600'),
                ('30 Urban centre', '#FF0000'),
            ]
            self._save_categorical_legend(out_dir / f"13_ST-BESA_{self.province}_legend_smod_l2_{timestamp}.png", "SMOD L2", l2_items)
            return True

        # EE layers to save
        ee_layers = [
            ("02", "smod-l2", smod_l2_vis),
            ("03", "smod-l1", smod_l1_vis),
            ("04", "population", pop_vis),
            ("05", "surface", sur_vis),
            ("06", "volume", vol_vis),
            ("07", "boundary", outline),
        ]
        
        futures_map = {}
        
        if progress: progress(0.2, desc="Starting parallel downloads (600 DPI)...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            # Submit EE tasks
            for item in ee_layers:
                futures_map[executor.submit(task_save_ee, item)] = f"Layer {item[1]}"
            
            # Submit OSM tasks
            futures_map[executor.submit(task_save_osm_bg)] = "OSM Background"
            futures_map[executor.submit(task_save_osm_lbl)] = "OSM Labels"
            
            # Submit Legend task
            futures_map[executor.submit(task_save_legends)] = "Legends"
            
            completed = 0
            total = len(futures_map)
            
            for future in concurrent.futures.as_completed(futures_map):
                name = futures_map[future]
                completed += 1
                try:
                    future.result()
                    print(f"[LayerExporter] Completed {name} ({completed}/{total})")
                except Exception as e:
                    print(f"[WARN] Task {name} failed: {e}")

        # Generate Photoshop script
        print("[LayerExporter] Creating Photoshop script...")
        try:
            self._generate_photoshop_script(out_dir)
        except Exception as e:
            print(f"[WARN] Photoshop script creation failed: {e}")
            
        # Create ZIP archive in exports folder
        exports_dir = Path.cwd() / "exports"
        exports_dir.mkdir(exist_ok=True)
        
        zip_filename = f"STBESA_LAYERS_{self.province}_{timestamp}.zip"
        zip_path = exports_dir / zip_filename
        
        # List files to zip
        files_to_zip = [f for f in out_dir.iterdir() if f.is_file()]
        print(f"[LayerExporter] Found {len(files_to_zip)} files to ZIP")
        
        if len(files_to_zip) == 0:
            print("[WARN] No files found in export directory!")
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for file_path in files_to_zip:
                print(f"[LayerExporter] Adding to ZIP: {file_path.name}")
                zf.write(file_path, file_path.name)
        
        print(f"[LayerExporter] ZIP created: {zip_path} ({zip_path.stat().st_size} bytes)")
        
        # Cleanup directory
        import time
        time.sleep(1) # Wait for file handles to release
        try:
            shutil.rmtree(out_dir)
        except Exception as e:
            print(f"[WARN] Could not clean up {out_dir}: {e}")
            
        return zip_path

