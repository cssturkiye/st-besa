"""
Folium-based Earth Engine map rendering helpers.

This module avoids importing geemap.foliumap, which is currently incompatible
with the installed geemap/xyzservices combination used by the app environment.
"""

from typing import Any, Dict


class EarthEngineFoliumMap:
    """Small compatibility wrapper for the subset of geemap.Map used by ST-BESA."""

    def __init__(self, analysis_service, height: str = "700px"):
        import folium

        self.analysis = analysis_service
        self._map = folium.Map(
            location=[20, 0],
            zoom_start=2,
            tiles="OpenStreetMap",
            height=height,
            control_scale=True,
        )

    def addLayer(
        self,
        ee_object,
        vis_params: Dict[str, Any] | None = None,
        name: str = "Layer",
        shown: bool = True,
        opacity: float = 1.0,
    ) -> None:
        """Add an Earth Engine object as a Folium tile layer."""
        import folium

        map_id = self.analysis._ee_get_mapid(ee_object, vis_params or {})
        tile_fetcher = map_id["tile_fetcher"]
        tile_url = getattr(tile_fetcher, "url_format", None)
        if tile_url is None:
            tile_url = tile_fetcher["url_format"]

        folium.TileLayer(
            tiles=tile_url,
            attr="Google Earth Engine",
            name=name,
            overlay=True,
            control=True,
            show=shown,
            opacity=opacity,
            max_zoom=24,
        ).add_to(self._map)

    def centerObject(self, ee_object, zoom: int = 9) -> None:
        """Center the map around an Earth Engine geometry-like object."""
        import ee

        try:
            bounds = self.analysis._ee_getinfo(
                ee.Geometry(ee_object).bounds().coordinates()
            )
            ring = bounds[0] if isinstance(bounds, list) and bounds else []
            xs = [float(pt[0]) for pt in ring]
            ys = [float(pt[1]) for pt in ring]
            if xs and ys:
                self._map.fit_bounds([[min(ys), min(xs)], [max(ys), max(xs)]])
                return
        except Exception:
            pass

        try:
            lon, lat = self.analysis._ee_getinfo(
                ee.Geometry(ee_object).centroid(1).coordinates()
            )
            self._map.location = [float(lat), float(lon)]
            self._map.zoom_start = zoom
        except Exception:
            self._map.zoom_start = zoom

    def addLayerControl(self) -> None:
        """Add a standard Folium layer control."""
        import folium

        folium.LayerControl(collapsed=False).add_to(self._map)

    def _repr_html_(self) -> str:
        return self._map._repr_html_()
