import os
from typing import Any, Dict, List, Tuple

import ee
import xarray as xr
from xee import helpers
import shapely.geometry


def bbox_to_geometry(bbox: List[float] | Tuple[float, ...]) -> ee.Geometry:
    """
    Converts a bounding box list or tuple to an Earth Engine Geometry.

    Args:
        bbox: A list or tuple of 4 floats [min_lon, min_lat, max_lon, max_lat].

    Returns:
        An ee.Geometry.BBox object.

    Raises:
        ValueError: If the bounding box does not contain exactly 4 elements.
    """
    if len(bbox) != 4:
        raise ValueError(
            "Bounding box must contain exactly 4 elements: [min_lon, min_lat, max_lon, max_lat]"
        )
    return ee.Geometry.BBox(*bbox)


def geojson_to_geometry(geojson: Dict[str, Any]) -> ee.Geometry:
    """
    Converts a GeoJSON dictionary to an Earth Engine Geometry.

    Args:
        geojson: A dictionary representing a GeoJSON FeatureCollection or Polygon.

    Returns:
        An ee.Geometry or ee.FeatureCollection geometry object.

    Raises:
        ValueError: If the GeoJSON format is unrecognized.
    """
    if "type" not in geojson:
        raise ValueError("Invalid GeoJSON: missing 'type' key.")

    if geojson["type"] == "FeatureCollection":
        return ee.FeatureCollection(geojson).geometry()
    elif geojson["type"] in ["Polygon", "MultiPolygon", "Feature"]:
        return ee.Geometry(geojson)
    else:
        raise ValueError(f"Unsupported GeoJSON type: {geojson['type']}")


class BlackMarbleRetriever:
    """
    A seamless Earth Engine retriever for NASA's Black Marble NTL product suite.
    """

    _PRODUCT_CATALOG = {
        "VNP46A1": "NOAA/VIIRS/001/VNP46A1",
        "VNP46A2": "NASA/VIIRS/002/VNP46A2",
    }

    _NATIVE_SCALE_DEGREES = 15.0 / 3600.0

    def __init__(self, project_name: str | None = None):
        """
        Initializes the retriever and handles Earth Engine authentication.

        Args:
            project_name: The GCP project name for EE initialization. If None,
                it attempts to read the 'EE_PROJECT' environment variable.
        """
        self.project_name = project_name or os.environ.get("EE_PROJECT")

        if not self.project_name:
            raise ValueError(
                "Earth Engine project name not provided. Pass it to the constructor "
                "or set the 'EE_PROJECT' environment variable."
            )

        self._initialize_ee()

    def _initialize_ee(self) -> None:
        """
        Attempts to initialize Earth Engine. Falls back to authentication if required.
        """
        try:
            ee.Initialize(project=self.project_name)
        except ee.ee_exception.EEException:
            ee.Authenticate()
            ee.Initialize(project=self.project_name)

    def get_data(
        self,
        product: str,
        start_date: str,
        end_date: str,
        region: ee.Geometry,
        bands: List[str] | None = None,
    ) -> xr.Dataset:
        """
        Retrieves Black Marble data from Earth Engine into an xarray dataset.

        Args:
            product: The product name (e.g., 'VNP46A2', 'VNP46A4').
            start_date: The start date string (e.g., '2021-01-01').
            end_date: The end date string (e.g., '2025-01-01').
            region: The spatial bounding area explicitly as an ee.Geometry.
            bands: An optional list of specific bands to select.

        Returns:
            An xarray Dataset containing the requested data.
        """
        # validate the product against the catalog
        ee_collection_path = self._PRODUCT_CATALOG.get(product.upper())
        if not ee_collection_path:
            valid_products = ", ".join(self._PRODUCT_CATALOG.keys())
            raise ValueError(
                f"Product '{product}' is not supported. Valid options: {valid_products}"
            )

        ic = (
            ee.ImageCollection(ee_collection_path)
            .filterBounds(region)
            .filterDate(start_date, end_date)
        )

        if bands:
            ic = ic.select(bands)

        shapely_geom = shapely.geometry.shape(region.getInfo())

        grid = helpers.fit_geometry(
            shapely_geom,
            grid_scale=(self._NATIVE_SCALE_DEGREES, -self._NATIVE_SCALE_DEGREES),
        )

        ds = xr.open_dataset(filename_or_obj=ic, engine="ee", **grid)

        return ds
