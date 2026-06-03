import os
import warnings
from typing import Any, Dict, List, Literal, Tuple

import ee
import geopandas as gpd
import pandas as pd
import shapely.geometry
import xarray as xr
from shapely.geometry import shape
from xee import helpers

warnings.filterwarnings(
    "ignore",
    message=".*Earth Engine is not initialized on worker.*",
    category=UserWarning,
)


def gdf_to_geometry(gdf: gpd.GeoDataFrame) -> ee.Geometry:
    """
    Convert a GeoPandas GeoDataFrame to an Earth Engine Geometry.

    Args:
        gdf: A GeoDataFrame containing the spatial data to convert.

    Returns:
        An ee.Geometry object representing the combined geometries of the dataframe.

    Raises:
        ValueError: If the GeoDataFrame is empty.
    """
    if gdf.empty:
        raise ValueError("Invalid GeoDataFrame empty dataframe provided.")

    geojson_dict = gdf.__geo_interface__

    return ee.FeatureCollection(geojson_dict).geometry()


def geojson_to_gdf(geojson_dict: Dict[str, Any], epsg: int = 4326) -> gpd.GeoDataFrame:
    """
    Converts a GeoJSON dictionary into a GeoPandas GeoDataFrame.

    Args:
        geojson_dict: Dictionary representing a GeoJSON FeatureCollection or Feature.
        epsg: The EPSG code for the coordinate reference system. Defaults to 4326.

    Returns:
        A GeoDataFrame containing the parsed spatial data.
    """
    if "features" in geojson_dict:
        features = geojson_dict["features"]
    else:
        features = [geojson_dict]

    gdf = gpd.GeoDataFrame.from_features(features)

    gdf.set_crs(epsg=epsg, inplace=True)

    return gdf


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


def geometry_to_gdf(ee_geom):
    """
    Convert an Earth Engine geometry to a GeoDataFrame.

    Args:
        ee_geom: The Earth Engine geometry to convert.

    Returns:
        A GeoDataFrame containing the converted geometry with WGS84 CRS.
    """
    geojson_dict = ee_geom.getInfo()

    shapely_geom = shape(geojson_dict)

    gdf = gpd.GeoDataFrame(index=[0], geometry=[shapely_geom], crs="EPSG:4326")

    return gdf


class BlackMarbleRetriever:
    """
    An Earth Engine retriever for NASA's Black Marble NTL product suite.
    """

    _PRODUCT_CATALOG = {
        "VNP46A1": "NOAA/VIIRS/001/VNP46A1",
        "VNP46A2": "NASA/VIIRS/002/VNP46A2",
        "NOAA/VIIRS/DNB/ANNUAL_V22": "NOAA/VIIRS/DNB/ANNUAL_V22",
    }

    _NATIVE_SCALE_DEGREES = 15.0 / 3600.0

    _PRODUCT_AVAILABILITY = {
        "VNP46A1": ("2012-01-19", "2025-01-02"),
        "VNP46A2": ("2012-01-19", None),
        "NOAA/VIIRS/DNB/ANNUAL_V22": ("2012-04-01", "2025-01-01"),
    }

    def _check_dataset_availability(
        self, product: str, start_date: str, end_date: str
    ) -> None:
        """Check if the requested dates fall within the known available dataset range."""
        if product not in self._PRODUCT_AVAILABILITY:
            return

        avail_start, avail_end = self._PRODUCT_AVAILABILITY[product]

        try:
            start_dt = pd.to_datetime(start_date, utc=True)
            end_dt = pd.to_datetime(end_date, utc=True)

            avail_start_dt = pd.to_datetime(avail_start, utc=True)
            avail_end_dt = (
                pd.to_datetime(avail_end, utc=True)
                if avail_end
                else pd.Timestamp.now(tz="UTC")
            )

            if start_dt < avail_start_dt or end_dt > avail_end_dt:
                end_str = avail_end if avail_end else "present"
                warnings.warn(
                    f"{product} data is generally available from {avail_start} to {end_str}.",
                    UserWarning,
                )
        except Exception:
            pass

    def get_data(
        self,
        product: Literal["VNP46A1", "VNP46A2", "NOAA/VIIRS/DNB/ANNUAL_V22"],
        start_date: str,
        end_date: str,
        region: ee.Geometry,
        bands: List[str] | None = None,
        chunks: Any = "auto",
        scale: float | None = None,
    ) -> xr.Dataset:
        """
        Retrieves Black Marble data from Earth Engine into an xarray dataset.

        Args:
            product: The product name (e.g., 'VNP46A2', 'VNP46A4').
            start_date: The start date string (e.g., '2021-01-01').
            end_date: The end date string (e.g., '2025-01-01').
            region: The spatial bounding area explicitly as an ee.Geometry.
            bands: An optional list of specific bands to select.
            chunks: Chunk size or scheme for the resulting xarray dataset. Defaults to "auto".
            scale: Spatial resolution in degrees. Defaults to native scale (approx 15 arc-seconds).

        Returns:
            An xarray Dataset containing the requested data.
        """
        if product not in self._PRODUCT_CATALOG:
            raise ValueError(f"Product '{product}' is not supported.")

        self._check_dataset_availability(product, start_date, end_date)

        if not os.environ.get("GOOGLE_CLOUD_PROJECT"):
            raise RuntimeError(
                "Earth Engine project is not set in the environment. "
                "Please set the 'GOOGLE_CLOUD_PROJECT' environment variable "
                "to your Earth Engine project name so that the background workers can initialize."
            )

        ee_collection_path = self._PRODUCT_CATALOG[product]

        ic = (
            ee.ImageCollection(ee_collection_path)
            .filterBounds(region)
            .filterDate(start_date, end_date)
        )

        if bands:
            ic = ic.select(bands)

        shapely_geom = shapely.geometry.shape(region.getInfo())

        actual_scale = scale or self._NATIVE_SCALE_DEGREES
        grid = helpers.fit_geometry(
            shapely_geom,
            grid_scale=(actual_scale, -actual_scale),
        )

        ds = xr.open_dataset(
            filename_or_obj=ic,
            engine="ee",
            chunks=chunks,
            ee_init_if_necessary=True,
            ee_init_kwargs={"project": os.environ["GOOGLE_CLOUD_PROJECT"]},
            **grid,
        )

        return ds
