from typing import Literal, Optional, Tuple, Union

import geopandas as gpd
import xarray as xr
from geocube.api.core import make_geocube


def get_spatial_dims(ds: Union[xr.Dataset, xr.DataArray]) -> Tuple[str, str]:
    """
    Dynamically identifiy the spatial dimensions (x/lon, y/lat) of a dataset.

    Args:
        ds: The input xarray Dataset or DataArray.

    Returns:
        A tuple containing the names of the x and y dimensions.
    """
    if hasattr(ds, "rio"):
        try:
            x_dim = ds.rio.x_dim
            y_dim = ds.rio.y_dim
            if x_dim and y_dim:
                return str(x_dim), str(y_dim)
        except Exception:
            pass

    for x_cand, y_cand in [("x", "y"), ("lon", "lat"), ("longitude", "latitude")]:
        if x_cand in ds.dims and y_cand in ds.dims:
            return x_cand, y_cand

    raise ValueError(
        f"Could not determine spatial dimensions for dataset with dims: {list(ds.dims)}"
    )


def get_agg_per_shape(
    ds: xr.Dataset,
    mask: xr.Dataset,
    variable: str,
    agg_type: Literal["mean", "median"] = "mean",
    is_valid_pct: bool = False,
    valid_pct_threshold: Optional[float] = None,
    geo_id_col: str = "geonameid",
) -> xr.Dataset:
    """
    Memory-safe aggregation using Dask and Zarr.

    Args:
        ds: The dataset containing the variable to aggregate.
        mask: The spatial mask dataset containing the geometry IDs.
        variable: The name of the data variable to aggregate.
        agg_type: The type of aggregation to perform ('mean' or 'median').
        is_valid_pct: Whether to calculate the percentage of valid (non-null) pixels.
        valid_pct_threshold: Optional threshold between 0 and 1. Aggregations with 
            valid pixel percentage below this threshold will be masked as NaN.
        geo_id_col: The column name uniquely identifying the shapes.

    Returns:
        A dataset containing the spatially aggregated values and optionally the valid percentages.
    """
    x_dim, y_dim = get_spatial_dims(ds)
    ds = ds.assign_coords({geo_id_col: mask[geo_id_col]})

    grouped = ds[variable].groupby(geo_id_col)

    if agg_type == "mean":
        agg_da = grouped.mean(dim=[x_dim, y_dim])
    else:
        agg_da = grouped.quantile(0.5, dim=[x_dim, y_dim]).drop_vars(
            "quantile", errors="ignore"
        )

    output_vars = {variable: agg_da}

    if is_valid_pct or valid_pct_threshold is not None:
        valid_pct_da = (
            ds[variable]
            .notnull()
            .astype(float)
            .groupby(geo_id_col)
            .mean(dim=[x_dim, y_dim])
        )

        if valid_pct_threshold is not None:
            if not (0 <= valid_pct_threshold <= 1):
                raise ValueError("valid_pct_threshold must be between 0 and 1.")
            output_vars[variable] = agg_da.where(valid_pct_da >= valid_pct_threshold)

        if is_valid_pct:
            output_vars["valid_pct"] = valid_pct_da

    return xr.Dataset(output_vars)


def get_gdf_mask_for_ds(
    ds: xr.Dataset, gdf: gpd.GeoDataFrame, geo_id_col: str = "geonameid"
) -> xr.Dataset:
    """
    Creates a spatial mask for an xarray Dataset based on a given GeoDataFrame.

    Args:
        ds: The reference dataset to match the spatial grid.
        gdf: The GeoDataFrame containing the vector shapes.
        geo_id_col: The column name in the GeoDataFrame that uniquely identifies each shape.

    Returns:
        A rasterized dataset mask where pixel values correspond to the geometry IDs.
    """
    x_dim, y_dim = get_spatial_dims(ds)
    ds = ds.rio.set_spatial_dims(x_dim=x_dim, y_dim=y_dim)

    if not ds.rio.crs:
        ds = ds.rio.write_crs("EPSG:4326")

    mask = make_geocube(gdf, measurements=[geo_id_col], like=ds)

    if x_dim != "x" or y_dim != "y":
        mask = mask.rename({"x": x_dim, "y": y_dim})

    return mask.assign_coords({x_dim: ds[x_dim], y_dim: ds[y_dim]})


def aggregate_to_shapes(
    ds: xr.Dataset,
    gdf: gpd.GeoDataFrame,
    variable: str = "DNB_BRDF-Corrected_NTL",
    agg_type: Literal["mean", "median"] = "mean",
    is_valid_pct: bool = False,
    valid_pct_threshold: Optional[float] = None,
    geo_id_col: str = "geonameid",
) -> xr.Dataset:
    """
    High-level helper to aggregate an xarray dataset to vector shapes.

    Args:
        ds: The input dataset.
        gdf: The GeoDataFrame containing the vector shapes.
        variable: The name of the data variable to aggregate.
        agg_type: The type of aggregation to perform ('mean' or 'median').
        is_valid_pct: Whether to calculate the percentage of valid pixels.
        valid_pct_threshold: Optional threshold to mask out geometries with 
            insufficient valid data.
        geo_id_col: The column name uniquely identifying the shapes.

    Returns:
        A dataset containing the spatially aggregated values.
    """
    if variable not in ds.data_vars:
        raise ValueError(f"Variable '{variable}' not found in dataset.")

    ds_subset = ds[[variable]]
    mask = get_gdf_mask_for_ds(ds_subset, gdf, geo_id_col=geo_id_col)

    zonal_ds = get_agg_per_shape(
        ds_subset,
        mask,
        variable,
        agg_type=agg_type,
        is_valid_pct=is_valid_pct,
        valid_pct_threshold=valid_pct_threshold,
        geo_id_col=geo_id_col,
    )

    return zonal_ds
