import os

import ee
import geopandas as gpd
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage


def initialize_ee(project_name: str | None = None) -> None:
    """
    Attempts to initialize Earth Engine. Falls back to authentication if required.

    Args:
        project_name: Optional explicit project name. If not provided,
            it attempts to read the 'GOOGLE_CLOUD_PROJECT' environment variable.
    """
    project = project_name or os.environ.get("GOOGLE_CLOUD_PROJECT")

    if not project:
        raise ValueError(
            "Earth Engine project name not provided. Pass it to the function "
            "or set the 'GOOGLE_CLOUD_PROJECT' environment variable."
        )

    try:
        ee.Initialize(project=project)
    except ee.ee_exception.EEException:
        ee.Authenticate()
        ee.Initialize(project=project)


def cluster_geometries(
    gdf: gpd.GeoDataFrame, max_distance_deg: float = 2.0
) -> gpd.GeoDataFrame:
    """
    Cluster geometries in a GeoDataFrame based on their distance.

    Args:
        gdf: GeoDataFrame containing the shapes to cluster.
        max_distance_deg: Maximum separation distance to belong to the same cluster (in degrees).

    Returns:
        A copy of the GeoDataFrame with an assigned 'cluster_id' column.
    """
    if len(gdf) <= 1:
        gdf_clustered = gdf.copy()
        gdf_clustered["cluster_id"] = [1] * len(gdf)
        return gdf_clustered

    centroids = gdf.geometry.centroid
    coords = np.column_stack((centroids.x, centroids.y))

    Z = linkage(coords, method="single", metric="euclidean")

    labels = fcluster(Z, t=max_distance_deg, criterion="distance")

    gdf_clustered = gdf.copy()
    gdf_clustered["cluster_id"] = labels
    return gdf_clustered
