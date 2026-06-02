import os
from unittest.mock import MagicMock, patch

import ee
import pytest
import xarray as xr

from blackmarble_toolkit.retrieval import (
    BlackMarbleRetriever,
    bbox_to_geometry,
    geojson_to_geometry,
)


@pytest.fixture
def mock_env():
    """
    temporarily sets the environment variable for testing.
    """
    os.environ["GOOGLE_CLOUD_PROJECT"] = "test-project-123"
    yield
    del os.environ["GOOGLE_CLOUD_PROJECT"]


class TestGeometryHelpers:
    """
    Test the explicit geometry helper functions.
    """

    @patch("blackmarble_toolkit.retrieval.ee")
    def test_bbox_to_geometry(self, mock_ee):
        """Test that a 4-element list/tuple correctly calls ee.Geometry.BBox."""
        bbox = [-72.0, 18.0, -71.0, 19.0]
        bbox_to_geometry(bbox)
        mock_ee.Geometry.BBox.assert_called_once_with(*bbox)

    def test_bbox_to_geometry_invalid_length(self):
        """
        Ensure passing a bounding box with the wrong number of elements raises an error.
        """
        with pytest.raises(ValueError, match="exactly 4 elements"):
            bbox_to_geometry([-72.0, 18.0, -71.0])

    @patch("blackmarble_toolkit.retrieval.ee")
    def test_geojson_to_geometry_feature_collection(self, mock_ee):
        """Test that a geojson FeatureCollection routes to ee.FeatureCollection."""
        geojson = {"type": "FeatureCollection", "features": []}
        geojson_to_geometry(geojson)
        mock_ee.FeatureCollection.assert_called_once_with(geojson)

    @patch("blackmarble_toolkit.retrieval.ee")
    def test_geojson_to_geometry_polygon(self, mock_ee):
        """Tests that a geojson Polygon routes to ee.Geometry."""
        geojson = {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]}
        geojson_to_geometry(geojson)
        mock_ee.Geometry.assert_called_once_with(geojson)

    def test_geojson_to_geometry_missing_type(self):
        """Ensures an invalid geojson dictionary raises an error."""
        with pytest.raises(ValueError, match="missing 'type' key"):
            geojson_to_geometry({"coordinates": []})


class TestBlackMarbleRetriever:
    """Test the main BlackMarbleRetriever class."""

    def test_get_data_missing_project(self):
        """Ensure an error is raised if GOOGLE_CLOUD_PROJECT is missing."""
        if "GOOGLE_CLOUD_PROJECT" in os.environ:
            del os.environ["GOOGLE_CLOUD_PROJECT"]

        retriever = BlackMarbleRetriever()
        mock_region = MagicMock(spec=ee.Geometry)

        with pytest.raises(RuntimeError, match="Earth Engine project is not set"):
            retriever.get_data(
                product="VNP46A2",
                start_date="2021-01-01",
                end_date="2021-02-01",
                region=mock_region,
            )

    @patch("blackmarble_toolkit.retrieval.xr.open_dataset")
    @patch("blackmarble_toolkit.retrieval.ee.ImageCollection")
    def test_get_data_execution(
        self, mock_image_collection, mock_open_dataset, mock_env
    ):
        """Test the complete data retrieval assuming a pre-built ee.Geometry."""
        # mock the ee.ImageCollection chaining (.filterBounds().filterDate())
        mock_ic_instance = MagicMock()
        mock_image_collection.return_value = mock_ic_instance
        mock_ic_instance.filterBounds.return_value = mock_ic_instance
        mock_ic_instance.filterDate.return_value = mock_ic_instance
        mock_ic_instance.select.return_value = mock_ic_instance

        mock_ds = xr.Dataset({"DNB_BRDF_Corrected_NTL": xr.DataArray([1, 2, 3])})
        mock_open_dataset.return_value = mock_ds

        retriever = BlackMarbleRetriever()

        mock_region = MagicMock(spec=ee.Geometry)
        mock_region.getInfo.return_value = {
            "type": "Polygon",
            "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]],
        }

        result_ds = retriever.get_data(
            product="VNP46A2",
            start_date="2021-01-01",
            end_date="2021-02-01",
            region=mock_region,
            bands=["DNB_BRDF_Corrected_NTL"],
        )

        # verify ee.ImageCollection was instantiated with the right path
        mock_image_collection.assert_called_with("NASA/VIIRS/002/VNP46A2")
        mock_ic_instance.filterBounds.assert_called_with(mock_region)

        assert "DNB_BRDF_Corrected_NTL" in result_ds.data_vars

    def test_get_data_invalid_product(self, mock_env):
        """Ensure requesting an unsupported product raises an error."""
        retriever = BlackMarbleRetriever()

        mock_region = MagicMock(spec=ee.Geometry)

        with pytest.raises(ValueError, match="Product 'MODIS' is not supported"):
            retriever.get_data(
                product="MODIS",
                start_date="2021-01-01",
                end_date="2021-02-01",
                region=mock_region,
            )
