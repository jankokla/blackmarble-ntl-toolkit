import numpy as np
import xarray as xr

from blackmarble_toolkit.methods.filters import CloudSnowFilter, FilterLowNTL
from blackmarble_toolkit.methods.geometric import AveragePooling2D


def test_filter_low_ntl(mock_vnp46a2_dataset: xr.Dataset) -> None:
    method = FilterLowNTL(threshold=1.0)
    result = method.transform(mock_vnp46a2_dataset)

    ntl = result["ntl"].values
    assert np.isnan(ntl[0, 0, 0])
    assert ntl[0, 0, 1] == 1.0
    assert ntl[0, 2, 2] == 8.0


def test_cloud_snow_filter(mock_vnp46a2_dataset: xr.Dataset) -> None:
    method = CloudSnowFilter(buffer_size=1)
    result = method.transform(mock_vnp46a2_dataset)

    ntl = result["ntl"].values
    assert np.isnan(ntl[0, 1, 1])
    assert ntl[0, 0, 0] == 0.0
    assert ntl[0, 0, 1] == 1.0
    assert ntl[0, 2, 2] == 8.0


def test_average_pooling_2d(mock_vnp46a2_dataset: xr.Dataset) -> None:
    # AveragePooling2D typically takes filter_size as tuple or int
    method = AveragePooling2D(filter_size=(3, 3))
    result = method.transform(mock_vnp46a2_dataset)

    ntl = result["ntl"].values
    # The whole 3x3 array has values 0 to 8, so the mean is 4.0
    assert ntl[0, 1, 1] == 4.0
