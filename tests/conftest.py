import dask.array as da
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from typing import Dict


@pytest.fixture(params=["numpy", "dask"])
def array_backend(request):
    """Fixture to parameterize tests over both numpy and dask backends."""
    return request.param


@pytest.fixture
def synthetic_dataset(array_backend: str) -> xr.Dataset:
    """
    Generates a small synthetic Dataset with x, y, and time dimensions.
    It contains a primary 'ntl' variable.
    """
    times = pd.date_range("2023-01-01", periods=3)
    ys = np.arange(10)
    xs = np.arange(10)

    # Create some structured synthetic data (e.g. ones with some noise)
    np.random.seed(42)
    data = (
        np.ones((len(times), len(ys), len(xs)))
        + np.random.rand(len(times), len(ys), len(xs)) * 0.1
    )

    if array_backend == "dask":
        # Chunk it into 1x5x5 chunks to test dask block logic
        data = da.from_array(data, chunks=(1, 5, 5))

    ds = xr.Dataset(
        data_vars={"ntl": (["time", "y", "x"], data)},
        coords={"time": times, "y": ys, "x": xs},
    )
    return ds


@pytest.fixture
def synthetic_catalog(array_backend: str) -> Dict[str, xr.Dataset]:
    """
    Generates a mock catalog (dict of datasets) for testing dependencies.
    """
    ys = np.arange(10)
    xs = np.arange(10)

    # Just a simple static 2D mask
    mask_data = np.ones((len(ys), len(xs)))
    if array_backend == "dask":
        mask_data = da.from_array(mask_data, chunks=(5, 5))

    mask_ds = xr.Dataset(
        data_vars={"qa_mask": (["y", "x"], mask_data)}, coords={"y": ys, "x": xs}
    )

    return {"VNP46A2": mask_ds}
