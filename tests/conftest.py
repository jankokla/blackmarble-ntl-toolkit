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


@pytest.fixture
def mock_vnp46a2_dataset() -> xr.Dataset:
    times = pd.date_range("2022-01-01", periods=1)

    ntl = np.arange(9, dtype=np.float32).reshape((1, 3, 3))

    qf = np.zeros((1, 3, 3), dtype=np.uint16)
    qf[0, 1, 1] = 128

    mqf = np.zeros((1, 3, 3), dtype=np.uint8)
    mqf[0, 0, 0] = 1

    snow_flag = np.zeros((1, 3, 3), dtype=np.uint8)

    moon = np.full((1, 3, 3), 50.0, dtype=np.float32)

    ds = xr.Dataset(
        data_vars={
            "DNB_BRDF_Corrected_NTL": (["time", "y", "x"], ntl),
            "QF_Cloud_Mask": (["time", "y", "x"], qf),
            "Mandatory_Quality_Flag": (["time", "y", "x"], mqf),
            "Snow_Flag": (["time", "y", "x"], snow_flag),
            "DNB_Lunar_Irradiance": (["time", "y", "x"], moon),
        },
        coords={"time": times, "y": [10, 20, 30], "x": [100, 110, 120]},
    )
    return ds


@pytest.fixture
def mock_vnp46a1_dataset() -> xr.Dataset:
    times = pd.date_range("2022-01-01", periods=1)

    sensor_z = np.full((1, 3, 3), 1000.0, dtype=np.float32)
    sensor_a = np.full((1, 3, 3), 2000.0, dtype=np.float32)
    solar_z = np.full((1, 3, 3), 11000.0, dtype=np.float32)
    moon_frac = np.full((1, 3, 3), 50.0, dtype=np.float32)

    ds = xr.Dataset(
        data_vars={
            "Sensor_Zenith": (["time", "y", "x"], sensor_z),
            "Sensor_Azimuth": (["time", "y", "x"], sensor_a),
            "Solar_Zenith": (["time", "y", "x"], solar_z),
            "Moon_Illumination_Fraction": (["time", "y", "x"], moon_frac),
        },
        coords={"time": times, "y": [10, 20, 30], "x": [100, 110, 120]},
    )
    return ds
