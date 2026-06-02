import pytest
import xarray as xr
from xarray.testing import assert_allclose

from blackmarble_toolkit.methods.base import PaperImplementation


class MockStep(PaperImplementation):
    @property
    def required_products_and_bands(self):
        return {}

    def _transform(self, ds: xr.Dataset, **kwargs) -> xr.Dataset:
        return ds


def test_standardize_dataset() -> None:
    step = MockStep()

    # test 1: with hyphen
    ds1 = xr.Dataset({"DNB_BRDF-Corrected_NTL": (["x"], [1, 2])})
    out1 = step._standardize_dataset(ds1)
    assert "ntl" in out1.data_vars
    assert "DNB_BRDF-Corrected_NTL" not in out1.data_vars

    # test 2: with underscore
    ds2 = xr.Dataset({"DNB_BRDF_Corrected_NTL": (["x"], [1, 2])})
    out2 = step._standardize_dataset(ds2)
    assert "ntl" in out2.data_vars
    assert "DNB_BRDF_Corrected_NTL" not in out2.data_vars

    # test 3: already ntl
    ds3 = xr.Dataset({"ntl": (["x"], [1, 2])})
    out3 = step._standardize_dataset(ds3)
    assert "ntl" in out3.data_vars

    # test 4: None
    with pytest.raises(ValueError, match="Input dataset cannot be None."):
        step._standardize_dataset(None)


def test_spatial_rolling_mean(synthetic_dataset: xr.Dataset) -> None:
    """
    Test that the custom _spatial_rolling_mean matches the standard
    xarray .rolling().mean() implementation.
    """
    da_ntl = synthetic_dataset["ntl"]

    custom_mean = PaperImplementation._spatial_rolling_mean(
        da_ntl, window_y=3, window_x=3
    )

    standard_mean = da_ntl.rolling(y=3, x=3, center=True, min_periods=1).mean()

    assert custom_mean.shape == da_ntl.shape
    assert custom_mean.dims == da_ntl.dims

    # compare inner section to avoid boundary condition differences
    inner_custom = custom_mean.isel(y=slice(1, -1), x=slice(1, -1))
    inner_standard = standard_mean.isel(y=slice(1, -1), x=slice(1, -1))

    assert_allclose(inner_custom, inner_standard)
