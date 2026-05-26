from abc import ABC, abstractmethod
from typing import Any, Dict, Set

import dask.array as da
import numpy as np
import xarray as xr
from scipy.ndimage import uniform_filter


class PaperImplementation(ABC):
    """
    Base class for research paper logic.
    Focuses on declaring data dependencies and applying scientific transformations.
    """

    def __init__(self, **params: Any):
        """
        Stores configuration parameters for metadata tracking.

        Args:
            **params: Configuration parameters specific to the implementation.
        """
        self.params = params

    @property
    def name(self) -> str:
        """Returns the class name as a identifier for the transformation."""
        return self.__class__.__name__

    @property
    @abstractmethod
    def required_products_and_bands(self) -> Dict[str, Set[str]]:
        """Declare dependencies (e.g., {'VNP46A2': {'DNB_BRDF-Corrected_NTL'}})"""
        pass

    def _standardize_dataset(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Standardizes NTL band names to ensure robustness to hyphen vs underscore variations.
        """
        if (
            ds is not None
            and "DNB_BRDF_Corrected_NTL" in ds.data_vars
            and "DNB_BRDF-Corrected_NTL" not in ds.data_vars
        ):
            return ds.rename({"DNB_BRDF_Corrected_NTL": "DNB_BRDF-Corrected_NTL"})
        return ds

    def transform(self, ds: xr.Dataset, **kwargs) -> xr.DataArray | xr.Dataset:
        """
        Apply the science transformation.
        Standardizes the dataset and then calls _transform.
        """
        if ds is not None:
            ds = self._standardize_dataset(ds)
        return self._transform(ds, **kwargs)

    @abstractmethod
    def _transform(self, ds: xr.Dataset, **kwargs) -> xr.DataArray | xr.Dataset:
        """
        Apply the science transformation.
        Returns either a single xarray object or a dictionary of objects.
        """
        pass

    @staticmethod
    def _spatial_rolling_mean(
        da_ntl: xr.DataArray, window_y: int, window_x: int
    ) -> xr.DataArray:
        """
        Applies a spatial 2D rolling mean using dask's map_overlap.
        This avoids the massive graph generation and memory overhead
        of xarray's `.rolling().mean()` on large spatiotemporal datasets
        chunked along spatial dimensions.
        """
        if "y" not in da_ntl.dims or "x" not in da_ntl.dims:
            raise ValueError("Spatial rolling mean requires 'y' and 'x' dimensions.")

        y_axis = da_ntl.dims.index("y")
        x_axis = da_ntl.dims.index("x")

        def chunk_filter(
            block: np.ndarray, w_y: int, w_x: int, ax_y: int, ax_x: int
        ) -> np.ndarray:
            valid = ~np.isnan(block)
            filled = np.nan_to_num(block)

            size = [1] * block.ndim
            size[ax_y] = w_y
            size[ax_x] = w_x

            sum_filled = uniform_filter(filled, size=size, mode="constant", cval=0.0) * (
                w_y * w_x
            )
            count_valid = uniform_filter(
                valid.astype(float), size=size, mode="constant", cval=0.0
            ) * (w_y * w_x)

            with np.errstate(divide="ignore", invalid="ignore"):
                res = sum_filled / count_valid

            res[count_valid < 1.0] = np.nan
            return res

        depth = {y_axis: window_y // 2, x_axis: window_x // 2}

        if isinstance(da_ntl.data, da.Array):
            res = da.map_overlap(
                chunk_filter,
                da_ntl.data,
                depth=depth,
                boundary="none",
                dtype=da_ntl.dtype,
                w_y=window_y,
                w_x=window_x,
                ax_y=y_axis,
                ax_x=x_axis,
            )
        else:
            # fallback for numpy arrays
            res = chunk_filter(da_ntl.data, window_y, window_x, y_axis, x_axis)

        return xr.DataArray(res, dims=da_ntl.dims, coords=da_ntl.coords)
