from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Set

import dask.array as da
import numpy as np
import xarray as xr
from scipy.ndimage import uniform_filter


class PaperImplementation(ABC):
    """
    Base class for research paper logic.
    Focuses on declaring data dependencies and applying scientific transformations.
    """

    # Note: Google Earth Engine uses "DNB_BRDF_Corrected_NTL",
    # whereas blackmarblepy uses "DNB_BRDF-Corrected_NTL".
    target_var_name = "DNB_BRDF_Corrected_NTL"

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

    def __str__(self) -> str:
        return self.name

    @property
    @abstractmethod
    def required_products_and_bands(self) -> Dict[str, Set[str]]:
        """Declare dependencies (e.g., {'VNP46A2': {'DNB_BRDF_Corrected_NTL'}})"""
        pass

    def _standardize_dataset(self, ds: xr.Dataset) -> xr.Dataset:
        """Standardize NTL band names to the canonical 'ntl' name."""
        if ds is None:
            raise ValueError("Input dataset cannot be None.")

        mapping = {
            k: "ntl"
            for k in [self.target_var_name, "DNB_BRDF-Corrected_NTL"]
            if k in ds.data_vars
        }
        return ds.rename(mapping) if mapping else ds

    def _get_expected_ntl_name(self) -> Optional[str]:
        """Finds the raw NTL band name this step expects from its required dependencies."""
        raw_names = {self.target_var_name, "DNB_BRDF-Corrected_NTL"}
        reqs = self.required_products_and_bands
        if reqs:
            for bands in reqs.values():
                for band in bands:
                    if band in raw_names:
                        return band
        return None

    def _get_band(
        self, ds: xr.Dataset, band_name: str, catalog: dict, align_time: bool = True
    ) -> xr.DataArray:
        """
        Helper method to retrieve a band either from the primary dataset or from
        an auxiliary dataset provided in the catalog (passed via kwargs).
        Ensures temporal alignment with the primary dataset if applicable.
        """
        result = None
        if band_name in ds.data_vars or band_name in ds.coords:
            result = ds[band_name]
        else:
            for cat_ds in catalog.values():
                if band_name in cat_ds.data_vars or band_name in cat_ds.coords:
                    result = cat_ds[band_name]
                    break

        if result is None:
            raise KeyError(
                f"Band '{band_name}' not found in the primary dataset or the provided catalog."
            )

        if align_time and "time" in ds.dims and "time" in result.dims:
            result = result.reindex(time=ds.time)

        return result

    def transform(self, ds: xr.Dataset, **kwargs) -> xr.Dataset:
        """
        Apply the science transformation.
        If the step expects a raw NTL name and the pipeline provides 'ntl', it maps it temporarily.
        After the transformation, it standardizes the output back to 'ntl'.
        """
        if ds is not None:
            ds = self._standardize_dataset(ds)
            expected = self._get_expected_ntl_name()
            if expected and "ntl" in ds.data_vars:
                ds = ds.rename({"ntl": expected})

        ds = self._transform(ds, **kwargs)

        if ds is not None:
            if isinstance(ds, xr.DataArray):
                ds = ds.to_dataset(name=self.target_var_name)

            ds = self._standardize_dataset(ds)

        return ds

    @abstractmethod
    def _transform(self, ds: xr.Dataset, **kwargs) -> xr.Dataset:
        """
        Apply the science transformation.
        Returns a dataset with the applied changes.
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

            sum_filled = uniform_filter(
                filled, size=size, mode="constant", cval=0.0
            ) * (w_y * w_x)
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
