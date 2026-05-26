from abc import abstractmethod
from typing import Dict, Set

import xarray as xr

from blackmarble_toolkit import PaperImplementation


class LinearInterpolationGapFilling(PaperImplementation):
    """
    Simple linear interpolation imputation baseline.

    This class uses linear interpolation along the time dimension to fill
    missing values in the data.
    """

    @property
    def required_products_and_bands(self) -> Dict[str, Set[str]]:
        """
        Declares the required data products and bands.

        Returns:
            Dictionary with product names as keys and a set of required bands as values.
        """
        return {"VNP46A2": {"DNB_BRDF-Corrected_NTL"}}

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

    def _transform(
        self, ds: xr.Dataset | None = None, **kwargs
    ) -> xr.DataArray | xr.Dataset:
        """
        Apply the science transformation.
        Standardizes the dataset and then calls _transform.
        """
        if ds is not None:
            ds = self._standardize_dataset(ds)
        return self._transform(ds, **kwargs)

    @abstractmethod
    def _transform(
        self, ds: xr.Dataset | None = None, **kwargs
    ) -> xr.DataArray | xr.Dataset:
        """
        Applies linear interpolation over the time dimension.

        Args:
            ds: Dataset to be gap-filled.

        Returns:
            The gap-filled Dataset or DataArray.
        """
        return ds[["DNB_BRDF-Corrected_NTL"]].interpolate_na(dim="time", method="linear")
