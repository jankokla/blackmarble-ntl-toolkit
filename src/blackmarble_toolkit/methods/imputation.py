from typing import Any, Dict, Optional, Set

import xarray as xr

from blackmarble_toolkit import PaperImplementation


class LinearInterpolationGapFilling(PaperImplementation):
    """
    Simple linear interpolation imputation baseline.

    This class uses linear interpolation along the time dimension to fill
    missing values in the data.

    Args:
        limit (Optional[int]): Maximum number of consecutive NaNs to fill. Defaults to None.
        max_gap (Optional[int]): Maximum gap size to fill. Defaults to None.
        ffill (bool): Whether to apply forward fill after interpolation. Defaults to False.
        bfill (bool): Whether to apply backward fill after interpolation. Defaults to False.
        **kwargs: Additional parameters passed to the base class.
    """

    def __init__(
        self,
        limit: Optional[int] = None,
        max_gap: Optional[int] = None,
        ffill: bool = False,
        bfill: bool = False,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.limit = limit
        self.max_gap = max_gap
        self.ffill = ffill
        self.bfill = bfill

    @property
    def required_products_and_bands(self) -> Dict[str, Set[str]]:
        """
        Declares the required data products and bands.

        Returns:
            Dictionary with product names as keys and a set of required bands as values.
        """
        return {"VNP46A2": {"DNB_BRDF_Corrected_NTL"}}

    def _transform(self, ds: xr.Dataset, **kwargs: Any) -> xr.Dataset:
        """
        Applies linear interpolation over the time dimension.

        Args:
            ds: Dataset to be gap-filled.

        Returns:
            The gap-filled Dataset.
        """
        res = ds[[self.target_var_name]].interpolate_na(
            dim="time", method="linear", limit=self.limit, max_gap=self.max_gap
        )

        if self.ffill:
            res = res.ffill(dim="time")
        if self.bfill:
            res = res.bfill(dim="time")

        return res
