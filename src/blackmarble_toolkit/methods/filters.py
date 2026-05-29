from typing import Any, Dict, Set

import bottleneck as bn
import dask.array as da
import numpy as np
import xarray as xr
from scipy.ndimage import maximum_filter

from blackmarble_toolkit import PaperImplementation


class BlackMarbleHighQualityFilter(PaperImplementation):
    """
    A specialized filter for VNP46A2 NTL data focusing on primary quality indicator.

    Unlike broader screening implementations, this class relies strictly on the
    instrument's internal quality assessments. It isolates pixels that are
    labeled as 'High Quality' by the mandatory sensor checks.
    """

    def __init__(
        self,
    ):
        """
        Initializes the High Quality filter with default NASA VNP46A2 flag requirements.
        """
        super().__init__()

    @property
    def required_products_and_bands(self) -> Dict[str, Set[str]]:
        """
        Required bands from VNP46A2 (radiance/quality).
        """
        return {
            "VNP46A2": {
                "DNB_BRDF_Corrected_NTL",
                "Mandatory_Quality_Flag",
            },
        }

    def _transform(self, ds: xr.Dataset, **kwargs) -> xr.Dataset:
        """
        Applies the filter to isolate high-quality NTL observations.
        """
        ntl = ds["DNB_BRDF_Corrected_NTL"]

        quality_mask = ds["Mandatory_Quality_Flag"] == 0
        positive_mask = ntl >= 0

        masked_ntl = ntl.where(quality_mask & positive_mask)
        return ds.assign(DNB_BRDF_Corrected_NTL=masked_ntl)


class FilterLowNTL(PaperImplementation):
    """
    Filters out low nighttime light radiance values.

    Implements a low radiance background filter as described in:

    Li, T., Zhu, Z., Wang, Z., Román, M. O., Kalb, V. L., & Zhao, Y. (2022).
    Continuous monitoring of nighttime light changes based on daily NASA's Black
    Marble product suite. Remote Sensing of Environment, 282, 113269.
    https://doi.org/10.1016/j.rse.2022.113269
    """

    @property
    def required_products_and_bands(self) -> Dict[str, Set[str]]:
        """
        Declare dependencies.

        Returns:
            A dictionary specifying the required product and band.
        """
        return {"VNP46A2": {"DNB_BRDF_Corrected_NTL"}}

    def _transform(self, ds: xr.Dataset, **kwargs) -> xr.Dataset:
        """
        Applies the low radiance filter to the dataset.

        Args:
            ds: The input dataset containing the NTL band.
            **kwargs: Additional keyword arguments passed to the transform.

        Returns:
            A Dataset with values below 1 replaced by NaN in the NTL band.
        """
        var_name = "DNB_BRDF_Corrected_NTL"

        da_ntl = ds[var_name]

        filtered_da = da_ntl.where(da_ntl >= 1.0)

        filtered_da.attrs = da_ntl.attrs

        return ds.assign({var_name: filtered_da})


class Jia2023HighQualityFilter(PaperImplementation):
    """
    Implements the four-step high-quality data retrieval process to filter
    daily Black Marble NTL radiance as described in:

    Jia, M., Li, X., Gong, Y., Belabbes, S., & Dell'Oro, L. (2023).
    Estimating natural disaster loss using improved daily night-time light data.
    International Journal of Applied Earth Observation and Geoinformation, 120, 103359.
    https://doi.org/10.1016/j.jag.2023.103359

    This class filters out observations affected by solar stray light, clouds,
    abnormal sensor values, and high lunar illumination.
    """

    def __init__(
        self,
        sza_threshold: float = 108.0,
        moon_fraction_threshold: float = 60,
    ):
        """
        Initializes the Jia et al. (2023) filter with paper-defined thresholds.

        Args:
            sza_threshold: Solar Zenith Angle threshold to ensure deep night.
                Paper uses 108°.
            moon_fraction_threshold: Max moon illumination fraction allowed.
                Paper uses 0.6 (60%).
        """
        super().__init__(
            sza_threshold=sza_threshold,
            moon_fraction_threshold=moon_fraction_threshold,
        )
        self.sza_threshold = sza_threshold
        self.moon_fraction_threshold = moon_fraction_threshold

    @property
    def required_products_and_bands(self) -> Dict[str, Set[str]]:
        """
        Required bands spanning VNP46A1 (geometry) and VNP46A2 (radiance/quality).
        """
        return {
            "VNP46A1": {"Solar_Zenith", "Moon_Illumination_Fraction"},
            "VNP46A2": {
                "DNB_BRDF_Corrected_NTL",
                "Mandatory_Quality_Flag",
                "QF_Cloud_Mask",
            },
        }

    def _transform(self, ds: xr.Dataset, **kwargs) -> xr.Dataset:
        """
        Applies the four screening steps to isolate high-quality NTL observations.
        """
        ntl = ds["DNB_BRDF_Corrected_NTL"]
        solar_zenith = self._get_band(ds, "Solar_Zenith", kwargs)
        moon_fraction = self._get_band(ds, "Moon_Illumination_Fraction", kwargs)

        # 1. solar filtering: pixels with SZA < 108° are removed to avoid stray light
        solar_mask = solar_zenith >= self.sza_threshold

        # 2. cloud labeling
        qf_safe = ds["QF_Cloud_Mask"].fillna(192).astype(np.uint16)
        cloud_free_mask = ((qf_safe >> 6) & 3) == 0

        # 3. quality check (Mandatory_Quality_Flag == 0)
        quality_mask = ds["Mandatory_Quality_Flag"] == 0

        # 4. moonlight mitigation: discards pixels with moon illumination > 60%
        moon_mask = moon_fraction <= self.moon_fraction_threshold

        final_mask = solar_mask & cloud_free_mask & quality_mask & moon_mask

        masked_ntl = ntl.where(final_mask)
        return ds.assign(DNB_BRDF_Corrected_NTL=masked_ntl)


class CloudSnowFilter(PaperImplementation):
    """
    Applies a spatial buffer for removing cloud and snow edge pixels.

    Implements the cloud and snow masking process for daily
    NASA's Black Marble NTL data as described in:

    Li, T., Zhu, Z., Wang, Z., Román, M. O., Kalb, V., & Zhao, Y. (2022).
    Continuous monitoring of nighttime light changes based on daily NASA's
    Black Marble product suite. Remote Sensing of Environment, 282, 113269.
    https://doi.org/10.1016/j.rse.2022.113269

    This class filters out observations affected by clouds, cirrus, and snow/ice
    surfaces, and dilates the resulting mask to ensure edge pixels are removed.
    """

    def __init__(self, buffer_size: int = 5, return_mask: bool = False, **kwargs):
        super().__init__(buffer_size=buffer_size, **kwargs)
        self.buffer_size = buffer_size
        self.return_mask = return_mask

    @property
    def required_products_and_bands(self) -> Dict[str, Set[str]]:
        """
        Specifiy the required products and bands for the filter.

        Returns:
            dictionary of required data.
        """
        return {
            "VNP46A2": {
                "DNB_BRDF_Corrected_NTL",
                "QF_Cloud_Mask",
                "Snow_Flag",
            }
        }

    def _transform(self, ds: xr.Dataset, **kwargs) -> xr.Dataset:
        """
        Apply cloud and snow masking, dilates the mask, and masks the NTL data.

        Args:
            ds: input dataset containing VIIRS bands.
            **kwargs: additional arguments.

        Returns:
            masked NTL data or a dataset containing NTL and the buffered mask.
        """
        if "y" not in ds.dims or "x" not in ds.dims:
            raise ValueError("CloudSnowFilter requires 'y' and 'x' dimensions.")

        qf = ds["QF_Cloud_Mask"]
        snow_flag = ds["Snow_Flag"]
        ntl = ds["DNB_BRDF_Corrected_NTL"]

        # bit 6-7: cloud detection results & confidence indicator
        # 10 = probably cloudy, 11 = confident cloudy
        cloud_conf = (qf.fillna(0).astype(int) >> 6) & 3
        cloud_mask = cloud_conf >= 2

        # bit 9: cirrus detection
        cirrus_mask = ((qf.fillna(0).astype(int) >> 9) & 1) == 1

        # bit 10: snow/ ice surface
        snow_ice_mask = ((qf.fillna(0).astype(int) >> 10) & 1) == 1

        # snow_flag variable
        snow_flag_mask = snow_flag == 1

        combined_mask = cloud_mask | cirrus_mask | snow_ice_mask | snow_flag_mask

        y_axis = combined_mask.dims.index("y")
        x_axis = combined_mask.dims.index("x")

        def _buffer_chunk(block, w_y, w_x, ax_y, ax_x):
            size = [1] * block.ndim
            size[ax_y] = w_y
            size[ax_x] = w_x
            return maximum_filter(block, size=size, mode="constant", cval=0.0)

        depth = {y_axis: self.buffer_size // 2, x_axis: self.buffer_size // 2}

        if isinstance(combined_mask.data, da.Array):
            buffered_mask_data = da.map_overlap(
                _buffer_chunk,
                combined_mask.data,
                depth=depth,
                boundary="none",
                dtype=bool,
                w_y=self.buffer_size,
                w_x=self.buffer_size,
                ax_y=y_axis,
                ax_x=x_axis,
            )
        else:
            buffered_mask_data = _buffer_chunk(
                combined_mask.data, self.buffer_size, self.buffer_size, y_axis, x_axis
            )

        buffered_mask = xr.DataArray(
            buffered_mask_data, dims=combined_mask.dims, coords=combined_mask.coords
        ).rename("buffered_mask")

        masked_ntl = xr.where(~buffered_mask, ntl, np.nan).rename(
            "DNB_BRDF_Corrected_NTL"
        )

        if self.return_mask:
            return ds.assign(
                {
                    "DNB_BRDF_Corrected_NTL": masked_ntl,
                    "raw_ntl": ntl,
                    "cloud_mask": cloud_mask,
                    "cirrus_mask": cirrus_mask,
                    "snow_ice_mask": snow_ice_mask,
                    "snow_flag_mask": snow_flag_mask,
                    "combined_mask": combined_mask,
                    "buffered_mask": buffered_mask,
                }
            )

        return ds[["DNB_BRDF_Corrected_NTL"]].assign(DNB_BRDF_Corrected_NTL=masked_ntl)


class ModifiedZScoreOutlierRemoval(PaperImplementation):
    """
    Implements a rolling modified z-score outlier removal as described in:

    Zheng, Q., Jiang, W., Wang, W., Lei, X., & Li, Z. (2021). Daily consistent
    NOAA-20 VIIRS nighttime light data in China. International Journal of
    Remote Sensing, 42(22), 8538-8561.
    https://doi.org/10.1080/01431161.2021.1969057

    This method utilizes a rolling window to calculate the median
    and the Median Absolute Deviation to identify temporal spikes
    (outliers).

    By using a 30-day backward-looking window, the filter remains compatible
    with walk-forward processing while remaining robust to outliers that
    typically bias a standard mean-based z-score.
    """

    def __init__(self, threshold: float = 3.5, window: int = 30, **params: Any):
        """
        Stores configuration parameters for metadata tracking.

        Args:
            threshold: Modified z-score threshold above which pixels are identified as outliers.
            window: Number of backward-looking days to evaluate in the rolling window.
            **params: Additional configuration parameters.
        """
        super().__init__(threshold=threshold, window=window, **params)
        self.threshold = threshold
        self.window = window

    @property
    def required_products_and_bands(self) -> Dict[str, Set[str]]:
        """Declare dependencies."""
        return {"VNP46A2": {"DNB_BRDF_Corrected_NTL"}}

    @staticmethod
    def _calc_rolling_zscore_numpy(
        arr: np.ndarray, window: int, threshold: float
    ) -> np.ndarray:
        """
        NumPy/Bottleneck logic to compute a rolling modified z-score along the last axis (-1).
        This runs block-wise internally within Dask, maintaining lower memory overhead.
        """
        # Use Bottleneck's move_median for fast, NaNs-aware rolling median over the last axis.
        # min_count=1 ensures that if there's at least 1 valid observation in the window,
        # it computes the median without propagating NaNs globally.
        median = bn.move_median(arr, window=window, min_count=1, axis=-1)
        abs_dev = np.abs(arr - median)
        mad = bn.move_median(abs_dev, window=window, min_count=1, axis=-1)

        # Output z_score array
        z_score = np.full_like(arr, np.nan)

        valid_mask = ~np.isnan(arr)

        # Broadcast mad to shape of arr for element-wise boolean operations (if needed)
        mad_b = mad

        mad_zero = mad_b == 0
        mad_non_zero = mad_b > 0

        # case 1: MAD > 0
        mask_mad_non_zero = valid_mask & mad_non_zero
        # prevent division by zero runtime warning by replacing zeros in mad_b
        # we know mad_b is > 0 in this mask, so it's safe.
        safe_divisor = np.where(mask_mad_non_zero, mad_b, 1.0)
        z_score[mask_mad_non_zero] = (
            0.6745 * abs_dev[mask_mad_non_zero] / safe_divisor[mask_mad_non_zero]
        )

        # case 2: MAD == 0
        mask_mad_zero = valid_mask & mad_zero
        mask_outlier = mask_mad_zero & (abs_dev > 0)
        z_score[mask_outlier] = threshold + 1.0
        mask_ok = mask_mad_zero & (abs_dev == 0)
        z_score[mask_ok] = 0.0

        return z_score

    def _transform(self, ds: xr.Dataset, **kwargs) -> xr.Dataset:
        """
        Apply the rolling modified z-score outlier removal transformation.

        Args:
            ds: Input xarray dataset containing DNB_BRDF_Corrected_NTL.

        Returns:
            An xarray Dataset with the outlier pixels replaced by np.nan in
                DNB_BRDF_Corrected_NTL.
        """
        da = ds["DNB_BRDF_Corrected_NTL"]

        if hasattr(da.data, "rechunk"):
            da = da.chunk({"time": -1})

        z_scores = xr.apply_ufunc(
            self._calc_rolling_zscore_numpy,
            da,
            kwargs={"window": self.window, "threshold": self.threshold},
            input_core_dims=[["time"]],
            output_core_dims=[["time"]],
            dask="parallelized",
            output_dtypes=[float],
            dask_gufunc_kwargs={"allow_rechunk": True},
        )

        # restore the original dimension order correctly across all Xarray versions.
        z_scores = z_scores.transpose(*da.dims)

        return ds.assign(
            DNB_BRDF_Corrected_NTL=xr.where(z_scores <= self.threshold, da, np.nan)
        )
