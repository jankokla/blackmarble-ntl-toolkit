from blackmarble_toolkit import PaperImplementation
from typing import Dict, Set

import numpy as np
import xarray as xr


class AveragePooling2D(PaperImplementation):
    """
    Implements a 2D spatial moving average (mean pooling) to reduce noise and
    enhance the signal-to-noise ratio in daily Nighttime Light (NTL) radiance.

    This spatial filtering technique is detailed in the Black Marble product
    documentation:

    Román, M. O., Wang, Z., Sun, Q., Kalb, V., Miller, S. D., Molthan, A., ...
    & Enenkel, S. (2018). NASA's Black Marble nighttime lights product suite.
    Remote Sensing of Environment, 210, 113-143.
    https://doi.org/10.1016/j.rse.2018.03.017

    This class applies a rolling mean across the spatial dimensions (x, y) of the
    dataset. This is often used to mitigate high-frequency noise, such as
    atmospheric artifacts or sub-pixel flickering, while preserving the
    underlying socioeconomic light signals.

    CAUTION:
    While spatial pooling improves stability, it acts as a low-pass filter that
    reduces spatial resolution. Larger filter sizes will result in 'blurring'
    of fine-scale urban structures.
    """

    def __init__(self, filter_size: tuple = (3, 3)):
        """
        Initializes the pooling operation.

        Args:
            filter_size: A tuple representing the window size (x, y) for
                the rolling mean. Default is (3, 3).
        """
        super().__init__(filter_size=filter_size)
        self.filter_size = filter_size

    @property
    def required_products_and_bands(self) -> Dict[str, Set[str]]:
        """
        Required data layers from NASA's Black Marble suite.
        """
        return {"VNP46A2": {"DNB_BRDF_Corrected_NTL"}}

    def _transform(self, ds: xr.Dataset, **kwargs) -> xr.Dataset:
        """
        Applies a spatial rolling mean to the NTL radiance layer.

        Args:
            ds: xarray.Dataset containing the 'DNB_BRDF_Corrected_NTL' band.

        Returns:
            xr.Dataset: The spatially smoothed NTL radiance dataset.
        """
        var_name = "DNB_BRDF_Corrected_NTL"

        x, y = self.filter_size
        smoothed = self._spatial_rolling_mean(ds[var_name], window_x=x, window_y=y)

        return ds.assign({var_name: smoothed})


class Hu2024AAveraging(PaperImplementation):
    """
    Implements the A-Averaging filter to reduce high temporal variation and
    blooming effects in daily NTL data as described in:

    Hu, Y., Zhou, X., Yamazaki, D., & Chen, J. (2024). A self-adjusting method
    to generate daily consistent nighttime light data for the detection of
    short-term rapid human activities. Remote Sensing of Environment, 304, 114077.
    https://doi.org/10.1016/j.rse.2024.114077

    The A-Averaging method decomposes NTL radiance into 'fixed' (stable) and
    'mismatch' (variable) components. By applying spatial averaging only to
    the mismatch component, it reduces blooming and temporal noise while
    preserving the sharp gradients of urban light structures.

    CAUTION:
    The identification of 'Fixed Light' requires a sufficient temporal baseline
    (ideally one year) to calculate meaningful quantiles.
    """

    def __init__(self, filter_size: tuple = (3, 3)):
        """
        Initializes the A-Averaging filter.

        Args:
            filter_size: The spatial window for smoothing the mismatch component.
                Standard paper implementation uses (3, 3).
        """
        super().__init__(filter_size=filter_size)
        self.filter_size = filter_size

    @property
    def required_products_and_bands(self) -> Dict[str, Set[str]]:
        """
        Required data layers from NASA's Black Marble suite[cite: 933].
        """
        return {"VNP46A2": {"DNB_BRDF_Corrected_NTL"}}

    def _transform(self, ds: xr.Dataset, **kwargs) -> xr.Dataset:
        """
        Applies the A-Averaging filter logic to the NTL radiance.

        The procedure involves:
        1. Removing outliers (ephemeral noise) using a 3rd standard deviation mask.
        2. Extracting 'Fixed Light' via the bottom 5% quantile mean.
        3. Isolating 'Mismatch Light' (Total - Fixed).
        4. Spatially smoothing only the Mismatch Light to mitigate blooming.
        5. Reconstructing the final radiance.
        """
        ntl = ds["DNB_BRDF_Corrected_NTL"]

        # 1. remove readings more than 3 standard deviations from the mean
        mean_ntl = ntl.mean(dim="time")
        std_ntl = ntl.std(dim="time")

        # mask outliers as NaN
        is_outlier = np.abs(ntl - mean_ntl) > 3 * std_ntl
        ntl_cleaned = ntl.where(~is_outlier)

        # 2. define Fixed Light as the mean of the 5% minimum light values
        q05 = ntl_cleaned.quantile(0.05, dim="time")
        ntl_fix = ntl_cleaned.where(ntl_cleaned <= q05).mean(dim="time")

        # 3. calculate Mismatch Light (Total - Fixed)
        ntl_mismatch = ntl_cleaned - ntl_fix

        # 4. apply 3x3 averaging ONLY to the mismatch component
        ntl_mismatch_smoothed = self._spatial_rolling_mean(
            ntl_mismatch, window_x=3, window_y=3
        )

        # 5. add the smoothed noise/mismatch back to the sharp fixed light
        ntl_consistent = ntl_fix + ntl_mismatch_smoothed

        # ff it was an outlier day, keep original NTL
        ntl_reconstructed = xr.where(is_outlier, ntl, ntl_consistent)

        return ds[["DNB_BRDF_Corrected_NTL"]].assign(
            DNB_BRDF_Corrected_NTL=ntl_reconstructed.transpose(*ntl.dims)
        )
