import warnings
from typing import Any, Dict, Set

import numpy as np
import pandas as pd
import xarray as xr

from blackmarble_toolkit import PaperImplementation


class QuadraticVZACorrection(PaperImplementation):
    """
    Implements a View Zenith Angle (VZA) correction for Nighttime Light radiance
    following the quadratic regression normalization approach detailed in:

    Zheng, Q., Zeng, Y., Zhou, Y., Wang, Z., Mu, T., & Weng, Q. (2025).
    Nighttime lights reveal substantial spatial heterogeneity and inequality
    in post-hurricane recovery. Remote Sensing of Environment, 319, 114645.
    https://doi.org/10.1016/j.rse.2025.114645

    This class normalizes pixel-wise radiance to a nadir observation (VZA = 0°) by
    calculating a multiplicative correction factor 'f' derived from the ratio between
    predicted nadir radiance and the predicted radiance at the observed angle.
    """

    def __init__(self, return_factor: bool = False):
        super().__init__(return_factor=return_factor)
        self.return_factor = return_factor

    @property
    def required_products_and_bands(self) -> Dict[str, Set[str]]:
        """
        Required data layers from NASA's Black Marble suite.
        """
        return {"VNP46A2": {"DNB_BRDF_Corrected_NTL"}, "VNP46A1": {"Sensor_Zenith"}}

    @staticmethod
    def _pixel_vza_logic(ntl, zenith):
        """
        Processes a single pixel's time-series.
        Inputs: 1D numpy arrays (time-axis).
        """
        # we need BOTH NTL and Zenith to be finite
        mask = np.isfinite(ntl) & np.isfinite(zenith)

        x_train = zenith[mask]
        y_train = ntl[mask]

        corrected_ntl = ntl.copy()
        f_values = np.full(ntl.shape, 1.0, dtype=np.float32)

        # check if we have enough data points to fit a 2nd-degree polynomial
        if len(x_train) >= 10 and np.ptp(x_train) >= 5:
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("error", category=np.exceptions.RankWarning)
                    # Fit: y = ax^2 + bx + c
                    a, b, c = np.polyfit(x_train, y_train, 2)

                # predict radiance across the whole time series (including NaNs)
                pred = a * (zenith**2) + b * zenith + c

                # only apply correction if the c is physically plausible (positive)
                if c > 0:
                    pred = a * (zenith**2) + b * zenith + c
                    f_values = np.where(pred > 0, c / pred, 1.0)
                else:
                    f_values = np.full(ntl.shape, 1.0, dtype=np.float32)

                corrected_ntl = ntl * f_values

            except (np.exceptions.RankWarning, Exception):
                pass

        # final Masking: Ensure original NaNs (and our new ones) propagate to the factor
        f_values[~np.isfinite(ntl)] = np.nan

        return corrected_ntl, f_values

    def _transform(self, ds: xr.Dataset, **kwargs) -> xr.Dataset:
        """
        Rechunks to time-contiguous chunks and applies the correction.
        """
        sensor_zenith = self._get_band(ds, "Sensor_Zenith", kwargs)

        corrected_ntl, correction_factor = xr.apply_ufunc(
            self._pixel_vza_logic,
            ds[self.target_var_name],
            sensor_zenith,
            input_core_dims=[["time"], ["time"]],
            output_core_dims=[["time"], ["time"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[np.float32, np.float32],
            dask_gufunc_kwargs={"allow_rechunk": True},
        )

        res_vars = {self.target_var_name: corrected_ntl}
        if self.return_factor:
            res_vars["VZA_Correction_Factor"] = correction_factor

        return xr.Dataset(
            data_vars=res_vars,
            coords=ds.coords,
        ).transpose(*ds.dims)


class Hu2024AngularCorrection(PaperImplementation):
    """
    Implementation of the angular correction method for daily NTL data:

    Hu, Y., Zhou, X., Yamazaki, D., & Chen, J. (2024). A self-adjusting method
    to generate daily consistent nighttime light data for the detection of
    short-term rapid human activities. Remote Sensing of Environment, 304, 114077.
    https://doi.org/10.1016/j.rse.2024.114077
    """

    @property
    def required_products_and_bands(self) -> Dict[str, Set[str]]:
        """Declare dependencies."""
        return {
            "VNP46A2": {"DNB_BRDF_Corrected_NTL"},
            "NOAA/VIIRS/DNB/ANNUAL_V22": {"average"},
        }

    @staticmethod
    def _process_angular_correction_block(
        ntl_block: np.ndarray, annual_block: np.ndarray, phase_vals: np.ndarray
    ) -> np.ndarray:
        """
        process a single spatial block over all time for angular correction.

        Args:
            ntl_block: (..., time) numpy array of NTL data.
            annual_block: (...) numpy array of annual mean data.
            phase_vals: (time,) numpy array of orbit phases.

        Returns:
            The corrected NTL data with shape (..., time).
        """
        original_shape = ntl_block.shape
        t = original_shape[-1]

        # Flatten spatial dimensions to simplify processing
        ntl_flat = ntl_block.reshape(-1, t)
        ann_flat = annual_block.reshape(-1)
        out_flat = np.full_like(ntl_flat, np.nan)

        for p in range(16):
            idx = np.where(phase_vals == p)[0]
            if len(idx) == 0:
                continue

            phase_data = ntl_flat[:, idx]

            # safely compute mean over time, ignoring warnings for all-NaN slices
            all_nan = np.isnan(phase_data).all(axis=1)
            safe_data = np.where(all_nan[:, None], 0.0, phase_data)
            p_mean = np.nanmean(safe_data, axis=1)
            p_mean = np.where(all_nan, np.nan, p_mean)

            # calculate a_i with safe division
            denom1 = np.where(ann_flat > 0, ann_flat, 1.0)
            a_i = np.where(ann_flat > 0, p_mean / denom1, np.nan)

            # calculate final ntl scaling factor
            denom2 = np.where(a_i > 0, a_i, 1.0)
            valid_mask = a_i > 0

            for i in idx:
                out_flat[:, i] = np.where(valid_mask, ntl_flat[:, i] / denom2, np.nan)

        return out_flat.reshape(original_shape)

    def _transform(self, ds: xr.Dataset, **kwargs: Any) -> xr.Dataset:
        """
        Apply the angular correction transformation.

        Args:
            ds: The daily dataset containing NTL and VZA.
            **kwargs: Additional arguments containing catalog.

        Returns:
            The corrected NTL Dataset.
        """
        ntl = ds[self.target_var_name]
        annual = self._get_band(ds, "average", kwargs)

        # group all the days of a year into 16 groups according to the daily vza.
        # since snpp repeats every 16 days, group by day modulo 16.
        days = ds.time.dt.floor("D") - xr.DataArray(pd.Timestamp("2012-01-01"))
        orbit_phase = (days.dt.days % 16).astype(int).values

        annual_mean = annual.mean(dim="time") if "time" in annual.dims else annual

        # rechunk efficiently for apply_ufunc
        spatial_dims = [dim for dim in ntl.dims if dim != "time"]
        chunk_dict = {"time": -1}
        annual_chunk_dict = {}
        for dim in spatial_dims:
            chunk_dict[dim] = "auto"
            if dim in annual_mean.dims:
                annual_chunk_dict[dim] = "auto"

        ntl_rechunked = ntl.chunk(chunk_dict)
        annual_rechunked = annual_mean.chunk(annual_chunk_dict)

        # ensure spatial chunks align before applying apply_ufunc
        ntl_rechunked, annual_rechunked = xr.align(
            ntl_rechunked, annual_rechunked, join="exact"
        )

        spatial_chunks = {
            dim: ntl_rechunked.chunksizes[dim]
            for dim in spatial_dims
            if dim in annual_rechunked.dims
        }
        annual_rechunked = annual_rechunked.chunk(spatial_chunks)

        ntl_sfac = xr.apply_ufunc(
            self._process_angular_correction_block,
            ntl_rechunked,
            annual_rechunked,
            kwargs={"phase_vals": orbit_phase},
            input_core_dims=[["time"], []],
            output_core_dims=[["time"]],
            dask="parallelized",
            output_dtypes=[float],
            dask_gufunc_kwargs={"allow_rechunk": True},
        )

        ntl_sfac = ntl_sfac.transpose(*ntl.dims)

        original_chunks = {dim: ntl.chunks[i][0] for i, dim in enumerate(ntl.dims)}

        ntl_sfac.encoding = {}
        return ds[[self.target_var_name]].assign(
            **{self.target_var_name: ntl_sfac.chunk(original_chunks)}
        )
