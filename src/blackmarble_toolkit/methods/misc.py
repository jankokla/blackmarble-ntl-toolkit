from typing import Dict, Set

import numpy as np
import statsmodels.api as sm
import xarray as xr

from blackmarble_toolkit import PaperImplementation


class Yue2026DisturbanceFactorCorrection(PaperImplementation):
    """
    Implements the Disturbance Factor Correction (DFC) model as described in:

    Yue, H., et al. (2026). Improved Daily Nighttime Light Data as
    High-Frequency Economic Indicator. Applied Sciences, 16(2), 947.
    https://doi.org/10.3390/app16020947

    It uses a Local Ordinary Least Squares (OLS) regression to
    decompose daily NTL radiance into three components: physical disturbances,
    socioeconomic signals (trends/seasonality), and random noise.

    The model accounts for:
    - Lunar Illumination (LI)
    - Sensor Geometry: Viewing Zenith Angle (VZA) and Viewing Azimuth Angle (VAA)
    - Anisotropic Interaction: VZA x VAA
    - Cloud-contamination artifacts
    - Long-term Day Trend and Seasonal Dummies (Spring, Summer, Autumn, Winter)

    LOGIC:
    The regression acts as a 'control' framework. While all factors are regressed
    simultaneously to prevent omitted variable bias, only the physical
    disturbances (Moon, Geometry, Clouds) are subtracted from the raw signal if
    they are statistically significant (p < 0.05). The socioeconomic signals
    (trend and seasonality) are preserved in the corrected output.
    """

    def __init__(self, return_coeffs: bool = False):
        super().__init__(return_coeffs=return_coeffs)
        self.return_coeffs = return_coeffs

    @property
    def required_products_and_bands(self) -> Dict[str, Set[str]]:
        return {
            "VNP46A1": {
                "Sensor_Zenith",
                "Sensor_Azimuth",
            },
            "VNP46A2": {
                "DNB_BRDF_Corrected_NTL",
                "DNB_Lunar_Irradiance",
                "QF_Cloud_Mask",
            },
        }

    @staticmethod
    def _regress_pixel(
        ntl,
        moon,
        vaa,
        vza,
        cloud,
        daytrend,
        spring_dummy,
        summer_dummy,
        autumn_dummy,
        winter_dummy,
    ):
        """
        Calculates the DFC for a single pixel time-series.
        Returns only the corrected NTL array.
        """
        corrected_ntl = ntl.copy()

        interaction = vza * vaa
        X = np.column_stack(
            [
                moon,
                vaa,
                vza,
                interaction,
                cloud,
                daytrend,
                spring_dummy,
                summer_dummy,
                autumn_dummy,
                winter_dummy,
            ]
        )

        # identify valid observations (no NaNs in Y or X)
        mask = np.isfinite(ntl) & ~np.any(~np.isfinite(X), axis=1)

        # statsmodels needs at least N_params + 1 observations
        if np.sum(mask) < 15:
            return (
                corrected_ntl,
                np.full(11, np.nan, dtype=np.float32),
                np.full(11, np.nan, dtype=np.float32),
            )

        # fit OLS with constant
        X_fit = sm.add_constant(X[mask], has_constant="add")
        model = sm.OLS(ntl[mask], X_fit).fit()

        # extract params and p-values
        # index 0: const, 1: moon, 2: vza, 3: vaa, 4: interact, 5: cloud
        params = model.params
        pvals = model.pvalues

        # subtract only the physical disturbance factors if they are significant
        for i in range(1, 6):
            if pvals[i] < 0.05:
                corrected_ntl -= params[i] * X[:, i - 1]

        corrected_ntl = np.maximum(0, corrected_ntl)

        return corrected_ntl, params.astype(np.float32), pvals.astype(np.float32)

    def _transform(self, ds: xr.Dataset, **kwargs) -> xr.Dataset:
        """
        Applies DFC correction pixel-by-pixel across the dataset.
        """
        # Generate 1D temporal features
        day_trend = (ds.time - ds.time).dt.days.astype(np.float32)
        seasons = ds.time.dt.season

        spr = (seasons == "MAM").astype(np.float32)
        sum_ = (seasons == "JJA").astype(np.float32)
        aut = (seasons == "SON").astype(np.float32)
        win = (seasons == "DJF").astype(np.float32)

        sensor_azimuth = self._get_band(ds, "Sensor_Azimuth", kwargs)
        sensor_zenith = self._get_band(ds, "Sensor_Zenith", kwargs)

        # normalize variables for numerical stability as per Table 1 of the paper
        moon_norm = ds["DNB_Lunar_Irradiance"] / 1e4
        sensor_azimuth_norm = sensor_azimuth / 1e2
        sensor_zenith_norm = sensor_zenith / 1e2
        day_trend_norm = day_trend / 1e2

        # apply via apply_ufunc
        corrected_ntl, params, pvals = xr.apply_ufunc(
            self._regress_pixel,
            ds[self.target_var_name],
            moon_norm,
            sensor_azimuth_norm,
            sensor_zenith_norm,
            ds["QF_Cloud_Mask"],
            day_trend_norm,
            spr,
            sum_,
            aut,
            win,
            input_core_dims=[["time"]] * 10,
            output_core_dims=[["time"], ["param"], ["param"]],
            vectorize=True,  # Loops over spatial dims
            dask="parallelized",  # Enables multi-core execution
            dask_gufunc_kwargs={"output_sizes": {"param": 11}},
            output_dtypes=[np.float32, np.float32, np.float32],
        )

        spatial_dims = [dim for dim in ds.dims if dim != "time"]

        corrected_ntl = corrected_ntl.rename(self.target_var_name).transpose(
            "time", *spatial_dims
        )

        if not self.return_coeffs:
            return corrected_ntl

        param_names = [
            "const",
            "moon",
            "vaa",
            "vza",
            "vaa:vza",
            "cloud",
            "daytrend",
            "spring_dummy",
            "summer_dummy",
            "autumn_dummy",
            "winter_dummy",
        ]
        params = (
            params.assign_coords(param=param_names)
            .rename("params")
            .transpose("param", *spatial_dims)
        )
        pvals = (
            pvals.assign_coords(param=param_names)
            .rename("pvalues")
            .transpose("param", *spatial_dims)
        )

        out_ds = xr.Dataset(
            {
                self.target_var_name: corrected_ntl,
                "params": params,
                "pvalues": pvals,
            }
        )
        return out_ds
