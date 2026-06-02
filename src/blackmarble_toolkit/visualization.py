from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from matplotlib.figure import Figure


def plot_multiple_timeseries(
    datasets: List[xr.Dataset],
    variable: str = "ntl",
    indexers: Optional[dict] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    titles: Optional[List[str]] = None,
    y_max: Optional[float] = None,
    title: Optional[str] = None,
    font_scale: float = 1.0,
    moving_average: Optional[int] = None,
) -> Figure:
    """Plots one or multiple aggregated time-series datasets as vertically stacked subplots.

    All subplots will share the same y-axis boundaries for direct comparability.

    Args:
        datasets: A list of spatially aggregated xarray Datasets.
        variable: The variable name to plot (e.g. "ntl").
        indexers: Optional dictionary of dimension coordinates to select specific data (e.g. `{"geonameid": "Region_A"}` or `{"x": 10.5, "y": 20.1}`).
        start_date: Optional start date string (e.g., '2022-01-01') to slice the data.
        end_date: Optional end date string (e.g., '2022-12-31') to slice the data.
        titles: Optional list of titles corresponding to each subplot. If not provided,
            will attempt to use the `.attrs['step']` from each dataset.
        y_max: Optional maximum limit for the y-axis.
        title: Optional string for the overall figure title.
        font_scale: The scaling factor for the plot's font sizes.
        moving_average: Optional moving average window size (days).

    Returns:
        The generated matplotlib Figure.
    """
    num_plots = len(datasets)
    if titles is None or len(titles) != num_plots:
        steps = []
        titles = []
        for i, ds in enumerate(datasets):
            step_name = str(ds.attrs.get("step", f"Time Series {i + 1}"))
            steps.append(step_name)
            titles.append(" $\\rightarrow$ ".join(steps))

    def preprocess_ds(ds):
        # Extract variable and slice time if requested
        da = ds[variable]
        if start_date or end_date:
            da = da.sel(time=slice(start_date, end_date))

        if indexers:
            valid_indexers = {
                k: v for k, v in indexers.items() if k in da.dims or k in da.coords
            }
            if valid_indexers:
                try:
                    da = da.sel(valid_indexers)
                except KeyError:
                    # Fallback to nearest if exact match fails (useful for floats like lat/lon)
                    da = da.sel(valid_indexers, method="nearest")

        # Ensure the dataset has been reduced to a 1D timeseries
        remaining_dims = [d for d in da.dims if d != "time"]
        for d in remaining_dims:
            if da.sizes[d] > 1:
                raise ValueError(
                    f"Dataset contains unreduced dimension '{d}' of size {da.sizes[d]}. "
                    "You must aggregate the data or provide specific `indexers` before plotting."
                )
        if remaining_dims:
            da = da.squeeze(dim=remaining_dims)

        return pd.Series(da.values, index=da.time.values)

    processed_series = []
    smoothed_series = []
    valid_vals = []

    for ds in datasets:
        series = preprocess_ds(ds)
        processed_series.append(series)

        if series.notna().any():
            valid_vals.append(series.values)
            if moving_average is not None:
                smoothed = series.rolling(
                    window=moving_average, center=True, min_periods=1
                ).mean()
            else:
                smoothed = None
        else:
            smoothed = None
        smoothed_series.append(smoothed)

    # calculate global y-limits for the primary axis
    if valid_vals:
        global_min = min(np.nanmin(v) for v in valid_vals)
        global_max = max(np.nanmax(v) for v in valid_vals)
    else:
        global_min, global_max = 0.0, 1.0

    y_padding = (global_max - global_min) * 0.1
    if y_padding == 0:
        y_padding = 1.0

    upper_bound = global_max + y_padding
    if y_max is not None:
        upper_bound = y_max

    y_lim = (0, upper_bound)

    with sns.axes_style(
        "whitegrid",
        {
            "grid.color": ".9",
            "grid.linestyle": ":",
        },
    ):
        sns.set_context("paper", font_scale=font_scale)
        fig_height = max(2.5, num_plots * 2.5)
        fig, axes = plt.subplots(
            nrows=num_plots, ncols=1, figsize=(12, fig_height), sharex=False
        )
        if num_plots == 1:
            axes = [axes]

        for ax, series, smoothed, sub_title in zip(
            axes, processed_series, smoothed_series, titles
        ):
            # primary axis plotting (ntl data)
            lns1 = ax.scatter(
                series.index,
                series.values,
                color="blue",
                alpha=0.6,
                s=20,
                edgecolors="none",
                label="Daily Data",
                zorder=3,
            )
            all_lns = [lns1]
            if moving_average is not None and smoothed is not None:
                (lns2,) = ax.plot(
                    series.index,
                    smoothed,
                    color="black",
                    linewidth=1.5,
                    label=f"Smoothed ({moving_average}-day)",
                    zorder=4,
                )
                all_lns.append(lns2)

            ax.set_ylabel("nW/cm²/sr", fontsize=11)
            ax.set_ylim(y_lim)
            ax.set_title(sub_title, fontsize=12, loc="left")

            if start_date or end_date:
                # If explicit bounds are provided, enforce them
                s_date = (
                    pd.to_datetime(start_date) if start_date else series.index.min()
                )
                e_date = pd.to_datetime(end_date) if end_date else series.index.max()
                ax.set_xlim(s_date, e_date)

        if title is not None:
            fig.suptitle(title, fontsize=16)

        plt.tight_layout()

    return fig
