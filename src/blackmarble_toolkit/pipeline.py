import json
import logging
from typing import List, Optional

import geopandas as gpd
import xarray as xr

from blackmarble_toolkit.aggregation import get_agg_per_shape, get_gdf_mask_for_ds

logger = logging.getLogger(__name__)


class NTLPipeline:
    """Orchestrate the sequential execution of NTL transformations."""

    def __init__(self, steps):
        """
        Initialize the pipeline with a sequence of processing steps.

        Args:
            steps: A list of instantiated processing step objects.
        """
        self.steps = steps
        self._preprocessed_ds = None
        self._intermediates: List[xr.Dataset] = []
        self._aggregated_ds: Optional[List[xr.Dataset]] = None

    def _validate_dependencies(self, ds, catalog=None):
        """
        Validates that all required bands are present in the provided datasets.

        Args:
            ds: The primary dataset.
            catalog: Dictionary containing auxiliary datasets.
        """
        required_bands = set()
        for step in self.steps:
            reqs = step.required_products_and_bands
            if reqs:
                for product, bands in reqs.items():
                    required_bands.update(bands)

        available_vars = set(ds.data_vars).union(set(ds.coords))
        if catalog:
            for cat_ds in catalog.values():
                available_vars.update(cat_ds.data_vars)
                available_vars.update(cat_ds.coords)

        missing = required_bands - available_vars
        if missing:
            raise ValueError(f"Missing required bands for pipeline execution {missing}")

    def run(
        self,
        ds: xr.Dataset,
        catalog: Optional[dict] = None,
        cache_intermediates: bool = False,
    ) -> xr.Dataset:
        """
        Execute the pipeline sequentially over the input dataset.

        Args:
            ds: The primary dataset to process.
            catalog: Dictionary containing auxiliary datasets.
            cache_intermediates: If True, caches intermediate datasets for later aggregation.

        Returns:
            The fully processed dataset containing the execution history.
        """
        if catalog is None:
            catalog = {}

        self._validate_dependencies(ds, catalog)

        history = ds.attrs.get("processing_history", [])

        if isinstance(history, str):
            try:
                history = json.loads(history)
            except json.JSONDecodeError:
                history = []

        current_ds = ds

        self._intermediates = []
        if cache_intermediates:
            self._intermediates.append(current_ds.assign_attrs(step="raw"))

        for step in self.steps:
            current_ds = step.transform(current_ds, **catalog)

            step_name = str(step)
            if cache_intermediates:
                self._intermediates.append(current_ds.assign_attrs(step=step_name))

            step_meta = {"step": step_name, "params": getattr(step, "params", {})}
            history.append(step_meta)

        current_ds.attrs["processing_history"] = json.dumps(history)
        self._preprocessed_ds = current_ds

        return current_ds

    @property
    def preprocessed_ds(self) -> Optional[xr.Dataset]:
        """Returns the final preprocessed dataset."""
        if self._preprocessed_ds is None:
            raise ValueError("No data has been preprocessed. Run the pipeline first.")
        return self._preprocessed_ds

    @property
    def aggregated_ds(self) -> List[xr.Dataset]:
        """Returns the cached aggregated datasets."""
        if self._aggregated_ds is None:
            raise ValueError("No aggregated data available. Run aggregate() first.")
        return self._aggregated_ds

    def aggregate(
        self,
        track_geometries: gpd.GeoDataFrame,
        geo_id_col: str = "geonameid",
    ) -> List[xr.Dataset]:
        """
        Aggregate cached intermediate datasets over the provided geometries.

        Args:
            track_geometries: GeoDataFrame containing the regions to aggregate over.
            geo_id_col: The column name in the GeoDataFrame that uniquely identifies each shape.

        Returns:
            A list of datasets, where each dataset contains the spatial aggregation
            for a specific processing step. The step name is stored in the
            `.attrs['step']` of each returned dataset.
        """
        if not self._intermediates:
            if self._preprocessed_ds is None:
                raise ValueError("No data to aggregate. Run the pipeline first.")
            to_aggregate = [self._preprocessed_ds.assign_attrs(step="final")]
        else:
            to_aggregate = self._intermediates

        track_var = "ntl"
        first_ds = to_aggregate[0]

        mask_var = "ntl" if "ntl" in first_ds.data_vars else "DNB_BRDF_Corrected_NTL"
        if mask_var not in first_ds.data_vars:
            raise ValueError(f"Expected NTL variable not found in the dataset.")

        # create mask once to avoid repeating expensive rasterizations
        mask = get_gdf_mask_for_ds(
            first_ds[[mask_var]], track_geometries, geo_id_col=geo_id_col
        )

        aggregated_results = []
        for ds_step in to_aggregate:
            step_var = "ntl" if "ntl" in ds_step.data_vars else "DNB_BRDF_Corrected_NTL"

            if step_var not in ds_step.data_vars:
                logger.warning(
                    f"Expected NTL variable not found in step "
                    f"'{ds_step.attrs.get('step', 'unknown')}'. Skipping."
                )
                continue

            zonal_ds = get_agg_per_shape(
                ds_step[[step_var]],
                mask,
                step_var,
                agg_type="mean",
                is_valid_pct=True,
                geo_id_col=geo_id_col,
            )

            if step_var != track_var:
                zonal_ds = zonal_ds.rename({step_var: track_var})

            zonal_ds = zonal_ds.assign_attrs(step=ds_step.attrs.get("step", "unknown"))
            aggregated_results.append(zonal_ds)

        self._aggregated_ds = aggregated_results
        return aggregated_results
