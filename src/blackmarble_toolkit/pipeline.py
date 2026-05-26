import json
from typing import Dict, List, Optional

import xarray as xr


class NTLPipeline:
    """Orchestrate the sequential execution of NTL transformations."""

    def __init__(self, steps):
        """
        Initialize the pipeline with a sequence of processing steps.

        Args:
            steps: A list of instantiated processing step objects.
        """
        self.steps = steps

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

    def run(self, ds, catalog=None):
        """
        Execute the pipeline sequentially over the input dataset.

        Args:
            ds: The primary dataset to process.
            catalog: Dictionary containing auxiliary datasets.

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

        for step in self.steps:
            result = step.transform(current_ds, **catalog)

            if isinstance(result, xr.DataArray):
                var_name = result.name if result.name else "DNB_BRDF-Corrected_NTL"
                current_ds = current_ds.assign({var_name: result})
            else:
                current_ds = result

            step_meta = {"step": step.name, "params": step.params}
            history.append(step_meta)

        current_ds.attrs["processing_history"] = json.dumps(history)

        return current_ds
