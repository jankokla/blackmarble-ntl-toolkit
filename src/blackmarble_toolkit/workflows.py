import json
import logging
from typing import Any, List, Optional

import geopandas as gpd
import xarray as xr
import yaml

from blackmarble_toolkit.pipeline import NTLPipeline
from blackmarble_toolkit.retrieval import BlackMarbleRetriever

logger = logging.getLogger(__name__)


def download_workflow(
    product: str,
    start_date: str,
    end_date: str,
    region_file: str,
    out_zarr: str,
    bands: Optional[List[str]] = None,
    scale: Optional[float] = None,
    chunks: Any = "auto",
):
    """
    Workflow to download Black Marble data and save to a Zarr store.
    """
    logger.info(f"Loading region from {region_file}...")
    gdf = gpd.read_file(region_file)

    # Use the combined geometry of the GeoDataFrame
    from blackmarble_toolkit.retrieval import gdf_to_geometry

    region_geom = gdf_to_geometry(gdf)

    logger.info(f"Retrieving data for {product} from {start_date} to {end_date}...")
    retriever = BlackMarbleRetriever()
    ds = retriever.get_data(
        product=product,
        start_date=start_date,
        end_date=end_date,
        region=region_geom,
        bands=bands,
        scale=scale,
        chunks=chunks,
    )

    logger.info(f"Writing uncomputed lazy dataset to {out_zarr}...")
    ds.to_zarr(out_zarr, compute=False)
    logger.info(
        "Download workflow completed successfully. Data is ready for preprocessing."
    )


def _load_steps_from_config(config_path: str) -> List[Any]:
    """Dynamically load processing steps based on a YAML/JSON configuration."""
    import importlib

    with open(config_path, "r") as f:
        if config_path.endswith(".json"):
            config = json.load(f)
        else:
            config = yaml.safe_load(f)

    steps_config = config.get("steps", [])
    steps = []

    steps_module = importlib.import_module("blackmarble_toolkit.steps")

    for step_cfg in steps_config:
        if isinstance(step_cfg, str):
            step_name = step_cfg
            params = {}
        else:
            step_name = list(step_cfg.keys())[0]
            params = step_cfg[step_name] or {}

        step_class = getattr(steps_module, step_name)
        steps.append(step_class(**params))

    return steps


def preprocess_workflow(
    input_zarr: str,
    config_file: str,
    out_zarr: str,
):
    """
    Workflow to preprocess downloaded data using NTLPipeline.
    """
    logger.info(f"Loading raw data from {input_zarr}...")
    ds = xr.open_zarr(input_zarr)

    logger.info(f"Loading pipeline steps from {config_file}...")
    steps = _load_steps_from_config(config_file)

    pipeline = NTLPipeline(steps=steps)

    logger.info("Running preprocessing pipeline...")
    # Compute happens when saving to Zarr
    preprocessed_ds = pipeline.run(ds, cache_intermediates=False)

    logger.info(f"Writing preprocessed dataset to {out_zarr}...")
    preprocessed_ds.to_zarr(out_zarr, compute=True)
    logger.info("Preprocess workflow completed successfully.")


def aggregate_workflow(
    input_zarr: str,
    region_file: str,
    geo_id_col: str,
    out_zarr: str,
    out_csv: Optional[str] = None,
):
    """
    Workflow to aggregate preprocessed data over vector geometries.
    """
    logger.info(f"Loading preprocessed data from {input_zarr}...")
    ds = xr.open_zarr(input_zarr)

    logger.info(f"Loading region shapes from {region_file}...")
    gdf = gpd.read_file(region_file)

    if geo_id_col not in gdf.columns:
        raise ValueError(
            f"Geometry ID column '{geo_id_col}' not found in {region_file}."
        )

    logger.info("Initializing aggregation pipeline...")

    # reuse NTLPipeline by artificially populating it with our loaded preprocessed ds
    pipeline = NTLPipeline(steps=[])
    pipeline._preprocessed_ds = ds

    logger.info("Aggregating data...")
    pipeline.aggregate(gdf, geo_id_col=geo_id_col, compute=True)

    logger.info(f"Writing aggregated dataset to {out_zarr}...")

    # since aggregate() returns a list of datasets (for each step if intermediates cached),
    # we just take the last one (the final output) and save to Zarr
    final_agg_ds = pipeline.aggregated_ds[-1]
    final_agg_ds.to_zarr(out_zarr, compute=True)

    if out_csv:
        logger.info(f"Exporting aggregated data to {out_csv}...")
        pipeline.to_csv(out_csv)

    logger.info("Aggregate workflow completed successfully.")
