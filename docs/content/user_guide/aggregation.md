---
icon: lucide/sigma
---

# Spatial Aggregation

Once the Nighttime Light (NTL) data has been [retrieved](data_retrieval.md) and [preprocessed](preprocessing.md), the final step in many workflows is to aggregate the pixel-level data into administrative boundaries or custom vector shapes. This reduces the data dimensionality and allows for longitudinal analysis at the regional level.

The toolkit provides built-in spatial aggregation directly within the `NTLPipeline` object. This makes it incredibly easy to track how the data evolves throughout the preprocessing stages by aggregating the intermediate steps as well as the final dataset.

## Usage Example

The typical workflow involves running your preprocessing pipeline with `cache_intermediates=True`, loading your vector data using GeoPandas, and applying the `aggregate()` function.

```python
import geopandas as gpd
from blackmarble_toolkit.pipeline import NTLPipeline

pipeline = NTLPipeline(steps)

processed_ds = pipeline.run(ds=raw_ds, cache_intermediates=True) # (1)! 

gdf = gpd.read_file("path/to/regions.geojson") # (2)! 

aggregated_results = pipeline.aggregate( # (3)! 
    track_geometries=gdf,
    geo_id_col="geonameid" # (4)! 
)
```

1. Run the pipeline, caching the intermediate results
2. Load your vector shapes (e.g., administrative boundaries)
3. Perform spatial aggregation across all cached steps
4. The unique identifier column in the GeoDataFrame

!!! tip "Unique Geometry Identifiers"

    When aggregating across multiple regions or shapes, each geometry **must** have a unique identifier (specified via the `geo_id_col` parameter). If your `GeoDataFrame` does not already contain a unique ID column, you can easily create one by resetting the index before aggregation:
    
    ```python
    gdf = gdf.reset_index(names="geo_id") # (1)! 
    ```

    1. Then pass geo_id_col="geo_id" to pipeline.aggregate().

### Intermediate vs. Final Results

When you call `pipeline.run()`, you can control whether you want to aggregate just the final output or all the intermediate stages of your pipeline:

- **`cache_intermediates=True`**: The pipeline saves a snapshot of the dataset after *each* transformation step. Calling `pipeline.aggregate()` will then return a list of datasets (one for the raw data, and one for every step applied). This is extremely useful if you want to perform downstream comparisons to analyze how each individual filter or correction affects your aggregated metrics.
- **`cache_intermediates=False`**: (Default behavior) The pipeline only keeps the final, fully-processed dataset in memory. Calling `pipeline.aggregate()` will return a list containing just one dataset—the final result. This is more memory-efficient and faster if you only care about the end product.

## How It Works

Under the hood, the pipeline uses `geocube` to rasterize your vector shapes onto the identical grid of your xarray dataset just once to avoid repeating expensive rasterizations. It then iterates through the cached steps:

1. **`get_gdf_mask_for_ds`**: Creates a spatial mask aligning the `GeoDataFrame` to the grid of the `xarray.Dataset`.
2. **`get_agg_per_shape`**: Groups the data by the geometry ID and performs a memory-safe aggregation using Dask.

By default, the pipeline computes both the **mean** radiance value and the **valid pixel percentage** (`valid_pct`) for every shape at each time step. The valid pixel percentage helps you identify shapes where too many pixels were masked out (e.g., due to heavy cloud cover or snow), ensuring you can filter out statistically unreliable aggregations downstream.

## API Reference

For detailed documentation, refer to the [Pipeline API Reference](../api_reference/pipeline.md) or the underlying [Aggregation Methods API Reference](../api_reference/aggregation.md).
