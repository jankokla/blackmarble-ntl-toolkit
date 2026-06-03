---
icon: lucide/square-terminal
---

# Command Line Interface (CLI)

While the [`NTLPipeline`](../api_reference/pipeline.md) is primarily designed for interactive case studies involving smaller datasets or regions, the toolkit also provides a Command Line Interface (CLI) tailored for computationally heavy, long-running jobs.

By utilizing out-of-core computation and saving intermediate states as Zarr stores, the CLI allows you to decouple data retrieval, preprocessing, and aggregation into distinct steps. This makes it much easier to resume failed downloads or experiment with different preprocessing pipelines across large spatial extents without having to redownload the raw data.

## Workflows

### 1. Download Data

The `download` command fetches Black Marble data for a specific region and time period from Google Earth Engine and saves it lazily to a local Zarr store.

```bash
blackmarble download \
    --product VNP46A2 \
    --start-date 2023-01-01 \
    --end-date 2023-01-31 \
    --region path/to/districts.geojson \
    --out raw_vnp46a2.zarr
```

If a preprocessing step requires an auxiliary catalog (like the `QuadraticVZACorrection` which requires the `Sensor_Zenith` band from the `VNP46A1` product), you can use the `--bands` argument to only download the data you specifically need:

```bash
blackmarble download \
    --product VNP46A1 \
    --start-date 2023-01-01 \
    --end-date 2023-01-31 \
    --region path/to/districts.geojson \
    --bands Sensor_Zenith \
    --out auxiliary_vnp46a1.zarr
```

**Arguments:**

- `--product`: The Black Marble product to download (e.g., `VNP46A2`).
- `--start-date`: Start date (YYYY-MM-DD).
- `--end-date`: End date (YYYY-MM-DD).
- `--region`: Path to a GeoJSON or Shapefile defining the spatial bounds.
- `--out`: Output Zarr store path.
- `--bands`: (Optional) List of specific bands to retrieve, separated by spaces (e.g., `--bands Sensor_Zenith Sensor_Azimuth`).
- `--scale`: (Optional) Spatial resolution in degrees.
- `--chunks`: (Optional) Chunking scheme (defaults to `auto`).

### 2. Preprocess Data

The `preprocess` command runs a series of preprocessing steps over the raw Zarr store. The steps are defined in a YAML or JSON configuration file.

```bash
blackmarble preprocess \
    --input raw.zarr \
    --config pipeline_config.yaml \
    --out preprocessed.zarr
```

**Arguments:**

- `--input`: Path to the raw Zarr store.
- `--config`: Path to the pipeline configuration file.
- `--out`: Output Zarr store path for the cleaned data.

**Configuration File Example:**

The configuration file dictates the exact `NTLPipeline` steps and their parameters. It can also define auxiliary datasets (catalogs) required by specific steps.

```yaml
catalog:
  VNP46A1: "path/to/auxiliary_vnp46a1.zarr"

steps:
  - filters.CloudSnowFilter: {}
  - angular.QuadraticVZACorrection: {}
  - imputation.LinearInterpolationGapFilling: {}
```

### 3. Aggregate Data

The `aggregate` command performs zonal statistics over vector geometries (like administrative districts) using the preprocessed data, and exports the results to Zarr and/or CSV.

```bash
blackmarble aggregate \
    --input preprocessed.zarr \
    --region path/to/districts.geojson \
    --geo-id district_id \
    --out-zarr aggregated.zarr \
    --out-csv aggregated.csv
```

**Arguments:**

- `--input`: Path to the preprocessed Zarr store.
- `--region`: Path to a GeoJSON or Shapefile defining the aggregation polygons.
- `--geo-id`: Column name in the region file that uniquely identifies each shape (defaults to `geonameid`).
- `--out-zarr`: Output Zarr store path for the aggregated multi-dimensional dataset.
- `--out-csv`: (Optional) Output CSV path to export the flattened results for downstream tabular analysis.
