# Black Marble NTL Toolkit

[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://jankokla.github.io/blackmarble-ntl-toolkit/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Black Marble NTL Toolkit** is a Python package designed to streamline the retrieval, cleaning, aggregation, and visualization of NASA's Nighttime Light (NTL) data. 

Deploying NTL data for conflict monitoring or socioeconomic research demands a combination of deep domain expertise and advanced technical proficiency to retrieve, process, and analyze large-scale datasets. Consequently, effective research relies on close collaboration between conflict researchers, machine learning specialists, and data engineers. To lower these technical barriers and facilitate independent experimentation, the codebase contains helper functions designed to streamline the retrieval, cleaning, and visualization of NTL data.

## 📚 Documentation

**Full documentation, tutorials, and API reference can be found here:**  
🔗 **[https://jankokla.github.io/blackmarble-ntl-toolkit/](https://jankokla.github.io/blackmarble-ntl-toolkit/)**

---

## ✨ Main Functionality

### 1. Earth Engine Authentication & Data Retrieval
Before retrieving data, you must authenticate and initialize the Google Earth Engine Python API. Once initialized, you can seamlessly pull down daily and annual NASA Black Marble products.

```python
from blackmarble_toolkit.utils import initialize_ee
from blackmarble_toolkit.retrieval import BlackMarbleRetriever

initialize_ee(project_name="your-gee-project-id")

retriever = BlackMarbleRetriever()

raw_ds = retriever.get_data(
    product="VNP46A2",
    start_date="2022-01-01", 
    end_date="2022-12-31",
    region=geometry
)
```

### 2. Preprocessing Pipeline
Create a robust, memory-efficient pipeline leveraging Dask and Xarray to filter out clouds, snow, and background noise, and correct geometric or angular distortions.

```python
from blackmarble_toolkit.pipeline import NTLPipeline
from blackmarble_toolkit.methods import filters, angular, geometric, imputation

steps = [
    filters.CloudSnowFilter(),
    angular.QuadraticVZACorrection(),
    geometric.AveragePooling2D(filter_size=(3, 3)),
    imputation.LinearInterpolationGapFilling()
]

pipeline = NTLPipeline(steps)
processed_ds = pipeline.run(raw_ds, cache_intermediates=True)
```

### 3. Spatial Aggregation
Map your pixel-level NTL radiance to custom geometries (like administrative regions) for time-series analysis.

```python
import geopandas as gpd

regions = gpd.read_file("regions.geojson")
regions = regions.reset_index(names="region_id")

aggregated_results = pipeline.aggregate(
    gdf=regions,
    geo_id_col="region_id"
)
```

### 4. Exporting to CSV

If you prefer doing downstream analysis and plotting in pandas or R, you can easily export the aggregated results to a CSV file.

```python
df = pipeline.to_csv("ntl_aggregated_results.csv")
```

## 🖥️ Command Line Interface (CLI)

The toolkit provides a powerful CLI designed for batch processing large regions by utilizing out-of-core computation and saving intermediate states as Zarr stores.

### 1. Download Data

Download Black Marble data for a specific region and time period and save it lazily to a Zarr store.
```bash
blackmarble download \
    --product VNP46A2 \
    --start-date 2023-01-01 \
    --end-date 2023-01-31 \
    --region path/to/districts.geojson \
    --out raw.zarr
```

### 2. Preprocess Data

Run a series of preprocessing steps (defined in a YAML/JSON configuration file) over the raw Zarr store.
```bash
blackmarble preprocess \
    --input raw.zarr \
    --config pipeline_config.yaml \
    --out preprocessed.zarr
```
*Example `pipeline_config.yaml`:*
```yaml
steps:
  - filters.CloudSnowFilter: {}
  - imputation.LinearInterpolationGapFilling: {}
```

### 3. Aggregate Data

Perform zonal statistics over your vector geometries and export to Zarr/CSV.
```bash
blackmarble aggregate \
    --input preprocessed.zarr \
    --region path/to/districts.geojson \
    --geo-id district_id \
    --out-zarr aggregated.zarr \
    --out-csv aggregated.csv
```

## 🛠️ Installation

You can install the toolkit directly via PyPI:

```bash
pip install blackmarble-ntl-toolkit
```

For more details on setting up your environment and configuring Earth Engine, please consult the [Installation Guide](https://jankokla.github.io/blackmarble-ntl-toolkit/content/get_started/installation/).
