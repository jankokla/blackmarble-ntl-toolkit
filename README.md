# Black Marble NTL Toolkit

[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://jankokla.github.io/blackmarble-ntl-toolkit/)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
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

## 🛠️ Installation

You can install the toolkit directly via PyPI:

```bash
pip install blackmarble-ntl-toolkit
```

For more details on setting up your environment and configuring Earth Engine, please consult the [Installation Guide](https://jankokla.github.io/blackmarble-ntl-toolkit/content/get_started/installation.md).
