---
icon: lucide/brush-cleaning
---

# Preprocessing

The `blackmarble-ntl-toolkit` uses a modular pipeline to preprocess Nighttime Lights (NTL) data. Once you have fetched your raw data following the [Data Retrieval](data_retrieval.md) guide, you can use this pipeline to clean and prepare it. The core of this system is the `NTLPipeline`, which takes a sequence of processing steps (methods) and applies them in order to an `xarray.Dataset`.

## Using the Pipeline

To use the preprocessing pipeline, initialize it with a list of processing steps and call `.run()` with your raw dataset.

```python
from blackmarble_toolkit.pipeline import NTLPipeline
from blackmarble_toolkit.methods.filters import (
    BlackMarbleHighQualityFilter,
    FilterLowNTL,
    ModifiedZScoreOutlierRemoval,
)

steps = [
    BlackMarbleHighQualityFilter(),
    FilterLowNTL(threshold=0.5),
    ModifiedZScoreOutlierRemoval(threshold=3.0),
]

pipeline = NTLPipeline(steps)
processed_ds = pipeline.run(ds)
```

## Auxiliary Data

Some transformations require additional datasets that are not included in the main [`VNP46A2`](https://developers.google.com/earth-engine/datasets/catalog/NASA_VIIRS_002_VNP46A2) product. You can retrieve these and pass them to the pipeline as a `catalog` dictionary.

### Daily Auxiliary Data

Methods like `QuadraticVZACorrection`, `Jia2023HighQualityFilter`, and `Yue2026DisturbanceFactorCorrection` rely on daily viewing zenith angles, solar angles, or lunar illumination. These are found in the [`VNP46A1`](https://developers.google.com/earth-engine/datasets/catalog/NOAA_VIIRS_001_VNP46A1) product.

!!! warning

    Google Earth Engine has not migrated the `VNP46A1` product to Collection 2. As a result, this product is currently only accessible for dates up until **2025-01-02**.

```python
from blackmarble_toolkit.retrieval import BlackMarbleRetriever
from blackmarble_toolkit.pipeline import NTLPipeline
from blackmarble_toolkit.methods.angular import QuadraticVZACorrection

retriever = BlackMarbleRetriever()

ds_a2 = retriever.get_data(# (1)! 
    product="VNP46A2",
    start_date="2022-01-01",
    end_date="2022-01-31",
    region=region,
)

ds_a1 = retriever.get_data(# (2)! 
    product="VNP46A1",
    start_date="2022-01-01",
    end_date="2022-01-31",
    region=region,
    bands=["Sensor_Zenith", "Lunar_Illuminated_Fraction"],
)

catalog = {"VNP46A1": ds_a1}

pipeline = NTLPipeline([QuadraticVZACorrection()])
processed_ds = pipeline.run(ds_a2, catalog=catalog)
```

1. Retrieve the main product (VNP46A2).
2. Retrieve the auxiliary product (VNP46A1) containing necessary bands.

### Annual Auxiliary Data

Other methods, such as `Hu2024AngularCorrection`, require an annual baseline dataset. For this, you need to provide the annual NTL product (identified by the catalog key [`NOAA/VIIRS/DNB/ANNUAL_V22`](https://developers.google.com/earth-engine/datasets/catalog/NOAA_VIIRS_DNB_ANNUAL_V22#bands)).

```python
from blackmarble_toolkit.methods.angular import Hu2024AngularCorrection

ds_a2 = retriever.get_data(# (1)! 
    product="VNP46A2",
    start_date="2022-01-01",
    end_date="2022-01-31",
    region=region,
) 

annual_ds = retriever.get_data(# (2)! 
    product="NOAA/VIIRS/DNB/ANNUAL_V22",
    start_date="2022-01-01",
    end_date="2022-12-31",
    region=region,
) 

catalog = {"NOAA/VIIRS/DNB/ANNUAL_V22": annual_ds}

pipeline = NTLPipeline([Hu2024AngularCorrection()])
processed_ds = pipeline.run(ds_a2, catalog=catalog)
```

1. Retrieve the main product
2. Retrieve the annual baseline

## Available Processing Methods

The toolkit provides several categories of preprocessing methods, many of which implement algorithms from published remote sensing literature.

### Filters

Filters are used to remove low-quality observations, clouds, snow, or outliers from the dataset.

- `BlackMarbleHighQualityFilter`: Strictly relies on the VNP46A2 mandatory quality flags to retain only high-quality, cloud-free, and snow-free pixels.
- `CloudSnowFilter`: Removes observations affected by clouds, cirrus, and snow/ice based on specific quality flag bits.[^Li2022]
- `Jia2023HighQualityFilter`: Implements the high-quality filtering approach from Jia et al. (2023), handling solar stray light, clouds, and snow.[^Jia2023]
- `FilterLowNTL`: Masks out pixels with NTL values below a specified threshold (often used to remove background noise).[^Li2022]
- `ModifiedZScoreOutlierRemoval`: Removes temporal outliers at the pixel level using the Modified Z-Score method.[^Hong2021]

### Geometric Corrections

- `AveragePooling2D`: Applies a rolling mean across spatial dimensions to smooth the data.[^Romn2018]
- `Hu2024AAveraging`: Advanced spatial averaging based on Hu et al. (2024).[^Hu2024]

### Angular Correction

Angular correction normalizes the NTL radiance to account for variations in the satellite's viewing zenith angle (VZA).

- `QuadraticVZACorrection`: Normalizes pixel-wise radiance to a nadir observation (VZA = 0°) by modeling the relationship between NTL and VZA with a quadratic function.[^Zheng2025]
- `Hu2024AngularCorrection`: Implements the angular correction algorithm described by Hu et al. (2024).[^Hu2024]

### Other Transformations

- `Yue2026DisturbanceFactorCorrection`: Implements the disturbance factor correction from Yue et al. (2026).[^Yue2026]

### Imputation

Imputation methods fill spatial or temporal gaps created by the filtering steps.

- `LinearInterpolationGapFilling`: Fills missing values using linear interpolation along the time dimension.

## Pipeline Execution History

The `NTLPipeline` automatically tracks the sequence of applied methods in the resulting dataset's attributes. You can access the history to verify which steps were applied:

```python
import json

history = json.loads(processed_ds.attrs.get("processing_history", "[]"))
for step in history:
    print(step["step"], step["params"])
```

## API Reference

For detailed documentation of all available preprocessing filters, corrections, and imputation methods, refer to the [Preprocessing Methods API Reference](../api_reference/preprocessing_methods.md).


[^Hong2021]: Hong, Yuchen, et al. "A monthly night-time light composite dataset of NOAA-20 in China: a multi-scale comparison with S-NPP." International Journal of Remote Sensing 42.20 (2021): 7931-7951. [DOI: 10.1080/01431161.2021.1969057](https://doi.org/10.1080/01431161.2021.1969057)
[^Li2022]: Li, Tian, et al. "Continuous monitoring of nighttime light changes based on daily NASA’s Black Marble product suite." Remote Sensing of Environment 282 (2022): 113269. [DOI: 10.1016/j.rse.2022.113269](https://doi.org/10.1016/j.rse.2022.113269)
[^Jia2023]: Jia, M., Li, X., Gong, Y., Belabbes, S., & Dell'Oro, L. "Estimating natural disaster loss using improved daily night-time light data." International Journal of Applied Earth Observation and Geoinformation 120 (2023): 103359. [DOI: 10.1016/j.jag.2023.103359](https://doi.org/10.1016/j.jag.2023.103359)
[^Hu2024]: Hu, Yang, et al. "A self-adjusting method to generate daily consistent nighttime light data for the detection of short-term rapid human activities." Remote Sensing of Environment 304 (2024): 114077. [DOI: 10.1016/j.rse.2024.114077](https://doi.org/10.1016/j.rse.2024.114077)
[^Zheng2025]: Zheng, Qiming, et al. "Nighttime lights reveal substantial spatial heterogeneity and inequality in post-hurricane recovery." Remote Sensing of Environment 319 (2025): 114645. [DOI: 10.1016/j.rse.2025.114645](https://doi.org/10.1016/j.rse.2025.114645)
[^Romn2018]: Román, Miguel O., et al. "NASA’s Black Marble nighttime lights product suite." Remote Sensing of Environment 210 (2018): 113-143. [DOI: 10.1016/j.rse.2018.03.017](https://doi.org/10.1016/j.rse.2018.03.017)
[^Yue2026]: Yue, Xiangqi, et al. "Improved Daily Nighttime Light Data as High-Frequency Economic Indicator." Applied Sciences 16.2 (2026): 947. [DOI: 10.3390/app16020947](https://doi.org/10.3390/app16020947)

