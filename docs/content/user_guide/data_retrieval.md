---
icon: lucide/folder-down
---

# Data Retrieval

## Authentication

Before fetching data, you must authenticate and initialize Google Earth Engine. You can do this using the `initialize_ee` helper from the toolkit:

```python
from blackmarble_toolkit.utils import initialize_ee

initialize_ee(project_name="your-google-cloud-project-id")
```

See the [Google Earth Engine](../get_started/gee.md) setup guide for more details on acquiring a project ID.

## Specifying Region of Interest (ROI)

The `BlackMarbleRetriever` requires an `ee.Geometry` object to specify the spatial bounds for data retrieval. There are several ways to define or acquire this region.

### GADM

You can download administrative boundaries from [GADM](https://gadm.org/) (e.g., as GeoPackage or GeoJSON). Read the file using `geopandas`, and convert it using the provided `gdf_to_geometry` helper:

```python
import geopandas as gpd
from blackmarble_toolkit.retrieval import gdf_to_geometry

gdf = gpd.read_file("gadm41_CHE_0.json")

region = gdf_to_geometry(gdf)
```

### geemap

You can use [geemap](https://geemap.org/) to interactively draw a region of interest in a Jupyter notebook and retrieve the geometry.

!!! warning

    `geemap` is not included as a dependency of this toolkit and must be installed separately.

```python
import geemap

Map = geemap.Map()
Map

# ... interactively draw a polygon on the map ...

region = map.draw_last_feature.geometry()
```

### geojson.io

You can manually draw a polygon on [geojson.io](https://geojson.io/) and copy the resulting JSON. You can then use `geojson_to_geometry` to convert the dictionary to an `ee.Geometry`.

```python
from blackmarble_toolkit.retrieval import geojson_to_geometry

# paste the FeatureCollection or Polygon dictionary from geojson.io
geojson_dict = {
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {},
      "geometry": {
        "coordinates": [
          [
            [10.0, 45.0],
            [11.0, 45.0],
            [11.0, 46.0],
            [10.0, 46.0],
            [10.0, 45.0]
          ]
        ],
        "type": "Polygon"
      }
    }
  ]
}

region = geojson_to_geometry(geojson_dict)
```

## Parsing the Data

Once you have defined your region of interest and initialized Earth Engine, you can use the `BlackMarbleRetriever` to pull the Nighttime Lights data. 

The `get_data` method returns an `xarray.Dataset` loaded with the requested temporal and spatial bounds.

```python
from blackmarble_toolkit.retrieval import BlackMarbleRetriever

retriever = BlackMarbleRetriever()

ds = retriever.get_data(
    product="VNP46A2",
    start_date="2022-01-01",
    end_date="2022-01-31",
    region=region,
    bands=["DNB_BRDF_Corrected_NTL"],
    chunks="auto",  # (1)! 
)
```

1. enable dynamic chunking for memory efficiency

The resulting `xarray.Dataset` includes spatial and temporal dimensions (`time`, `lat`, `lon`) and is immediately ready for downstream processing, masking, or visualization.

!!! note "Lazy Evaluation with Dask"

    By using chunking (e.g., `chunks="auto"`), the returned dataset is backed by Dask. This means the data is **lazily loaded** and the code above executes instantly without actually fetching the data from Earth Engine yet.

    The actual data download is deferred until you trigger an action that requires the data in memory, such as:

    - Plotting the data
    - Writing the dataset to disk (e.g., `.to_netcdf()`)
    - Explicitly calling `ds.compute()` or `ds.load()`
