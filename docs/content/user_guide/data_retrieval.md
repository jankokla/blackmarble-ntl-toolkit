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

!!! tip "Visualizing your ROI"

    If you load your Region of Interest as a GeoPandas `GeoDataFrame`, you can quickly visualize it in a Jupyter notebook by calling `gdf.explore()`. This renders an interactive map to help verify your boundaries before fetching the data.

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

!!! warning "Sparse Geometries and Performance"

    When passing a multi-polygon or several distant geometries as the region, you may notice that data is retrieved for the entire bounding box encompassing them. This happens because of how `xee` bridges Google Earth Engine with Xarray.
    
    Xarray is fundamentally designed to handle dense, rectilinear n-dimensional arrays. It cannot inherently represent multiple disconnected spatial regions in a single continuous array without filling in the empty space between them. When you pass a multi-polygon or several distant geometries to `xee`, it calculates a single bounding box encompassing all of them and requests a regular grid for that entire spatial extent.

## Processing Sparse Geometries

For now, the package is more focused on a case study-based analysis (i.e. focused on a single contiguous area at a time). However, if you want to include a lot of scattered geometries around the world, you have two primary alternatives to avoid downloading large empty spaces.

### Spatial Clustering

If you want to use the toolkit's pipeline for gap-filling and preprocessing, you can use the `cluster_geometries` helper. This groups nearby geometries together, allowing you to fetch and process tightly-bound datasets iteratively.

```python
import geopandas as gpd
import pandas as pd
from blackmarble_toolkit.utils import cluster_geometries
from blackmarble_toolkit.retrieval import BlackMarbleRetriever, gdf_to_geometry
from blackmarble_toolkit.pipeline import NTLPipeline

gdf = gpd.read_file('regions.geojson')
clustered_gdf = cluster_geometries(gdf, max_distance_deg=2.0)  # (1)!

retriever = BlackMarbleRetriever()
pipeline = NTLPipeline([...])  # (2)!

results = []
for cluster_id, cluster_group in clustered_gdf.groupby('cluster_id'):
    cluster_region = gdf_to_geometry(cluster_group)  # (3)!
    
    ds = retriever.get_data(
        product="VNP46A2",
        start_date="2023-01-01",
        end_date="2023-01-31",
        region=cluster_region
    )
    
    pipeline.run(ds)
    pipeline.aggregate(cluster_group, geo_id_col="id", compute=True)
    results.append(pipeline.to_csv())

final_df = pd.concat(results, ignore_index=True)
```

1. Cluster geometries within 2 degrees of each other.
2. Provide your pipeline steps.
3. Get the bounding box *just* for this cluster, rather than the entire global dataset.

!!! tip "Visualizing Clusters"

    You can easily visualize the resulting clusters to verify their grouping by calling `clustered_gdf.explore(column='cluster_id', categorical=True)` in a Jupyter notebook.

### Earth Engine

If you don't need to apply any preprocessing and just want to retrieve the raw time-series for your geometries, you can bypass `xarray` and use Earth Engine reducers directly.

```python
import ee
import geopandas as gpd
import pandas as pd

gdf = gpd.read_file('regions.geojson')
fc = ee.FeatureCollection(gdf.__geo_interface__)

ic = (
    ee.ImageCollection("NASA/VIIRS/002/VNP46A2")
    .filterDate("2023-01-01", "2023-01-31")
    .select("DNB_BRDF_Corrected_NTL")
)

def extract_time_series(image):
    reduced = image.reduceRegions(
        collection=fc,
        reducer=ee.Reducer.mean(),
    )
    return reduced.map(lambda f: f.set('date', image.date().format('YYYY-MM-dd')))  # (1)!

time_series = ic.map(extract_time_series).flatten()  # (2)!

df = pd.DataFrame([feat['properties'] for feat in time_series.getInfo()['features']])  # (3)!
```

1. Add the image date to each feature.
2. Extract data and flatten the collection.
3. Retrieve as a pandas DataFrame.
