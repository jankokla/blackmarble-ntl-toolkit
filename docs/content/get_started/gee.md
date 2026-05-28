---
icon: lucide/earth
---

# Google Earth Engine

To download the data, `blackmarble-ntl-toolkit` uses `Xee`, which makes requests to [Google Earth Engine](https://developers.google.com/earth-engine/guides) for data. To use Earth Engine, you'll need to create and register a Google Cloud project, authenticate with Google, and initialize the service.

!!! note

    if you're having trouble creating the project, refer to the Earth Engine
    [Authentication and Initialization guide](https://developers.google.com/earth-engine/guides/auth).

## 1. Create a Project

Follow instructions in the [Earth Engine Access guide](https://developers.google.com/earth-engine/guides/access#get_access_to_earth_engine) to create and register a Google Cloud project.

## 2. Authenticate

### With Environment Variable

Google needs to know who is accessing Earth Engine to determine what services are available and what permissions are granted. The goal of authentication is to establish credentials that can be used during initialization. To do that, you need to specify the project name. For that purpose we use an environment variable. You can create an `.env` file and inside the file, set your project ID:

```env
# Earth Engine Project ID required for authentication and retrieval
EE_PROJECT="your-google-cloud-project-id"
```

Alternatively, you can export it directly in your terminal:

```bash
export EE_PROJECT="your-google-cloud-project-id"
```

### Specifying it in Data Retrieval

You can also specify it during the data retrieval:

```python
from blackmarble_toolkit.retrieval import BlackMarbleRetriever, geojson_to_geometry

EXAMPLE_GEOJSON = {...}

retriever = BlackMarbleRetriever(project_name="your-google-cloud-project-id")
region = geojson_to_geometry(EXAMPLE_GEOJSON)


ds = retriever.get_data(
    product="VNP46A2",
    start_date="2022-01-01",
    end_date="2023-12-31",
    region=region,
    bands=["DNB_BRDF_Corrected_NTL", "QF_Cloud_Mask", "Snow_Flag"],
)
```
