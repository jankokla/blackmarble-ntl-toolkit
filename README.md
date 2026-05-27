# Black Marble NTL Toolkit

A Python toolkit for retrieving and preprocessing NASA's Black Marble Nighttime Light (NTL) data using Google Earth Engine, xarray, and Dask.

## Configuration

This project requires authentication with Google Earth Engine to retrieve NASA Black Marble data. You must provide your Earth Engine Project ID as an environment variable before running the pipeline.

You can set this up by copying the `.env.example` file to a `.env` file in the root of the project:

```bash
cp .env.example .env
```

Inside the `.env` file, set your project ID:

```env
# Earth Engine Project ID required for authentication and retrieval
EE_PROJECT="your-google-cloud-project-id"
```

Alternatively, you can export it directly in your terminal:

```bash
export EE_PROJECT="your-google-cloud-project-id"
```
