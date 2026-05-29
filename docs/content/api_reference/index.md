# API Reference

This section provides detailed documentation for the classes, methods, and functions available in the package. 

The API is divided into the following core modules:

- **[Retrieval](retrieval.md):** Tools for interfacing with Google Earth Engine (`BlackMarbleRetriever`) and downloading raw datasets.
- **[Preprocessing Methods](preprocessing_methods.md):** The individual filtering, correction, and imputation methods available for cleaning the NTL data.
- **[Pipeline](pipeline.md):** The `NTLPipeline` orchestration class, responsible for running sequences of transformations and caching intermediate steps.
- **[Aggregation](aggregation.md):** Functions for rasterizing vector shapes and performing memory-safe spatial aggregations using Dask.

For higher-level workflows and practical examples of how these components fit together, please see the [User Guide](../user_guide/index.md).
