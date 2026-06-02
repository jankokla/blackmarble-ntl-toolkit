---
icon: lucide/table
---

# Exporting

While the `blackmarble-ntl-toolkit` uses `xarray` internally to manage multi-dimensional datasets, many downstream workflows in data science and economics rely on tabular data structures. Because some users might prefer **pandas** (in Python) or **R** for downstream statistical analysis and plotting, we provide a convenient option to convert your aggregated results directly into a CSV file or a pandas `DataFrame`.

## Exporting to CSV

Once you have run the `.aggregate()` method on your `NTLPipeline`, you can easily export the results using the `.to_csv()` method.

```python
import geopandas as gpd
from blackmarble_toolkit.pipeline import NTLPipeline

pipeline = NTLPipeline(steps)
pipeline.run(ds=raw_ds, cache_intermediates=True) # (1)!

regions = gpd.read_file("path/to/regions.geojson")
regions['numeric_id'] = range(len(regions))

pipeline.aggregate(gdf=regions, geo_id_col="numeric_id") # (2)!

pipeline.to_csv("ntl_results.csv") # (3)!
```

1. Run pipeline and optionally cache intermediates
2. Aggregate the data over regions
3. Save directly to a CSV file

### Understanding the Output

The `to_csv()` method automatically converts the multi-dimensional dataset into a **long-format** tabular structure. It tracks the chronological order of your preprocessing pipeline so you can easily compare the effects of different filters.

The output will contain the following columns:

- **`time`**: The date of the observation.
- **`<geo_id_col>`**: The identifier for the vector shape (whatever you passed to `aggregate()`).
- **`step_index`**: The numerical sequence of the preprocessing step (e.g., `0` for Raw, `1` for the first filter).
- **`step`**: The string name of the preprocessing step applied.
- **`ntl`**: The aggregated Nighttime Light radiance value.
- **`valid_pct`**: The percentage of valid pixels for the region (if you enabled `is_valid_pct=True` during aggregation).

## Returning a DataFrame

If you don't provide a `file_path`, the `.to_csv()` method will simply return the `pandas.DataFrame` in memory. This is particularly useful if you want to immediately feed the data into libraries like **Seaborn**, **Statsmodels**, or **scikit-learn** without touching the disk.

```python
import seaborn as sns
import matplotlib.pyplot as plt

df = pipeline.to_csv() # (1)!

final_step_df = df[df['step'] == 'LinearInterpolationGapFilling'] # (2)!

sns.lineplot(data=final_step_df, x='time', y='ntl', hue='numeric_id') # (3)!
plt.show()
```

1. Get the dataframe without saving it to disk
2. Filter for the final processing step
3. Plot the timeseries using seaborn
