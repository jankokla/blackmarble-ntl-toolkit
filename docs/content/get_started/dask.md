---
icon: lucide/network
---

# Using Dask

The `blackmarble-ntl-toolkit` leverages [Dask](https://www.dask.org/) under the hood to handle large arrays of Nighttime Light (NTL) data. Since satellite imagery and daily time-series can quickly exceed the memory capacity of a standard machine, Dask allows us to process this data in chunks and parallelize the computations.

## Why We Use Dask

Dask is designed to scale Python workflows. When using the toolkit:

- **Lazy Evaluation**: Transformations added to the `NTLPipeline` are lazily evaluated. This means that Dask builds a "computation graph" of all the operations you want to perform but doesn't actually process the data until you explicitly request the result (e.g., by calling `.compute()` or `.plot()`).
- **Distributed Computing**: Dask shines when dealing with massive datasets that don't fit into RAM. However, it's worth noting that if you are running this on a typical personal laptop or desktop, the advantages of a distributed system might be limited. The true power of Dask is unlocked when running on a cluster or a dedicated server where workloads can be distributed across many workers.
- **Memory Management**: Even on a single machine, Dask's chunking helps prevent out-of-memory errors by loading only the necessary parts of the data at any given time.

## Monitoring with the Client

To gain insight into how Dask is executing your tasks, managing memory, and parallelizing operations, you can initialize a local `Client` before running your pipeline. The Dask Client provides an interactive dashboard.

```python
from dask.distributed import Client

client = Client()

# The dashboard link will be printed to your console,
# typically at http://127.0.0.1:8787/status
print(client.dashboard_link)
```

Once the client is running, you can open the dashboard link in your web browser. When you execute an operation that triggers computation (like `pipeline.aggregate(..., compute=True)`), the dashboard will show you:

- **Task Stream**: A real-time visualization of tasks running across your CPU cores.
- **Progress**: A progress bar for the overall computation.
- **Memory Usage**: How much memory each worker is currently consuming.

Using the dashboard is highly recommended if you are processing large regions or long time periods, as it helps you identify bottlenecks or memory issues in your workflow.
