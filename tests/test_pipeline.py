import json
from typing import Dict

import pytest
import xarray as xr

from blackmarble_toolkit.methods.base import PaperImplementation
from blackmarble_toolkit.pipeline import NTLPipeline


class DummyTransformation(PaperImplementation):
    @property
    def required_products_and_bands(self):
        return {"VNP46A2": {"qa_mask"}}

    def _transform(self, ds: xr.Dataset, **kwargs) -> xr.Dataset:
        # just mock a transformation by adding 1 to ntl
        return ds.assign(ntl=ds["ntl"] + 1)


class DummyTransformation2(PaperImplementation):
    @property
    def required_products_and_bands(self):
        return {}

    def _transform(self, ds: xr.Dataset, **kwargs) -> xr.Dataset:
        # Add another variable to test dataset extension
        return ds.assign(extra_band=ds["ntl"] * 2)


def test_validate_dependencies(
    synthetic_dataset: xr.Dataset, synthetic_catalog: Dict[str, xr.Dataset]
) -> None:
    pipeline = NTLPipeline([DummyTransformation()])

    pipeline._validate_dependencies(synthetic_dataset, synthetic_catalog)

    with pytest.raises(ValueError, match="Missing required bands"):
        pipeline._validate_dependencies(synthetic_dataset, {})


def test_pipeline_run(
    synthetic_dataset: xr.Dataset, synthetic_catalog: Dict[str, xr.Dataset]
) -> None:
    pipeline = NTLPipeline([DummyTransformation(), DummyTransformation2(threshold=10)])

    # standardize dataset first to test if standardizing works during the run
    ds_input = synthetic_dataset.rename({"ntl": "DNB_BRDF-Corrected_NTL"})

    result = pipeline.run(ds_input, synthetic_catalog)
    assert "ntl" in result.data_vars
    assert "DNB_BRDF-Corrected_NTL" not in result.data_vars

    assert (result["ntl"] == synthetic_dataset["ntl"] + 1).all()

    assert "extra_band" in result.data_vars
    assert (result["extra_band"] == result["ntl"] * 2).all()

    history_str = result.attrs.get("processing_history")
    assert history_str is not None

    history = json.loads(history_str)
    assert len(history) == 2
    assert history[0]["step"] == "DummyTransformation"
    assert history[1]["step"] == "DummyTransformation2"
    assert history[1]["params"] == {"threshold": 10}


def test_pipeline_cache_intermediates(
    synthetic_dataset: xr.Dataset, synthetic_catalog: Dict[str, xr.Dataset]
) -> None:
    pipeline = NTLPipeline([DummyTransformation(), DummyTransformation2()])
    
    pipeline.run(synthetic_dataset, synthetic_catalog, cache_intermediates=True)
    
    # 2 steps + initial standardized dataset = 3 intermediates
    assert len(pipeline._intermediates) == 3
    assert pipeline._intermediates[0].attrs.get("step") == "Raw"
    assert pipeline._intermediates[1].attrs.get("step") == "DummyTransformation"
    assert pipeline._intermediates[2].attrs.get("step") == "DummyTransformation2"

