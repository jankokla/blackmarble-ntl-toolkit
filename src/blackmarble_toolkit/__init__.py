import warnings

from .methods.base import PaperImplementation
from .methods.angular import Hu2024AngularCorrection, QuadraticVZACorrection
from .methods.filters import (
    BlackMarbleHighQualityFilter,
    CloudSnowFilter,
    FilterLowNTL,
    Jia2023HighQualityFilter,
)
from .methods.geometric import AveragePooling2D, Hu2024AAveraging
from .methods.imputation import LinearInterpolationGapFilling
from .pipeline import NTLPipeline
from .retrieval import BlackMarbleRetriever

warnings.filterwarnings(
    "ignore",
    message=".*Earth Engine is not initialized on worker.*",
    category=UserWarning,
)

__all__ = [
    "NTLPipeline",
    "PaperImplementation",
    "BlackMarbleRetriever",
    "BlackMarbleHighQualityFilter",
    "CloudSnowFilter",
    "Jia2023HighQualityFilter",
    "FilterLowNTL",
    "AveragePooling2D",
    "Hu2024AAveraging",
    "TemporalRollingAverage",
    "Jia2023GapFilling",
    "LinearInterpolationGapFilling",
    "QuadraticVZACorrection",
    "Yue2026DisturbanceFactorCorrection",
    "Hu2024AngularCorrection",
]
