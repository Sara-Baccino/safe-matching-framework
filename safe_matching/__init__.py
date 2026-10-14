"""safe-matching-framework: matching di popolazioni (PS, hungarian, sorted, partial OT, ...)."""

from safe_matching.pipeline import PipelineResult, run_pipeline
from safe_matching.schemas import DataConfig, MatchingConfig
from safe_matching.version import __version__

__all__ = ["run_pipeline", "PipelineResult", "DataConfig", "MatchingConfig", "__version__"]
