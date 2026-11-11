"""safe-matching-framework: matching di popolazioni (PS, hungarian, sorted, partial OT, ...).

Lettura, mappatura, imputazione e conversione in uscita sono delegate a safe-data-processing.
"""

from safe_matching.pipeline import PipelineResult, run_pipeline
from safe_matching.schemas import MatchingConfig
from safe_matching.version import __version__

__all__ = ["run_pipeline", "PipelineResult", "MatchingConfig", "__version__"]
