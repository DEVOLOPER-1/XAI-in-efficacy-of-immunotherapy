"""src/inference.py — compatibility wrapper for the explainability pipeline.
The project is being rebuilt around shared patient splits and aligned
experiment settings. The real orchestration now lives in
`src.explainability`, while this module preserves the public entry points
used by `main.py` and older notebooks.
"""

from __future__ import annotations

from src.explainability import (
    ExplainabilityArtifact,
    ExplainabilityReport,
    evaluate,
    predict,
    run_explainability,
)

__all__ = [
    "ExplainabilityArtifact",
    "ExplainabilityReport",
    "evaluate",
    "predict",
    "run_explainability",
]
