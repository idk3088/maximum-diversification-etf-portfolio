"""Compatibility exports for project metrics.

Step 4's implementation lives in :mod:`src.model_metrics`; future covariance
and portfolio metrics may extend this public module without duplicating logic.
"""

from src.model_metrics import (
    calculate_model_metrics,
    directional_accuracy,
    prediction_correlation,
)

__all__ = [
    "calculate_model_metrics",
    "directional_accuracy",
    "prediction_correlation",
]
