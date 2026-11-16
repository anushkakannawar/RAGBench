"""Evaluation module - metrics and error taxonomy."""

from .metrics import (
    RetrievalMetrics,
    RAGMetrics,
    evaluate_retrieval
)

from .error_taxonomy import (
    FailureMode,
    ErrorAnnotation,
    ErrorAnalyzer
)

__all__ = [
    "RetrievalMetrics",
    "RAGMetrics",
    "evaluate_retrieval",
    "FailureMode",
    "ErrorAnnotation",
    "ErrorAnalyzer"
]
