"""
RetrieveAudit: RAG Retrieval Architecture & Failure Mode Analysis

A systematic empirical study of retrieval architectures and LLM hallucination 
patterns in Retrieval-Augmented Generation systems.
"""

__version__ = "0.1.0"
__author__ = "RetrieveAudit Contributors"

# Core modules
from . import retrieval
from . import evaluation
from . import data_processing
from . import generation

__all__ = [
    "retrieval",
    "evaluation", 
    "data_processing",
    "generation"
]
