"""
Error Taxonomy and Failure Mode Classification for RAG systems.
Categorizes retrieval and generation failures into actionable classes.
"""
from enum import Enum
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import json


class FailureMode(Enum):
    """Five core failure categories for RAG systems."""
    
    CHUNK_BOUNDARY_FRAGMENTATION = "chunk_boundary_fragmentation"
    LEXICAL_SEMANTIC_MISMATCH = "lexical_semantic_mismatch"
    MULTI_HOP_REASONING_COLLAPSE = "multi_hop_reasoning_collapse"
    LOST_IN_THE_MIDDLE = "lost_in_the_middle"
    EXTRINSIC_HALLUCINATION = "extrinsic_hallucination"
    OTHER = "other"


@dataclass
class ErrorAnnotation:
    """
    Structured annotation for a failed RAG response.
    """
    query_id: str
    query: str
    generated_answer: str
    ground_truth_answer: str
    retrieved_chunks: List[Dict]
    
    # Error classification
    failure_mode: FailureMode
    failure_description: str
    
    # Context analysis
    correct_chunks_retrieved: bool
    correct_chunk_positions: List[int] = field(default_factory=list)
    
    # Additional metadata
    retriever_type: str = "unknown"
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'query_id': self.query_id,
            'query': self.query,
            'generated_answer': self.generated_answer,
            'ground_truth_answer': self.ground_truth_answer,
            'retrieved_chunks': self.retrieved_chunks,
            'failure_mode': self.failure_mode.value,
            'failure_description': self.failure_description,
            'correct_chunks_retrieved': self.correct_chunks_retrieved,
            'correct_chunk_positions': self.correct_chunk_positions,
            'retriever_type': self.retriever_type,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ErrorAnnotation':
        """Create from dictionary."""
        return cls(
            query_id=data['query_id'],
            query=data['query'],
            generated_answer=data['generated_answer'],
            ground_truth_answer=data['ground_truth_answer'],
            retrieved_chunks=data['retrieved_chunks'],
            failure_mode=FailureMode(data['failure_mode']),
            failure_description=data['failure_description'],
            correct_chunks_retrieved=data['correct_chunks_retrieved'],
            correct_chunk_positions=data.get('correct_chunk_positions', []),
            retriever_type=data.get('retriever_type', 'unknown'),
            metadata=data.get('metadata', {})
        )


class ErrorAnalyzer:
    """
    Automated heuristics + manual annotation support for error taxonomy.
    """
    
    def __init__(self):
        self.annotations: List[ErrorAnnotation] = []
    
    def suggest_failure_mode(
        self,
        query: str,
        generated_answer: str,
        retrieved_chunks: List[Dict],
        ground_truth_chunks: Optional[List[str]] = None
    ) -> FailureMode:
        """
        Heuristic-based failure mode suggestion.
        Still requires manual verification.
        
        Args:
            query: User query
            generated_answer: LLM's generated response
            retrieved_chunks: Retrieved context chunks
            ground_truth_chunks: Known correct chunk IDs (if available)
            
        Returns:
            Suggested FailureMode
        """
        # Heuristic 1: Check if correct chunks were retrieved at all
        if ground_truth_chunks:
            retrieved_ids = [c['chunk_id'] for c in retrieved_chunks]
            correct_retrieved = any(gt in retrieved_ids for gt in ground_truth_chunks)
            
            if not correct_retrieved:
                # Likely lexical/semantic mismatch
                return FailureMode.LEXICAL_SEMANTIC_MISMATCH
            
            # Check if correct chunks are buried (positions 3+)
            correct_positions = [
                i for i, cid in enumerate(retrieved_ids) 
                if cid in ground_truth_chunks
            ]
            if correct_positions and min(correct_positions) >= 2:
                return FailureMode.LOST_IN_THE_MIDDLE
        
        # Heuristic 2: Check for multi-hop keywords
        multi_hop_keywords = ['compare', 'difference', 'both', 'and', 'versus']
        if any(kw in query.lower() for kw in multi_hop_keywords):
            return FailureMode.MULTI_HOP_REASONING_COLLAPSE
        
        # Heuristic 3: Check if answer is very different from retrieved chunks
        # (potential extrinsic hallucination)
        answer_tokens = set(generated_answer.lower().split())
        chunk_tokens = set()
        for chunk in retrieved_chunks:
            chunk_tokens.update(chunk['text'].lower().split())
        
        overlap = len(answer_tokens & chunk_tokens)
        if len(answer_tokens) > 0:
            overlap_ratio = overlap / len(answer_tokens)
            if overlap_ratio < 0.3:  # Less than 30% overlap
                return FailureMode.EXTRINSIC_HALLUCINATION
        
        # Default to OTHER for manual inspection
        return FailureMode.OTHER
    
    def add_annotation(self, annotation: ErrorAnnotation):
        """Add manually annotated error."""
        self.annotations.append(annotation)
    
    def get_failure_distribution(self) -> Dict[str, int]:
        """Get count of each failure mode."""
        distribution = {}
        for mode in FailureMode:
            count = sum(1 for a in self.annotations if a.failure_mode == mode)
            if count > 0:
                distribution[mode.value] = count
        return distribution
    
    def get_failures_by_retriever(self) -> Dict[str, Dict[str, int]]:
        """Break down failures by retriever type."""
        by_retriever = {}
        
        for annotation in self.annotations:
            retriever = annotation.retriever_type
            if retriever not in by_retriever:
                by_retriever[retriever] = {}
            
            mode = annotation.failure_mode.value
            by_retriever[retriever][mode] = by_retriever[retriever].get(mode, 0) + 1
        
        return by_retriever
    
    def save_annotations(self, filepath: str):
        """Save annotations to JSON file."""
        data = [a.to_dict() for a in self.annotations]
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def load_annotations(self, filepath: str):
        """Load annotations from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.annotations = [ErrorAnnotation.from_dict(item) for item in data]
    
    def generate_report(self) -> str:
        """Generate human-readable error analysis report."""
        total = len(self.annotations)
        if total == 0:
            return "No error annotations found."
        
        report = f"# RAG Error Analysis Report\n\n"
        report += f"**Total Errors Analyzed:** {total}\n\n"
        
        # Failure distribution
        report += "## Failure Mode Distribution\n\n"
        distribution = self.get_failure_distribution()
        for mode, count in sorted(distribution.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total) * 100
            report += f"- **{mode}**: {count} ({percentage:.1f}%)\n"
        
        report += "\n## Failures by Retriever Type\n\n"
        by_retriever = self.get_failures_by_retriever()
        for retriever, modes in by_retriever.items():
            report += f"### {retriever}\n"
            for mode, count in sorted(modes.items(), key=lambda x: x[1], reverse=True):
                report += f"  - {mode}: {count}\n"
            report += "\n"
        
        # Key insights
        report += "## Key Insights\n\n"
        
        # Find most common failure
        if distribution:
            most_common = max(distribution.items(), key=lambda x: x[1])
            report += f"- Most common failure: **{most_common[0]}** ({most_common[1]} cases)\n"
        
        # Check retrieval success rate
        correct_retrieved = sum(1 for a in self.annotations if a.correct_chunks_retrieved)
        retrieval_success = (correct_retrieved / total) * 100 if total > 0 else 0
        report += f"- Correct chunks retrieved: {correct_retrieved}/{total} ({retrieval_success:.1f}%)\n"
        
        return report


if __name__ == "__main__":
    # Example usage
    analyzer = ErrorAnalyzer()
    
    # Create sample annotation
    annotation = ErrorAnnotation(
        query_id="q001",
        query="What was the total revenue in 2024?",
        generated_answer="The company had strong performance in 2024.",
        ground_truth_answer="Total revenue in 2024 was $5.2 billion.",
        retrieved_chunks=[
            {'chunk_id': 'doc1_chunk_5', 'text': 'The company expanded operations...', 'rank': 1},
            {'chunk_id': 'doc1_chunk_12', 'text': 'Total revenue in 2024 was $5.2 billion.', 'rank': 2}
        ],
        failure_mode=FailureMode.LOST_IN_THE_MIDDLE,
        failure_description="Correct chunk retrieved at rank 2 but LLM generated vague response.",
        correct_chunks_retrieved=True,
        correct_chunk_positions=[1],
        retriever_type="bm25"
    )
    
    analyzer.add_annotation(annotation)
    
    # Generate report
    print(analyzer.generate_report())
    
    # Save annotations
    # analyzer.save_annotations("error_annotations.json")
