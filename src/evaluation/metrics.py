"""
Evaluation metrics for RAG retrieval quality.
Implements Recall@K, MRR, Precision, and custom RAG metrics.
"""
from typing import List, Dict, Set, Optional
import numpy as np
from collections import defaultdict


class RetrievalMetrics:
    """Calculate standard retrieval evaluation metrics."""
    
    @staticmethod
    def recall_at_k(
        retrieved_ids: List[str], 
        relevant_ids: List[str], 
        k: int
    ) -> float:
        """
        Recall@K: Fraction of relevant documents retrieved in top-k.
        
        Args:
            retrieved_ids: List of retrieved chunk IDs (in rank order)
            relevant_ids: List of ground-truth relevant chunk IDs
            k: Cutoff position
            
        Returns:
            Recall score [0, 1]
        """
        if not relevant_ids:
            return 0.0
        
        retrieved_set = set(retrieved_ids[:k])
        relevant_set = set(relevant_ids)
        
        hits = len(retrieved_set & relevant_set)
        return hits / len(relevant_set)
    
    @staticmethod
    def precision_at_k(
        retrieved_ids: List[str], 
        relevant_ids: List[str], 
        k: int
    ) -> float:
        """
        Precision@K: Fraction of retrieved documents that are relevant.
        
        Args:
            retrieved_ids: List of retrieved chunk IDs
            relevant_ids: List of ground-truth relevant chunk IDs
            k: Cutoff position
            
        Returns:
            Precision score [0, 1]
        """
        if k == 0:
            return 0.0
        
        retrieved_set = set(retrieved_ids[:k])
        relevant_set = set(relevant_ids)
        
        hits = len(retrieved_set & relevant_set)
        return hits / k
    
    @staticmethod
    def mean_reciprocal_rank(
        retrieved_ids: List[str], 
        relevant_ids: List[str]
    ) -> float:
        """
        Mean Reciprocal Rank: 1/rank of first relevant document.
        
        Args:
            retrieved_ids: List of retrieved chunk IDs (in rank order)
            relevant_ids: List of ground-truth relevant chunk IDs
            
        Returns:
            MRR score [0, 1]
        """
        relevant_set = set(relevant_ids)
        
        for rank, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in relevant_set:
                return 1.0 / rank
        
        return 0.0
    
    @staticmethod
    def average_precision(
        retrieved_ids: List[str], 
        relevant_ids: List[str]
    ) -> float:
        """
        Average Precision: Mean of precision values at each relevant document.
        
        Args:
            retrieved_ids: List of retrieved chunk IDs
            relevant_ids: List of ground-truth relevant chunk IDs
            
        Returns:
            AP score [0, 1]
        """
        if not relevant_ids:
            return 0.0
        
        relevant_set = set(relevant_ids)
        precisions = []
        num_relevant_seen = 0
        
        for rank, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in relevant_set:
                num_relevant_seen += 1
                precisions.append(num_relevant_seen / rank)
        
        if not precisions:
            return 0.0
        
        return sum(precisions) / len(relevant_set)
    
    @staticmethod
    def ndcg_at_k(
        retrieved_ids: List[str], 
        relevant_ids: List[str],
        relevance_scores: Optional[Dict[str, float]] = None,
        k: int = 10
    ) -> float:
        """
        Normalized Discounted Cumulative Gain at K.
        
        Args:
            retrieved_ids: List of retrieved chunk IDs
            relevant_ids: List of ground-truth relevant chunk IDs
            relevance_scores: Optional dict mapping chunk_id to relevance (0-1)
            k: Cutoff position
            
        Returns:
            NDCG score [0, 1]
        """
        # Default binary relevance if no scores provided
        if relevance_scores is None:
            relevance_scores = {rid: 1.0 for rid in relevant_ids}
        
        def dcg(ids, k):
            score = 0.0
            for rank, doc_id in enumerate(ids[:k], start=1):
                rel = relevance_scores.get(doc_id, 0.0)
                score += rel / np.log2(rank + 1)
            return score
        
        # Calculate DCG for retrieved results
        dcg_score = dcg(retrieved_ids, k)
        
        # Calculate ideal DCG
        ideal_order = sorted(
            relevant_ids, 
            key=lambda x: relevance_scores.get(x, 0.0), 
            reverse=True
        )
        idcg_score = dcg(ideal_order, k)
        
        if idcg_score == 0.0:
            return 0.0
        
        return dcg_score / idcg_score


class RAGMetrics:
    """Metrics specific to RAG system evaluation."""
    
    @staticmethod
    def context_coverage(
        retrieved_chunks: List[str],
        answer_text: str,
        ground_truth_chunks: List[str]
    ) -> float:
        """
        Measure how much of the ground truth context was retrieved.
        
        Args:
            retrieved_chunks: Retrieved chunk texts
            answer_text: Generated answer (not used, for future faithfulness checks)
            ground_truth_chunks: Ground truth chunk texts
            
        Returns:
            Coverage score [0, 1]
        """
        if not ground_truth_chunks:
            return 1.0
        
        retrieved_set = set(retrieved_chunks)
        gt_set = set(ground_truth_chunks)
        
        return len(retrieved_set & gt_set) / len(gt_set)
    
    @staticmethod
    def chunk_utilization(
        retrieved_chunks: List[Dict],
        answer_text: str,
        min_overlap_ratio: float = 0.3
    ) -> float:
        """
        Estimate what fraction of retrieved chunks were actually used.
        Simple heuristic: check if chunk tokens appear in answer.
        
        Args:
            retrieved_chunks: List of chunk dicts with 'text' key
            answer_text: Generated answer
            min_overlap_ratio: Minimum token overlap to consider chunk "used"
            
        Returns:
            Utilization ratio [0, 1]
        """
        if not retrieved_chunks:
            return 0.0
        
        answer_tokens = set(answer_text.lower().split())
        used_count = 0
        
        for chunk in retrieved_chunks:
            chunk_tokens = set(chunk['text'].lower().split())
            overlap = len(answer_tokens & chunk_tokens)
            
            if len(chunk_tokens) > 0:
                overlap_ratio = overlap / len(chunk_tokens)
                if overlap_ratio >= min_overlap_ratio:
                    used_count += 1
        
        return used_count / len(retrieved_chunks)


def evaluate_retrieval(
    results: List[Dict],
    ground_truth: List[Dict],
    k_values: List[int] = [1, 3, 5, 10]
) -> Dict[str, float]:
    """
    Evaluate retrieval results against ground truth.
    
    Args:
        results: List of dicts with 'query_id', 'retrieved_ids' keys
        ground_truth: List of dicts with 'query_id', 'relevant_ids' keys
        k_values: List of k values for Recall@K and Precision@K
        
    Returns:
        Dict of metric name -> average score
    """
    # Build ground truth lookup
    gt_map = {item['query_id']: item['relevant_ids'] for item in ground_truth}
    
    metrics = defaultdict(list)
    retrieval_metrics = RetrievalMetrics()
    
    for result in results:
        query_id = result['query_id']
        retrieved_ids = result['retrieved_ids']
        relevant_ids = gt_map.get(query_id, [])
        
        if not relevant_ids:
            continue
        
        # Calculate metrics for different k values
        for k in k_values:
            recall = retrieval_metrics.recall_at_k(retrieved_ids, relevant_ids, k)
            precision = retrieval_metrics.precision_at_k(retrieved_ids, relevant_ids, k)
            
            metrics[f'recall@{k}'].append(recall)
            metrics[f'precision@{k}'].append(precision)
        
        # Calculate MRR and AP
        mrr = retrieval_metrics.mean_reciprocal_rank(retrieved_ids, relevant_ids)
        ap = retrieval_metrics.average_precision(retrieved_ids, relevant_ids)
        
        metrics['mrr'].append(mrr)
        metrics['map'].append(ap)
    
    # Average all metrics
    avg_metrics = {
        name: np.mean(values) for name, values in metrics.items()
    }
    
    return avg_metrics


if __name__ == "__main__":
    # Example usage
    retrieved_ids = ['doc1_chunk_0', 'doc2_chunk_1', 'doc3_chunk_0', 'doc1_chunk_2']
    relevant_ids = ['doc1_chunk_0', 'doc1_chunk_2', 'doc5_chunk_1']
    
    metrics = RetrievalMetrics()
    
    print("Evaluation Metrics:")
    print(f"  Recall@3: {metrics.recall_at_k(retrieved_ids, relevant_ids, 3):.3f}")
    print(f"  Precision@3: {metrics.precision_at_k(retrieved_ids, relevant_ids, 3):.3f}")
    print(f"  MRR: {metrics.mean_reciprocal_rank(retrieved_ids, relevant_ids):.3f}")
    print(f"  AP: {metrics.average_precision(retrieved_ids, relevant_ids):.3f}")
