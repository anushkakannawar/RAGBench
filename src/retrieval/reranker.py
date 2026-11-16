"""
Cross-Encoder Reranking for improved precision.
Two-stage retrieval: broad recall → precise reranking.
"""
from typing import List, Dict, Optional
from sentence_transformers import CrossEncoder
import numpy as np


class CrossEncoderReranker:
    """
    Rerank retrieved chunks using cross-encoder model.
    More accurate but computationally expensive than bi-encoders.
    """
    
    def __init__(
        self, 
        model_name: str = "BAAI/bge-reranker-base",
        device: Optional[str] = None
    ):
        """
        Initialize cross-encoder reranker.
        
        Args:
            model_name: Cross-encoder model name
            device: 'cuda', 'cpu', or None (auto-detect)
        """
        print(f"[CrossEncoderReranker] Loading model: {model_name}")
        self.model = CrossEncoder(model_name, device=device)
        self.model_name = model_name
        
        print(f"[CrossEncoderReranker] Model loaded on device: {self.model.device}")
    
    def rerank(
        self, 
        query: str, 
        chunks: List[Dict], 
        top_k: Optional[int] = None
    ) -> List[Dict]:
        """
        Rerank chunks using cross-encoder.
        
        Args:
            query: Search query
            chunks: List of retrieved chunks (from any retriever)
            top_k: Number of top chunks to return (None = return all)
            
        Returns:
            Reranked chunks with updated scores and ranks
        """
        if not chunks:
            return []
        
        # Prepare query-chunk pairs
        pairs = [[query, chunk['text']] for chunk in chunks]
        
        # Get reranking scores
        scores = self.model.predict(pairs, show_progress_bar=False)
        
        # Add scores to chunks
        for chunk, score in zip(chunks, scores):
            chunk['rerank_score'] = float(score)
        
        # Sort by rerank score
        reranked = sorted(chunks, key=lambda x: x['rerank_score'], reverse=True)
        
        # Update ranks and retriever name
        for rank, chunk in enumerate(reranked, start=1):
            chunk['rank'] = rank
            chunk['original_retriever'] = chunk.get('retriever', 'unknown')
            chunk['retriever'] = 'reranked'
        
        # Return top-k if specified
        if top_k is not None:
            return reranked[:top_k]
        
        return reranked
    
    def batch_rerank(
        self, 
        queries: List[str], 
        chunks_batch: List[List[Dict]], 
        top_k: Optional[int] = None
    ) -> List[List[Dict]]:
        """
        Rerank multiple queries.
        
        Args:
            queries: List of search queries
            chunks_batch: List of chunk lists, one per query
            top_k: Number of top chunks per query
            
        Returns:
            List of reranked chunk lists
        """
        results = []
        for query, chunks in zip(queries, chunks_batch):
            reranked = self.rerank(query, chunks, top_k=top_k)
            results.append(reranked)
        
        return results


class HybridWithReranker:
    """
    Two-stage retrieval: Hybrid retrieval → Cross-encoder reranking.
    Best precision but most computationally expensive.
    """
    
    def __init__(
        self, 
        hybrid_retriever,
        reranker: Optional[CrossEncoderReranker] = None,
        reranker_model: str = "BAAI/bge-reranker-base"
    ):
        """
        Initialize two-stage retriever.
        
        Args:
            hybrid_retriever: HybridRetriever instance
            reranker: CrossEncoderReranker instance (optional)
            reranker_model: Model name if creating new reranker
        """
        self.hybrid_retriever = hybrid_retriever
        
        if reranker is None:
            self.reranker = CrossEncoderReranker(model_name=reranker_model)
        else:
            self.reranker = reranker
        
        print(f"[HybridWithReranker] Two-stage retrieval initialized")
    
    def retrieve(
        self, 
        query: str, 
        top_k: int = 5, 
        retrieval_depth: int = 20
    ) -> List[Dict]:
        """
        Two-stage retrieval and reranking.
        
        Args:
            query: Search query
            top_k: Final number of chunks to return
            retrieval_depth: How many candidates to retrieve before reranking
            
        Returns:
            Reranked top-k results
        """
        # Stage 1: Broad retrieval
        candidates = self.hybrid_retriever.retrieve(
            query, 
            top_k=retrieval_depth,
            retrieval_depth=retrieval_depth
        )
        
        # Stage 2: Precise reranking
        reranked = self.reranker.rerank(query, candidates, top_k=top_k)
        
        return reranked
    
    def batch_retrieve(
        self, 
        queries: List[str], 
        top_k: int = 5,
        retrieval_depth: int = 20
    ) -> List[List[Dict]]:
        """
        Batch two-stage retrieval.
        
        Args:
            queries: List of search queries
            top_k: Final chunks per query
            retrieval_depth: Candidate depth before reranking
            
        Returns:
            List of reranked result lists
        """
        # Stage 1: Batch retrieval
        candidates_batch = self.hybrid_retriever.batch_retrieve(
            queries,
            top_k=retrieval_depth,
            retrieval_depth=retrieval_depth
        )
        
        # Stage 2: Batch reranking
        reranked_batch = self.reranker.batch_rerank(queries, candidates_batch, top_k=top_k)
        
        return reranked_batch


def create_reranker(
    model_name: str = "BAAI/bge-reranker-base",
    device: Optional[str] = None
) -> CrossEncoderReranker:
    """Factory function to create reranker."""
    return CrossEncoderReranker(model_name=model_name, device=device)


def create_hybrid_with_reranker(
    hybrid_retriever,
    reranker_model: str = "BAAI/bge-reranker-base"
) -> HybridWithReranker:
    """Factory function to create two-stage retriever."""
    return HybridWithReranker(hybrid_retriever, reranker_model=reranker_model)


if __name__ == "__main__":
    # Example usage
    sample_chunks = [
        {
            'chunk_id': 'doc1_chunk_0',
            'text': 'Cross-encoders process query and document together for accurate relevance scoring.',
            'score': 0.8,
            'rank': 1,
            'retriever': 'hybrid'
        },
        {
            'chunk_id': 'doc2_chunk_0',
            'text': 'Bi-encoders encode queries and documents separately for efficient retrieval.',
            'score': 0.7,
            'rank': 2,
            'retriever': 'hybrid'
        },
        {
            'chunk_id': 'doc3_chunk_0',
            'text': 'Reranking improves precision by applying expensive models to top candidates.',
            'score': 0.6,
            'rank': 3,
            'retriever': 'hybrid'
        }
    ]
    
    reranker = create_reranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    reranked = reranker.rerank("What are cross-encoders used for?", sample_chunks, top_k=2)
    
    print("Query: What are cross-encoders used for?")
    print("\nReranked Results:")
    for r in reranked:
        print(f"  Rank {r['rank']}: {r['chunk_id']} (score: {r['rerank_score']:.4f})")
        print(f"    {r['text'][:80]}...")
