"""
Hybrid Retrieval using Reciprocal Rank Fusion (RRF).
Combines BM25 sparse and dense semantic retrieval.
"""
from typing import List, Dict
from collections import defaultdict


class HybridRetriever:
    """
    Combines sparse (BM25) and dense retrievers using Reciprocal Rank Fusion.
    Balances lexical and semantic matching strengths.
    """
    
    def __init__(self, sparse_retriever, dense_retriever, k: int = 60):
        """
        Initialize hybrid retriever.
        
        Args:
            sparse_retriever: SparseRetriever instance
            dense_retriever: DenseRetriever instance
            k: RRF parameter (typically 60)
        """
        self.sparse_retriever = sparse_retriever
        self.dense_retriever = dense_retriever
        self.k = k
        
        print(f"[HybridRetriever] Initialized with RRF k={k}")
    
    def _reciprocal_rank_fusion(
        self, 
        sparse_results: List[Dict], 
        dense_results: List[Dict]
    ) -> List[Dict]:
        """
        Fuse results using Reciprocal Rank Fusion.
        
        RRF score = sum(1 / (k + rank)) for each retriever
        
        Args:
            sparse_results: Results from BM25
            dense_results: Results from dense retriever
            
        Returns:
            Fused and re-ranked results
        """
        # Calculate RRF scores
        rrf_scores = defaultdict(float)
        chunk_data = {}
        
        # Add sparse scores
        for result in sparse_results:
            chunk_id = result['chunk_id']
            rank = result['rank']
            rrf_scores[chunk_id] += 1.0 / (self.k + rank)
            chunk_data[chunk_id] = result
        
        # Add dense scores
        for result in dense_results:
            chunk_id = result['chunk_id']
            rank = result['rank']
            rrf_scores[chunk_id] += 1.0 / (self.k + rank)
            
            # Store chunk data if not already present
            if chunk_id not in chunk_data:
                chunk_data[chunk_id] = result
        
        # Sort by RRF score
        sorted_chunks = sorted(
            rrf_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Build final results
        fused_results = []
        for rank, (chunk_id, rrf_score) in enumerate(sorted_chunks, start=1):
            result = chunk_data[chunk_id].copy()
            result['rrf_score'] = rrf_score
            result['rank'] = rank
            result['retriever'] = 'hybrid_rrf'
            
            # Keep original scores for analysis
            result['original_retriever'] = chunk_data[chunk_id]['retriever']
            
            fused_results.append(result)
        
        return fused_results
    
    def retrieve(self, query: str, top_k: int = 5, retrieval_depth: int = 20) -> List[Dict]:
        """
        Retrieve using hybrid approach.
        
        Args:
            query: Search query
            top_k: Final number of chunks to return
            retrieval_depth: How many chunks to retrieve from each retriever before fusion
            
        Returns:
            Fused and re-ranked results
        """
        # Retrieve from both retrievers
        sparse_results = self.sparse_retriever.retrieve(query, top_k=retrieval_depth)
        dense_results = self.dense_retriever.retrieve(query, top_k=retrieval_depth)
        
        # Fuse results
        fused_results = self._reciprocal_rank_fusion(sparse_results, dense_results)
        
        # Return top-k
        return fused_results[:top_k]
    
    def batch_retrieve(
        self, 
        queries: List[str], 
        top_k: int = 5,
        retrieval_depth: int = 20
    ) -> List[List[Dict]]:
        """
        Retrieve for multiple queries.
        
        Args:
            queries: List of search queries
            top_k: Final number of chunks per query
            retrieval_depth: Retrieval depth per retriever
            
        Returns:
            List of fused result lists
        """
        # Batch retrieve from both retrievers
        sparse_results_batch = self.sparse_retriever.batch_retrieve(queries, top_k=retrieval_depth)
        dense_results_batch = self.dense_retriever.batch_retrieve(queries, top_k=retrieval_depth)
        
        # Fuse results for each query
        all_results = []
        for sparse_res, dense_res in zip(sparse_results_batch, dense_results_batch):
            fused = self._reciprocal_rank_fusion(sparse_res, dense_res)
            all_results.append(fused[:top_k])
        
        return all_results


def create_hybrid_retriever(sparse_retriever, dense_retriever, k: int = 60) -> HybridRetriever:
    """Factory function to create hybrid retriever."""
    return HybridRetriever(sparse_retriever, dense_retriever, k=k)


if __name__ == "__main__":
    # Example usage (requires sparse and dense retrievers)
    from sparse_retriever import create_sparse_retriever
    from dense_retriever import create_dense_retriever
    
    sample_chunks = [
        {
            'chunk_id': 'doc1_chunk_0',
            'doc_id': 'doc1',
            'text': 'Retrieval-Augmented Generation combines information retrieval with generation.'
        },
        {
            'chunk_id': 'doc1_chunk_1',
            'doc_id': 'doc1',
            'text': 'BM25 algorithm uses term frequency and inverse document frequency for ranking.'
        },
        {
            'chunk_id': 'doc2_chunk_0',
            'doc_id': 'doc2',
            'text': 'Dense retrieval leverages neural embeddings for semantic similarity search.'
        },
        {
            'chunk_id': 'doc3_chunk_0',
            'doc_id': 'doc3',
            'text': 'Reciprocal Rank Fusion combines multiple ranking signals effectively.'
        }
    ]
    
    sparse = create_sparse_retriever(sample_chunks)
    dense = create_dense_retriever(sample_chunks, model_name="sentence-transformers/all-MiniLM-L6-v2")
    hybrid = create_hybrid_retriever(sparse, dense, k=60)
    
    results = hybrid.retrieve("How does hybrid search combine different methods?", top_k=3)
    
    print("Query: How does hybrid search combine different methods?")
    print("\nTop Results:")
    for r in results:
        print(f"  Rank {r['rank']}: {r['chunk_id']} (RRF: {r['rrf_score']:.4f})")
        print(f"    From: {r['original_retriever']}")
        print(f"    {r['text'][:80]}...")
