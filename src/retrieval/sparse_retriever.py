"""
BM25 Sparse Retrieval Implementation.
Pure keyword-based retrieval using BM25 algorithm.
"""
from typing import List, Dict, Tuple
from rank_bm25 import BM25Okapi
import numpy as np


class SparseRetriever:
    """
    BM25-based sparse retrieval.
    Strong on exact entity/jargon matches, weak on synonyms.
    """
    
    def __init__(self, chunks: List[Dict[str, str]]):
        """
        Initialize BM25 index with document chunks.
        
        Args:
            chunks: List of dicts with 'chunk_id' and 'text' keys
        """
        self.chunks = chunks
        self.chunk_ids = [c['chunk_id'] for c in chunks]
        
        # Tokenize documents for BM25
        tokenized_corpus = [doc['text'].lower().split() for doc in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        print(f"[SparseRetriever] Indexed {len(chunks)} chunks")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Retrieve top-k most relevant chunks using BM25.
        
        Args:
            query: Search query
            top_k: Number of chunks to retrieve
            
        Returns:
            List of retrieved chunks with scores, sorted by relevance
        """
        # Tokenize query
        tokenized_query = query.lower().split()
        
        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_k_idx = np.argsort(scores)[::-1][:top_k]
        
        # Build results
        results = []
        for idx in top_k_idx:
            results.append({
                'chunk_id': self.chunk_ids[idx],
                'text': self.chunks[idx]['text'],
                'score': float(scores[idx]),
                'rank': len(results) + 1,
                'retriever': 'bm25',
                'doc_id': self.chunks[idx].get('doc_id', 'unknown')
            })
        
        return results
    
    def batch_retrieve(self, queries: List[str], top_k: int = 5) -> List[List[Dict]]:
        """
        Retrieve for multiple queries.
        
        Args:
            queries: List of search queries
            top_k: Number of chunks per query
            
        Returns:
            List of result lists, one per query
        """
        return [self.retrieve(q, top_k) for q in queries]


def create_sparse_retriever(chunks: List[Dict[str, str]]) -> SparseRetriever:
    """Factory function to create sparse retriever."""
    return SparseRetriever(chunks)


if __name__ == "__main__":
    # Example usage
    sample_chunks = [
        {
            'chunk_id': 'doc1_chunk_0',
            'doc_id': 'doc1',
            'text': 'Retrieval-Augmented Generation combines retrieval with generation. RAG systems retrieve relevant documents first.'
        },
        {
            'chunk_id': 'doc1_chunk_1',
            'doc_id': 'doc1',
            'text': 'BM25 is a sparse retrieval algorithm based on term frequency and inverse document frequency.'
        },
        {
            'chunk_id': 'doc2_chunk_0',
            'doc_id': 'doc2',
            'text': 'Dense retrieval uses neural embeddings to represent queries and documents in semantic space.'
        }
    ]
    
    retriever = create_sparse_retriever(sample_chunks)
    results = retriever.retrieve("What is BM25?", top_k=2)
    
    print("Query: What is BM25?")
    print("\nTop Results:")
    for r in results:
        print(f"  Rank {r['rank']}: {r['chunk_id']} (score: {r['score']:.3f})")
        print(f"    {r['text'][:80]}...")
