"""
Dense Retrieval Implementation using sentence embeddings.
Semantic search strong on paraphrasing, weak on exact matches.
"""
from typing import List, Dict, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss


class DenseRetriever:
    """
    Embedding-based dense retrieval using FAISS for efficient search.
    Uses BGE or E5 embeddings for semantic similarity.
    """
    
    def __init__(
        self, 
        chunks: List[Dict[str, str]], 
        model_name: str = "BAAI/bge-base-en-v1.5",
        device: Optional[str] = None
    ):
        """
        Initialize dense retriever with embedding model.
        
        Args:
            chunks: List of dicts with 'chunk_id' and 'text' keys
            model_name: Sentence transformer model name
            device: 'cuda', 'cpu', or None (auto-detect)
        """
        self.chunks = chunks
        self.chunk_ids = [c['chunk_id'] for c in chunks]
        
        print(f"[DenseRetriever] Loading model: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)
        
        # Encode all chunks
        print(f"[DenseRetriever] Encoding {len(chunks)} chunks...")
        chunk_texts = [c['text'] for c in chunks]
        self.embeddings = self.model.encode(
            chunk_texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  # For cosine similarity
        )
        
        # Build FAISS index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product = cosine similarity with normalized vectors
        self.index.add(self.embeddings)
        
        print(f"[DenseRetriever] Indexed {len(chunks)} chunks with dimension {dimension}")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Retrieve top-k most semantically similar chunks.
        
        Args:
            query: Search query
            top_k: Number of chunks to retrieve
            
        Returns:
            List of retrieved chunks with scores, sorted by similarity
        """
        # Encode query
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # Search in FAISS
        scores, indices = self.index.search(query_embedding, top_k)
        
        # Build results
        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
            results.append({
                'chunk_id': self.chunk_ids[idx],
                'text': self.chunks[idx]['text'],
                'score': float(score),
                'rank': rank,
                'retriever': 'dense',
                'doc_id': self.chunks[idx].get('doc_id', 'unknown')
            })
        
        return results
    
    def batch_retrieve(self, queries: List[str], top_k: int = 5) -> List[List[Dict]]:
        """
        Retrieve for multiple queries efficiently.
        
        Args:
            queries: List of search queries
            top_k: Number of chunks per query
            
        Returns:
            List of result lists, one per query
        """
        # Encode all queries at once
        query_embeddings = self.model.encode(
            queries,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # Batch search
        scores, indices = self.index.search(query_embeddings, top_k)
        
        # Build results for each query
        all_results = []
        for q_idx in range(len(queries)):
            query_results = []
            for rank, (idx, score) in enumerate(zip(indices[q_idx], scores[q_idx]), start=1):
                query_results.append({
                    'chunk_id': self.chunk_ids[idx],
                    'text': self.chunks[idx]['text'],
                    'score': float(score),
                    'rank': rank,
                    'retriever': 'dense',
                    'doc_id': self.chunks[idx].get('doc_id', 'unknown')
                })
            all_results.append(query_results)
        
        return all_results


def create_dense_retriever(
    chunks: List[Dict[str, str]], 
    model_name: str = "BAAI/bge-base-en-v1.5",
    device: Optional[str] = None
) -> DenseRetriever:
    """Factory function to create dense retriever."""
    return DenseRetriever(chunks, model_name=model_name, device=device)


if __name__ == "__main__":
    # Example usage
    sample_chunks = [
        {
            'chunk_id': 'doc1_chunk_0',
            'doc_id': 'doc1',
            'text': 'Retrieval-Augmented Generation combines information retrieval with text generation.'
        },
        {
            'chunk_id': 'doc1_chunk_1',
            'doc_id': 'doc1',
            'text': 'BM25 is a lexical matching algorithm for search based on term frequencies.'
        },
        {
            'chunk_id': 'doc2_chunk_0',
            'doc_id': 'doc2',
            'text': 'Neural embeddings enable semantic search by representing text in vector space.'
        }
    ]
    
    retriever = create_dense_retriever(sample_chunks, model_name="sentence-transformers/all-MiniLM-L6-v2")
    results = retriever.retrieve("How does semantic search work?", top_k=2)
    
    print("Query: How does semantic search work?")
    print("\nTop Results:")
    for r in results:
        print(f"  Rank {r['rank']}: {r['chunk_id']} (similarity: {r['score']:.3f})")
        print(f"    {r['text'][:80]}...")
