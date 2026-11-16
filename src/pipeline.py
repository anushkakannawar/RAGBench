"""
End-to-end RAG pipeline combining retrieval and generation.
"""
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class RAGResponse:
    """Complete RAG response with metadata."""
    query: str
    answer: str
    retrieved_chunks: List[Dict]
    retriever_name: str
    metadata: Optional[Dict] = None


class RAGPipeline:
    """
    Complete RAG pipeline: Query → Retrieve → Generate → Response
    """
    
    def __init__(
        self,
        retriever,
        generator,
        retriever_name: str = "unknown",
        top_k: int = 5
    ):
        """
        Initialize RAG pipeline.
        
        Args:
            retriever: Any retriever (sparse, dense, hybrid, reranked)
            generator: LLM generator instance
            retriever_name: Name for logging
            top_k: Number of chunks to retrieve
        """
        self.retriever = retriever
        self.generator = generator
        self.retriever_name = retriever_name
        self.top_k = top_k
    
    def query(
        self,
        query: str,
        top_k: Optional[int] = None,
        return_chunks: bool = True
    ) -> RAGResponse:
        """
        Execute full RAG pipeline.
        
        Args:
            query: User query
            top_k: Override default top_k
            return_chunks: Whether to include retrieved chunks in response
            
        Returns:
            RAGResponse with answer and metadata
        """
        k = top_k if top_k is not None else self.top_k
        
        # Step 1: Retrieve
        retrieved_chunks = self.retriever.retrieve(query, top_k=k)
        
        # Step 2: Generate
        answer = self.generator.generate(query, retrieved_chunks)
        
        # Step 3: Package response
        response = RAGResponse(
            query=query,
            answer=answer,
            retrieved_chunks=retrieved_chunks if return_chunks else [],
            retriever_name=self.retriever_name,
            metadata={
                'num_chunks_retrieved': len(retrieved_chunks),
                'top_k': k
            }
        )
        
        return response
    
    def batch_query(
        self,
        queries: List[str],
        top_k: Optional[int] = None
    ) -> List[RAGResponse]:
        """
        Execute pipeline on multiple queries.
        
        Args:
            queries: List of queries
            top_k: Override default top_k
            
        Returns:
            List of RAGResponse objects
        """
        k = top_k if top_k is not None else self.top_k
        
        # Batch retrieve
        retrieved_chunks_batch = self.retriever.batch_retrieve(queries, top_k=k)
        
        # Batch generate
        answers = self.generator.batch_generate(queries, retrieved_chunks_batch)
        
        # Package responses
        responses = []
        for query, answer, chunks in zip(queries, answers, retrieved_chunks_batch):
            response = RAGResponse(
                query=query,
                answer=answer,
                retrieved_chunks=chunks,
                retriever_name=self.retriever_name,
                metadata={
                    'num_chunks_retrieved': len(chunks),
                    'top_k': k
                }
            )
            responses.append(response)
        
        return responses


def create_pipeline(
    retriever,
    generator,
    retriever_name: str = "unknown",
    top_k: int = 5
) -> RAGPipeline:
    """Factory function to create RAG pipeline."""
    return RAGPipeline(retriever, generator, retriever_name, top_k)


if __name__ == "__main__":
    print("RAG Pipeline Module")
    print("=" * 60)
    print("\nThis module provides end-to-end RAG functionality.")
    print("\nExample usage:")
    print("""
from pipeline import create_pipeline
from retrieval.sparse_retriever import create_sparse_retriever
from generation.llm_generator import create_generator

# Initialize components
retriever = create_sparse_retriever(chunks)
generator = create_generator(generator_type="local")

# Create pipeline
pipeline = create_pipeline(
    retriever=retriever,
    generator=generator,
    retriever_name="bm25",
    top_k=5
)

# Query
response = pipeline.query("What is RAG?")
print(f"Answer: {response.answer}")
    """)
