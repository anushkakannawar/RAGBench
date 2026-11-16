"""
Create a sample dataset for testing the RAG pipelines.
Generates synthetic Q&A pairs or loads from FinanceBench.
"""
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_processing.chunker import DocumentChunker


def create_synthetic_dataset():
    """Create a small synthetic dataset for quick testing."""
    
    documents = [
        {
            "doc_id": "doc_001",
            "text": """
            Artificial Intelligence and Machine Learning have revolutionized many industries. 
            Deep learning models, particularly neural networks, have achieved remarkable success 
            in computer vision, natural language processing, and speech recognition. 
            
            Retrieval-Augmented Generation (RAG) is a technique that combines information retrieval 
            with language generation. The system first retrieves relevant documents from a knowledge 
            base, then uses those documents as context for generating responses. This approach helps 
            reduce hallucinations and grounds the model's outputs in factual information.
            
            RAG systems can fail in several ways. Chunk boundary fragmentation occurs when relevant 
            information is split across multiple chunks, causing the retriever to fetch only partial 
            context. Multi-hop reasoning failures happen when the question requires combining facts 
            from different sources, and the retriever only finds one of them.
            
            BM25 is a sparse retrieval algorithm based on term frequency and inverse document frequency. 
            It excels at matching exact terms and entities but struggles with synonyms and paraphrasing. 
            Dense retrieval, using neural embeddings, handles semantic similarity better but may miss 
            exact entity matches that BM25 would catch.
            """
        },
        {
            "doc_id": "doc_002",
            "text": """
            Vector databases enable efficient similarity search over high-dimensional embeddings. 
            Popular options include FAISS, Weaviate, Pinecone, and ChromaDB. These systems use 
            approximate nearest neighbor algorithms like HNSW or IVF to achieve sub-linear search times.
            
            Cross-encoder reranking improves retrieval precision by applying a more expensive model 
            to the top candidate documents. Unlike bi-encoders that encode queries and documents 
            separately, cross-encoders process query-document pairs jointly for more accurate scoring.
            
            Hybrid search combines multiple retrieval strategies. Reciprocal Rank Fusion (RRF) is 
            a simple yet effective method for merging ranked lists from different retrievers. 
            It assigns each document a score of 1/(k + rank), where k is typically 60, then sums 
            scores across all retrievers.
            """
        },
        {
            "doc_id": "doc_003",
            "text": """
            Evaluation metrics for retrieval systems include Recall@K, Precision@K, and Mean 
            Reciprocal Rank (MRR). Recall@K measures the fraction of relevant documents retrieved 
            in the top-K results. MRR is 1 divided by the rank of the first relevant document.
            
            For RAG systems specifically, we also care about answer quality metrics like 
            faithfulness (does the answer stick to retrieved context?) and answer correctness 
            (is it factually accurate?). The RAGAS framework provides automated evaluation tools.
            
            Common failure modes in RAG include: lexical-semantic mismatch (dense retrieval finds 
            topically similar but wrong documents), lost-in-the-middle (correct context retrieved 
            but LLM ignores it), and extrinsic hallucination (LLM uses parametric knowledge instead 
            of retrieved context).
            """
        },
        {
            "doc_id": "doc_004",
            "text": """
            Company ABC reported strong financial results for Q4 2024. Total revenue reached $5.2 billion,
            representing a 15% increase year-over-year. Operating expenses were $3.1 billion, resulting
            in an operating income of $2.1 billion. Net income for the quarter was $1.8 billion.
            
            The cloud services division was the primary growth driver, with revenue of $2.8 billion,
            up 25% from the previous year. The traditional software business generated $1.9 billion,
            growing at a modest 5%. Other revenue sources contributed $500 million.
            
            For the full year 2024, Company ABC achieved total revenue of $18.5 billion, with 
            net income of $6.2 billion. The company returned $2.5 billion to shareholders through
            dividends and share buybacks during the year.
            """
        }
    ]
    
    # Chunk documents
    print("Chunking documents...")
    chunker = DocumentChunker(chunk_size=128, overlap=16)
    all_chunks = chunker.chunk_documents(documents)
    
    # Convert to dict format
    chunks_dict = [
        {
            'chunk_id': c.chunk_id,
            'doc_id': c.doc_id,
            'text': c.text,
            'metadata': c.metadata
        }
        for c in all_chunks
    ]
    
    # Create Q&A pairs with ground truth
    qa_pairs = [
        {
            "query_id": "q001",
            "query": "What is Retrieval-Augmented Generation?",
            "answer": "RAG is a technique that combines information retrieval with language generation.",
            "relevant_chunks": ["doc_001_chunk_1"]
        },
        {
            "query_id": "q002",
            "query": "How does BM25 differ from dense retrieval?",
            "answer": "BM25 excels at exact term matching but struggles with synonyms, while dense retrieval handles semantic similarity better.",
            "relevant_chunks": ["doc_001_chunk_3"]
        },
        {
            "query_id": "q003",
            "query": "What is Reciprocal Rank Fusion used for?",
            "answer": "RRF is a method for merging ranked lists from different retrievers in hybrid search.",
            "relevant_chunks": ["doc_002_chunk_2"]
        },
        {
            "query_id": "q004",
            "query": "What are common RAG failure modes?",
            "answer": "Common failures include chunk boundary fragmentation, multi-hop reasoning failures, lexical-semantic mismatch, lost-in-the-middle, and extrinsic hallucination.",
            "relevant_chunks": ["doc_001_chunk_2", "doc_003_chunk_2"]
        },
        {
            "query_id": "q005",
            "query": "What is cross-encoder reranking?",
            "answer": "Cross-encoders process query-document pairs jointly for more accurate relevance scoring, unlike bi-encoders that encode separately.",
            "relevant_chunks": ["doc_002_chunk_1"]
        },
        {
            "query_id": "q006",
            "query": "What was Company ABC's total revenue in Q4 2024?",
            "answer": "$5.2 billion",
            "relevant_chunks": ["doc_004_chunk_0"]
        },
        {
            "query_id": "q007",
            "query": "Which division drove Company ABC's growth?",
            "answer": "The cloud services division with revenue of $2.8 billion, up 25%.",
            "relevant_chunks": ["doc_004_chunk_1"]
        },
        {
            "query_id": "q008",
            "query": "What is the difference between Recall and MRR?",
            "answer": "Recall@K measures fraction of relevant docs in top-K, while MRR is 1 divided by rank of first relevant document.",
            "relevant_chunks": ["doc_003_chunk_0"]
        },
        {
            "query_id": "q009",
            "query": "Name three vector database options",
            "answer": "FAISS, Weaviate, and Pinecone are popular vector database options.",
            "relevant_chunks": ["doc_002_chunk_0"]
        },
        {
            "query_id": "q010",
            "query": "What was Company ABC's full year 2024 net income?",
            "answer": "$6.2 billion",
            "relevant_chunks": ["doc_004_chunk_2"]
        }
    ]
    
    return chunks_dict, qa_pairs


def load_financebench_dataset(n_samples: int = 100):
    """Load FinanceBench dataset from Hugging Face - using Squad format as alternative."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: datasets library not installed. Run: pip install datasets")
        return None, None
    
    print("Loading SQuAD v2 dataset (real Q&A with Wikipedia context)...")
    # Use SQuAD - it has actual context passages
    dataset = load_dataset("rajpurkar/squad_v2", split="validation")
    
    documents = []
    qa_pairs = []
    seen_contexts = set()
    
    # Limit to n_samples
    sample_size = min(n_samples, len(dataset))
    samples = dataset.select(range(sample_size))
    
    print(f"Processing {sample_size} samples from SQuAD v2...")
    
    for i, example in enumerate(samples):
        # SQuAD format: has context (passage), question, and answers
        context = example.get('context', '')
        question = example.get('question', '')
        answers = example.get('answers', {})
        
        # Get answer text
        if answers and 'text' in answers and len(answers['text']) > 0:
            answer_text = answers['text'][0]
        else:
            answer_text = "No answer"
        
        # Create unique doc_id for each context
        doc_id = f'squad_doc_{i}'
        
        # Store document (each context is a separate document)
        if context and len(context) > 100:
            documents.append({
                'doc_id': doc_id,
                'text': context
            })
            
            # Create Q&A pair
            qa_pairs.append({
                'query_id': f'q{i:04d}',
                'query': question,
                'answer': answer_text,
                'doc_id': doc_id,
                'relevant_chunks': []  # Will be populated after chunking
            })
    
    # Chunk documents
    print(f"Chunking {len(documents)} documents...")
    chunker = DocumentChunker(chunk_size=512, overlap=64)
    all_chunks = chunker.chunk_documents(documents)
    
    chunks_dict = [
        {
            'chunk_id': c.chunk_id,
            'doc_id': c.doc_id,
            'text': c.text,
            'metadata': c.metadata
        }
        for c in all_chunks
    ]
    
    # Map chunks back to Q&A pairs based on doc_id
    # For each Q&A pair, find chunks from the same document
    print("Mapping chunks to queries...")
    for qa in qa_pairs:
        qa_doc_id = qa['doc_id']
        # Find all chunks from this document
        relevant_chunk_ids = [
            c['chunk_id'] for c in chunks_dict 
            if c['doc_id'] == qa_doc_id
        ]
        qa['relevant_chunks'] = relevant_chunk_ids[:3]  # Use first 3 chunks as ground truth
    
    return chunks_dict, qa_pairs


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Create sample dataset")
    parser.add_argument('--type', choices=['synthetic', 'financebench'], default='synthetic',
                       help='Dataset type')
    parser.add_argument('--samples', type=int, default=100,
                       help='Number of samples for FinanceBench')
    parser.add_argument('--output_dir', type=str, default='data/processed',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate or load dataset
    if args.type == 'synthetic':
        chunks, qa_pairs = create_synthetic_dataset()
    else:
        chunks, qa_pairs = load_financebench_dataset(args.samples)
        
        if chunks is None:
            print("Failed to load FinanceBench dataset")
            return
    
    # Save chunks
    chunks_file = output_dir / 'chunks.json'
    with open(chunks_file, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Saved {len(chunks)} chunks to {chunks_file}")
    
    # Save Q&A pairs
    qa_file = output_dir / 'qa_pairs.json'
    with open(qa_file, 'w', encoding='utf-8') as f:
        json.dump({'queries': qa_pairs}, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Saved {len(qa_pairs)} Q&A pairs to {qa_file}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Dataset Summary")
    print("=" * 60)
    print(f"Type: {args.type}")
    print(f"Chunks: {len(chunks)}")
    print(f"Q&A Pairs: {len(qa_pairs)}")
    print(f"\nSample Query:")
    print(f"  Q: {qa_pairs[0]['query']}")
    print(f"  A: {qa_pairs[0]['answer']}")


if __name__ == "__main__":
    main()
