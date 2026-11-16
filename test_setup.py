"""
Quick test script to verify all components are working.
Run this after installation to check your setup.
"""
import sys
from pathlib import Path

print("=" * 80)
print("RetrieveAudit Setup Test")
print("=" * 80)

# Test imports
print("\n[1/6] Testing imports...")
try:
    from src.data_processing.chunker import DocumentChunker
    from src.retrieval.sparse_retriever import SparseRetriever
    from src.retrieval.dense_retriever import DenseRetriever
    from src.retrieval.hybrid_retriever import HybridRetriever
    from src.retrieval.reranker import HybridWithReranker
    from src.evaluation.metrics import RetrievalMetrics
    from src.evaluation.error_taxonomy import ErrorAnalyzer, FailureMode
    print("  ✓ All imports successful")
except Exception as e:
    print(f"  ✗ Import failed: {e}")
    sys.exit(1)

# Test chunking
print("\n[2/6] Testing document chunking...")
try:
    chunker = DocumentChunker(chunk_size=128, overlap=16)
    test_doc = {
        'doc_id': 'test_doc',
        'text': 'This is a test document. ' * 50
    }
    chunks = chunker.chunk_documents([test_doc])
    print(f"  ✓ Created {len(chunks)} chunks")
except Exception as e:
    print(f"  ✗ Chunking failed: {e}")
    sys.exit(1)

# Convert to dict format
chunks_dict = [
    {
        'chunk_id': c.chunk_id,
        'doc_id': c.doc_id,
        'text': c.text
    }
    for c in chunks
]

# Test sparse retrieval
print("\n[3/6] Testing BM25 sparse retrieval...")
try:
    sparse = SparseRetriever(chunks_dict)
    results = sparse.retrieve("test document", top_k=3)
    print(f"  ✓ Retrieved {len(results)} results")
except Exception as e:
    print(f"  ✗ Sparse retrieval failed: {e}")
    sys.exit(1)

# Test dense retrieval
print("\n[4/6] Testing dense retrieval...")
print("  (This may take a few minutes on first run - downloading model)")
try:
    dense = DenseRetriever(
        chunks_dict,
        model_name="sentence-transformers/all-MiniLM-L6-v2",  # Small model for testing
        device="cpu"
    )
    results = dense.retrieve("test document", top_k=3)
    print(f"  ✓ Retrieved {len(results)} results")
except Exception as e:
    print(f"  ✗ Dense retrieval failed: {e}")
    print(f"  Note: This requires sentence-transformers. Install: pip install sentence-transformers")
    sys.exit(1)

# Test hybrid retrieval
print("\n[5/6] Testing hybrid retrieval...")
try:
    hybrid = HybridRetriever(sparse, dense, k=60)
    results = hybrid.retrieve("test document", top_k=3)
    print(f"  ✓ Retrieved {len(results)} results with RRF scores")
except Exception as e:
    print(f"  ✗ Hybrid retrieval failed: {e}")
    sys.exit(1)

# Test metrics
print("\n[6/6] Testing evaluation metrics...")
try:
    metrics = RetrievalMetrics()
    retrieved_ids = ['test_doc_chunk_0', 'test_doc_chunk_1', 'test_doc_chunk_2']
    relevant_ids = ['test_doc_chunk_1', 'test_doc_chunk_5']
    
    recall = metrics.recall_at_k(retrieved_ids, relevant_ids, k=3)
    mrr = metrics.mean_reciprocal_rank(retrieved_ids, relevant_ids)
    
    print(f"  ✓ Computed metrics (Recall@3: {recall:.2f}, MRR: {mrr:.2f})")
except Exception as e:
    print(f"  ✗ Metrics computation failed: {e}")
    sys.exit(1)

# All tests passed
print("\n" + "=" * 80)
print("✅ ALL TESTS PASSED!")
print("=" * 80)
print("\nYour setup is ready. Next steps:")
print("1. Create a dataset:")
print("   python scripts/create_sample_dataset.py --type synthetic")
print("\n2. Run all pipelines:")
print("   python experiments/run_all_pipelines.py \\")
print("       --chunks data/processed/chunks.json \\")
print("       --queries data/processed/qa_pairs.json")
print("\n3. Visualize results:")
print("   python scripts/visualize_results.py \\")
print("       --metrics results/metrics/metrics_*.json")
print("\nSee QUICKSTART.md for detailed instructions.")
