"""
Run all 4 retrieval pipelines and compare results.
Controlled experiment with identical settings except retrieval method.
"""
import json
import sys
from pathlib import Path
from typing import List, Dict
from datetime import datetime
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_processing.chunker import DocumentChunker
from retrieval.sparse_retriever import SparseRetriever
from retrieval.dense_retriever import DenseRetriever
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import HybridWithReranker
from evaluation.metrics import evaluate_retrieval


class RAGPipelineComparison:
    """
    Orchestrates comparison of all 4 retrieval architectures.
    """
    
    def __init__(
        self,
        chunks: List[Dict],
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        dense_model: str = "BAAI/bge-base-en-v1.5",
        reranker_model: str = "BAAI/bge-reranker-base",
        device: str = None
    ):
        """
        Initialize all 4 pipelines.
        
        Args:
            chunks: Pre-chunked documents
            chunk_size: Chunk size (for documentation)
            chunk_overlap: Chunk overlap (for documentation)
            dense_model: Dense retriever model name
            reranker_model: Reranker model name
            device: Device for models
        """
        self.chunks = chunks
        self.config = {
            'chunk_size': chunk_size,
            'chunk_overlap': chunk_overlap,
            'dense_model': dense_model,
            'reranker_model': reranker_model,
            'device': device
        }
        
        print("=" * 80)
        print("Initializing RAG Pipeline Comparison")
        print("=" * 80)
        
        # Pipeline 1: Sparse (BM25)
        print("\n[1/4] Building Sparse Retriever (BM25)...")
        self.sparse_retriever = SparseRetriever(chunks)
        
        # Pipeline 2: Dense
        print("\n[2/4] Building Dense Retriever...")
        self.dense_retriever = DenseRetriever(
            chunks, 
            model_name=dense_model,
            device=device
        )
        
        # Pipeline 3: Hybrid (RRF)
        print("\n[3/4] Building Hybrid Retriever (RRF)...")
        self.hybrid_retriever = HybridRetriever(
            self.sparse_retriever,
            self.dense_retriever,
            k=60
        )
        
        # Pipeline 4: Hybrid + Reranking
        print("\n[4/4] Building Hybrid + Reranker...")
        self.hybrid_reranker = HybridWithReranker(
            self.hybrid_retriever,
            reranker_model=reranker_model
        )
        
        print("\n" + "=" * 80)
        print("All pipelines initialized successfully!")
        print("=" * 80 + "\n")
    
    def run_comparison(
        self,
        queries: List[Dict],
        top_k: int = 5,
        retrieval_depth: int = 20
    ) -> Dict[str, List[Dict]]:
        """
        Run all 4 pipelines on the same queries.
        
        Args:
            queries: List of dicts with 'query_id' and 'query' keys
            top_k: Number of chunks to retrieve
            retrieval_depth: Depth for hybrid methods before reranking
            
        Returns:
            Dict mapping pipeline_name -> list of results
        """
        results = {
            'bm25': [],
            'dense': [],
            'hybrid_rrf': [],
            'hybrid_reranked': []
        }
        
        print(f"Running comparison on {len(queries)} queries...")
        print(f"Settings: top_k={top_k}, retrieval_depth={retrieval_depth}\n")
        
        # Extract query texts
        query_texts = [q['query'] for q in queries]
        query_ids = [q['query_id'] for q in queries]
        
        # Pipeline 1: BM25
        print("[1/4] Running BM25...")
        bm25_results = self.sparse_retriever.batch_retrieve(query_texts, top_k=top_k)
        for qid, retrieved in zip(query_ids, bm25_results):
            results['bm25'].append({
                'query_id': qid,
                'retrieved_chunks': retrieved,
                'retrieved_ids': [r['chunk_id'] for r in retrieved]
            })
        
        # Pipeline 2: Dense
        print("[2/4] Running Dense Retrieval...")
        dense_results = self.dense_retriever.batch_retrieve(query_texts, top_k=top_k)
        for qid, retrieved in zip(query_ids, dense_results):
            results['dense'].append({
                'query_id': qid,
                'retrieved_chunks': retrieved,
                'retrieved_ids': [r['chunk_id'] for r in retrieved]
            })
        
        # Pipeline 3: Hybrid RRF
        print("[3/4] Running Hybrid (RRF)...")
        hybrid_results = self.hybrid_retriever.batch_retrieve(
            query_texts, 
            top_k=top_k,
            retrieval_depth=retrieval_depth
        )
        for qid, retrieved in zip(query_ids, hybrid_results):
            results['hybrid_rrf'].append({
                'query_id': qid,
                'retrieved_chunks': retrieved,
                'retrieved_ids': [r['chunk_id'] for r in retrieved]
            })
        
        # Pipeline 4: Hybrid + Reranking
        print("[4/4] Running Hybrid + Reranking...")
        reranked_results = self.hybrid_reranker.batch_retrieve(
            query_texts,
            top_k=top_k,
            retrieval_depth=retrieval_depth
        )
        for qid, retrieved in zip(query_ids, reranked_results):
            results['hybrid_reranked'].append({
                'query_id': qid,
                'retrieved_chunks': retrieved,
                'retrieved_ids': [r['chunk_id'] for r in retrieved]
            })
        
        print("\nAll pipelines completed!\n")
        return results
    
    def evaluate_all(
        self,
        results: Dict[str, List[Dict]],
        ground_truth: List[Dict],
        k_values: List[int] = [1, 3, 5, 10]
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all pipeline results.
        
        Args:
            results: Results from run_comparison
            ground_truth: List of dicts with 'query_id' and 'relevant_ids'
            k_values: K values for metrics
            
        Returns:
            Dict mapping pipeline_name -> metrics dict
        """
        print("=" * 80)
        print("Evaluating All Pipelines")
        print("=" * 80 + "\n")
        
        all_metrics = {}
        
        for pipeline_name, pipeline_results in results.items():
            print(f"Evaluating {pipeline_name}...")
            metrics = evaluate_retrieval(pipeline_results, ground_truth, k_values)
            all_metrics[pipeline_name] = metrics
            
            # Print summary
            print(f"  Recall@5: {metrics.get('recall@5', 0):.3f}")
            print(f"  MRR: {metrics.get('mrr', 0):.3f}")
            print()
        
        return all_metrics
    
    def save_results(
        self,
        results: Dict,
        metrics: Dict,
        output_dir: Path
    ):
        """Save all results and metrics to files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_file = output_dir / f"retrieval_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved detailed results to: {results_file}")
        
        # Save metrics
        metrics_file = output_dir / f"metrics_{timestamp}.json"
        with open(metrics_file, 'w') as f:
            json.dump({
                'config': self.config,
                'metrics': metrics,
                'timestamp': timestamp
            }, f, indent=2)
        print(f"Saved metrics to: {metrics_file}")
        
        # Save summary report
        report_file = output_dir / f"summary_{timestamp}.txt"
        with open(report_file, 'w') as f:
            f.write("RAG Pipeline Comparison Report\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("Configuration:\n")
            for key, value in self.config.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")
            
            f.write("Results:\n")
            for pipeline, pipeline_metrics in metrics.items():
                f.write(f"\n{pipeline.upper()}:\n")
                for metric_name, value in sorted(pipeline_metrics.items()):
                    f.write(f"  {metric_name}: {value:.4f}\n")
        
        print(f"Saved summary report to: {report_file}\n")


def load_qa_dataset(filepath: str) -> tuple:
    """
    Load Q&A dataset with queries and ground truth.
    
    Expected format:
    {
        "queries": [
            {"query_id": "q001", "query": "What is...", "relevant_chunks": ["doc1_chunk_0", ...]}
        ]
    }
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    queries = []
    ground_truth = []
    
    for item in data.get('queries', []):
        queries.append({
            'query_id': item['query_id'],
            'query': item['query']
        })
        ground_truth.append({
            'query_id': item['query_id'],
            'relevant_ids': item.get('relevant_chunks', [])
        })
    
    return queries, ground_truth


def main():
    parser = argparse.ArgumentParser(description="Run all RAG pipelines comparison")
    parser.add_argument('--chunks', type=str, required=True, help='Path to chunked documents JSON')
    parser.add_argument('--queries', type=str, required=True, help='Path to queries JSON')
    parser.add_argument('--output_dir', type=str, default='results/metrics', help='Output directory')
    parser.add_argument('--top_k', type=int, default=5, help='Number of chunks to retrieve')
    parser.add_argument('--retrieval_depth', type=int, default=50, help='Retrieval depth before reranking')
    parser.add_argument('--dense_model', type=str, default='BAAI/bge-base-en-v1.5', help='Dense model')
    parser.add_argument('--reranker_model', type=str, default='BAAI/bge-reranker-base', help='Reranker model')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Load data
    print("Loading data...")
    with open(args.chunks, 'r') as f:
        chunks = json.load(f)
    
    queries, ground_truth = load_qa_dataset(args.queries)
    print(f"Loaded {len(chunks)} chunks and {len(queries)} queries\n")
    
    # Initialize comparison
    comparison = RAGPipelineComparison(
        chunks=chunks,
        dense_model=args.dense_model,
        reranker_model=args.reranker_model,
        device=args.device
    )
    
    # Run all pipelines
    results = comparison.run_comparison(
        queries=queries,
        top_k=args.top_k,
        retrieval_depth=args.retrieval_depth
    )
    
    # Evaluate
    metrics = comparison.evaluate_all(results, ground_truth)
    
    # Save results
    comparison.save_results(results, metrics, args.output_dir)
    
    print("=" * 80)
    print("Comparison Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
