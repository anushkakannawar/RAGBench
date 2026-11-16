"""
Interactive failure analysis tool.
Helps manually annotate and categorize RAG failures.
"""
import json
import sys
from pathlib import Path
from typing import List, Dict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from evaluation.error_taxonomy import (
    ErrorAnalyzer, 
    ErrorAnnotation, 
    FailureMode
)


def load_results(results_path: str) -> Dict:
    """Load retrieval results from JSON."""
    with open(results_path, 'r') as f:
        return json.load(f)


def find_failures(
    results: List[Dict],
    ground_truth: List[Dict],
    threshold: float = 0.5
) -> List[Dict]:
    """
    Identify failed queries for manual inspection.
    
    Args:
        results: Retrieval results
        ground_truth: Ground truth with correct answers
        threshold: Score threshold for considering a failure
        
    Returns:
        List of failed query data
    """
    gt_map = {item['query_id']: item for item in ground_truth}
    
    failures = []
    
    for result in results:
        query_id = result['query_id']
        retrieved_ids = result.get('retrieved_ids', [])
        
        if query_id not in gt_map:
            continue
        
        gt = gt_map[query_id]
        relevant_ids = set(gt.get('relevant_ids', []))
        
        # Check if any relevant docs were retrieved
        retrieved_set = set(retrieved_ids)
        hits = len(retrieved_set & relevant_ids)
        
        # Consider it a failure if recall is below threshold
        recall = hits / len(relevant_ids) if relevant_ids else 0
        
        if recall < threshold:
            failures.append({
                'query_id': query_id,
                'query': gt.get('query', ''),
                'retrieved_chunks': result.get('retrieved_chunks', []),
                'relevant_ids': list(relevant_ids),
                'recall': recall
            })
    
    return failures


def interactive_annotation(failures: List[Dict], analyzer: ErrorAnalyzer):
    """
    Interactive CLI for annotating failures.
    
    Args:
        failures: List of failed queries
        analyzer: ErrorAnalyzer instance
    """
    print("=" * 80)
    print("INTERACTIVE FAILURE ANALYSIS")
    print("=" * 80)
    print(f"\nTotal failures to analyze: {len(failures)}")
    print("\nFailure Modes:")
    for i, mode in enumerate(FailureMode, 1):
        print(f"  {i}. {mode.value}")
    print("  0. Skip this example")
    print("  q. Quit and save\n")
    
    for idx, failure in enumerate(failures, 1):
        print("\n" + "=" * 80)
        print(f"Example {idx}/{len(failures)}")
        print("=" * 80)
        
        print(f"\nQuery ID: {failure['query_id']}")
        print(f"Query: {failure['query']}")
        print(f"Recall: {failure['recall']:.2f}")
        
        print(f"\nRetrieved Chunks:")
        for i, chunk in enumerate(failure['retrieved_chunks'][:5], 1):
            print(f"  [{i}] {chunk['chunk_id']}")
            print(f"      {chunk['text'][:100]}...")
        
        print(f"\nRelevant IDs: {', '.join(failure['relevant_ids'][:5])}")
        
        # Get user input
        choice = input("\nSelect failure mode (1-6, 0 to skip, q to quit): ").strip()
        
        if choice.lower() == 'q':
            print("\nSaving and quitting...")
            break
        
        if choice == '0':
            continue
        
        try:
            mode_idx = int(choice) - 1
            modes = list(FailureMode)
            if 0 <= mode_idx < len(modes):
                failure_mode = modes[mode_idx]
            else:
                print("Invalid choice, skipping...")
                continue
        except ValueError:
            print("Invalid input, skipping...")
            continue
        
        # Get description
        description = input("Brief description of the failure: ").strip()
        
        # Check if correct chunks were retrieved
        retrieved_ids = [c['chunk_id'] for c in failure['retrieved_chunks']]
        correct_retrieved = any(
            rid in retrieved_ids 
            for rid in failure['relevant_ids']
        )
        
        # Find positions of correct chunks
        correct_positions = [
            i for i, cid in enumerate(retrieved_ids)
            if cid in failure['relevant_ids']
        ]
        
        # Create annotation
        annotation = ErrorAnnotation(
            query_id=failure['query_id'],
            query=failure['query'],
            generated_answer="",  # Would need actual generated answer
            ground_truth_answer="",  # Would need from dataset
            retrieved_chunks=failure['retrieved_chunks'],
            failure_mode=failure_mode,
            failure_description=description,
            correct_chunks_retrieved=correct_retrieved,
            correct_chunk_positions=correct_positions,
            retriever_type="unknown"  # Would need from results metadata
        )
        
        analyzer.add_annotation(annotation)
        print(f"✓ Annotated as: {failure_mode.value}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze RAG failures")
    parser.add_argument('--results', type=str, required=True, help='Path to results JSON')
    parser.add_argument('--ground_truth', type=str, required=True, help='Path to ground truth JSON')
    parser.add_argument('--output', type=str, default='results/error_analysis/annotations.json', 
                       help='Output path for annotations')
    parser.add_argument('--sample_size', type=int, default=50, 
                       help='Number of failures to sample')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Recall threshold for considering failure')
    
    args = parser.parse_args()
    
    # Load data
    print("Loading results...")
    results_data = load_results(args.results)
    
    # Extract results for a specific retriever (e.g., bm25)
    # Adjust based on your results structure
    results = results_data.get('bm25', [])
    
    with open(args.ground_truth, 'r') as f:
        gt_data = json.load(f)
    
    ground_truth = gt_data.get('queries', [])
    
    # Find failures
    print("Identifying failures...")
    failures = find_failures(results, ground_truth, args.threshold)
    print(f"Found {len(failures)} failures")
    
    # Sample if needed
    if len(failures) > args.sample_size:
        import random
        failures = random.sample(failures, args.sample_size)
        print(f"Sampled {args.sample_size} for analysis")
    
    # Initialize analyzer
    analyzer = ErrorAnalyzer()
    
    # Interactive annotation
    try:
        interactive_annotation(failures, analyzer)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user...")
    
    # Save annotations
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    analyzer.save_annotations(str(output_path))
    print(f"\n✓ Saved {len(analyzer.annotations)} annotations to {output_path}")
    
    # Generate report
    report = analyzer.generate_report()
    
    report_path = output_path.parent / "analysis_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"✓ Saved report to {report_path}")
    print("\n" + report)


if __name__ == "__main__":
    main()
