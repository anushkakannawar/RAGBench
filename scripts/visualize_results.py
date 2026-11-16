"""
Visualization utilities for RAG experiment results.
Creates comparison charts, failure mode breakdowns, and ablation plots.
"""
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11


def load_metrics(metrics_file: str) -> Dict:
    """Load metrics from JSON file."""
    with open(metrics_file, 'r') as f:
        return json.load(f)


def plot_retriever_comparison(
    metrics: Dict[str, Dict[str, float]],
    output_path: str = "retriever_comparison.png"
):
    """
    Create bar chart comparing all retrievers across metrics.
    
    Args:
        metrics: Dict mapping retriever_name -> {metric_name: value}
        output_path: Where to save the plot
    """
    # Prepare data
    retrievers = list(metrics.keys())
    metric_names = ['recall@5', 'precision@5', 'mrr', 'map']
    
    # Extract values
    data = []
    for metric in metric_names:
        values = [metrics[r].get(metric, 0) for r in retrievers]
        data.append(values)
    
    # Create subplots
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    for idx, (metric, values) in enumerate(zip(metric_names, data)):
        ax = axes[idx]
        bars = ax.bar(retrievers, values, color=colors)
        
        ax.set_ylabel(metric.upper(), fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1.0)
        ax.set_title(metric.replace('_', ' ').title(), fontsize=13)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=10)
        
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved comparison plot to {output_path}")
    plt.close()


def plot_failure_modes(
    annotations_file: str,
    output_path: str = "failure_modes.png"
):
    """
    Create stacked bar chart showing failure modes by retriever.
    
    Args:
        annotations_file: Path to error annotations JSON
        output_path: Where to save the plot
    """
    with open(annotations_file, 'r') as f:
        annotations = json.load(f)
    
    # Count failures by retriever and mode
    failure_counts = {}
    
    for ann in annotations:
        retriever = ann.get('retriever_type', 'unknown')
        mode = ann.get('failure_mode', 'other')
        
        if retriever not in failure_counts:
            failure_counts[retriever] = {}
        
        failure_counts[retriever][mode] = failure_counts[retriever].get(mode, 0) + 1
    
    # Create DataFrame
    df = pd.DataFrame(failure_counts).fillna(0).T
    
    # Plot stacked bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    df.plot(kind='bar', stacked=True, ax=ax, 
            colormap='Set3', edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Retriever', fontsize=13, fontweight='bold')
    ax.set_ylabel('Number of Failures', fontsize=13, fontweight='bold')
    ax.set_title('Failure Mode Distribution by Retriever', fontsize=15, fontweight='bold')
    ax.legend(title='Failure Mode', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved failure modes plot to {output_path}")
    plt.close()


def plot_recall_at_k(
    metrics: Dict[str, Dict[str, float]],
    output_path: str = "recall_at_k.png"
):
    """
    Plot Recall@K curves for all retrievers.
    
    Args:
        metrics: Dict mapping retriever_name -> {metric_name: value}
        output_path: Where to save the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    k_values = [1, 3, 5, 10]
    
    for retriever, retriever_metrics in metrics.items():
        recall_values = [
            retriever_metrics.get(f'recall@{k}', 0)
            for k in k_values
        ]
        
        ax.plot(k_values, recall_values, marker='o', linewidth=2.5,
                markersize=8, label=retriever.replace('_', ' ').title())
    
    ax.set_xlabel('K (Number of Retrieved Documents)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Recall@K', fontsize=13, fontweight='bold')
    ax.set_title('Recall@K Comparison Across Retrievers', fontsize=15, fontweight='bold')
    ax.set_xticks(k_values)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved Recall@K plot to {output_path}")
    plt.close()


def create_summary_table(
    metrics: Dict[str, Dict[str, float]],
    output_path: str = "metrics_table.png"
):
    """
    Create a formatted table of all metrics.
    
    Args:
        metrics: Dict mapping retriever_name -> {metric_name: value}
        output_path: Where to save the table image
    """
    # Prepare data
    data = []
    for retriever, retriever_metrics in metrics.items():
        row = {
            'Retriever': retriever.replace('_', ' ').title(),
            'Recall@1': f"{retriever_metrics.get('recall@1', 0):.3f}",
            'Recall@5': f"{retriever_metrics.get('recall@5', 0):.3f}",
            'Precision@5': f"{retriever_metrics.get('precision@5', 0):.3f}",
            'MRR': f"{retriever_metrics.get('mrr', 0):.3f}",
            'MAP': f"{retriever_metrics.get('map', 0):.3f}"
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, len(data) * 0.8))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=df.values, colLabels=df.columns,
                     cellLoc='center', loc='center',
                     colColours=['#f0f0f0'] * len(df.columns))
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#4ECDC4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved metrics table to {output_path}")
    plt.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize RAG experiment results")
    parser.add_argument('--metrics', type=str, required=True,
                       help='Path to metrics JSON file')
    parser.add_argument('--annotations', type=str, default=None,
                       help='Path to error annotations JSON (optional)')
    parser.add_argument('--output_dir', type=str, default='results/figures',
                       help='Output directory for plots')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading metrics...")
    data = load_metrics(args.metrics)
    metrics = data.get('metrics', {})
    
    # Generate plots
    print("\nGenerating visualizations...")
    
    plot_retriever_comparison(
        metrics,
        output_path=str(output_dir / 'retriever_comparison.png')
    )
    
    plot_recall_at_k(
        metrics,
        output_path=str(output_dir / 'recall_at_k.png')
    )
    
    create_summary_table(
        metrics,
        output_path=str(output_dir / 'metrics_table.png')
    )
    
    if args.annotations and Path(args.annotations).exists():
        plot_failure_modes(
            args.annotations,
            output_path=str(output_dir / 'failure_modes.png')
        )
    
    print("\n" + "=" * 60)
    print("✅ All visualizations created successfully!")
    print(f"Check {output_dir} for the plots")


if __name__ == "__main__":
    main()
