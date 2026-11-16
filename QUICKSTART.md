# Quick Start Guide

Get the RAG failure analysis project running in **under 10 minutes**.

## Prerequisites

- Python 3.8+
- 8GB+ RAM (16GB recommended)
- GPU optional (for faster embeddings/generation)

## Installation

### 1. Clone & Setup Environment

```bash
cd RetrieveAudit

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Create Sample Dataset

```bash
# Quick synthetic dataset for testing
python scripts/create_sample_dataset.py --type synthetic --output_dir data/processed

# Or load FinanceBench (requires internet)
python scripts/create_sample_dataset.py --type financebench --samples 100 --output_dir data/processed
```

This creates:
- `data/processed/chunks.json` - Chunked documents
- `data/processed/qa_pairs.json` - Q&A pairs with ground truth

### 3. Run All Pipelines

```bash
python experiments/run_all_pipelines.py \
    --chunks data/processed/chunks.json \
    --queries data/processed/qa_pairs.json \
    --output_dir results/metrics \
    --top_k 5
```

This will:
1. Initialize all 4 retrievers (BM25, Dense, Hybrid, Hybrid+Rerank)
2. Run retrieval on all queries
3. Compute evaluation metrics
4. Save results to `results/metrics/`

**Expected runtime:** 5-15 minutes depending on dataset size and hardware.

### 4. Visualize Results

```bash
python scripts/visualize_results.py \
    --metrics results/metrics/metrics_<timestamp>.json \
    --output_dir results/figures
```

Creates comparison charts in `results/figures/`:
- `retriever_comparison.png` - Bar charts of all metrics
- `recall_at_k.png` - Recall curves
- `metrics_table.png` - Summary table

## Using Google Colab (Recommended for GPU)

1. Upload the project to Google Drive or GitHub
2. Open `notebooks/colab_starter.ipynb` in Colab
3. Enable GPU: Runtime → Change runtime type → T4 GPU
4. Run all cells

## Next Steps

### Manual Error Analysis

Once you have results, analyze failures:

```bash
python experiments/analyze_failures.py \
    --results results/metrics/retrieval_results_<timestamp>.json \
    --ground_truth data/processed/qa_pairs.json \
    --output results/error_analysis/annotations.json \
    --sample_size 50
```

This starts an **interactive CLI** where you:
1. Review failed queries
2. Classify failure modes
3. Document error patterns

### Ablation Study: Chunk Size

Test different chunk sizes:

```bash
# Test 256 tokens
python scripts/create_sample_dataset.py --type synthetic --chunk_size 256 --output_dir data/256

python experiments/run_all_pipelines.py \
    --chunks data/256/chunks.json \
    --queries data/256/qa_pairs.json \
    --output_dir results/ablation/chunk_256

# Test 1024 tokens
python scripts/create_sample_dataset.py --type synthetic --chunk_size 1024 --output_dir data/1024

python experiments/run_all_pipelines.py \
    --chunks data/1024/chunks.json \
    --queries data/1024/qa_pairs.json \
    --output_dir results/ablation/chunk_1024
```

Compare results across chunk sizes.

## Project Structure Quick Reference

```
RetrieveAudit/
├── src/
│   ├── retrieval/          # 4 retrieval implementations
│   │   ├── sparse_retriever.py      (BM25)
│   │   ├── dense_retriever.py       (Embeddings)
│   │   ├── hybrid_retriever.py      (RRF)
│   │   └── reranker.py              (Cross-encoder)
│   ├── evaluation/
│   │   ├── metrics.py               (Recall, MRR, etc.)
│   │   └── error_taxonomy.py        (Failure modes)
│   └── data_processing/
│       ├── chunker.py               (Document chunking)
│       └── pdf_parser.py            (PDF extraction)
├── experiments/
│   ├── run_all_pipelines.py         (Main orchestrator)
│   └── analyze_failures.py          (Error analysis)
├── scripts/
│   ├── create_sample_dataset.py     (Data prep)
│   └── visualize_results.py         (Plotting)
└── notebooks/
    └── colab_starter.ipynb          (Colab notebook)
```

## Testing Individual Components

### Test Sparse Retrieval

```python
from src.retrieval.sparse_retriever import create_sparse_retriever

chunks = [
    {'chunk_id': 'c1', 'doc_id': 'd1', 'text': 'RAG combines retrieval and generation.'},
    {'chunk_id': 'c2', 'doc_id': 'd1', 'text': 'BM25 is a sparse retrieval algorithm.'}
]

retriever = create_sparse_retriever(chunks)
results = retriever.retrieve("What is BM25?", top_k=2)

for r in results:
    print(f"Rank {r['rank']}: {r['chunk_id']} (score: {r['score']:.3f})")
```

### Test Dense Retrieval

```python
from src.retrieval.dense_retriever import create_dense_retriever

retriever = create_dense_retriever(
    chunks, 
    model_name="sentence-transformers/all-MiniLM-L6-v2"  # Smaller model for testing
)

results = retriever.retrieve("semantic search", top_k=2)
```

## Common Issues

### Out of Memory
- Use smaller embedding models: `all-MiniLM-L6-v2` instead of `bge-base-en-v1.5`
- Reduce batch size in retrievers
- Use CPU instead of GPU for small datasets

### Slow Downloads
- Models are downloaded from Hugging Face on first run
- Use Colab or cached models after first download

### Import Errors
- Make sure you're in the project root directory
- Activate virtual environment
- Check `requirements.txt` installation

## Tips for Best Results

1. **Start Small**: Test on 10-20 queries first
2. **Use GPU**: Colab T4 GPU is free and 5-10x faster
3. **Ground Truth Quality**: Manually verify 20-30 ground truth annotations
4. **Document Everything**: Keep notes on interesting failure patterns

## Getting Help

- Check example notebooks in `notebooks/`
- Review test code at bottom of each `.py` file (run with `python filename.py`)
- Open an issue on GitHub with error messages

---

**Ready to start?** Run the 3-command quick test:

```bash
python scripts/create_sample_dataset.py --type synthetic
python experiments/run_all_pipelines.py --chunks data/processed/chunks.json --queries data/processed/qa_pairs.json
python scripts/visualize_results.py --metrics results/metrics/metrics_*.json
```

Check `results/figures/` for your first comparison plots!
