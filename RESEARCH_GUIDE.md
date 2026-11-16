# Research Guide: RAG Failure Mode Analysis

**How to turn this project into a compelling research contribution for ML interviews and graduate applications.**

## The Research Question

> **"How do different retrieval architectures fail in RAG systems, and what systematic patterns can we identify in LLM hallucinations?"**

This is not just "build a chatbot." You're conducting a **controlled empirical study** with:
- Independent variables: Retrieval architecture (BM25, Dense, Hybrid, Reranked)
- Dependent variables: Retrieval metrics + failure mode distribution
- Control variables: LLM, prompt template, chunk size, dataset

## Why This Stands Out

1. **Scientific Rigor**: You're treating RAG as a measurement problem, not an engineering task
2. **Novel Taxonomy**: You're creating a reusable framework for classifying RAG failures
3. **Actionable Insights**: Your findings tell practitioners *when* to use which retriever
4. **Reproducible**: Everything is documented, versioned, and testable

## 8-Day Research Plan

### Days 1-2: Data Preparation & Ground Truth

**Goal**: Clean dataset with 100-150 Q&A pairs and verified ground truth.

**Tasks**:
- [ ] Load FinanceBench or another messy-document dataset
- [ ] Manually verify 20 ground truth annotations (correct chunks per query)
- [ ] Create reproducible chunking pipeline (512 tokens, 64 overlap)
- [ ] Document data quality issues in `notebooks/01_data_exploration.ipynb`

**Deliverable**: `data/processed/` with chunks and verified Q&A pairs

**Red Flags to Avoid**:
- Don't skip ground truth verification - bad labels = bad conclusions
- Don't use only synthetic data - real documents have real failure modes
- Don't change chunk size mid-experiment

---

### Days 3-4: Pipeline Implementation

**Goal**: All 4 retrievers working and tested.

**Tasks**:
- [ ] Implement and test each retriever individually
- [ ] Run `test_setup.py` to verify all components
- [ ] Ensure identical settings: same LLM, same prompt, same context length
- [ ] Save retrieval results with full metadata (rank, score, retriever type)

**Deliverable**: Working pipelines in `src/retrieval/`

**Key Code Quality Checks**:
```python
# Always log configuration
config = {
    'chunk_size': 512,
    'chunk_overlap': 64,
    'dense_model': 'BAAI/bge-base-en-v1.5',
    'reranker_model': 'BAAI/bge-reranker-base',
    'top_k': 5,
    'retrieval_depth': 20
}

# Always include metadata
result = {
    'query_id': 'q001',
    'retrieved_chunks': [...],
    'retriever_name': 'hybrid_rrf',
    'config': config,
    'timestamp': '2024-11-16T14:30:00'
}
```

---

### Day 5: Automated Evaluation

**Goal**: Quantitative comparison of all architectures.

**Tasks**:
- [ ] Run `run_all_pipelines.py` on full dataset
- [ ] Compute Recall@K (K=1,3,5,10), MRR, MAP for each retriever
- [ ] Create comparison tables and plots
- [ ] Identify which queries each retriever fails on

**Deliverable**: Metrics in `results/metrics/`, plots in `results/figures/`

**Expected Findings** (based on similar studies):
- BM25: ~45-55% Recall@5, strong on exact entity names
- Dense: ~50-60% Recall@5, better on paraphrased queries
- Hybrid RRF: ~60-70% Recall@5, balanced
- Hybrid + Reranking: ~70-80% Recall@5, best overall

**If Your Results Differ**: That's research! Document *why* (dataset characteristics, domain specifics).

---

### Days 6-7: Manual Error Analysis (The Critical Part)

**Goal**: Annotate 50-100 failures with taxonomy labels.

**Process**:

1. **Sample Failed Queries**: Use `analyze_failures.py` to identify low-recall queries
   
2. **For Each Failure, Ask**:
   - Was the correct chunk retrieved? (Yes/No)
   - If yes, at what rank? (Lost-in-the-middle if rank ≥ 3)
   - If no, why? (Lexical mismatch? Chunk boundary split?)
   - Did the LLM use retrieved context or its own knowledge? (Extrinsic hallucination)
   - Did the question require multiple facts? (Multi-hop failure)

3. **Annotate Using the CLI**:
   ```bash
   python experiments/analyze_failures.py \
       --results results/metrics/retrieval_results.json \
       --ground_truth data/processed/qa_pairs.json \
       --sample_size 60
   ```

4. **Build Pattern Library**: Keep a spreadsheet with:
   | Query ID | Failure Mode | Retriever | Pattern Description | Example |
   |----------|--------------|-----------|---------------------|---------|
   | q023 | Chunk Boundary | BM25 | Revenue number split across chunks | "Total revenue was" in chunk 5, "$5.2B" in chunk 6 |

**Deliverable**: 
- `results/error_analysis/annotations.json` (structured data)
- `results/error_analysis/analysis_report.txt` (human-readable)
- Notebook with case studies: `notebooks/03_failure_analysis.ipynb`

**Quality Check**: Inter-annotator agreement
- Have a friend/colleague annotate 20 examples
- Compute agreement: `(agreements / total) > 0.75` is good
- Document disagreements and refine taxonomy

---

### Day 8: Ablation & Documentation

**Goal**: Test one architectural choice + polished documentation.

**Mini-Ablation Study**: Chunk Size Impact

```bash
# Test 256, 512, 1024 token chunks
for size in 256 512 1024; do
    python scripts/create_sample_dataset.py \
        --chunk_size $size \
        --output_dir data/chunk_$size
    
    python experiments/run_all_pipelines.py \
        --chunks data/chunk_$size/chunks.json \
        --queries data/chunk_$size/qa_pairs.json \
        --output_dir results/ablation/chunk_$size
done
```

**Expected Finding**: 
- Small chunks (256): Higher precision, more boundary issues
- Large chunks (1024): Lower precision, better context coverage

**Documentation Checklist**:
- [ ] Update README.md with key findings
- [ ] Add result plots to README
- [ ] Write `FINDINGS.md` with 3-5 key insights
- [ ] Clean up code comments
- [ ] Add docstrings to all functions
- [ ] Create requirements.txt with exact versions

---

## Key Insights to Highlight

### 1. Failure Mode Distribution by Retriever

Create this chart:

```
Failure Modes by Retriever Type

BM25:
  ████████████ 35% Lexical-Semantic Mismatch
  ████████ 25% Multi-Hop Failure
  ████ 15% Lost-in-Middle
  ... 

Dense:
  ████████ 25% Chunk Boundary Fragmentation
  ██████ 20% Lexical-Semantic Mismatch
  ...

Hybrid + Rerank:
  ████ 15% Multi-Hop Failure
  ██ 10% Lost-in-Middle
  ...
```

**Insight**: "Reranking reduced lost-in-the-middle failures by 60% compared to BM25."

### 2. When Each Retriever Wins

| Retriever | Best For | Example Query |
|-----------|----------|---------------|
| BM25 | Exact entity names, product codes | "What was AAPL revenue in Q4?" |
| Dense | Paraphrased/semantic queries | "How did the company perform last quarter?" |
| Hybrid RRF | General-purpose, unknown query types | Balanced performance |
| Hybrid + Rerank | High-stakes, latency-tolerant | When precision matters most |

### 3. Chunk Size Trade-Off

**Finding**: "Optimal chunk size depends on retrieval architecture. BM25 benefits from larger chunks (1024 tokens) for full context, while reranking works best with smaller chunks (256 tokens) to reduce noise."

---

## Writing the Research Summary

### For Resume (2-3 bullets)

> **RAG Retrieval & Hallucination Analysis** | Python, Transformers, FAISS  
> Benchmarked four retrieval architectures (BM25, BGE embeddings, Reciprocal Rank Fusion, Cross-Encoder Reranking) on 150 financial document queries; improved Recall@5 by 28% via two-stage reranking
> 
> Developed 5-category error taxonomy for RAG failure modes through manual analysis of 60 failed cases; identified chunk boundary fragmentation as responsible for 35% of BM25 failures, reduced to 12% with hybrid retrieval

### For GitHub README

See the generated `README.md` - it includes:
- Clear problem statement
- Architecture diagrams
- Results tables
- Example failure cases
- Instructions to reproduce

### For Blog Post / Medium Article

**Title**: "Why Your RAG System Fails: A Systematic Study of Retrieval Architectures"

**Structure**:
1. Motivation: RAG is everywhere, but failures are poorly understood
2. Method: Controlled experiment + failure taxonomy
3. Key Findings: Show failure mode charts
4. Case Studies: 3-4 interesting failure examples
5. Recommendations: Decision tree for choosing retriever
6. Limitations & Future Work

---

## Presenting Your Work

### Interview Talking Points

**"Tell me about a project you're proud of."**

> "I built a systematic study of retrieval architectures for RAG systems. Most people treat RAG as a black box, but I wanted to understand *how* and *why* different retrievers fail. I compared four architectures—sparse BM25, dense embeddings, hybrid search, and two-stage reranking—on 150 queries from financial documents.
> 
> The interesting part was the error analysis. I manually categorized 60 failures into five modes: chunk boundary issues, lexical mismatches, lost-in-the-middle, multi-hop reasoning collapse, and extrinsic hallucination. I found that reranking reduced 'lost-in-the-middle' failures by 60%, which makes sense because cross-encoders do better fine-grained relevance scoring.
> 
> I also ran an ablation on chunk size and found that optimal size depends on your retriever—BM25 needs larger chunks for full context, but reranking benefits from smaller, focused chunks. Everything is reproducible on GitHub with documented experiments."

**Follow-up they'll ask**: "What would you do differently?"

> "I'd love to test this on multiple domains—legal, medical, code—to see if failure patterns generalize. I'd also want to incorporate a generation step to measure end-to-end answer quality, not just retrieval. And I'd implement automated failure detection using LLM-as-a-judge to scale beyond manual annotation."

---

## Common Pitfalls to Avoid

### ❌ Don't Do This

1. **P-hacking**: Don't cherry-pick the one chunk size / model that gives best results
2. **Weak Baseline**: Don't compare against random retrieval or broken BM25
3. **No Ground Truth**: Don't rely on automated metrics alone without manual verification
4. **Inconsistent Settings**: Don't change prompts/LLMs between retrievers
5. **No Error Analysis**: Quantitative metrics alone don't tell the story

### ✅ Do This Instead

1. **Pre-register**: Document your hypotheses before running experiments
2. **Strong Baseline**: Use well-tuned BM25 and SOTA dense models
3. **Mixed Methods**: Combine automated metrics + manual error analysis
4. **Controlled Experiment**: Change ONE variable at a time
5. **Failure Case Studies**: Include 3-5 interesting failure examples in writeup

---

## Success Metrics

**You've succeeded when**:

- ✅ You can explain to a non-expert why reranking helps
- ✅ Your error taxonomy is reusable by others
- ✅ Your results contradict or refine conventional wisdom
- ✅ Someone can reproduce your work from GitHub alone
- ✅ You have 3-5 "TIL" (Today I Learned) insights
- ✅ You can confidently discuss limitations

**Red flags you're off track**:

- ❌ "My retriever is 10% better" without understanding why
- ❌ Results are too good (>95% accuracy on hard queries)
- ❌ You have no interesting failure cases
- ❌ You can't explain what's inside your error bars
- ❌ Your ablation studies confirm the obvious

---

## Resources & References

### Must-Read Papers

1. **"Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"** (Lewis et al., 2020)
   - Original RAG paper

2. **"Lost in the Middle"** (Liu et al., 2023)
   - LLMs ignore relevant context when it's buried

3. **"Benchmarking Large Language Models in Retrieval-Augmented Generation"** (Chen et al., 2023)
   - RGB benchmark for RAG evaluation

4. **"Contextual Document Embeddings"** (Saad-Falcon et al., 2024)
   - Why chunk boundaries matter

### Tools & Frameworks

- **RAGAS**: Automated RAG evaluation
- **FlashRAG**: Modular RAG framework
- **HuggingFace Datasets**: FinanceBench, HotpotQA, etc.
- **BGE Models**: SOTA embeddings from BAAI

---

## Timeline Checkpoint

**End of Day 2**: You should have clean data and verified ground truth  
**End of Day 4**: All retrievers working, tested on 10 sample queries  
**End of Day 5**: Full results, comparison charts, metrics computed  
**End of Day 7**: 50+ annotated failures, error taxonomy validated  
**End of Day 8**: Ablation complete, README polished, ready to share

**If you're behind**: Cut scope (50 queries instead of 150), but don't skip error analysis—that's the research contribution.

---

## Questions to Answer in Your Writeup

1. **What did you find?** (Results summary)
2. **Why is this surprising/useful?** (Significance)
3. **How confident are you?** (Limitations, error bars)
4. **What should practitioners do?** (Recommendations)
5. **What's next?** (Future work)

---

**Good luck! You're building something genuinely interesting. 🚀**
