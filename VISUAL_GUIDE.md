# Test Suite - Visual Quick Reference

## 🎯 What is This?

A complete framework for evaluating how well your RAG system retrieves relevant documents.

```
User Query
    ↓
[Baseline Retrieval] ←→ [HyDE Retrieval]
    ↓                        ↓
Get Results          Get Results
    ↓                        ↓
Calculate Metrics    Calculate Metrics
    ↓                        ↓
  Compare Performance & Show Improvement
```

## 📍 Where to Start

```
START HERE
    ↓
┌─────────────────────────────────┐
│  TEST_SUITE_README.md           │ ← First-time users
│  (this directory)               │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│  GETTING_STARTED.md             │ ← 5-minute setup
│  tests/evaluation/              │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│  Populate queries.json          │ ← Add your queries
│  tests/evaluation/ground_truth/ │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│  Run pytest                     │ ← Execute tests
│  tests/evaluation/...           │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│  Analyze results                │ ← View report
│  analyze_report.py              │
└─────────────────────────────────┘
```

## 🚀 3-Step Quick Start

### Step 1: View Available Data
```bash
python -m tests.evaluation.validate_ground_truth --list-documents
```
Shows what elicitations you can use in ground truth.

### Step 2: Add Test Queries
Edit: `tests/evaluation/ground_truth/queries.json`

Add queries like:
```json
{
  "query_id": "Q001",
  "query_text": "Your question here?",
  "relevant_elicitations": ["doc_id_1", "doc_id_2"]
}
```

### Step 3: Run Tests
```bash
pytest tests/evaluation/test_retrieval_ablation.py::test_generate_final_report -v
```

## 📊 Output Example

```
Baseline Q001: P@1=1.00, AP=1.00
HyDE     Q001: P@1=1.00, AP=1.00
         ↓
    ============================================================
    ABLATION STUDY RESULTS
    ============================================================
    P@1    | Baseline: 0.750 | HyDE: 0.850 | Improvement: +13.3%
    ...
    MAP    | Baseline: 0.670 | HyDE: 0.760 | Improvement: +13.4%
```

## 📁 Key Files

```
tests/evaluation/
├── test_retrieval_ablation.py    ← Main test (RUN THIS)
├── ground_truth/
│   ├── queries.json              ← EDIT THIS (add your queries)
│   └── loader.py                 ← Loads & validates
├── metrics/
│   ├── precision_recall.py       ← Calculates P@K, R@K
│   └── mean_average_precision.py ← Calculates MAP
├── README.md                     ← Full documentation
└── GETTING_STARTED.md            ← Quick start guide
```

## ✅ What Metrics You Get

| Metric | What It Means |
|--------|---------------|
| **P@K** | % of top-K results that are relevant |
| **R@K** | % of relevant results found in top-K |
| **MAP** | Overall quality of ranking (0-1) |

Higher scores = better retrieval!

## 🔄 Typical Workflow

```
1. Understand Your Data
   └─ What documents do you have?
      Run: validate_ground_truth --list-documents

2. Create Test Cases
   └─ Write test queries
      Edit: ground_truth/queries.json

3. Run Evaluation
   └─ Compare baseline vs HyDE
      Run: pytest test_retrieval_ablation.py

4. Analyze Results
   └─ Review metrics & patterns
      Run: analyze_report.py
```

## 💡 Key Concepts

### Ground Truth
Your manually-curated list of:
- Test queries (what users ask)
- Relevant documents (what should be retrieved)

### Baseline Retrieval
Raw query → embedding → search results

### HyDE Retrieval
Query → generate hypothetical answer → embedding → search results

### Metrics
Numbers that measure retrieval quality:
- Precision: accuracy of results
- Recall: completeness of results  
- Average Precision: overall ranking quality

## 🎯 Success Indicators

```
GOOD ✅
├─ HyDE MAP > Baseline MAP
├─ Improvement > 10%
├─ P@1 improvement significant
└─ Fewer HyDE failures than baseline

NEEDS WORK ⚠️
├─ HyDE MAP < Baseline MAP
├─ Improvement < 5%
├─ Inconsistent performance
└─ HyDE creates new failures
```

## 🛠️ Customization

### Add Custom Query
1. Open `tests/evaluation/ground_truth/queries.json`
2. Add new entry with query_id, query_text, relevant_elicitations
3. Run validation: `python -m tests.evaluation.validate_ground_truth --verbose`
4. Re-run tests

### Modify HyDE Prompt
1. Edit `services/rag_service.py`
2. Find `generate_hypothetical_document()` method
3. Update French prompt
4. Re-run tests

### Change Retrieval K
1. Edit `tests/evaluation/test_retrieval_ablation.py`
2. Find `k=5` in `_retrieve_baseline()` and `_retrieve_with_hyde()`
3. Change to desired value
4. Re-run tests

## 📚 Documentation

- **Quick Start** (5 min): `GETTING_STARTED.md`
- **Full Guide** (15 min): `README.md`
- **Ground Truth** (10 min): `ground_truth/README.md`
- **Architecture** (10 min): `ARCHITECTURE.md`
- **Navigation** (5 min): `INDEX.md`

## 🐛 Common Issues & Fixes

| Issue | Solution |
|-------|----------|
| No documents in vector store | Run `main.py` to load documents |
| Ground truth validation fails | Check JSON syntax, run with `--verbose` |
| Tests can't find backend | Ensure backend is initialized |
| Report not generated | Check pytest output for errors |

## ✨ Features

✅ No frontend required (direct backend integration)
✅ Automated report generation
✅ Standard IR metrics included
✅ HyDE ablation study
✅ Per-query breakdown
✅ Error analysis
✅ Easy to extend
✅ Comprehensive documentation

## 🎓 Learning Path

```
Beginner
  ├─ Read: GETTING_STARTED.md
  ├─ Do: Edit queries.json
  ├─ Do: Run pytest
  └─ Do: Analyze results
       ↓
Intermediate
  ├─ Read: README.md (full guide)
  ├─ Understand: Metrics (P@K, R@K, MAP)
  ├─ Understand: Ablation study
  └─ Do: Refine ground truth
       ↓
Advanced
  ├─ Read: ARCHITECTURE.md
  ├─ Modify: HyDE prompt
  ├─ Add: Custom metrics
  ├─ Extend: Custom retrieval
  └─ Do: Full pipeline optimization
```

## 🚀 Next Steps

1. **Right now**
   ```bash
   python -m tests.evaluation.validate_ground_truth --example
   ```

2. **Next (5 min)**
   Read: `tests/evaluation/GETTING_STARTED.md`

3. **Then (10 min)**
   Edit: `tests/evaluation/ground_truth/queries.json`

4. **Finally (10 min)**
   Run: `pytest tests/evaluation/test_retrieval_ablation.py -v`

---

**Ready to evaluate your RAG system?** Let's go! 🚀
