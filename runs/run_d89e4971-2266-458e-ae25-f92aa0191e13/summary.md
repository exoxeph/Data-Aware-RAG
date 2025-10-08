# Evaluation Report: qa_small

## Metrics

| Metric | Value |
|--------|-------|
| Accept@1 | 30.00% |
| Avg Score | 0.694 |
| Repair Rate | 70.00% |
| Total Queries | 20 |
| Avg Retrieved | 8.0 |
| Avg Context Chars | 587 |
| Avg Pruned Chars | 689 |

## Latency

| Statistic | Value (ms) |
|-----------|------------|
| P50 | 0.0 |
| P95 | 1.0 |
| Mean | 0.2 |
| Max | 1.0 |

## Top 3 Worst Cases (by verify score)

### 1. Query ID: q4

**Query:** How do I fine-tune a pretrained model?

**Intent:** how_to

**Verify Score:** 0.655

**Accepted:** False

**Repair Used:** True

**Path:** retrieve → rerank → prune → contextualize → generate → verify → repair

**Answer:** Based on the provided context, this is a generated response that addresses your question using the available information.

### 2. Query ID: q7

**Query:** Explain the role of attention mechanisms

**Intent:** explanation

**Verify Score:** 0.655

**Accepted:** False

**Repair Used:** True

**Path:** retrieve → rerank → prune → contextualize → generate → verify → repair

**Answer:** Based on the provided context, this is a generated response that addresses your question using the available information.

### 3. Query ID: q9

**Query:** How does gradient descent work?

**Intent:** explanation

**Verify Score:** 0.655

**Accepted:** False

**Repair Used:** True

**Path:** retrieve → rerank → prune → contextualize → generate → verify → repair

**Answer:** Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed.

## Top 3 Best Cases (by verify score)

### 1. Query ID: q1

**Query:** What is transfer learning?

**Intent:** definition

**Verify Score:** 0.820

**Accepted:** True

**Path:** retrieve → rerank → prune → contextualize → generate → verify

### 2. Query ID: q6

**Query:** What is backpropagation?

**Intent:** definition

**Verify Score:** 0.770

**Accepted:** True

**Path:** retrieve → rerank → prune → contextualize → generate → verify

### 3. Query ID: q11

**Query:** Compare supervised and unsupervised learning

**Intent:** comparison

**Verify Score:** 0.745

**Accepted:** True

**Path:** retrieve → rerank → prune → contextualize → generate → verify

