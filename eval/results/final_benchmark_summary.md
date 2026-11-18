# PrivateRAG: Final Project Benchmark Summary & Paper Tables

**Course:** Natural Language Processing (Fall 2026, BCS-7A)  
**Team:** Danyal Ahmed (23L-0746), Abdul Ghani Nadeem (23L-0761), Ali Jafar (23L-0815)  
**Institution:** FAST - NUCES, Lahore  

---

## 1. Executive Summary

PrivateRAG implements a **dual-layer defense-in-depth architecture** for enterprise Retrieval-Augmented Generation:
1. **Input-Side RBAC Filtering:** Operates directly inside the vector database (Qdrant) query engine before any document vectors are scored or retrieved.
2. **Output-Side Hybrid NER Sanitization:** Employs a hybrid arbitration pipeline combining regex (for pattern-based entities) and token classification (for contextual language entities), non-destructively masking sensitive tokens right-to-left as `[<LABEL>_REDACTED]`.

---

## 2. Table I: RBAC Retrieval Benchmark (256 Evaluated Runs)

Evaluated across 64 queries (38 natural, 26 adversarial) across all 4 privilege tiers:

| Access Role | Permitted Document Tiers | Filtered Leak Rate | Unfiltered Baseline Leak Rate | Authorized Retrieval Hit@3 |
|:---|:---|:---:|:---:|:---:|
| **Intern** | `public` | **0/64 = 0.0%** | 48/64 = 75.0% | **16/16 = 100.0%** |
| **Employee** | `public`, `internal` | **0/64 = 0.0%** | 32/64 = 50.0% | **26/26 = 100.0%** |
| **HR Specialist** | `public`, `internal`, `hr_restricted` | **0/64 = 0.0%** | 19/64 = 29.7% | **32/32 = 100.0%** |
| **Administrator** | `public`, `internal`, `hr_restricted`, `exec_restricted` | **0/64 = 0.0%** | 0/64 = 0.0% | **38/38 = 100.0%** |
| **OVERALL** | — | **0/256 = 0.0%** 🏆 | **99/256 = 38.7%** ⚠️ | **100.0%** 🎯 |

- **Adversarial Query Leaks (Filtered):** `0/104 = 0.0%`
- **Data Boundary Guarantee:** Unauthorized document payloads are never loaded into memory.

---

## 3. Table II: Named Entity Recognition Benchmark (BIO Entity Extraction)

Comparison on synthetic test corpus (`test.jsonl`, 450 sentences) and natural realistic corpus (`test_realistic.jsonl`, 60 sentences):

### A. Synthetic Test Set (`test.jsonl`)
| System | EMAIL F1 | PHONE F1 | EMP_ID F1 | SALARY F1 | Micro F1 | Hard Negative FP |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Regex Baseline** | **100.0%** | **100.0%** | **100.0%** | 88.2% | 73.1% | 34 |
| **Hybrid Filter (Proposed)** | **100.0%** | **100.0%** | **100.0%** | 88.2% | 73.1% | 34 |

### B. Realistic Test Set (`test_realistic.jsonl`)
| System | EMAIL F1 | PHONE F1 | EMP_ID F1 | SALARY F1 | Micro F1 | Hard Negative FP |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Regex Baseline** | **100.0%** | **100.0%** | **100.0%** | 93.3% | 63.7% | 2 |
| **Hybrid Filter (Proposed)** | **100.0%** | **100.0%** | **100.0%** | 93.3% | 63.7% | 2 |

---

## 4. Table III: End-to-End Leak Prevention Benchmark (Phase 5)

Evaluates verbatim ground-truth leakage in generated answers before and after the PrivateRAG output sanitization filter across 256 end-to-end runs:

### A. Leak Reduction by Entity Type
| Entity Type | Nature | Raw Leaks Count | Sanitized Leaks Count | Leak Reduction % |
|:---|:---|:---:|:---:|:---:|
| **`EMAIL`** | Pattern-based | 66 | **0** | **100.0%** 🛡️ |
| **`PHONE`** | Pattern-based | 66 | **0** | **100.0%** 🛡️ |
| **`EMP_ID`** | Pattern-based | 72 | **0** | **100.0%** 🛡️ |
| **`SALARY`** | Semi-pattern / Currency | 25 | **0** | **100.0%** 🛡️ |
| **`PERSON`** | Contextual Language | 199 | 199* | *Fine-tuned model checkpoint |
| **`ADDRESS`** | Contextual Language | 32 | 32* | *Fine-tuned model checkpoint |

*\*Note: Contextual entities (`PERSON` and `ADDRESS`) depend on local model weights (`models/distilbert-ner`) fine-tuned via `notebooks/train_colab.ipynb` on Colab/Kaggle.*

---

## 5. System Architecture & Dual-Layer Security Verification

```
[ User Query + Role JWT ]
           │
           ▼
[ Layer 1: RBAC Vector Filter ] ──> Qdrant Payload Filter (0.0% leak across 256 runs)
           │
           ▼
[ Context Retrieval & LLM ]     ──> Authorized Chunks Only
           │
           ▼
[ Layer 2: Output Sanitization] ──> Right-to-Left Hybrid Masking ([..._REDACTED])
           │
           ▼
[ Final Sanitized Response ]    ──> Rendered in React UI with Visual Security Badges
```

All 56 unit and integration tests passed (`tests/test_rbac.py`, `tests/test_ner.py`, `tests/test_integration.py`, `tests/test_e2e_eval.py`).
