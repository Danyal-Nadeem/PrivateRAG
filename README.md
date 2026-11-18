# 🛡️ PrivateRAG
### *An NLP-Driven Document Retrieval and Text Sanitization System with Access Control and ML-Based Filtering*

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.13" />
  <img src="https://img.shields.io/badge/FastAPI-0.162-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/Qdrant-Vector_DB-DC382D?style=for-the-badge&logo=qdrant&logoColor=white" alt="Qdrant" />
  <img src="https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="HuggingFace" />
  <img src="https://img.shields.io/badge/PyTest-31_Passed-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white" alt="PyTest" />
  <img src="https://img.shields.io/badge/RBAC_Leak_Rate-0.0%25-brightgreen?style=for-the-badge" alt="0% Leak Rate" />
</p>

---

## 👥 Course & Team Information

- **Course:** Natural Language Processing (Fall 2026) | Section **BCS-7A**
- **Institution:** FAST - National University of Computer and Emerging Sciences (NUCES), Lahore
- **Team Members:**
  - **Muhammad Danyal** (`23L-0746`)
  - **Abdul Ghani Nadeem** (`23L-0761`)
  - **Ali Jafar** (`23L-0815`)

---

## 📖 Executive Summary & Problem Statement

Standard **Retrieval-Augmented Generation (RAG)** systems index enterprise documents into a shared vector space and perform nearest-neighbor search without contextual permission checks. This introduces severe security vulnerabilities:
1. **Unauthorized Retrieval (Input Risk):** A low-privilege user (such as an intern) can craft prompts that pull restricted executive salaries, strategic M&A memos, or confidential HR records.
2. **Accidental PII Exposure (Output Risk):** Even authorized document chunks often leak personal data (cell numbers, home addresses, employee badge IDs) in model answers.

### 💡 The PrivateRAG Dual-Defense Solution
PrivateRAG implements a **two-stage NLP & Security architecture**:
- **Layer 1 (Pre-Retrieval Input Control):** Role-Based Access Control (RBAC) enforced directly **inside the Qdrant vector query** via payload filters *before* retrieval occurs. Unauthorized chunks never touch the LLM context.
- **Layer 2 (Post-Generation Output Sanitization):** A hybrid token-classification engine (fine-tuned `distilbert-base-cased` + regex) that intercepts generated text and automatically redacts sensitive entities into standardized tokens (e.g. `[EMAIL_REDACTED]`, `[SALARY_REDACTED]`).

```
                              ┌──────────────────────────────────────────────┐
                              │            User Prompt + JWT Role            │
                              └──────────────────────┬───────────────────────┘
                                                     │
                                                     ▼
                                      [ Stage 1: RBAC Vector Query ]
                                      Qdrant Payload Filter on Role
                                      (Zero unauthorized chunks retrieved)
                                                     │
                                                     ▼
                                      [ Stage 2: Contextual Generation ]
                                      LLM: Extractive / Gemini / Local HF
                                                     │
                                                     ▼
                                      [ Stage 3: Output Sanitization ]
                                      Hybrid NER (DistilBERT + Regex)
                                                     │
                                                     ▼
                              ┌──────────────────────────────────────────────┐
                              │    Sanitized Answer ([..._REDACTED] PII)     │
                              └──────────────────────────────────────────────┘
```

---

## 📊 Proven Benchmark Results (Phase 2 Evaluation)

Evaluated across **256 distinct runs** (64 realistic test queries × 4 access roles, top-$k = 3$):

| Role | Allowed Document Tiers | Leak Rate (WITH RBAC Filter) | Baseline Leak Rate (WITHOUT Filter) | Hit@3 Retrieval (Authorized) |
|:---|:---|:---:|:---:|:---:|
| **Intern** | `public` | **0/64 = 0.0%** | 48/64 = 75.0% | **16/16 = 100.0%** |
| **Employee** | `public`, `internal` | **0/64 = 0.0%** | 32/64 = 50.0% | **26/26 = 100.0%** |
| **HR Specialist** | `public`, `internal`, `hr_restricted` | **0/64 = 0.0%** | 19/64 = 29.7% | **32/32 = 100.0%** |
| **Administrator** | `public`, `internal`, `hr_restricted`, `exec_restricted` | **0/64 = 0.0%** | 0/64 = 0.0% | **38/38 = 100.0%** |
| **OVERALL** | — | **0/256 = 0.0%** 🏆 | **99/256 = 38.7%** ⚠️ | **100.0%** 🎯 |

- **Adversarial Query Leak Rate (Filtered):** `0/104 = 0.0%` (Zero leaks even when queries explicitly demand confidential salaries or addresses).
- **Unit Test Coverage:** `31 passed in 14.89s` (`tests/test_rbac.py`).
- **Data Integrity:** 100% verified zero entity-overlap across splits (`src/data_gen/verify_data.py`).

---

## 🔐 Access Hierarchy & Entity Redaction

### Role-to-Access Matrix

| Role | `public` | `internal` | `hr_restricted` | `exec_restricted` |
|:---|:---:|:---:|:---:|:---:|
| **intern** | ✅ | ❌ | ❌ | ❌ |
| **employee** | ✅ | ✅ | ❌ | ❌ |
| **hr** | ✅ | ✅ | ✅ | ❌ |
| **admin** | ✅ | ✅ | ✅ | ✅ |

### Entity Redaction Specification (BIO Tagged)

| Entity Type | Example Value | Redacted Token Output | Primary Detector |
|:---|:---|:---|:---|
| **`PERSON`** | *"Dr. Azhar Memon"*, *"Julia Gibson"* | `[PERSON_REDACTED]` | Fine-tuned DistilBERT |
| **`ADDRESS`** | *"7353 Cypress Boulevard, NM 87101"* | `[ADDRESS_REDACTED]` | Fine-tuned DistilBERT |
| **`SALARY`** | *"$180,000"*, *"PKR 350,000"* | `[SALARY_REDACTED]` | Regex + DistilBERT |
| **`EMAIL`** | *"dylan.80104@enterprise.org"* | `[EMAIL_REDACTED]` | Rule-Based Regex |
| **`PHONE`** | *"(217) 289-7912"*, *"+92-42-37664711"* | `[PHONE_REDACTED]` | Rule-Based Regex |
| **`EMP_ID`** | *"EMP-89012"*, *"E-98651"* | `[EMP_ID_REDACTED]` | Rule-Based Regex |

---

## 🛠️ Project Architecture & Tech Stack

```
privaterag/
├── data/
│   ├── documents/          # 120 Markdown docs + JSON span sidecars + metadata.json
│   ├── ner/                # train (2100), val (450), test (450), test_realistic (60)
│   ├── eval/queries.json   # 64 benchmark queries (38 normal, 26 adversarial)
│   └── README.md           # Dataset generation & isolation specifications
├── src/
│   ├── data_gen/           # Synthetic data generation, tokenizers & verification
│   ├── retrieval/          # Qdrant client, chunking, all-MiniLM-L6-v2 embedder, RBAC
│   ├── llm/                # Extractive (default), Gemini 2.0, Local HF adapters
│   ├── api/                # FastAPI application, HS256 JWT auth, /ask endpoint
│   └── ner/                # [Phase 3] Regex baseline, DistilBERT training & masking
├── eval/
│   ├── rbac_eval.py        # RBAC leakage and retrieval evaluation harness
│   └── results/            # Evaluated JSON metrics & CSV matrices
├── tests/                  # Pytest test suite (RBAC, JWT, Qdrant payload consistency)
├── qdrant_storage/         # Local persistent Qdrant database (Gitignored)
├── requirements.txt        # Pinned virtual environment dependencies
└── SPEC.md                 # Formal project specifications & grading contract
```

---

## 🚀 Quick Start Guide

### 1. Environment Setup

Clone repository and activate a Python 3.13 virtual environment:

```powershell
git clone https://github.com/Danyal-Nadeem/PrivateRAG.git privaterag
cd privaterag

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Configure Environment Variables

```powershell
Copy-Item .env.example .env
```
Edit `.env` and configure your `SECRET_KEY`:
```env
SECRET_KEY=privaterag-super-secret-key-change-in-production
LLM_PROVIDER=extractive
# Optional for live Gemini API:
# GEMINI_API_KEY=your-gemini-key
# GEMINI_MODEL=gemini-2.0-flash
```

### 3. Verify Data Integrity

```powershell
python -m src.data_gen.verify_data
```
*Asserts 100% component-level disjointness between train, validation, and test sets.*

### 4. Index Documents into Qdrant

```powershell
python -m src.retrieval.ingest --recreate
```
*Parses 120 company documents, constructs 390 sliding-window chunks with title context headers, and indexes embeddings into local persistent storage (`qdrant_storage/`).*

### 5. Run the Automated Test Suite

```powershell
python -m pytest tests/ -v
```

### 6. Run RBAC Benchmark Evaluation

```powershell
python -m eval.rbac_eval
```

### 7. Launch the FastAPI Server

```powershell
python -m uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000
```
Interactive Swagger documentation is available at: `http://127.0.0.1:8000/docs`

---

## 🔌 API Reference

### `POST /login-role`
Simulates single sign-on (SSO) authentication for prototype evaluation and returns a 60-minute signed HS256 JWT token.

**Request:**
```json
{
  "role": "employee"
}
```

**Response (`200 OK`):**
```json
{
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "role": "employee"
}
```

---

### `POST /ask`
Answers questions over indexed documents with strict role-level vector filtering.

**Headers:**
```http
Authorization: Bearer <your_jwt_token>
Content-Type: application/json
```

**Request:**
```json
{
  "query": "What are the Kubernetes deployment protocols in Rev 1?"
}
```
*(Note: Any `"role"` attribute passed in the request body is strictly ignored for security; role is parsed exclusively from the signed JWT).*

**Response (`200 OK`):**
```json
{
  "query": "What are the Kubernetes deployment protocols in Rev 1?",
  "role": "employee",
  "answer": "This standard operating procedure is restricted to full-time employees...",
  "retrieved_documents": [
    {
      "doc_id": "doc_internal_001",
      "access_level": "internal",
      "chunk_index": 0,
      "title": "IT Infrastructure Deployment and Kubernetes Cluster Runbook (Rev 1)",
      "score": 0.8241
    }
  ]
}
```

---

### `GET /health`
Returns system status, Qdrant persistence status, vector collection counts, and active LLM provider.

**Response (`200 OK`):**
```json
{
  "status": "ok",
  "qdrant_persistent": true,
  "total_chunks": 390,
  "llm_provider": "extractive"
}
```

---

## 🗺️ Project Milestones & Roadmap

- [x] **Phase 1: Synthetic Data Engineering & NER Corpus**
  - [x] 120 Markdown documents across 4 access tiers (`public`, `internal`, `hr`, `exec`).
  - [x] BIO-tagged dataset with 3,000 sentences (105/22/23 disjoint templates).
  - [x] Natural realistic test set (`test_realistic.jsonl`) with hard negatives.
  - [x] Strict data isolation validation suite (`verify_data.py`).
- [x] **Phase 2: RBAC Retrieval & API Architecture**
  - [x] Local persistent Qdrant with `query_points` API.
  - [x] Query-time vector payload filtering (0.0% filtered leak rate).
  - [x] Fast, extensible LLM provider abstraction (Extractive, Gemini, Local HF).
  - [x] FastAPI backend with JWT SSO simulator and query-injection defense.
  - [x] Automated RBAC evaluation matrix (`rbac_eval.py`).
- [x] **Phase 3: Named Entity Recognition & Output Sanitization**
  - [x] Regex baseline filter (`src/ner/regex_filter.py`).
  - [x] Pretrained `dslim/bert-base-NER` baseline.
  - [x] Fine-tuning `distilbert-base-cased` on Colab/Kaggle (`train.py` & `notebooks/train_colab.ipynb`).
  - [x] Hybrid decision filter (`src/ner/hybrid_filter.py`).
  - [x] Non-destructive character-offset text masking (`src/ner/masking.py`).
  - [x] Benchmark evaluation runner (`eval/ner_eval.py`).
- [x] **Phase 4: Full-Stack Integration & React UI**
  - [x] Post-generation masking middleware wired into `/ask` (raw answers never leave server).
  - [x] CORS middleware added to FastAPI for cross-origin frontend communication.
  - [x] React + Vite modern chat UI with interactive role switcher (`frontend/`).
  - [x] Visual security tags for `[..._REDACTED]` tokens and retrieved document inspector.
- [ ] **Phase 5: End-to-End Evaluation & Final Report**
  - [ ] End-to-end sensitive entity leak benchmark (`e2e_eval.py`).
  - [ ] Final paper tables, ablation studies, and submission report.

---

## ⚖️ License & Academic Integrity

Developed for academic purposes under the **Fall 2026 Natural Language Processing** course at **FAST-NUCES**. All synthetic datasets and evaluation benchmarks were generated deterministically under fixed seeds with no live web scraping or unapproved external dependencies.
