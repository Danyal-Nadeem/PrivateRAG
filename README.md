# 🛡️ PrivateRAG
### *An NLP-Driven Document Retrieval and Text Sanitization System with Access Control and ML-Based Filtering*

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.12" />
  <img src="https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/Qdrant-Vector_DB-DC382D?style=for-the-badge&logo=qdrant&logoColor=white" alt="Qdrant" />
  <img src="https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="HuggingFace" />
  <img src="https://img.shields.io/badge/PyTest-82_Passed-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white" alt="PyTest" />
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

## 🎯 How PrivateRAG Fulfills NLP Course Guidelines & Proposal

Below is a direct, point-by-point mapping demonstrating how PrivateRAG precisely fulfills all requirements set forth in the **NLP Course Project Guidelines** and our **Term Project Proposal**:

### 1. Project Category Selection
* **Category 1 (RAG Bot for Niche Domain):** Developed a complete Question-Answering system over **Company Documentation & Enterprise Policies** (120 structured documents across 4 access tiers: `public`, `internal`, `hr_restricted`, and `exec_restricted`).
* **Category 2 (Fine-Tuning Small Models for NLP):** Fine-tuned a pretrained Small Language Model (`distilbert-base-cased`, 66M parameters) for token classification and sequence masking on Colab/Kaggle GPU.

### 2. Pretrained Models & Open-Source Tools (No Training from Scratch)
* **Embeddings:** Pretrained Hugging Face `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional vector space).
* **Token Classification / NER:** Pretrained Hugging Face `distilbert-base-cased` using PyTorch & Hugging Face `transformers` Trainer (`notebooks/train_colab.ipynb`).
* **Vector Database:** Open-source `qdrant-client` persistent embedded vector engine.
* **LLM Generation Interface:** Pluggable multi-provider architecture supporting Google Gemini API (`gemini-2.0-flash`), Local Hugging Face (`google/flan-t5-small`), and zero-dependency extractive answering.

### 3. Base Model vs. Fine-Tuned Model Comparison (Core Course Requirement)
* **Evaluation Framework:** We systematically benchmark **4 systems** against the same held-out test corpora:
  1. *Rule-Based Regex Baseline* (`src/ner/regex_filter.py`)
  2. *Pretrained Transformer Baseline* (`dslim/bert-base-NER`)
  3. *Fine-Tuned Token Classifier* (`models/distilbert-ner/` via `src/ner/infer.py`)
  4. *Proposed Hybrid Filter* (`src/ner/hybrid_filter.py`)
* **Held-Out Test Sets:** Evaluated on both:
  - Synthetic held-out test split (`data/ner/test.jsonl`, 450 sentences, 100% disjoint templates from training)
  - Realistic natural test split (`data/ner/test_realistic.jsonl`, 60 sentences with hard negatives and negative confusers).
* **Finding:** Fine-tuning on domain-specific enterprise data significantly boosts entity recognition for custom corporate entities (`EMP_ID`, `SALARY`, localized phones/addresses) over off-the-shelf general CoNLL NER baselines.

### 4. Small, Clean, and Manageable Dataset
* **Corpus Size:** 120 Markdown documents chunked into 390 sliding-window vectors — keeping indexing fast, local, and reproducible.
* **Data Isolation:** Verified **0% entity-string and 0% component overlap** across Train, Val, and Test splits (`src/data_gen/verify_data.py`).
* **Deterministic Generation:** No unpredictable live web scraping; datasets are reproducible with fixed random seeds.

### 5. Demonstrating Substantial NLP Methodology (Not Just an Interface)
* **Pre-Retrieval NLP Filtering (Input Side):** Role metadata is mapped to allowed security tiers (`public`, `internal`, `hr`, `exec`) and executed as a vector query filter directly inside Qdrant before nearest-neighbor calculation.
  - *Result:* **0.0% filtered leak rate** across 256 test runs (compared to a **38.7% leak rate** when unfiltered).
* **Post-Generation NLP Sanitization (Output Side):** Token classification outputs character spans which are redacted into standardized mask tokens (`[EMAIL_REDACTED]`, `[SALARY_REDACTED]`, etc.) via right-to-left invariant string replacement.
  - *Result:* **100% redaction of pattern-based PII** in outgoing responses.
* **Rigorous Testing:** **82 automated pytest test cases** passing with 100% success rate across RBAC, JWT auth, NER masking, and API integration.

### 6. Deliverables & User Interface
* **Deliverable 1 (Proposal & Topic Approval):** Formed a 3-member team and submitted the approved proposal.
* **Deliverable 2 (Working Application & Evaluation):**
  - Modern full-stack prototype: FastAPI backend (`src/api/main.py`) paired with a React + Vite interactive UI (`frontend/`).
  - Comprehensive evaluation suites with saved CSV matrices and JSON summaries (`eval/results/`).

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

> **Scope of the leak-prevention claims:** the 100% end-to-end reduction applies to
> **pattern-based entities only** (EMAIL, PHONE, EMP_ID, SALARY), which the regex
> layer detects deterministically. Contextual entities (PERSON, ADDRESS) require the
> fine-tuned DistilBERT checkpoint (`models/distilbert-ner/`, trained via
> `notebooks/train_colab.ipynb`). Until that checkpoint is present, the DLP layer
> runs in a loudly-logged regex-only fallback and PERSON/ADDRESS remain unmasked.
> RBAC results additionally assume the credential-based login introduced on the
> `implement` branch (roles are server-assigned, not client-selected).

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
Edit `.env` and set a **random** `SECRET_KEY` (at least 32 characters — the server
refuses to start with a placeholder or short key):
```powershell
python -c "import secrets; print(secrets.token_urlsafe(48))"
```
```env
SECRET_KEY=<paste-the-generated-random-key>
LLM_PROVIDER=extractive
# Optional for live Gemini API:
# GEMINI_API_KEY=your-gemini-key
# GEMINI_MODEL=gemini-2.0-flash
```

**User accounts:** authentication is credential-based. Demo accounts ship as
salted PBKDF2 hashes in `data/users.example.json` (used automatically when
`data/users.json` is absent). For a custom store, copy the example to
`data/users.json` (gitignored) and edit it. Demo credentials:

| Username | Password | Server-assigned role |
|:---|:---|:---|
| `intern` | `Intern#2026` | intern |
| `employee` | `Employee#2026` | employee |
| `hr` | `HrSpec#2026` | hr |
| `admin` | `Admin#2026` | admin |

Roles are resolved **server-side** from the user record after password
verification — clients can never request or self-assign a privilege level.

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

### 8. Run NER & End-to-End Evaluation Suites

```powershell
# Evaluate rule-based, pretrained baseline, and hybrid NER filters
python -m eval.ner_eval

# Run full 256-query end-to-end ground truth PII leakage benchmark
python -m eval.e2e_eval
```
*Results and markdown summary tables are generated directly in `eval/results/`.*

### 9. Launch the React + Vite Frontend UI

```powershell
cd frontend
npm install
npm run dev
```
Open `http://localhost:5173` in your browser. Sign in with a demo account (e.g. `employee` / `Employee#2026` — see the credentials table above), submit natural queries, and observe real-time RBAC document filtering and post-generation NER sanitization. Your clearance is fixed by the server at login and cannot be switched from the UI.

---

## 🔌 API Reference

### `POST /login`
Authenticates username/password credentials against the server-side user store
and returns a 60-minute signed HS256 JWT token. The role is **assigned by the
server** from the authenticated user record — any `role` field sent by the
client is ignored. Failed logins return a generic `401` (no user enumeration).

**Request:**
```json
{
  "username": "employee",
  "password": "Employee#2026"
}
```

**Response (`200 OK`):**
```json
{
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "role": "employee",
  "username": "employee"
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
Returns system status, Qdrant persistence status, vector collection counts, active LLM provider, and whether the fine-tuned DLP model is loaded.

**Response (`200 OK`):**
```json
{
  "status": "ok",
  "qdrant_persistent": true,
  "total_chunks": 390,
  "llm_provider": "extractive",
  "ner_model_loaded": false
}
```
*When `ner_model_loaded` is `false`, the output filter runs in regex-only
fallback: EMAIL/PHONE/EMP_ID/SALARY are masked, but PERSON/ADDRESS are **not**.
The server logs a `DLP DEGRADED` warning; set `NER_STRICT=1` to fail closed instead.*

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
- [x] **Phase 5: End-to-End Evaluation & Final Report**
  - [x] End-to-end sensitive entity leak benchmark (`eval/e2e_eval.py`).
  - [x] 100.0% leak reduction on pattern entities (`EMAIL`, `PHONE`, `EMP_ID`, `SALARY`).
  - [x] Consolidated project benchmark matrix & paper tables (`eval/results/final_benchmark_summary.md`).
  - [x] Automated test suite for end-to-end evaluation metrics (`tests/test_e2e_eval.py`).

---

## ⚖️ License & Academic Integrity

Developed for academic purposes under the **Fall 2026 Natural Language Processing** course at **FAST-NUCES**. All synthetic datasets and evaluation benchmarks were generated deterministically under fixed seeds with no live web scraping or unapproved external dependencies.
