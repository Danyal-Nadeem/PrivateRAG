# PrivateRAG

A role-based access controlled Retrieval-Augmented Generation system.

See [SPEC.md](SPEC.md) for the full project specification.

---

## Quick Start

### 1 — Set up environment

```powershell
python -m venv .venv
.\.venv\Scripts\pip.exe install -r requirements.txt
```

### 2 — Configure secrets

```powershell
cp .env.example .env
# Edit .env — set SECRET_KEY (required), optionally set LLM_PROVIDER
```

> **Note:** `SECRET_KEY` is required to start the server. `LLM_PROVIDER` defaults to `extractive` (no model, no key needed).

### 3 — Generate data (Phase 1)

```powershell
.\.venv\Scripts\python.exe src/data_gen/generate_docs.py
.\.venv\Scripts\python.exe src/data_gen/generate_queries.py
.\.venv\Scripts\python.exe src/data_gen/verify_data.py
```

### 4 — Ingest documents into Qdrant

```powershell
.\.venv\Scripts\python.exe -m src.retrieval.ingest --recreate
```

### 5 — Run the API server

```powershell
.\.venv\Scripts\python.exe -m uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000
```

> `/login-role` accepts a role without a password to simulate SSO for prototype purposes. This is **not** real authentication — in production, replace it with your identity provider.

### 6 — Run tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -v
```

### 7 — RBAC evaluation

```powershell
.\.venv\Scripts\python.exe -m eval.rbac_eval
```

---

## Role-to-Access-Level Mapping

| Role     | Sees                                                      |
|----------|-----------------------------------------------------------|
| intern   | public                                                    |
| employee | public, internal                                          |
| hr       | public, internal, hr_restricted                           |
| admin    | public, internal, hr_restricted, exec_restricted          |

---

## LLM Providers

Set `LLM_PROVIDER` in `.env`:

| Value       | Description                                             | Packages needed      |
|-------------|--------------------------------------------------------|----------------------|
| `extractive`| Returns the top retrieved chunk. **Default.** No keys. | None (built-in)      |
| `gemini`    | Calls the Gemini API                                    | `google-generativeai`|
| `local_hf`  | Runs a small HuggingFace model locally                  | `transformers`       |

---

## Project Layout

```
privaterag/
├── data/
│   ├── documents/          # 120 markdown docs + sidecar JSONs + metadata.json
│   ├── ner/                # train/val/test/test_realistic JSONL
│   ├── eval/queries.json   # 64 evaluation queries
│   └── README.md           # Data generation details
├── src/
│   ├── data_gen/           # Phase 1: data generation scripts
│   ├── retrieval/          # Phase 2: Qdrant, chunking, embedder, RBAC
│   ├── llm/                # Phase 2: LLM provider interface
│   └── api/                # Phase 2: FastAPI application
├── tests/                  # pytest suite
├── eval/                   # rbac_eval.py + results/
├── qdrant_storage/         # Qdrant persistent storage (gitignored)
├── .env.example            # Env var template
├── requirements.txt        # Pinned dependencies
└── SPEC.md                 # Full project specification
```
