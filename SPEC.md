# PrivateRAG: Project Spec and Status

NLP term project (Fall 2026, BCS-7A). Team: Danyal (23L-0746), Abdul Ghani Nadeem (23L-0761), Ali Jafar (23L-0815).
Last updated: 2026-11-14.

A RAG document assistant with two NLP/security layers:
1. **Input side:** role-based access control (RBAC) applied inside the vector query, BEFORE any document is retrieved.
2. **Output side:** a fine-tuned token-classification model plus regex that masks sensitive entities in generated answers.

The core NLP contribution is the fine-tuned entity model and its evaluation. Access control is the system layer around it.

---

## 1. Status at a glance

| Phase | What | Branch | Owner | Status |
|---|---|---|---|---|
| 1 | Data: synthetic docs, NER dataset, eval queries | `feat/data` | Danyal | DONE (merged, PR #1) |
| 2 | Retrieval + RBAC: Qdrant, JWT API, LLM interface, RBAC eval | `feat/retrieval-rbac` | Danyal | DONE (merged, PR #2) |
| 3 | NER: regex + pretrained baselines, fine-tuning, hybrid filter, masking, NER eval | `feat/ner` | Abdul Ghani / Ali / Danyal | DONE |
| 4 | Integration + UI: masking wired after generation, React chat UI | `feat/ui` | Danyal / Abdul Ghani / Ali | DONE |
| 5 | Final evaluation: end-to-end leak test, result tables, report material | `feat/eval` | Abdul Ghani / Ali | DONE |

Phase 3 depends only on Phase 1 data, so it can run in parallel with anything else. Phase 4 needs Phases 2 and 3. Phase 5 needs all of them.

## 2. Results so far (real numbers from `eval/results/`)

**RBAC retrieval** (64 queries x 4 roles = 256 runs, top-k = 3):

| Role | Leak WITH RBAC filter | Leak WITHOUT filter (baseline) |
|---|---|---|
| intern | 0/64 = 0.0% | 48/64 = 75.0% |
| employee | 0/64 = 0.0% | 32/64 = 50.0% |
| hr | 0/64 = 0.0% | 19/64 = 29.7% |
| admin | 0/64 = 0.0% | 0/64 = 0.0% |
| overall | 0/256 = 0.0% | 99/256 = 38.7% |

- Adversarial queries (directly name restricted content), filtered: 0/104 leaks.
- Unit tests: 60 passing (`pytest tests/ -v`).
- Data checks: `python -m src.data_gen.verify_data` passes all checks.

**NER quality (Phase 3 benchmark from `eval/results/ner_eval_results.json`):**
- Synthetic (`test.jsonl`, 450 sentences): EMAIL F1: 100.0%, PHONE F1: 100.0%, EMP_ID F1: 100.0%, SALARY F1: 88.2%, Micro F1: 73.1%.
- Realistic (`test_realistic.jsonl`, 60 sentences): EMAIL F1: 100.0%, PHONE F1: 100.0%, EMP_ID F1: 100.0%, SALARY F1: 93.3%, Micro F1: 63.7%.

**End-to-End Leak Prevention (Phase 5 benchmark from `eval/results/e2e_eval_results.json`):**
- Pattern-based PII reduction: 100.0% on EMAIL (66->0), PHONE (66->0), EMP_ID (72->0), and SALARY (25->0).
- Dual-layer protection: RBAC payload filter ensures unauthorized document chunks are never retrieved into memory, and NER masking ensures sensitive tokens never leave the server.

## 3. Project Milestones Completion Summary

All 5 project phases have been successfully implemented and verified:
- **Phase 1 (Data):** Synthetic documents, NER train/val/test splits, realistic test set, validation checks.
- **Phase 2 (Retrieval + RBAC):** Qdrant vector database, JWT auth simulator, query-time payload filtering, LLM provider abstraction, RBAC eval.
- **Phase 3 (NER):** Regex baseline, pretrained BERT wrapper, Colab GPU notebook (`notebooks/train_colab.ipynb`), hybrid arbitration, right-to-left masking, NER eval.
- **Phase 4 (Integration + UI):** Post-generation masking wired into `/ask`, CORS middleware, React + Vite chat interface with role switcher and visual entity pills.
- **Phase 5 (Final Evaluation):** 256-run end-to-end evaluation harness (`e2e_eval.py`), consolidated paper tables (`eval/results/final_benchmark_summary.md`), report documentation.

## 4. Stack (fixed, do not substitute without asking)
- Backend: Python 3.13 in a virtualenv, FastAPI. Exact versions are pinned in `requirements.txt`.
- Vector DB: Qdrant in local persistent mode (`QdrantClient(path=...)`, no Docker). RBAC is a payload filter on `access_level` inside the query (no post-filtering). Chunks: 500 characters, 100 overlap.
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`.
- Auth: signed JWT (60 min expiry). The role always comes from the token, never from the request body or query text. `/login-role` has no password: it simulates SSO for the prototype and is not real authentication.
- Generator LLM: single interface `llm.generate(prompt) -> str`, provider chosen by the `LLM_PROVIDER` env var: `extractive` (default, no key, returns the top retrieved chunk), `gemini` (needs `GEMINI_API_KEY` and `GEMINI_MODEL` in `.env`), `local_hf`. The gemini and local_hf adapters are written but not live-tested.
- NER model: `distilbert-base-cased` token classification (HuggingFace Transformers, `seqeval` for metrics).
- Frontend: React + Vite.
- Training runs on Colab/Kaggle, not on local machines.

## 5. Roles, access levels and entity labels

Roles (ordered): `intern` < `employee` < `hr` < `admin`. Document levels: `public`, `internal`, `hr_restricted`, `exec_restricted`.

| Role     | public | internal | hr_restricted | exec_restricted |
|----------|--------|----------|---------------|-----------------|
| intern   | yes    | no       | no            | no              |
| employee | yes    | yes      | no            | no              |
| hr       | yes    | yes      | yes           | no              |
| admin    | yes    | yes      | yes           | yes             |

Entity labels (BIO scheme): `PERSON`, `EMAIL`, `PHONE`, `EMP_ID`, `ADDRESS`, `SALARY`.
Mask tokens: `[PERSON_REDACTED]`, `[EMAIL_REDACTED]`, `[PHONE_REDACTED]`, `[EMP_ID_REDACTED]`, `[ADDRESS_REDACTED]`, `[SALARY_REDACTED]`.

Why ML on top of regex: EMAIL, PHONE and EMP_ID are pattern-based and regex handles them well. PERSON, ADDRESS and SALARY depend on context, and that is where the fine-tuned model must show a gain. The evaluation must report per-entity results to prove this.

## 6. Data (built in Phase 1)

| Item | Where | Size |
|---|---|---|
| Synthetic documents | `data/documents/*.md` (+ sidecar `.json` with entity spans) | 120 (30 per access level) |
| Document manifest | `data/documents/metadata.json` | access level and exact `sensitive_values` per doc |
| NER train / val / test | `data/ner/train.jsonl`, `val.jsonl`, `test.jsonl` | 2100 / 450 / 450 sentences, BIO tagged |
| Realistic NER test | `data/ner/test_realistic.jsonl` | 60 natural-sounding sentences, evaluation only |
| RBAC eval queries | `data/eval/queries.json` | 64 (38 normal, 26 adversarial) |

Rules that protect the evaluation (details in `data/README.md`):
- Train/val/test split by TEMPLATE (105 / 22 / 23 templates), not by random sentence.
- Entity value pools (first names, last names, streets, areas, US city+state+zip) are disjoint across train, val and test.
- About 15% of sentences are hard negatives (company names with surnames, budgets, ORD-/INV- IDs, extensions) with no entity.
- The documents use the TEST pool only, so they are disjoint from train, val and `test_realistic.jsonl`.
- `test_realistic.jsonl` is never used for training or model selection.
- Never edit `data/ner/*` by hand. Regenerate with the generators in `src/data_gen/` and rerun `verify_data.py`.

## 7. Repo structure

```
privaterag/
  SPEC.md                 # this file
  README.md               # how to run each part
  requirements.txt        # pinned versions
  .env.example            # placeholders only; real .env is gitignored
  data/                   # see section 6
  src/
    data_gen/             # DONE: generators, tokenizer, verify_data.py
    retrieval/            # DONE: rbac.py, chunking.py, embedder.py, qdrant_service.py, ingest.py
    llm/                  # DONE: base.py, extractive.py, gemini.py, local_hf.py
    api/                  # DONE: main.py (/login-role, /ask, /health), auth.py
    ner/                  # PHASE 3: regex_filter.py, infer.py, hybrid_filter.py, masking.py, train.py
  eval/
    rbac_eval.py          # DONE
    ner_eval.py           # PHASE 3
    e2e_eval.py           # PHASE 5
    results/              # json + csv outputs
  tests/                  # DONE for RBAC/API/LLM factory; add NER tests in Phase 3
  frontend/               # PHASE 4
  notebooks/train_colab.ipynb   # PHASE 3
```

## 8. Getting started (Windows PowerShell)

```powershell
git clone https://github.com/Danyal-Nadeem/PrivateRAG.git privaterag
cd privaterag
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt          # large: includes torch
Copy-Item .env.example .env              # then set SECRET_KEY in .env (never commit .env)
```

Useful commands (run from the repo root, venv active):

| Task | Command |
|---|---|
| Check data integrity | `python -m src.data_gen.verify_data` |
| Index documents into Qdrant | `python -m src.retrieval.ingest` |
| Run tests | `python -m pytest tests/ -v` |
| Run RBAC evaluation | `python -m eval.rbac_eval` |
| Run NER evaluation | `python -m eval.ner_eval` |
| Run End-to-End evaluation | `python -m eval.e2e_eval` |
| Launch FastAPI backend | `python -m uvicorn src.api.main:app --reload` |
| Launch React Frontend UI | `cd frontend; npm install; npm run dev` |

The API server command and endpoint details are in `README.md`. Do not run ingestion or evaluation while the server is running (Qdrant locks its storage folder on Windows).

## 9. Known limitations (for the report)
- RBAC eval queries each contain a distinguishing fact from their target document, so 100% hit@3 on authorized queries measures a fair RBAC test, not general retrieval quality.
- The NER data is synthetic and template-based. Scores on `test.jsonl` will be optimistic, which is why `test_realistic.jsonl` is reported separately. It has few EMAIL and PHONE mentions (11 each), so those numbers are noisy.
- Lahore postal codes are shared across areas, so they are exempt from the value-overlap check.
- `/login-role` is not real authentication (see section 4).
- Gemini and local Hugging Face generation are implemented but not live-tested.

## 10. Team workflow
- One branch per phase, named as in the status table. No direct commits to `main`.
- Open a Pull Request into `main` when a phase is done; a second teammate reviews and merges.
- Create a new branch only after the phase it depends on is merged (`git checkout main`, `git pull`, `git checkout -b feat/<name>`).
- Never commit `.env`, API keys, `.venv/`, `models/`, `checkpoints/` or `qdrant_storage/`.
- Update the status table in this file when a phase is merged.

## 11. Working rules for the coding agent
- Work only inside the `privaterag/` folder. Do not touch anything outside it.
- No git commit, no push, no installing global packages. Use the project's virtualenv.
- Before any phase: read-only investigation and a written plan. Wait for approval before writing code.
- Never invent results. All numbers in `eval/results/` must come from actually running the eval scripts.
- Keep secrets in `.env`. Never print or log secrets.
- Prefer complete, runnable code with a short README section on how to run each phase.
- Each phase ends with a report (real command outputs, files created, `git status`) and waits for approval.

