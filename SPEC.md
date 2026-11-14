# PrivateRAG: Project Spec

NLP term project (Fall 2026, BCS-7A). A RAG document assistant with two NLP/security layers:
1. Input side: role-based access control (RBAC) applied BEFORE vector retrieval.
2. Output side: a fine-tuned token-classification model (plus regex) that masks sensitive entities in generated answers.

The core NLP contribution is the fine-tuned entity model and its evaluation. Access control is the system layer around it.

## Stack (fixed, do not substitute without asking)
- Backend: Python 3.11, FastAPI
- Vector DB: Qdrant (local, Docker or in-memory mode), RBAC enforced via payload filter on `access_level`
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`
- Generator LLM: behind a single interface `llm.generate(prompt) -> str`, provider set by env var (default: Gemini free tier; fallback: a small local HF model). Never hardcode keys.
- NER model: `distilbert-base-cased` fine-tuned for token classification (HuggingFace Transformers, `seqeval` for metrics)
- Frontend: React + Vite (simple chat UI with a role switcher dropdown)
- Training runs on Colab/Kaggle. The repo only contains the training script and a notebook; do not train on the local machine.

## Roles and access levels
Roles (ordered): `intern` < `employee` < `hr` < `admin`
Document access levels: `public`, `internal`, `hr_restricted`, `exec_restricted`

| Role     | public | internal | hr_restricted | exec_restricted |
|----------|--------|----------|---------------|-----------------|
| intern   | yes    | no       | no            | no              |
| employee | yes    | yes      | no            | no              |
| hr       | yes    | yes      | yes           | no              |
| admin    | yes    | yes      | yes           | yes             |

## Entity labels (BIO scheme)
`PERSON`, `EMAIL`, `PHONE`, `EMP_ID`, `ADDRESS`, `SALARY`
Mask tokens: `[PERSON_REDACTED]`, `[EMAIL_REDACTED]`, `[PHONE_REDACTED]`, `[EMP_ID_REDACTED]`, `[ADDRESS_REDACTED]`, `[SALARY_REDACTED]`

Why ML on top of regex: EMAIL, PHONE and EMP_ID are pattern-based and regex handles them well. PERSON, ADDRESS and SALARY depend on context, and that is where the fine-tuned model must show a gain. The evaluation must report per-entity results to prove this.

## Repo structure
```
privaterag/
  SPEC.md
  data/
    documents/          # synthetic company docs (txt/md) + metadata.json
    ner/                # train.jsonl, val.jsonl, test.jsonl (BIO tagged)
  src/
    data_gen/           # synthetic docs + NER dataset generators (Faker + templates)
    retrieval/          # ingestion, Qdrant client, RBAC filter
    ner/                # train.py, infer.py, regex_filter.py, hybrid_filter.py
    llm/                # llm interface + provider adapters
    api/                # FastAPI app (/login-role, /ask, /health)
  eval/
    ner_eval.py         # entity-level P/R/F1 for 3 systems
    rbac_eval.py        # leak-rate test, role x query matrix
    e2e_eval.py         # end-to-end leak test on generated answers
    results/            # json + csv outputs
  frontend/
  notebooks/train_colab.ipynb
  README.md
```

## Phases
Each phase ends with a short report and waits for approval before the next one starts.

1. **Data**: synthetic documents (about 120 across the 4 levels, with realistic HR/payroll/exec content containing the 6 entity types), `metadata.json` with `access_level` per doc, and a BIO-tagged NER dataset (about 3000 sentences). Split 70/15/15 by TEMPLATE, not by random sentence, so test templates are unseen (avoids leakage).
2. **Retrieval + RBAC**: ingestion into Qdrant with `access_level` payload, retrieval function that takes a role and applies the filter, `/ask` endpoint returning answer plus retrieved doc ids.
3. **NER**: regex baseline, pretrained baseline (`dslim/bert-base-NER`, mapped to our labels where possible), fine-tuning script, hybrid filter (model + regex, union of spans), masking function.
4. **Integration + UI**: wire the masking filter after generation, React UI with role switcher, show retrieved docs and masked answer.
5. **Evaluation**: run all evals, export tables and a results summary for the report.

## Evaluation (what the report needs)
- NER: entity-level precision/recall/F1 (seqeval), overall and per entity, for: regex-only, pretrained NER, fine-tuned, hybrid (fine-tuned + regex).
- RBAC: leak rate = fraction of (role, query) pairs where a document above the role's level was retrieved. Target 0 percent. Include adversarial queries that directly name restricted content.
- End-to-end: fraction of generated answers that still contain a ground-truth sensitive entity after masking, with and without the filter.

## Working rules for the coding agent
- Work only inside the `privaterag/` folder. Do not touch anything outside it.
- No git commit, no push, no installing global packages. Use a virtualenv.
- Before any phase: read-only investigation and a written plan. Wait for approval before writing code.
- Never invent results. All numbers in `eval/results/` must come from actually running the eval scripts.
- Keep secrets in `.env`, add `.env` to `.gitignore`.
- Prefer complete, runnable code with a short README section on how to run each phase.
