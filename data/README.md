# PrivateRAG: Data Manifest & Specifications (Phase 1)

This directory contains the synthetic document corpus, the BIO-tagged Named Entity Recognition (NER) dataset, and the evaluation queries designed for PrivateRAG.

---

## 1. Directory Structure

```
data/
  README.md                     # This documentation file
  documents/                    # 120 synthetic enterprise documents
    doc_public_001.md ...       # 30 public documents
    doc_internal_001.md ...     # 30 internal documents
    doc_hr_001.md ...           # 30 HR restricted documents
    doc_exec_001.md ...         # 30 executive restricted documents
    doc_*.json                  # Sidecar JSON with exact entity character spans per document
    metadata.json               # Manifest mapping doc_id, filename, access_level, entities, sensitive_values
  ner/                          # BIO-tagged NER dataset (3,000 sentences)
    train.jsonl                 # 2,100 examples (105 train templates, 15% zero-entity hard negatives)
    val.jsonl                   # 450 examples (22 val templates, 15% zero-entity hard negatives)
    test.jsonl                  # 450 examples (23 test templates, 15% zero-entity hard negatives)
  eval/
    queries.json                # 64 evaluation queries (normal and adversarial across 4 access tiers)
```

---

## 2. Entity Formats & Tagging Standards (BIO Scheme)

The NER model and redaction pipeline target 6 sensitive entity classes:

| Entity Label | Mask Token | Description & Format Rules | Examples |
|---|---|---|---|
| `PERSON` | `[PERSON_REDACTED]` | Full personal names (Western and Pakistani mix). **Titles (Dr., Mr., Ms., Engr., CEO, Director) are strictly excluded from the span.** | `Sarah Jenkins`, `Muhammad Usman`, `Ahmad Raza`, `John Doe` |
| `EMP_ID` | `[EMP_ID_REDACTED]` | Employee identification badges: **80% `EMP-12345`**; **20% `EMP12345` or `E-12345`**. | `EMP-48219`, `EMP58210`, `E-19402` |
| `EMAIL` | `[EMAIL_REDACTED]` | Personal and corporate email addresses. | `jane.doe@corp.acme.com`, `usman.raza@techmatrix.net` |
| `PHONE` | `[PHONE_REDACTED]` | Telephone and mobile lines in US and Pakistani (+92) formats. | `(212) 555-0192`, `+1-415-555-8921`, `+92 300 1234567`, `0301-7654321`, `+92-42-35876543` |
| `ADDRESS` | `[ADDRESS_REDACTED]` | One-line full residential or mailing address treated as **ONE single entity span**. 70% US-style, 30% Lahore-style. | `742 Evergreen Terrace, Springfield, IL 62704`, `House 12, Street 5, Gulberg III, Lahore, 54000` |
| `SALARY` | `[SALARY_REDACTED]` | **Currency symbol/code + monetary amount only.** Qualifying words like "per annum", "annually", "/yr", or "base" are **strictly outside the span**. | `$125,000`, `USD 95,000`, `PKR 350,000`, `Rs. 1,200,000`, `$95k`, `EUR 85,000` |

---

## 3. Template-Based Data Splits & Generalization Guarantees

To ensure rigorous evaluation and prevent data leakage:
1. **Disjoint Template Partitioning**:
   - Total Templates: **150**
   - **Train**: 105 templates (89 entity-bearing, 16 zero-entity)
   - **Validation**: 22 templates (19 entity-bearing, 3 zero-entity)
   - **Test**: 23 templates (20 entity-bearing, 3 zero-entity)
   - Zero template overlap: No template string or syntactic frame in validation or test appears in train.
2. **Disjoint Entity Value Pools**:
   - First names and last names are partitioned into disjoint sets (70% train, 15% val, 15% test).
   - Street names, cities, and Lahore sector names are partitioned disjointly across splits.
   - Employee ID numerical ranges are partitioned: Train (10000–59999), Val (60000–79999), Test (80000–99999).
   - Phone area codes and exchange prefixes are strictly partitioned across splits.
   - **Zero String Overlap:** Asserted by `verify_data.py` for all entity types across splits (with the natural exception of recurring numerical `SALARY` values).
3. **Hard Negatives (Zero-Entity Sentences)**:
   - Exactly **15% of all sentences** contain zero sensitive entities.
   - These sentences feature non-entity hard negatives that challenge pattern matchers:
     - Company names (`Acme Global Corp`, `Vertex Solutions`)
     - Product names (`CloudCore v3.2`, `Sentinel AI`)
     - Standalone city names (`Chicago`, `Denver`, `Lahore`)
     - Corporate revenue & budgets (`$45M in annual revenue`, `$12.8M ARR`, `$850,000 budget`)
     - Headcounts (`team of 45 engineers`, `headcount of 1,200 employees`)
     - Dates & Version numbers (`October 14, 2024`, `v2.4.1`)
     - Ticket IDs (`TKT-48213`, `BUG-1092`, `INC-9941`) mimicking employee badge patterns.

---

## 4. Document Corpus (120 Documents)

Distributed equally across the 4 access tiers:
- **`public` (30 docs):** Public whitepapers, governance reports, and press releases. Contains **NO sensitive values** except the names of public corporate figures (`Dr. Arshad Khan`, `Sarah Jenkins`, etc.).
- **`internal` (30 docs):** Standard engineering runbooks, IT setup guides, facilities policies. Contains `PERSON`, `EMP_ID`, `EMAIL`, and `PHONE` of technical leads. Contains zero salary records.
- **`hr_restricted` (30 docs):** Performance reviews, payroll adjustments, compensation memos, offer letters. Contains all 6 entity types (`PERSON`, `EMP_ID`, `SALARY`, `ADDRESS`, `PHONE`, `EMAIL`).
- **`exec_restricted` (30 docs):** Board minutes, C-suite executive compensation resolutions, M&A due diligence, and residential security stipends. Contains all 6 entity types.

Each document has an accompanying sidecar `<doc_id>.json` specifying exact character-level entity offsets, and is cataloged in `metadata.json` with an explicit list of `sensitive_values`.

---

## 5. Evaluation Queries (`data/eval/queries.json`)

Contains 64 targeted queries (16 per access level):
- **Normal Queries:** Informational questions asking about procedures, technical documentation, or public governance without referencing restricted PII.
- **Adversarial Queries:** Deliberately name restricted entities or directly ask for sensitive attributes (e.g., asking for an employee's salary, phone number, or home address from an HR or executive document). Used in Phase 5 to measure RBAC leakage and redaction resilience.
