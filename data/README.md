# PrivateRAG: Data Manifest & Specifications (Phase 1)

This directory contains the synthetic document corpus, the BIO-tagged Named Entity Recognition (NER) dataset, the evaluation queries, and the realistic out-of-distribution evaluation set designed for PrivateRAG.

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
  ner/                          # BIO-tagged NER dataset (3,000 sentences + 60 realistic test sentences)
    train.jsonl                 # 2,100 examples (105 train templates: 75 entity, 30 zero-entity hard negatives)
    val.jsonl                   # 450 examples (22 val templates: 14 entity, 8 zero-entity hard negatives)
    test.jsonl                  # 450 examples (23 test templates: 15 entity, 8 zero-entity hard negatives)
    test_realistic.jsonl        # 60 natural conversational/email evaluation sentences (unseen names/addresses)
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
   - **Train**: 105 templates (75 entity-bearing, 30 zero-entity hard negatives)
   - **Validation**: 22 templates (14 entity-bearing, 8 zero-entity hard negatives)
   - **Test**: 23 templates (15 entity-bearing, 8 zero-entity hard negatives)
   - Zero template overlap: No template string or syntactic frame in validation or test appears in train.
   - Template concentration constraint: No single zero-entity template produces more than 15% of its split's zero-entity sentences (Train max: 3.49%, Val max: 13.24%, Test max: 13.24%).
2. **Disjoint Component-Level Value Pools**:
   - **First Names & Last Names:** Partitioned into strictly disjoint sets across train, val, and test. Zero overlap across all splits.
   - **Street Names & US City+State+Zip Combos:** Strictly partitioned across splits with zero overlap.
   - **Area Names (Lahore):** Partitioned into disjoint sets (e.g., Train has Gulberg, Model Town; Val has DHA Phase 1-2, Muslim Town; Test has DHA Phase 5-6, Bahria Town). Zero overlap across all splits.
   - **Lahore Postal Codes Limitation & Exemption:** Lahore postal codes (e.g., 54810, 54792, 54600) represent municipal postal delivery circles that span multiple adjacent neighborhoods (e.g., 54810 serves Cavalry Ground, Cantt, and Askari). While neighborhood area names and full address strings are 100% disjoint, standalone Lahore postal code integers naturally overlap across postal circles and are formally exempted from component-level isolation assertions.
   - **Employee IDs & Phone Prefixes:** Numerical ranges (10000–59999 for Train, 60000–79999 for Val, 80000–99999 for Test) and telephone area codes / mobile prefixes are strictly partitioned with zero overlap.
   - **Zero String Overlap:** Asserted by `verify_data.py` for all entity types across splits (with the natural exception of recurring numerical `SALARY` values).
3. **Hard Negatives & Confusers (Zero-Entity Sentences)**:
   - Exactly **15% of all sentences** contain zero sensitive entities.
   - Varied syntactic structures featuring challenging confusers:
     - Company names containing split-specific surnames (e.g. `Bukhari Textiles`, `Smith Logistics`)
     - Office locations / branches without full addresses (e.g. `the Gulberg branch`, `the Austin facility`)
     - Order, invoice, and SKU identifiers mimicking badge codes (e.g. `ORD-88124`, `INV-10928`, `SKU-49102`)
     - Extension codes and internal routing numbers (e.g. `ext. 4821`, `routing code 84920`)
     - Non-salary budget, revenue, and invoice totals (e.g. `an operating budget of $125,000`, `invoice total of PKR 450,000`)
     - Product names, dates, and version numbers.

---

## 4. Realistic Evaluation Set (`data/ner/test_realistic.jsonl`)

Contains 60 natural-sounding sentences formatted as chatbot conversational answers and professional email lines:
- **Evaluation Only:** Must never be used during model training.
- **Composition:** 10 zero-entity hard negatives and 50 entity-bearing sentences spanning all 6 entity types.
- **Unseen Entities:** Utilizes dedicated names, streets, and areas that do not appear in `train.jsonl` or `val.jsonl`.
- **Validation:** Checked with the same tokenization, offset alignment, and zero-string-overlap assertions as standard splits.

---

## 5. Document Corpus (120 Documents)

Distributed equally across the 4 access tiers:
- **`public` (30 docs):** Public whitepapers, governance reports, and press releases. Contains **NO sensitive values** except the names of public corporate figures (`Dr. Azhar Memon`, `Julia Gibson`, etc.).
- **`internal` (30 docs):** Standard engineering runbooks, IT setup guides, facilities policies. Contains `PERSON`, `EMP_ID`, `EMAIL`, and `PHONE` of technical leads. Contains zero salary records.
- **`hr_restricted` (30 docs):** Performance reviews, payroll adjustments, compensation memos, offer letters. Contains all 6 entity types (`PERSON`, `EMP_ID`, `SALARY`, `ADDRESS`, `PHONE`, `EMAIL`).
- **`exec_restricted` (30 docs):** Board minutes, C-suite executive compensation resolutions, M&A due diligence, and residential security stipends. Contains all 6 entity types.

**Leakage Protection & Pool Isolation:**
- All document entities and public figures are strictly generated using the **TEST pool** components (and zero from train or val pools).
- Document entities (`PERSON`, `EMAIL`, `PHONE`, `EMP_ID`, `ADDRESS`) have **zero exact string overlap** with `train.jsonl`, `val.jsonl`, and `test_realistic.jsonl`.
- Each document has an accompanying sidecar `<doc_id>.json` specifying exact character-level entity offsets, and is cataloged in `metadata.json` with an explicit list of `sensitive_values`.

---

## 6. Evaluation Queries (`data/eval/queries.json`)

Contains 64 targeted queries (16 per access level):
- **Normal Queries:** Informational questions asking about procedures, technical documentation, or public governance without referencing restricted PII.
- **Adversarial Queries:** Deliberately name restricted entities or directly ask for sensitive attributes (e.g., asking for an employee's salary, phone number, or home address from an HR or executive document). Used in Phase 5 to measure RBAC leakage and redaction resilience.
