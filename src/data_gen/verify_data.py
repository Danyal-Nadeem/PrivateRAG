import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_gen.tokenizer import tokenize_with_offsets, TOKEN_REGEX
from src.data_gen.ner_templates import TRAIN_TEMPLATES, VAL_TEMPLATES, TEST_TEMPLATES

ALL_ENTITIES = ["PERSON", "EMAIL", "PHONE", "EMP_ID", "ADDRESS", "SALARY"]

def verify_ner_splits() -> Dict[str, Any]:
    print("=================================================================")
    print("STEP 1: VERIFYING NER DATASETS (train.jsonl, val.jsonl, test.jsonl)")
    print("=================================================================")
    
    ner_dir = Path("data/ner")
    split_files = {
        "train": (ner_dir / "train.jsonl", 2100, 315, 300),
        "val":   (ner_dir / "val.jsonl", 450, 68, 60),
        "test":  (ner_dir / "test.jsonl", 450, 68, 60),
    }
    
    split_data = {}
    entity_strings_per_split = {s: {e: set() for e in ALL_ENTITIES} for s in split_files}
    template_ids_per_split = {s: set() for s in split_files}
    
    for split_name, (file_path, expected_total, expected_zeros, min_mentions) in split_files.items():
        assert file_path.exists(), f"File {file_path} does not exist!"
        with open(file_path, "r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f]
            
        print(f"\nChecking {split_name.upper()} split ({file_path.name}):")
        print(f"  Total records: {len(records)} (Expected: {expected_total})")
        assert len(records) == expected_total, f"Expected {expected_total} records in {split_name}, got {len(records)}"
        
        zero_entity_count = 0
        entity_counts = {e: 0 for e in ALL_ENTITIES}
        
        for idx, rec in enumerate(records):
            text = rec["text"]
            tokens = rec["tokens"]
            ner_tags = rec["ner_tags"]
            entities = rec["entities"]
            template_id = rec["template_id"]
            template_ids_per_split[split_name].add(template_id)
            
            # 1. Zero entity count
            if len(entities) == 0:
                zero_entity_count += 1
                assert all(tag == "O" for tag in ner_tags), f"Record {rec['id']} has 0 entities but non-O tags!"
            
            # 2. Tokenizer check
            tokens_with_offsets = tokenize_with_offsets(text)
            re_tokens = [t[0] for t in tokens_with_offsets]
            assert tokens == re_tokens, f"Record {rec['id']} tokens do not match regex tokenizer!"
            assert len(tokens) == len(ner_tags), f"Record {rec['id']} token and tag lengths mismatch!"
            
            # 3. BIO sequence validity check
            prev_tag = "O"
            for tag in ner_tags:
                if tag.startswith("I-"):
                    curr_type = tag.split("-")[1]
                    assert prev_tag in (f"B-{curr_type}", f"I-{curr_type}"), (
                        f"Record {rec['id']} invalid BIO transition: {prev_tag} -> {tag}"
                    )
                prev_tag = tag
                
            # 4. Entity offsets and span alignment check
            for ent in entities:
                e_start = ent["start"]
                e_end = ent["end"]
                e_label = ent["label"]
                e_text = ent["text"]
                
                # Check character slice matches text
                actual_slice = text[e_start:e_end]
                assert actual_slice == e_text, (
                    f"Record {rec['id']} span mismatch: text[{e_start}:{e_end}]='{actual_slice}' != '{e_text}'"
                )
                
                # Record entity text and count
                entity_counts[e_label] += 1
                entity_strings_per_split[split_name][e_label].add(e_text.strip())
                
                # Check that tokens within this span match BIO tags
                span_tokens = [
                    (tok_idx, t[0]) for tok_idx, t in enumerate(tokens_with_offsets)
                    if t[1] >= e_start and t[2] <= e_end
                ]
                assert len(span_tokens) > 0, f"Record {rec['id']} entity '{e_text}' has no overlapping tokens!"
                assert ner_tags[span_tokens[0][0]] == f"B-{e_label}", (
                    f"Record {rec['id']} first token of '{e_text}' is not B-{e_label}: {ner_tags[span_tokens[0][0]]}"
                )
                for tok_idx, tok_str in span_tokens[1:]:
                    assert ner_tags[tok_idx] == f"I-{e_label}", (
                        f"Record {rec['id']} token '{tok_str}' in '{e_text}' is not I-{e_label}: {ner_tags[tok_idx]}"
                    )
                    
        zero_pct = (zero_entity_count / len(records)) * 100
        print(f"  Zero-entity records: {zero_entity_count} ({zero_pct:.1f}%) [Target: ~15%]")
        assert abs(zero_entity_count - expected_zeros) <= 2, (
            f"Zero entity count {zero_entity_count} deviates from expected {expected_zeros}"
        )
        
        print("  Entity mention breakdown:")
        for ent_type in ALL_ENTITIES:
            count = entity_counts[ent_type]
            print(f"    - {ent_type:<10}: {count:>4} (Minimum required: {min_mentions})")
            assert count >= min_mentions, f"{split_name} has only {count} {ent_type} mentions (required: {min_mentions})"
            
        split_data[split_name] = records
        
    # Check disjoint templates
    print("\n-----------------------------------------------------------------")
    print("STEP 2: VERIFYING TEMPLATE ISOLATION ACROSS SPLITS")
    print("-----------------------------------------------------------------")
    train_tmpls = template_ids_per_split["train"]
    val_tmpls = template_ids_per_split["val"]
    test_tmpls = template_ids_per_split["test"]
    
    print(f"Unique templates instantiated: Train={len(train_tmpls)}, Val={len(val_tmpls)}, Test={len(test_tmpls)}")
    assert len(TRAIN_TEMPLATES) == 105, f"Expected 105 train templates, got {len(TRAIN_TEMPLATES)}"
    assert len(VAL_TEMPLATES) == 22, f"Expected 22 val templates, got {len(VAL_TEMPLATES)}"
    assert len(TEST_TEMPLATES) == 23, f"Expected 23 test templates, got {len(TEST_TEMPLATES)}"
    print(f"Total defined templates in code: {len(TRAIN_TEMPLATES) + len(VAL_TEMPLATES) + len(TEST_TEMPLATES)} (105 / 22 / 23)")
    
    tr_val_overlap = train_tmpls & val_tmpls
    tr_tst_overlap = train_tmpls & test_tmpls
    val_tst_overlap = val_tmpls & test_tmpls
    
    print(f"  Train & Val overlap:  {len(tr_val_overlap)} (Expected: 0)")
    print(f"  Train & Test overlap: {len(tr_tst_overlap)} (Expected: 0)")
    print(f"  Val & Test overlap:   {len(val_tst_overlap)} (Expected: 0)")
    assert len(tr_val_overlap) == 0, f"Template overlap between Train and Val: {tr_val_overlap}"
    assert len(tr_tst_overlap) == 0, f"Template overlap between Train and Test: {tr_tst_overlap}"
    assert len(val_tst_overlap) == 0, f"Template overlap between Val and Test: {val_tst_overlap}"
    print("  [PASS] Templates are 100% disjoint across all splits.")
    
    # Check zero exact entity string overlap (except SALARY)
    print("\n-----------------------------------------------------------------")
    print("STEP 3: VERIFYING VALUE POOL ISOLATION (ZERO STRING OVERLAP)")
    print("-----------------------------------------------------------------")
    non_salary_entities = ["PERSON", "EMAIL", "PHONE", "EMP_ID", "ADDRESS"]
    for ent in non_salary_entities:
        tr_val = entity_strings_per_split["train"][ent] & entity_strings_per_split["val"][ent]
        tr_tst = entity_strings_per_split["train"][ent] & entity_strings_per_split["test"][ent]
        val_tst = entity_strings_per_split["val"][ent] & entity_strings_per_split["test"][ent]
        
        print(f"  {ent:<10}: Unique Train={len(entity_strings_per_split['train'][ent]):>3}, Val={len(entity_strings_per_split['val'][ent]):>3}, Test={len(entity_strings_per_split['test'][ent]):>3}")
        print(f"              Overlap (Tr/Val: {len(tr_val)}, Tr/Tst: {len(tr_tst)}, Val/Tst: {len(val_tst)})")
        
        assert len(tr_val) == 0, f"Overlap in {ent} between Train and Val: {tr_val}"
        assert len(tr_tst) == 0, f"Overlap in {ent} between Train and Test: {tr_tst}"
        assert len(val_tst) == 0, f"Overlap in {ent} between Val and Test: {val_tst}"
        
    print("  [PASS] Non-salary entity strings have zero overlap across splits.")
    return split_data

def verify_documents():
    print("\n=================================================================")
    print("STEP 4: VERIFYING DOCUMENT CORPUS AND METADATA (120 DOCUMENTS)")
    print("=================================================================")
    
    docs_dir = Path("data/documents")
    meta_path = docs_dir / "metadata.json"
    assert meta_path.exists(), f"metadata.json missing at {meta_path}"
    
    with open(meta_path, "r", encoding="utf-8") as f:
        meta_list = json.load(f)
        
    print(f"Total documents listed in metadata.json: {len(meta_list)} (Expected: 120)")
    assert len(meta_list) == 120, f"Expected 120 documents, got {len(meta_list)}"
    
    tier_counts = {}
    for m in meta_list:
        lvl = m["access_level"]
        tier_counts[lvl] = tier_counts.get(lvl, 0) + 1
        
    print(f"Document counts per access level:")
    for tier in ["public", "internal", "hr_restricted", "exec_restricted"]:
        print(f"  - {tier:<16}: {tier_counts.get(tier, 0)} (Expected: 30)")
        assert tier_counts.get(tier, 0) == 30, f"Expected 30 docs for {tier}, got {tier_counts.get(tier, 0)}"
        
    total_spans_checked = 0
    public_sensitive_violations = 0
    
    for m in meta_list:
        doc_id = m["doc_id"]
        filename = m["filename"]
        access_level = m["access_level"]
        sensitive_vals = m["sensitive_values"]
        
        md_path = docs_dir / filename
        sidecar_path = docs_dir / f"{doc_id}.json"
        
        assert md_path.exists(), f"Markdown file {md_path} missing!"
        assert sidecar_path.exists(), f"Sidecar JSON {sidecar_path} missing!"
        
        with open(md_path, "r", encoding="utf-8") as f:
            doc_text = f.read()
            
        with open(sidecar_path, "r", encoding="utf-8") as f:
            sidecar = json.load(f)
            
        assert sidecar["doc_id"] == doc_id
        assert sidecar["access_level"] == access_level
        
        spans = sidecar["spans"]
        for sp in spans:
            extracted = doc_text[sp["start"]:sp["end"]]
            assert extracted == sp["text"], (
                f"Document {doc_id} span text mismatch: '{extracted}' != '{sp['text']}'"
            )
            total_spans_checked += 1
            
        # Assert public documents have NO sensitive values except public figure PERSON names
        if access_level == "public":
            for sp in spans:
                if sp["label"] != "PERSON":
                    public_sensitive_violations += 1
            assert m["entities_present"] == ["PERSON"] or m["entities_present"] == [], (
                f"Public document {doc_id} contains forbidden entities: {m['entities_present']}"
            )
            
    assert public_sensitive_violations == 0, f"Found {public_sensitive_violations} sensitive violations in public docs!"
    print(f"Verified {total_spans_checked} character spans across all 120 sidecar files.")
    print("  [PASS] Public docs contain zero sensitive values except public figure names.")
    print("  [PASS] Every document sidecar exactly aligns with markdown text.")

def verify_queries():
    print("\n=================================================================")
    print("STEP 5: VERIFYING EVALUATION QUERIES (queries.json)")
    print("=================================================================")
    
    queries_path = Path("data/eval/queries.json")
    assert queries_path.exists(), f"queries.json missing at {queries_path}"
    
    with open(queries_path, "r", encoding="utf-8") as f:
        queries = json.load(f)
        
    print(f"Total queries in queries.json: {len(queries)} (Expected: >= 60)")
    assert len(queries) >= 60, f"Expected >= 60 queries, got {len(queries)}"
    
    type_counts = {}
    tier_counts = {}
    
    for q in queries:
        assert "query_id" in q
        assert "query_text" in q
        assert "target_doc_id" in q
        assert "target_access_level" in q
        assert "type" in q
        assert q["type"] in ("normal", "adversarial"), f"Unknown type {q['type']} in query {q['query_id']}"
        
        type_counts[q["type"]] = type_counts.get(q["type"], 0) + 1
        tier_counts[q["target_access_level"]] = tier_counts.get(q["target_access_level"], 0) + 1
        
    print(f"Query distribution by type:")
    print(f"  - normal:      {type_counts.get('normal', 0)}")
    print(f"  - adversarial: {type_counts.get('adversarial', 0)}")
    assert type_counts.get("normal", 0) >= 20, "Insufficient normal queries!"
    assert type_counts.get("adversarial", 0) >= 20, "Insufficient adversarial queries!"
    
    print(f"Query distribution by target access level:")
    for tier in ["public", "internal", "hr_restricted", "exec_restricted"]:
        print(f"  - {tier:<16}: {tier_counts.get(tier, 0)}")
        assert tier_counts.get(tier, 0) >= 10, f"Insufficient queries targeting {tier}"
        
    print("  [PASS] Evaluation queries cover both normal and adversarial types across all 4 tiers.")

def main():
    print("STARTING FULL PHASE 1 VERIFICATION SUITE\n")
    verify_ner_splits()
    verify_documents()
    verify_queries()
    print("\n=================================================================")
    print(">>> ALL VERIFICATION CHECKS PASSED SUCCESSFULLY! (PHASE 1 COMPLETE) <<<")
    print("=================================================================")

if __name__ == "__main__":
    main()
