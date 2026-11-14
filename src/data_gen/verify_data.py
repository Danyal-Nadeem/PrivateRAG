import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_gen.tokenizer import tokenize_with_offsets
from src.data_gen.ner_templates import TRAIN_TEMPLATES, VAL_TEMPLATES, TEST_TEMPLATES

ALL_ENTITIES = ["PERSON", "EMAIL", "PHONE", "EMP_ID", "ADDRESS", "SALARY"]
NON_SALARY_ENTITIES = ["PERSON", "EMAIL", "PHONE", "EMP_ID", "ADDRESS"]

def verify_ner_splits() -> Tuple[Dict[str, Any], Dict[str, Dict[str, Set[str]]]]:
    print("=================================================================")
    print("STEP 1: VERIFYING NER DATASETS (train.jsonl, val.jsonl, test.jsonl)")
    print("=================================================================")
    
    ner_dir = Path("data/ner")
    split_files = {
        "train": (ner_dir / "train.jsonl", 2100, 315, 300, 30),
        "val":   (ner_dir / "val.jsonl", 450, 68, 60, 8),
        "test":  (ner_dir / "test.jsonl", 450, 68, 60, 8),
    }
    
    split_data = {}
    entity_strings_per_split = {s: {e: set() for e in ALL_ENTITIES} for s in split_files}
    template_ids_per_split = {s: set() for s in split_files}
    zero_template_counts = {s: Counter() for s in split_files}
    
    for split_name, (file_path, expected_total, expected_zeros, min_mentions, min_zero_tmpls) in split_files.items():
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
            
            # 1. Zero entity count & tracking
            if len(entities) == 0:
                zero_entity_count += 1
                zero_template_counts[split_name][template_id] += 1
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
        
        # Check zero-entity template concentration constraint (max <= 15%)
        distinct_zeros = len(zero_template_counts[split_name])
        max_template_count = max(zero_template_counts[split_name].values())
        max_share = (max_template_count / zero_entity_count) * 100
        print(f"  Distinct zero-entity templates: {distinct_zeros} (Required: >= {min_zero_tmpls})")
        print(f"  Max share from any single zero-entity template: {max_share:.2f}% (Limit: <= 15.00%)")
        assert distinct_zeros >= min_zero_tmpls, f"{split_name} has only {distinct_zeros} zero-entity templates (required: {min_zero_tmpls})"
        assert max_share <= 15.00, f"{split_name} template concentration exceeds 15%: {max_share:.2f}%"
        
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
    print("STEP 3: VERIFYING FULL-STRING VALUE ISOLATION (ZERO STRING OVERLAP)")
    print("-----------------------------------------------------------------")
    for ent in NON_SALARY_ENTITIES:
        tr_val = entity_strings_per_split["train"][ent] & entity_strings_per_split["val"][ent]
        tr_tst = entity_strings_per_split["train"][ent] & entity_strings_per_split["test"][ent]
        val_tst = entity_strings_per_split["val"][ent] & entity_strings_per_split["test"][ent]
        
        print(f"  {ent:<10}: Unique Train={len(entity_strings_per_split['train'][ent]):>3}, Val={len(entity_strings_per_split['val'][ent]):>3}, Test={len(entity_strings_per_split['test'][ent]):>3}")
        print(f"              Overlap (Tr/Val: {len(tr_val)}, Tr/Tst: {len(tr_tst)}, Val/Tst: {len(val_tst)})")
        
        assert len(tr_val) == 0, f"Overlap in {ent} between Train and Val: {tr_val}"
        assert len(tr_tst) == 0, f"Overlap in {ent} between Train and Test: {tr_tst}"
        assert len(val_tst) == 0, f"Overlap in {ent} between Val and Test: {val_tst}"
        
    print("  [PASS] Non-salary entity strings have zero overlap across splits.")
    return split_data, entity_strings_per_split

def verify_component_overlap(split_data: Dict[str, Any]):
    print("\n=================================================================")
    print("STEP 4: VERIFYING COMPONENT-LEVEL DISJOINTNESS ACROSS SPLITS")
    print("=================================================================")
    
    first_names = {s: set() for s in split_data}
    last_names = {s: set() for s in split_data}
    street_names = {s: set() for s in split_data}
    area_names = {s: set() for s in split_data}
    us_city_state_zip = {s: set() for s in split_data}
    
    lahore_area_keywords = [
        "Gulberg", "Model Town", "Garden Town", "Shadman", "Faisal Town", "Johar Town",
        "Cavalry Ground", "Samanabad", "Township", "DHA Phase 1", "DHA Phase 2", "Muslim Town",
        "Allama Iqbal Town", "Cantt Sector B", "PCSIR", "DHA Phase 5", "DHA Phase 6",
        "Bahria Town", "WAPDA Town", "Askari 10", "State Life Society"
    ]
    
    for s, recs in split_data.items():
        for r in recs:
            for ent in r["entities"]:
                lbl = ent["label"]
                val = ent["text"].strip()
                if lbl == "PERSON":
                    parts = val.split()
                    if len(parts) >= 2:
                        first_names[s].add(parts[0])
                        last_names[s].add(parts[-1])
                    elif len(parts) == 1:
                        first_names[s].add(parts[0])
                elif lbl == "ADDRESS":
                    if "Lahore" in val:
                        parts = [p.strip() for p in val.split(",")]
                        for p in parts:
                            if any(k in p for k in lahore_area_keywords):
                                area_names[s].add(p)
                    else:
                        parts = [p.strip() for p in val.split(",")]
                        if len(parts) >= 3:
                            st_parts = parts[0].split()
                            if len(st_parts) >= 2:
                                street_names[s].add(" ".join(st_parts[1:]))
                            us_city_state_zip[s].add(f"{parts[1]}, {parts[2]}")

    components = [
        ("FIRST NAMES", first_names),
        ("LAST NAMES", last_names),
        ("STREET NAMES", street_names),
        ("AREA NAMES", area_names),
        ("US CITY+STATE+ZIP COMBOS", us_city_state_zip)
    ]
    
    for comp_name, comp_dict in components:
        tv = comp_dict["train"] & comp_dict["val"]
        tt = comp_dict["train"] & comp_dict["test"]
        vt = comp_dict["val"] & comp_dict["test"]
        print(f"  {comp_name:<26}: Overlap Tr/Val={len(tv)}, Tr/Tst={len(tt)}, Val/Tst={len(vt)}")
        assert len(tv) == 0, f"Component overlap between Train and Val in {comp_name}: {tv}"
        assert len(tt) == 0, f"Component overlap between Train and Test in {comp_name}: {tt}"
        assert len(vt) == 0, f"Component overlap between Val and Test in {comp_name}: {vt}"
        
    print("  [PASS] All 5 required components have 100% zero overlap across splits.")
    print("  [NOTE] Lahore postal codes (e.g. 54810) are formally exempt as documented in data/README.md.")

def verify_test_realistic(entity_strings_per_split: Dict[str, Dict[str, Set[str]]]):
    print("\n=================================================================")
    print("STEP 5: VERIFYING REALISTIC TEST SET (test_realistic.jsonl)")
    print("=================================================================")
    
    realistic_path = Path("data/ner/test_realistic.jsonl")
    assert realistic_path.exists(), f"File {realistic_path} missing!"
    
    with open(realistic_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
        
    print(f"Total records in test_realistic.jsonl: {len(records)} (Expected: 60)")
    assert len(records) == 60, f"Expected 60 records, got {len(records)}"
    
    zero_count = 0
    entity_counts = {e: 0 for e in ALL_ENTITIES}
    realistic_strings = {e: set() for e in ALL_ENTITIES}
    
    for r in records:
        text = r["text"]
        tokens = r["tokens"]
        ner_tags = r["ner_tags"]
        entities = r["entities"]
        
        if len(entities) == 0:
            zero_count += 1
            assert all(t == "O" for t in ner_tags), f"Record {r['id']} has 0 entities but non-O tags!"
            
        # Tokenizer check
        tokens_with_offsets = tokenize_with_offsets(text)
        re_tokens = [t[0] for t in tokens_with_offsets]
        assert tokens == re_tokens, f"Record {r['id']} tokens mismatch regex tokenizer!"
        assert len(tokens) == len(ner_tags)
        
        # BIO sequence validity
        prev_tag = "O"
        for tag in ner_tags:
            if tag.startswith("I-"):
                curr_type = tag.split("-")[1]
                assert prev_tag in (f"B-{curr_type}", f"I-{curr_type}"), f"Invalid transition: {prev_tag} -> {tag}"
            prev_tag = tag
            
        # Span alignment
        for ent in entities:
            e_start = ent["start"]
            e_end = ent["end"]
            e_label = ent["label"]
            e_text = ent["text"]
            
            assert text[e_start:e_end] == e_text
            entity_counts[e_label] += 1
            realistic_strings[e_label].add(e_text.strip())
            
    print(f"Zero-entity negative records: {zero_count} (Expected: ~10)")
    assert 8 <= zero_count <= 12, f"Expected ~10 zero-entity records, got {zero_count}"
    
    print("Entity breakdown in test_realistic.jsonl:")
    for ent in ALL_ENTITIES:
        print(f"  - {ent:<10}: {entity_counts[ent]:>2} mentions")
        assert entity_counts[ent] > 0, f"Entity {ent} has 0 mentions in realistic set!"
        
    # Check zero exact string overlap with Train and Val
    print("\nChecking exact string isolation against Train and Val:")
    for ent in NON_SALARY_ENTITIES:
        tr_overlap = realistic_strings[ent] & entity_strings_per_split["train"][ent]
        val_overlap = realistic_strings[ent] & entity_strings_per_split["val"][ent]
        print(f"  {ent:<10}: Overlap vs Train={len(tr_overlap)}, vs Val={len(val_overlap)}")
        assert len(tr_overlap) == 0, f"test_realistic.jsonl overlaps with Train for {ent}: {tr_overlap}"
        assert len(val_overlap) == 0, f"test_realistic.jsonl overlaps with Val for {ent}: {val_overlap}"
        
    print("  [PASS] test_realistic.jsonl has zero exact string overlap with Train and Val.")

def verify_documents():
    print("\n=================================================================")
    print("STEP 6: VERIFYING DOCUMENT CORPUS AND METADATA (120 DOCUMENTS)")
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
    print("STEP 7: VERIFYING EVALUATION QUERIES (queries.json)")
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
    print("STARTING FULL PHASE 1 EXTENDED VERIFICATION SUITE\n")
    split_data, entity_strings = verify_ner_splits()
    verify_component_overlap(split_data)
    verify_test_realistic(entity_strings)
    verify_documents()
    verify_queries()
    print("\n=================================================================")
    print(">>> ALL VERIFICATION CHECKS PASSED SUCCESSFULLY! (PHASE 1 COMPLETE) <<<")
    print("=================================================================")

if __name__ == "__main__":
    main()
