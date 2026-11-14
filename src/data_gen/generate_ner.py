import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_gen.entity_pools import EntityPoolGenerator
from src.data_gen.tokenizer import tokenize_with_offsets, align_tokens_and_bio
from src.data_gen.ner_templates import (
    TRAIN_TEMPLATES,
    VAL_TEMPLATES,
    TEST_TEMPLATES,
    SENSITIVE_ENTITY_TYPES,
    HARD_NEGATIVE_SLOTS,
)

SLOT_PATTERN = re.compile(r"\{([A-Z0-9_]+)\}")

def fill_template(
    template: Dict[str, Any],
    pool_gen: EntityPoolGenerator,
    example_id: str
) -> Dict[str, Any]:
    text_pattern = template["text"]
    template_id = template["id"]
    
    slots_in_order = SLOT_PATTERN.findall(text_pattern)
    person_name = pool_gen.generate_person() if "PERSON" in slots_in_order else None
    
    curr_text = text_pattern
    curr_pos = 0
    entities = []
    
    while True:
        m = SLOT_PATTERN.search(curr_text, curr_pos)
        if not m:
            break
        slot_name = m.group(1)
        start_pos = m.start()
        end_pos = m.end()
        
        if slot_name in ("PERSON", "PERSON_2"):
            val = person_name if (slot_name == "PERSON" and person_name) else pool_gen.generate_person()
            label = "PERSON"
        elif slot_name in ("EMAIL", "EMAIL_2"):
            val = pool_gen.generate_email(person_name)
            label = "EMAIL"
        elif slot_name in ("PHONE", "PHONE_2"):
            val = pool_gen.generate_phone()
            label = "PHONE"
        elif slot_name in ("EMP_ID", "EMP_ID_2"):
            val = pool_gen.generate_emp_id()
            label = "EMP_ID"
        elif slot_name in ("ADDRESS", "ADDRESS_2"):
            val = pool_gen.generate_address()
            label = "ADDRESS"
        elif slot_name in ("SALARY", "SALARY_2"):
            val = pool_gen.generate_salary()
            label = "SALARY"
        elif slot_name in HARD_NEGATIVE_SLOTS:
            val = pool_gen.generate_hard_negative(HARD_NEGATIVE_SLOTS[slot_name])
            label = None
        else:
            val = "N/A"
            label = None
            
        val = val.strip()
        curr_text = curr_text[:start_pos] + val + curr_text[end_pos:]
        
        if label:
            entities.append({
                "start": start_pos,
                "end": start_pos + len(val),
                "label": label,
                "text": val
            })
            
        curr_pos = start_pos + len(val)
        
    tokens_with_offsets = tokenize_with_offsets(curr_text)
    tokens = [t[0] for t in tokens_with_offsets]
    ner_tags = align_tokens_and_bio(curr_text, entities, tokens_with_offsets)
    
    return {
        "id": example_id,
        "template_id": template_id,
        "text": curr_text,
        "tokens": tokens,
        "ner_tags": ner_tags,
        "entities": entities
    }

def generate_split(
    split_name: str,
    templates: List[Dict[str, Any]],
    total_count: int,
    zero_count: int,
    seed: int = 42
) -> List[Dict[str, Any]]:
    pool_gen = EntityPoolGenerator(split=split_name, seed=seed)
    
    zero_templates = [t for t in templates if "zero" in t["id"]]
    ent_templates = [t for t in templates if "zero" not in t["id"]]
    
    ent_count = total_count - zero_count
    samples = []
    
    # 1. Generate zero-entity sentences
    for i in range(zero_count):
        tmpl = zero_templates[i % len(zero_templates)]
        ex_id = f"{split_name}_{len(samples) + 1:04d}"
        samples.append(fill_template(tmpl, pool_gen, ex_id))
        
    # 2. Generate entity-bearing sentences
    for i in range(ent_count):
        tmpl = ent_templates[i % len(ent_templates)]
        ex_id = f"{split_name}_{len(samples) + 1:04d}"
        samples.append(fill_template(tmpl, pool_gen, ex_id))
        
    # Shuffle deterministically so zero-entity sentences are distributed throughout
    rng = pool_gen.rng
    rng.shuffle(samples)
    
    # Re-assign sequential IDs after shuffle
    for idx, s in enumerate(samples):
        s["id"] = f"{split_name}_{idx + 1:04d}"
        
    return samples

def main():
    out_dir = Path("data/ner")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    splits_spec = [
        ("train", TRAIN_TEMPLATES, 2100, 315, 1001),
        ("val", VAL_TEMPLATES, 450, 68, 2002),
        ("test", TEST_TEMPLATES, 450, 68, 3003),
    ]
    
    for split_name, templates, total, zeros, seed in splits_spec:
        print(f"Generating {split_name} split: total={total}, zero_entity={zeros}...")
        records = generate_split(split_name, templates, total, zeros, seed)
        file_path = out_dir / f"{split_name}.jsonl"
        with open(file_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Wrote {len(records)} records to {file_path}")

if __name__ == "__main__":
    main()
