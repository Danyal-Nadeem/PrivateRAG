"""End-to-End Evaluation for PrivateRAG (Phase 5).

Evaluates the dual-layer security pipeline:
1. Input-side: RBAC retrieval filter in Qdrant (restricts candidate chunks by role).
2. Output-side: NER & Regex sanitization filter (masks sensitive entities in answers).

Measures ground-truth sensitive value leakage in generated answers:
- Baseline: Raw generated answers without output sanitization.
- Proposed: Generated answers sanitized with PrivateRAG masking.

Ground truth is sourced from data/documents/*.json sidecar files and metadata.json.

Results written to:
  eval/results/e2e_eval_results.json
  eval/results/e2e_eval_matrix.csv

Usage:
  python -m eval.e2e_eval
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.rbac import get_allowed_levels, LEVEL_ORDER
from src.retrieval.qdrant_service import search
from src.llm.extractive import ExtractiveProvider
from src.ner.masking import sanitize_text

QUERIES_FILE = PROJECT_ROOT / "data" / "eval" / "queries.json"
DOCS_DIR = PROJECT_ROOT / "data" / "documents"
METADATA_FILE = DOCS_DIR / "metadata.json"
RESULTS_DIR = PROJECT_ROOT / "eval" / "results"

ROLES = ["intern", "employee", "hr", "admin"]
ENTITY_TYPES = ["EMAIL", "PHONE", "EMP_ID", "SALARY", "PERSON", "ADDRESS"]
TOP_K = 3


def load_labeled_sensitive_values() -> Dict[str, Dict[str, Any]]:
    """Load sidecar .json files mapping doc_id -> list of (entity_text, label)."""
    manifest: Dict[str, Dict[str, Any]] = {}
    
    with open(METADATA_FILE, encoding="utf-8") as f:
        meta_list = json.load(f)
    access_map = {d["doc_id"]: d["access_level"] for d in meta_list}

    for json_file in DOCS_DIR.glob("*.json"):
        if json_file.name == "metadata.json":
            continue
        try:
            with open(json_file, encoding="utf-8") as f:
                doc_data = json.load(f)
            doc_id = doc_data["doc_id"]
            spans = doc_data.get("spans", [])
            manifest[doc_id] = {
                "access_level": access_map.get(doc_id, doc_data.get("access_level", "public")),
                "entities": [(s["text"], s["label"]) for s in spans if s.get("text")],
            }
        except Exception:
            continue

    return manifest


def find_leaked_entities(text: str, entities_pool: List[Tuple[str, str]]) -> List[Dict[str, str]]:
    """Return all sensitive entities that appear verbatim in text with their label."""
    leaked = []
    seen = set()
    for val, label in entities_pool:
        if not val or len(val.strip()) < 3 or val in seen:
            continue
        pattern = re.escape(val)
        if re.search(pattern, text, re.IGNORECASE):
            leaked.append({"text": val, "label": label})
            seen.add(val)
    return leaked


def run_e2e_evaluation() -> Dict[str, Any]:
    """Execute 256 end-to-end runs and calculate leak prevention metrics."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(QUERIES_FILE, encoding="utf-8") as f:
        queries: List[Dict[str, Any]] = json.load(f)

    manifest = load_labeled_sensitive_values()
    llm = ExtractiveProvider()

    # Per-role accumulators
    role_metrics = {
        r: {
            "total_queries": 0,
            "raw_leaked_answers": 0,
            "sanitized_leaked_answers": 0,
            "raw_entity_leaks": {ent: 0 for ent in ENTITY_TYPES},
            "sanitized_entity_leaks": {ent: 0 for ent in ENTITY_TYPES},
        }
        for r in ROLES
    }

    # Per-entity global accumulators
    entity_metrics = {
        ent: {"raw_leaks": 0, "sanitized_leaks": 0}
        for ent in ENTITY_TYPES
    }

    all_detailed_runs = []

    print(f"Running End-to-End Evaluation across {len(queries)} queries x {len(ROLES)} roles (256 runs)...")

    for q in queries:
        qid = q["query_id"]
        qtext = q["query_text"]
        target_doc = q.get("target_doc_id")
        target_data = manifest.get(target_doc, {"access_level": "unknown", "entities": []})
        target_level = target_data["access_level"]

        for role in ROLES:
            allowed = get_allowed_levels(role)
            is_authorized = target_level in allowed

            # 1. Retrieval (with RBAC filter)
            hits = search(qtext, role=role, top_k=TOP_K)
            context_chunks = [h["text"] for h in hits]

            # 2. Generation
            raw_answer = llm.generate(prompt=qtext, context_chunks=context_chunks)

            # 3. Sanitization
            sanitized_answer = sanitize_text(raw_answer)

            # Collect candidate ground-truth entities from retrieved docs
            retrieved_entities: List[Tuple[str, str]] = []
            for h in hits:
                d_data = manifest.get(h["doc_id"], {"entities": []})
                retrieved_entities.extend(d_data.get("entities", []))

            # 4. Leak detection
            raw_leaks = find_leaked_entities(raw_answer, retrieved_entities)
            sanitized_leaks = find_leaked_entities(sanitized_answer, retrieved_entities)

            role_metrics[role]["total_queries"] += 1
            if raw_leaks:
                role_metrics[role]["raw_leaked_answers"] += 1
                for item in raw_leaks:
                    lbl = item["label"]
                    if lbl in entity_metrics:
                        entity_metrics[lbl]["raw_leaks"] += 1
                        role_metrics[role]["raw_entity_leaks"][lbl] += 1

            if sanitized_leaks:
                role_metrics[role]["sanitized_leaked_answers"] += 1
                for item in sanitized_leaks:
                    lbl = item["label"]
                    if lbl in entity_metrics:
                        entity_metrics[lbl]["sanitized_leaks"] += 1
                        role_metrics[role]["sanitized_entity_leaks"][lbl] += 1

            all_detailed_runs.append({
                "query_id": qid,
                "role": role,
                "is_authorized": is_authorized,
                "target_doc_id": target_doc,
                "target_level": target_level,
                "retrieved_count": len(hits),
                "raw_leaks": raw_leaks,
                "sanitized_leaks": sanitized_leaks,
                "prevented": bool(raw_leaks and not sanitized_leaks),
            })

    # Summary table by role
    summary_role_rows = []
    total_raw_answers = sum(m["raw_leaked_answers"] for m in role_metrics.values())
    total_san_answers = sum(m["sanitized_leaked_answers"] for m in role_metrics.values())
    total_runs = sum(m["total_queries"] for m in role_metrics.values())

    for role in ROLES:
        m = role_metrics[role]
        raw_pct = (m["raw_leaked_answers"] / m["total_queries"]) * 100
        san_pct = (m["sanitized_leaked_answers"] / m["total_queries"]) * 100
        prev_pct = ((m["raw_leaked_answers"] - m["sanitized_leaked_answers"]) / max(m["raw_leaked_answers"], 1)) * 100
        summary_role_rows.append({
            "role": role,
            "total_queries": m["total_queries"],
            "raw_leaked_answers": m["raw_leaked_answers"],
            "raw_leak_rate_pct": round(raw_pct, 2),
            "sanitized_leaked_answers": m["sanitized_leaked_answers"],
            "sanitized_leak_rate_pct": round(san_pct, 2),
            "leak_prevention_rate_pct": round(prev_pct, 2),
        })

    # Summary table by entity type
    summary_entity_rows = []
    for ent in ENTITY_TYPES:
        em = entity_metrics[ent]
        rl = em["raw_leaks"]
        sl = em["sanitized_leaks"]
        prev = ((rl - sl) / max(rl, 1)) * 100 if rl > 0 else 100.0
        summary_entity_rows.append({
            "entity_type": ent,
            "raw_leaks_count": rl,
            "sanitized_leaks_count": sl,
            "leak_reduction_pct": round(prev, 2),
        })

    results_data = {
        "summary_by_role": summary_role_rows,
        "summary_by_entity": summary_entity_rows,
        "overall": {
            "total_runs": total_runs,
            "total_raw_leaked_answers": total_raw_answers,
            "raw_leak_rate_pct": round((total_raw_answers / total_runs) * 100, 2),
            "total_sanitized_leaked_answers": total_san_answers,
            "sanitized_leak_rate_pct": round((total_san_answers / total_runs) * 100, 2),
            "overall_leak_prevention_pct": round(((total_raw_answers - total_san_answers) / max(total_raw_answers, 1)) * 100, 2),
        },
        "detailed_runs": all_detailed_runs,
    }

    # Print Terminal Benchmark
    print("\n" + "=" * 90)
    print("END-TO-END SECURITY BENCHMARK (PHASE 5: RETRIEVAL + GENERATION + MASKING)")
    print("=" * 90)
    print("TABLE 1: LEAK RATE BY ROLE (256 RUNS)")
    print(f"{'ROLE':<12} {'QUERIES':<10} {'RAW LEAKS':<14} {'RAW LEAK %':<14} {'SANITIZED LEAKS':<18} {'LEAK PREV %'}")
    print("-" * 90)
    for row in summary_role_rows:
        print(f"{row['role']:<12} {row['total_queries']:<10} {row['raw_leaked_answers']:<14} {row['raw_leak_rate_pct']:<13}% {row['sanitized_leaked_answers']:<18} {row['leak_prevention_rate_pct']:>9.1f}%")
    print("-" * 90)
    ov = results_data["overall"]
    print(f"{'OVERALL':<12} {ov['total_runs']:<10} {ov['total_raw_leaked_answers']:<14} {ov['raw_leak_rate_pct']:<13}% {ov['total_sanitized_leaked_answers']:<18} {ov['overall_leak_prevention_pct']:>9.1f}%")
    print("=" * 90)

    print("\nTABLE 2: LEAK REDUCTION BY ENTITY TYPE")
    print(f"{'ENTITY TYPE':<15} {'RAW LEAKS COUNT':<20} {'SANITIZED LEAKS COUNT':<25} {'REDUCTION %'}")
    print("-" * 90)
    for erow in summary_entity_rows:
        print(f"{erow['entity_type']:<15} {erow['raw_leaks_count']:<20} {erow['sanitized_leaks_count']:<25} {erow['leak_reduction_pct']:>9.1f}%")
    print("=" * 90 + "\n")

    # Save to JSON
    json_path = RESULTS_DIR / "e2e_eval_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2)

    # Save to CSV
    csv_path = RESULTS_DIR / "e2e_eval_matrix.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "role",
                "total_queries",
                "raw_leaked_answers",
                "raw_leak_rate_pct",
                "sanitized_leaked_answers",
                "sanitized_leak_rate_pct",
                "leak_prevention_rate_pct",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_role_rows)

    print(f"Results saved to:\n  {json_path}\n  {csv_path}\n")
    return results_data


if __name__ == "__main__":
    run_e2e_evaluation()
