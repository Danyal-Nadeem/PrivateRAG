"""RBAC evaluation script.

Runs every query in data/eval/queries.json under all 4 roles (256 runs),
both with and without the RBAC filter, and reports:
  - Leak rate (filtered and unfiltered) per role and overall
  - Hit@1 and Hit@3 for authorized queries, split by normal vs adversarial
  - Adversarial leak rate

Leaks are computed from data/documents/metadata.json (ground truth),
NOT from the Qdrant payload.

Results are written to:
  eval/results/rbac_eval_results.json
  eval/results/rbac_eval_matrix.csv

Usage:
    python -m eval.rbac_eval
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

from src.retrieval.rbac import ROLE_ACCESS_MAP, LEVEL_ORDER, get_allowed_levels
from src.retrieval.qdrant_service import search, search_no_filter

QUERIES_FILE  = PROJECT_ROOT / "data" / "eval" / "queries.json"
METADATA_FILE = PROJECT_ROOT / "data" / "documents" / "metadata.json"
RESULTS_DIR   = PROJECT_ROOT / "eval" / "results"

ROLES = ["intern", "employee", "hr", "admin"]
TOP_K = 3


def load_metadata_index(metadata_path: Path) -> dict[str, str]:
    """Return {doc_id: access_level} from metadata.json (ground truth)."""
    with open(metadata_path, encoding="utf-8") as f:
        docs: list[dict] = json.load(f)
    return {d["doc_id"]: d["access_level"] for d in docs}


def is_leak(retrieved_level: str, role: str) -> bool:
    allowed = get_allowed_levels(role)
    max_allowed = max(LEVEL_ORDER[lvl] for lvl in allowed)
    return LEVEL_ORDER.get(retrieved_level, 0) > max_allowed


def run_eval() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(QUERIES_FILE, encoding="utf-8") as f:
        queries: list[dict] = json.load(f)

    meta_index = load_metadata_index(METADATA_FILE)

    print("=" * 70)
    print("PrivateRAG — RBAC Evaluation")
    print(f"  Queries : {len(queries)}")
    print(f"  Roles   : {ROLES}")
    print(f"  Runs    : {len(queries) * len(ROLES)} (each query × each role)")
    print(f"  Top-K   : {TOP_K}")
    print("=" * 70)

    rows: list[dict] = []

    for q in queries:
        q_id    = q["query_id"]
        q_text  = q.get("query_text", q.get("query", ""))
        q_type  = q.get("type", "normal")
        t_doc   = q["target_doc_id"]
        t_level = q["target_access_level"]  # ground-truth from queries.json

        for role in ROLES:
            allowed_levels = get_allowed_levels(role)
            role_max = max(LEVEL_ORDER[lvl] for lvl in allowed_levels)

            # ---- FILTERED run ----
            try:
                hits_f = search(q_text, role=role, top_k=TOP_K)
            except Exception as e:
                print(f"  ERROR filtered {q_id} role={role}: {e}")
                hits_f = []

            # Map retrieved doc_ids → ground-truth levels
            retrieved_levels_f = [
                meta_index.get(h["doc_id"], "public") for h in hits_f
            ]
            retrieved_ids_f = [h["doc_id"] for h in hits_f]

            leak_f = any(is_leak(lvl, role) for lvl in retrieved_levels_f)
            hit1_f = t_doc in retrieved_ids_f[:1] if retrieved_ids_f else False
            hit3_f = t_doc in retrieved_ids_f[:3] if retrieved_ids_f else False

            # Authorized if the role can see the target document
            authorized = LEVEL_ORDER[t_level] <= role_max

            # ---- UNFILTERED (baseline) run ----
            try:
                hits_u = search_no_filter(q_text, top_k=TOP_K)
            except Exception as e:
                print(f"  ERROR unfiltered {q_id}: {e}")
                hits_u = []

            retrieved_levels_u = [
                meta_index.get(h["doc_id"], "public") for h in hits_u
            ]
            retrieved_ids_u = [h["doc_id"] for h in hits_u]
            leak_u = any(is_leak(lvl, role) for lvl in retrieved_levels_u)
            hit1_u = t_doc in retrieved_ids_u[:1] if retrieved_ids_u else False
            hit3_u = t_doc in retrieved_ids_u[:3] if retrieved_ids_u else False

            rows.append({
                "query_id"          : q_id,
                "role"              : role,
                "query_type"        : q_type,
                "target_doc_id"     : t_doc,
                "target_level"      : t_level,
                "authorized"        : authorized,
                # Filtered
                "filtered_retrieved_ids"    : retrieved_ids_f,
                "filtered_retrieved_levels" : retrieved_levels_f,
                "filtered_leak"             : leak_f,
                "filtered_hit@1"            : hit1_f,
                "filtered_hit@3"            : hit3_f,
                # Unfiltered
                "unfiltered_retrieved_ids"    : retrieved_ids_u,
                "unfiltered_retrieved_levels" : retrieved_levels_u,
                "unfiltered_leak"             : leak_u,
                "unfiltered_hit@1"            : hit1_u,
                "unfiltered_hit@3"            : hit3_u,
            })

    # ------------------------------------------------------------------
    # Aggregate statistics
    # ------------------------------------------------------------------
    total_runs = len(rows)

    def _rate(col: str, subset=None) -> float:
        data = rows if subset is None else subset
        if not data:
            return 0.0
        return sum(1 for r in data if r[col]) / len(data)

    # Per-role stats (filtered)
    per_role: dict[str, dict] = {}
    for role in ROLES:
        role_rows   = [r for r in rows if r["role"] == role]
        auth_rows   = [r for r in role_rows if r["authorized"]]
        normal_auth = [r for r in auth_rows if r["query_type"] == "normal"]
        adv_auth    = [r for r in auth_rows if r["query_type"] == "adversarial"]
        adv_rows    = [r for r in role_rows if r["query_type"] == "adversarial"]
        per_role[role] = {
            "total_runs"              : len(role_rows),
            "authorized_runs"         : len(auth_rows),
            "filtered_leak_rate"      : _rate("filtered_leak", role_rows),
            "unfiltered_leak_rate"    : _rate("unfiltered_leak", role_rows),
            "filtered_hit@1_normal"   : _rate("filtered_hit@1", normal_auth),
            "filtered_hit@3_normal"   : _rate("filtered_hit@3", normal_auth),
            "filtered_hit@1_adversarial" : _rate("filtered_hit@1", adv_auth),
            "filtered_hit@3_adversarial" : _rate("filtered_hit@3", adv_auth),
            "adversarial_leak_rate"   : _rate("filtered_leak", adv_rows),
            "unfiltered_hit@1_normal" : _rate("unfiltered_hit@1", normal_auth),
            "unfiltered_hit@3_normal" : _rate("unfiltered_hit@3", normal_auth),
        }

    overall_filtered_leak   = _rate("filtered_leak")
    overall_unfiltered_leak = _rate("unfiltered_leak")
    overall_adv_rows = [r for r in rows if r["query_type"] == "adversarial"]
    overall_adv_leak = _rate("filtered_leak", overall_adv_rows)

    # ------------------------------------------------------------------
    # Print report
    # ------------------------------------------------------------------
    print(f"\n{'ROLE':<12} {'F-LEAK%':>8} {'U-LEAK%':>8} {'HIT@1-N':>8} {'HIT@3-N':>8} {'HIT@1-A':>8} {'HIT@3-A':>8} {'ADV-LEAK%':>10}")
    print("-" * 76)
    for role, s in per_role.items():
        print(
            f"{role:<12} "
            f"{s['filtered_leak_rate']*100:>7.1f}% "
            f"{s['unfiltered_leak_rate']*100:>7.1f}% "
            f"{s['filtered_hit@1_normal']*100:>7.1f}% "
            f"{s['filtered_hit@3_normal']*100:>7.1f}% "
            f"{s['filtered_hit@1_adversarial']*100:>7.1f}% "
            f"{s['filtered_hit@3_adversarial']*100:>7.1f}% "
            f"{s['adversarial_leak_rate']*100:>9.1f}%"
        )
    print("-" * 76)
    print(f"{'OVERALL':<12} {overall_filtered_leak*100:>7.1f}% {overall_unfiltered_leak*100:>7.1f}%")
    print(f"\nOverall adversarial leak rate (filtered): {overall_adv_leak*100:.1f}%")
    print(f"\nTarget filtered leak rate: 0%")

    if overall_filtered_leak > 0:
        leaking = [r for r in rows if r["filtered_leak"]]
        print(f"\nWARNING: {len(leaking)} leak(s) detected!")
        for r in leaking[:5]:
            print(f"  query={r['query_id']} role={r['role']} levels={r['filtered_retrieved_levels']}")
    else:
        print("\n✓ Zero leaks detected with RBAC filter.")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    results_json = {
        "total_runs"              : total_runs,
        "overall_filtered_leak"   : overall_filtered_leak,
        "overall_unfiltered_leak" : overall_unfiltered_leak,
        "overall_adversarial_leak": overall_adv_leak,
        "per_role"                : per_role,
        "rows"                    : rows,
    }
    out_json = RESULTS_DIR / "rbac_eval_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=2)
    print(f"\nFull results saved to: {out_json}")

    # CSV matrix
    out_csv = RESULTS_DIR / "rbac_eval_matrix.csv"
    csv_fields = [
        "query_id", "role", "query_type", "target_doc_id", "target_level",
        "authorized",
        "filtered_leak", "filtered_hit@1", "filtered_hit@3",
        "unfiltered_leak", "unfiltered_hit@1", "unfiltered_hit@3",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in csv_fields})
    print(f"CSV matrix saved to  : {out_csv}")
    print("=" * 70)


if __name__ == "__main__":
    run_eval()
