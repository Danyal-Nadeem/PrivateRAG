"""NER evaluation benchmark for the 4 systems:
1. Regex Only (src/ner/regex_filter.py)
2. Pretrained Baseline (dslim/bert-base-NER)
3. Fine-tuned DistilBERT (src/ner/infer.py)
4. Hybrid Filter (Regex + Fine-tuned)

Evaluated across both:
- data/ner/test.jsonl (450 template sentences)
- data/ner/test_realistic.jsonl (60 natural sentences)

Reports:
- Per-entity Precision, Recall, F1
- Overall Micro and Macro Precision, Recall, F1
- False positives on zero-entity hard negatives
- Saves outputs to eval/results/ner_eval_results.json and ner_eval_matrix.csv
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ner.regex_filter import extract_regex_entities
from src.ner.pretrained_baseline import extract_pretrained_entities
from src.ner.infer import extract_model_entities
from src.ner.hybrid_filter import extract_hybrid_entities

ALL_ENTITIES = ["PERSON", "EMAIL", "PHONE", "EMP_ID", "ADDRESS", "SALARY"]
RESULTS_DIR = PROJECT_ROOT / "eval" / "results"


def load_dataset(path: Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def evaluate_system(
    extract_fn,
    records: list[dict],
    system_name: str,
) -> dict[str, Any]:
    """Evaluate *extract_fn* against *records* using exact character span matching."""
    tp_per_label = defaultdict(int)
    fp_per_label = defaultdict(int)
    fn_per_label = defaultdict(int)
    support_per_label = defaultdict(int)

    hard_negative_fp = 0
    hard_negative_sentences = 0
    extraction_errors = 0

    for r in records:
        text = r["text"]
        gt_spans = r.get("entities", [])
        is_hard_negative = len(gt_spans) == 0

        if is_hard_negative:
            hard_negative_sentences += 1

        try:
            pred_spans = extract_fn(text)
        except Exception as exc:
            # Never silently score a broken extractor as zero predictions.
            extraction_errors += 1
            if extraction_errors == 1:
                print(f"  [ERROR] {system_name}: extraction failed: {exc}")
            pred_spans = []

        if is_hard_negative and pred_spans:
            hard_negative_fp += len(pred_spans)

        # Standard representation: (label, start, end)
        gt_set: Set[Tuple[str, int, int]] = {
            (e["label"], e["start"], e["end"]) for e in gt_spans
        }
        pred_set: Set[Tuple[str, int, int]] = {
            (e["label"], e["start"], e["end"]) for e in pred_spans
        }

        for lbl, start, end in gt_set:
            support_per_label[lbl] += 1

        matched = gt_set & pred_set
        for lbl, _, _ in matched:
            tp_per_label[lbl] += 1

        fp_set = pred_set - gt_set
        for lbl, _, _ in fp_set:
            fp_per_label[lbl] += 1

        fn_set = gt_set - pred_set
        for lbl, _, _ in fn_set:
            fn_per_label[lbl] += 1

    # Compute metrics per label
    per_label_metrics = {}
    f1_list = []

    for lbl in ALL_ENTITIES:
        tp = tp_per_label[lbl]
        fp = fp_per_label[lbl]
        fn = fn_per_label[lbl]
        sup = support_per_label[lbl]

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        per_label_metrics[lbl] = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "support": sup,
        }
        if sup > 0:
            f1_list.append(f1)

    total_tp = sum(tp_per_label[l] for l in ALL_ENTITIES)
    total_fp = sum(fp_per_label[l] for l in ALL_ENTITIES)
    total_fn = sum(fn_per_label[l] for l in ALL_ENTITIES)
    total_sup = sum(support_per_label[l] for l in ALL_ENTITIES)

    micro_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_rec  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1   = (2 * micro_prec * micro_rec) / (micro_prec + micro_rec) if (micro_prec + micro_rec) > 0 else 0.0
    macro_f1   = sum(f1_list) / len(f1_list) if f1_list else 0.0

    return {
        "system": system_name,
        "per_label": per_label_metrics,
        "micro_precision": micro_prec,
        "micro_recall": micro_rec,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "total_support": total_sup,
        "hard_negatives_evaluated": hard_negative_sentences,
        "hard_negatives_fp": hard_negative_fp,
        "extraction_errors": extraction_errors,
    }


def print_evaluation_table(results_by_system: dict[str, dict], dataset_name: str) -> None:
    print("\n" + "=" * 90)
    print(f"BENCHMARK RESULTS: {dataset_name.upper()}")
    print("=" * 90)

    header = (
        f"{'SYSTEM':<20} "
        f"{'PERSON F1':>10} "
        f"{'EMAIL F1':>10} "
        f"{'PHONE F1':>10} "
        f"{'EMP_ID F1':>10} "
        f"{'ADDR F1':>10} "
        f"{'SAL F1':>10} "
        f"{'MICRO F1':>10} "
        f"{'NEG FP':>8}"
    )
    print(header)
    print("-" * len(header))

    for sys_name, res in results_by_system.items():
        pl = res["per_label"]
        print(
            f"{sys_name:<20} "
            f"{pl['PERSON']['f1']*100:>9.1f}% "
            f"{pl['EMAIL']['f1']*100:>9.1f}% "
            f"{pl['PHONE']['f1']*100:>9.1f}% "
            f"{pl['EMP_ID']['f1']*100:>9.1f}% "
            f"{pl['ADDRESS']['f1']*100:>9.1f}% "
            f"{pl['SALARY']['f1']*100:>9.1f}% "
            f"{res['micro_f1']*100:>9.1f}% "
            f"{res['hard_negatives_fp']:>8}"
        )
    print("-" * len(header))


def run_all_evaluations() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    test_path = PROJECT_ROOT / "data" / "ner" / "test.jsonl"
    realistic_path = PROJECT_ROOT / "data" / "ner" / "test_realistic.jsonl"

    test_records = load_dataset(test_path)
    realistic_records = load_dataset(realistic_path)

    systems = {
        "1. Regex Only": extract_regex_entities,
        "2. Pretrained NER": extract_pretrained_entities,
        "3. Fine-tuned Model": extract_model_entities,
        "4. Hybrid (Proposed)": extract_hybrid_entities,
    }

    # Honesty guards: never record all-zero rows for systems that cannot run.
    from src.ner.infer import model_available
    warnings: list[str] = []
    checkpoint_ok = model_available()
    if not checkpoint_ok:
        msg = (
            "Fine-tuned checkpoint NOT found (models/distilbert-ner). "
            "Skipping '3. Fine-tuned Model'; '4. Hybrid' runs in regex-only "
            "fallback (PERSON/ADDRESS unmasked). Train the model and re-run "
            "for a valid baseline-vs-fine-tuned comparison."
        )
        print("\n" + "!" * 90 + f"\n[WARNING] {msg}\n" + "!" * 90)
        warnings.append(msg)
        systems.pop("3. Fine-tuned Model")

    from src.ner.pretrained_baseline import pretrained_available
    if not pretrained_available():
        msg = (
            "Pretrained baseline (dslim/bert-base-NER) could not be loaded "
            "(offline mode or download failure). Skipping '2. Pretrained NER' "
            "instead of recording misleading zero scores."
        )
        print(f"[WARNING] {msg}")
        warnings.append(msg)
        systems.pop("2. Pretrained NER")

    full_results = {}
    run_metadata = {
        "model_checkpoint_available": checkpoint_ok,
        "warnings": warnings,
    }

    for d_name, records in [("test.jsonl (450 sents)", test_records), ("test_realistic.jsonl (60 sents)", realistic_records)]:
        ds_results = {}
        for s_name, fn in systems.items():
            print(f"Running {s_name} on {d_name}...")
            res = evaluate_system(fn, records, s_name)
            ds_results[s_name] = res
        print_evaluation_table(ds_results, d_name)
        full_results[d_name] = ds_results

    # Save to JSON (metadata records whether model-dependent systems could run)
    out_json = RESULTS_DIR / "ner_eval_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"metadata": run_metadata, "datasets": full_results}, f, indent=2)
    print(f"\nFull evaluation metrics saved to: {out_json}")

    # Save to CSV Matrix
    out_csv = RESULTS_DIR / "ner_eval_matrix.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "dataset", "system",
            "person_f1", "email_f1", "phone_f1", "emp_id_f1", "address_f1", "salary_f1",
            "micro_precision", "micro_recall", "micro_f1", "macro_f1",
            "hard_negatives_fp"
        ])
        for d_name, s_dict in full_results.items():
            for s_name, res in s_dict.items():
                pl = res["per_label"]
                writer.writerow([
                    d_name, s_name,
                    f"{pl['PERSON']['f1']*100:.1f}",
                    f"{pl['EMAIL']['f1']*100:.1f}",
                    f"{pl['PHONE']['f1']*100:.1f}",
                    f"{pl['EMP_ID']['f1']*100:.1f}",
                    f"{pl['ADDRESS']['f1']*100:.1f}",
                    f"{pl['SALARY']['f1']*100:.1f}",
                    f"{res['micro_precision']*100:.1f}",
                    f"{res['micro_recall']*100:.1f}",
                    f"{res['micro_f1']*100:.1f}",
                    f"{res['macro_f1']*100:.1f}",
                    res["hard_negatives_fp"],
                ])
    print(f"CSV summary matrix saved to   : {out_csv}")
    print("=" * 90)


if __name__ == "__main__":
    run_all_evaluations()
