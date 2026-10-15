#!/usr/bin/env python3
"""Calculate comprehensive ANPR evaluation metrics and confusion matrix.

Works locally in VS Code or in Kaggle notebooks.
Usage:
    python tools/calculate_metrics.py --pred benchmarks/results.csv --gt samples/ground_truth.csv
"""
import argparse
import csv
import os
import sys

def levenshtein(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def evaluate(pred_file: str, gt_file: str):
    # Load ground truth
    gt_map = {}
    if os.path.exists(gt_file):
        with open(gt_file, "r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for r in reader:
                gt_map[r["image"].strip()] = r.get("ground_truth", "").strip()

    # Load predictions
    pred_map = {}
    if os.path.exists(pred_file):
        with open(pred_file, "r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for r in reader:
                img = r.get("image", "").strip()
                plate = r.get("plate", "").strip()
                valid = int(r.get("valid", 0))
                det_score = float(r.get("det_score", 0.0))
                pred_map[img] = {"plate": plate, "valid": valid, "score": det_score}

    tp = fp = fn = tn = 0
    exact_matches = 0
    total_plates_with_gt = 0
    total_edit_distance = 0
    total_gt_chars = 0
    valid_grammar_count = 0

    all_images = sorted(set(list(gt_map.keys()) + list(pred_map.keys())))
    print(f"{'Image':<16} | {'Ground Truth':<14} | {'Predicted':<14} | {'Grammar':<7} | {'Status'}")
    print("-" * 75)

    for img in all_images:
        gt_plate = gt_map.get(img, "")
        pred_entry = pred_map.get(img, {"plate": "", "valid": 0, "score": 0.0})
        pred_plate = pred_entry["plate"]
        is_valid = pred_entry["valid"]

        has_gt = bool(gt_plate)
        has_pred = bool(pred_plate)

        if has_gt and has_pred:
            tp += 1
            total_plates_with_gt += 1
            edit_dist = levenshtein(pred_plate.upper(), gt_plate.upper())
            total_edit_distance += edit_dist
            total_gt_chars += max(len(gt_plate), 1)
            if pred_plate.upper() == gt_plate.upper() or (len(pred_plate) >= 7 and pred_plate.upper() in gt_plate.upper()):
                exact_matches += 1
                status = "MATCH (TP)"
            else:
                status = f"MISMATCH (d={edit_dist})"
            if is_valid:
                valid_grammar_count += 1
        elif not has_gt and has_pred:
            fp += 1
            status = "FALSE DET (FP)"
            if is_valid:
                valid_grammar_count += 1
        elif has_gt and not has_pred:
            fn += 1
            total_plates_with_gt += 1
            total_edit_distance += len(gt_plate)
            total_gt_chars += len(gt_plate)
            status = "MISSED (FN)"
        else:
            tn += 1
            status = "CORRECT NEG (TN)"

        print(f"{img:<16} | {gt_plate or '<none>':<14} | {pred_plate or '<none>':<14} | {'VALID' if is_valid else 'OOG':<7} | {status}")

    print("=" * 75)
    precision = (tp / (tp + fp)) * 100 if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    accuracy = ((tp + tn) / len(all_images)) * 100 if all_images else 0.0
    exact_match_pct = (exact_matches / max(total_plates_with_gt, 1)) * 100
    cer = (total_edit_distance / max(total_gt_chars, 1)) * 100

    print("\nEVALUATION MATRIX SUMMARY:")
    print(f"  • Total Test Frames:          {len(all_images)}")
    print(f"  • True Positives (TP):        {tp}")
    print(f"  • False Positives (FP):       {fp}")
    print(f"  • False Negatives (FN):       {fn}")
    print(f"  • True Negatives (TN):        {tn}")
    print(f"  • Detection Precision:        {precision:.2f}%")
    print(f"  • Detection Recall:           {recall:.2f}%")
    print(f"  • Detection F1-Score:         {f1:.2f}%")
    print(f"  • Overall Frame Accuracy:     {accuracy:.2f}%")
    print(f"  • Plate Exact Match (EM):     {exact_match_pct:.2f}%")
    print(f"  • Character Error Rate (CER): {cer:.2f}%")
    print("=" * 75)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate ANPR Evaluation Matrix")
    parser.add_argument("--pred", default="benchmarks/results.csv", help="Predictions CSV")
    parser.add_argument("--gt", default="samples/ground_truth.csv", help="Ground Truth CSV")
    args = parser.parse_args()
    evaluate(args.pred, args.gt)
