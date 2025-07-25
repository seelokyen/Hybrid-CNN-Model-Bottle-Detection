# src/utils.py

import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc, confusion_matrix, ConfusionMatrixDisplay

def yolo_to_xyxy(box, img_w, img_h):
    xc, yc, w, h = box
    x1 = int((xc - w / 2) * img_w)
    y1 = int((yc - h / 2) * img_h)
    x2 = int((xc + w / 2) * img_w)
    y2 = int((yc + h / 2) * img_h)
    return [x1, y1, x2, y2]

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def evaluate_detections(predictions, gt_dir, img_dir, iou_thresh=0.5, results_prefix=""):
    tp, fp, fn = 0, 0, 0
    y_scores, y_trues, y_preds = [], [], []
    
    os.makedirs('results', exist_ok=True)

    for img_file, dets in predictions.items():
        basename = os.path.splitext(img_file)[0]
        label_path = os.path.join(gt_dir, f"{basename}.txt")
        img_path = os.path.join(img_dir, img_file)

        if not os.path.exists(label_path) or not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        with open(label_path, 'r') as f:
            gt_boxes = [yolo_to_xyxy(list(map(float, line.strip().split()[1:])), w, h)
                        for line in f if line.startswith('0')]

        matched = [False] * len(gt_boxes)

        for det in dets:
            if det.get('pred', 0) != 1:
                continue

            pred_box = det['bbox']
            conf = det.get('conf', 0.5)
            max_iou = 0
            max_idx = -1

            for i, gt_box in enumerate(gt_boxes):
                iou = compute_iou(pred_box, gt_box)
                if iou > max_iou:
                    max_iou = iou
                    max_idx = i

            if max_iou >= iou_thresh and not matched[max_idx]:
                tp += 1
                matched[max_idx] = True
                y_trues.append(1)
                y_preds.append(1)
            else:
                fp += 1
                y_trues.append(0)
                y_preds.append(1)

            y_scores.append(conf)

        for unmatched in matched:
            if not unmatched:
                fn += 1
                y_trues.append(1)
                y_preds.append(0)
                y_scores.append(0)  # optional, to keep same length
    
    # Compute and Print Metrics
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    
    print(f"\nEvaluation Metrics for '{results_prefix}' (IoU ≥ {iou_thresh:.2f}):")
    print(f"TP: {tp}, FP: {fp}, FN: {fn}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    # Confusion Matrix
    if y_trues and y_preds:
        cm = confusion_matrix(y_trues, y_preds, labels=[1,0])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["bottle", "background"])
        # disp.plot(cmap='Blues')
        # plt.title("Hybrid Confusion Matrix (TP/FP/FN)")
        # plt.tight_layout()
        # plt.savefig("results/hybrid_confusion_matrix.png")
        # plt.close()
        # Plot with predicted on Y-axis and true on X-axis
        fig, ax = plt.subplots(figsize=(6, 5))
        disp.plot(cmap='Blues', ax=ax, values_format='d')
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        ax.set_title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig("results/hybrid_confusion_matrix.png")
        plt.close()

        print("Confusion Matrix saved to results/hybrid_confusion_matrix.png")

    # PR Curve & AP
    if y_scores and y_trues:
        precision_curve, recall_curve, _ = precision_recall_curve(y_trues, y_scores)
        ap = auc(recall_curve, precision_curve)
        print(f"AP (area under PR curve): {ap:.4f}")
        plt.figure()
        plt.plot(recall_curve, precision_curve, label=f"PR Curve (AP={ap:.4f})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Hybrid Precision-Recall Curve")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig("results/hybrid_pr_curve.png")
        plt.close()
        print("PR Curve saved to results/hybrid_pr_curve.png")
    else:
        ap = None

    return {
        "TP": tp, "FP": fp, "FN": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "ap": ap
    }

