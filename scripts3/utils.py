import numpy as np
import os
import glob
import matplotlib.pyplot as plt

def xywh2xyxy(x):
    """
    Convert bounding box format from center (x, y, w, h) to corner (x_min, y_min, x_max, y_max).
    """
    y = np.zeros_like(x)
    y[0] = x[0] - x[2] / 2  # x_min
    y[1] = x[1] - x[3] / 2  # y_min
    y[2] = x[0] + x[2] / 2  # x_max
    y[3] = x[1] + x[3] / 2  # y_max
    return y

def box_iou(box1, box2, eps=1e-7):
    """
    Calculate intersection-over-union (IoU) of two bounding boxes.
    """
    box1 = np.array(box1)
    box2 = np.array(box2)

    inter_xmin = max(box1[0], box2[0])
    inter_ymin = max(box1[1], box2[1])
    inter_xmax = min(box1[2], box2[2])
    inter_ymax = min(box1[3], box2[3])

    inter_width = max(inter_xmax - inter_xmin, 0)
    inter_height = max(inter_ymax - inter_ymin, 0)
    inter_area = inter_width * inter_height

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    iou = inter_area / (union_area + eps)

    return iou

def evaluate_predictions(pred_folder, gt_folder, conf_th=0.1):
    nb_fp, nb_tp, nb_fn = 0, 0, 0

    gt_filenames = [
        os.path.splitext(os.path.basename(f))[0]
        for f in glob.glob(os.path.join(gt_folder, "*.txt"))
    ]
    pred_filenames = [
        os.path.splitext(os.path.basename(f))[0]
        for f in glob.glob(os.path.join(pred_folder, "*.txt"))
    ]

    all_filenames = set(gt_filenames + pred_filenames)
    
    for filename in all_filenames:
        gt_file = os.path.join(gt_folder, f"{filename}.txt")
        pred_file = os.path.join(pred_folder, f"{filename}.txt")

        gt_boxes = []
        if os.path.isfile(gt_file) and os.path.getsize(gt_file) > 0:
            with open(gt_file, "r") as f:
                gt_boxes = [
                    xywh2xyxy(np.array(line.strip().split(" ")[1:5]).astype(float))
                    for line in f.readlines()
                ]

        gt_matches = np.zeros(len(gt_boxes), dtype=bool)

        if os.path.isfile(pred_file) and os.path.getsize(pred_file) > 0:
            with open(pred_file, "r") as f:
                pred_boxes = [line.strip().split(" ") for line in f.readlines()]

            for pred_box in pred_boxes:
                try:
                    _, x, y, w, h, conf = map(float, pred_box)
                except:
                    print(pred_file)
                if conf < conf_th:
                    continue
                pred_box = xywh2xyxy(np.array([x, y, w, h]))

                if gt_boxes:
                    matches = [box_iou(pred_box, gt_box) > 0.1 for gt_box in gt_boxes]
                    if any(matches):
                        nb_tp += 1
                        gt_matches = gt_matches | matches
                    else:
                        nb_fp += 1
                else:
                    nb_fp += 1

        if gt_boxes:
            nb_fn += len(gt_boxes) - np.sum(gt_matches)

    precision = nb_tp / (nb_tp + nb_fp) if (nb_tp + nb_fp) > 0 else 0
    recall = nb_tp / (nb_tp + nb_fn) if (nb_tp + nb_fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {"precision": precision, "recall": recall, "f1_score": f1_score}

def find_best_conf_threshold_and_plot(pred_folder, gt_folder, conf_thres_range):
    f1_scores, precisions, recalls = [], [], []

    for conf_thres in conf_thres_range:
        results = evaluate_predictions(pred_folder, gt_folder, conf_thres)
        f1_scores.append(results["f1_score"])
        precisions.append(results["precision"])
        recalls.append(results["recall"])

    best_idx = np.argmax(f1_scores)
    best_conf_thres = conf_thres_range[best_idx]
    best_f1_score = f1_scores[best_idx]
    best_precision = precisions[best_idx]
    best_recall = recalls[best_idx]

    plt.figure(figsize=(10, 6))
    plt.plot(conf_thres_range, f1_scores, label="F1 Score", color="blue", marker="o")
    plt.plot(conf_thres_range, precisions, label="Precision", color="green", linestyle="--")
    plt.plot(conf_thres_range, recalls, label="Recall", color="red", linestyle="-.")

    plt.scatter(best_conf_thres, best_f1_score, color="blue", s=100, edgecolor="k", zorder=5)
    plt.scatter(best_conf_thres, best_precision, color="green", s=100, edgecolor="k", zorder=5)
    plt.scatter(best_conf_thres, best_recall, color="red", s=100, edgecolor="k", zorder=5)

    plt.text(best_conf_thres, best_f1_score, f" Best F1: {best_f1_score:.2f}\n Precision: {best_precision:.2f}\n Recall: {best_recall:.2f}", fontsize=9, verticalalignment="bottom")

    plt.title("F1 Score, Precision, and Recall vs. Confidence Threshold")
    plt.xlabel("Confidence Threshold")
    plt.ylabel("Metric Value")
    plt.legend()
    plt.grid(True)

    # Save the plot to the prediction folder
    plot_path = os.path.join(pred_folder, 'threshold_metrics_plot.png')
    plt.savefig(plot_path)
    plt.close()

    print(f"Plot saved at: {plot_path}")

    return best_conf_thres, best_f1_score, best_precision, best_recall
